from typing import List

import numpy as np
import torch
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.utils.rwml.alfworld_reward import ALFWorldRWMLRewardScorer
from verl.workers.rollout.agent_vllm_rollout.vllm_rollout import vLLMRollout
from verl.workers.rollout.schemas import RolloutHandler


class ALFWorldRWMLvLLMRollout(vLLMRollout):
    """vLLM rollout for offline ALFWorld RWML using embedding reward instead of env stepping."""

    def __init__(self, actor_module, rollout_config, agentgym_config, tokenizer, model_hf_config, reward_scorer: ALFWorldRWMLRewardScorer, **kwargs):
        super().__init__(
            actor_module=actor_module,
            rollout_config=rollout_config,
            agentgym_config=agentgym_config,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            **kwargs,
        )
        self.reward_scorer = reward_scorer

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        cur_device = prompts.batch["input_ids"].device
        batch_size = prompts.batch["input_ids"].size(0) * self.config.n
        rollout_handler_ls: List[RolloutHandler] = self.preprocess_prompt_to_rollout_handler(prompts, n=self.config.n)

        generation_prompt_idxs = [handler.get_generation_prompt(self.tokenizer) for handler in rollout_handler_ls]
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,
                prompt_token_ids=generation_prompt_idxs,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
        generated_token_ids = output[0].tolist()

        response_ids, response_attention_mask, response_position_ids, response_loss_mask, response_observation_mask = [], [], [], [], []
        scores, raw_scores, messages, task_dones = [], [], [], []

        repeated_targets: List[str] = []
        for target in prompts.non_tensor_batch["rwml_target_next_observation"]:
            repeated_targets.extend([target] * self.config.n)

        decoded_responses: List[str] = []
        for i, rollout_handler in enumerate(rollout_handler_ls):
            token_ids = generated_token_ids[i]
            content = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            rollout_handler.add_assistant_message(self.tokenizer, content)
            rollout_handler.truncate_output_ids()

            decoded_responses.append(content)
            response_ids.append(torch.tensor(rollout_handler.response_ids, dtype=torch.int, device=cur_device))
            response_attention_mask.append(torch.tensor(rollout_handler.response_attention_mask, dtype=torch.int, device=cur_device))
            response_position_ids.append(torch.tensor(rollout_handler.response_position_ids, dtype=torch.int, device=cur_device))
            response_loss_mask.append(torch.tensor(rollout_handler.response_loss_mask, dtype=torch.int, device=cur_device))
            response_observation_mask.append(torch.tensor(rollout_handler.response_observation_mask, dtype=torch.int, device=cur_device))
            messages.append(rollout_handler.messages)

        reward_scores = self.reward_scorer.score_texts(decoded_responses, repeated_targets)
        for score in reward_scores:
            raw_scores.append(float(score))
            scores.append(float(score))
            task_dones.append(1.0)

        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if self.pad_to_max_response_length and response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if self.pad_to_max_response_length and response_attention_mask.shape[1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if self.pad_to_max_response_length and response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
        response_observation_mask = pad_sequence(response_observation_mask, batch_first=True, padding_value=0)
        if self.pad_to_max_response_length and response_observation_mask.shape[1] < self.config.response_length:
            response_observation_mask = pad_sequence_to_length(response_observation_mask, self.config.response_length, 0)

        response_length = response_ids.size(1)
        delta_position_ids = torch.arange(1, response_length + 1, device=cur_device)
        delta_position_ids = delta_position_ids.unsqueeze(0).repeat(batch_size, 1)
        input_ids = prompts.batch["input_ids"].repeat_interleave(self.config.n, dim=0)
        attention_mask = prompts.batch["attention_mask"].repeat_interleave(self.config.n, dim=0)
        position_ids = prompts.batch["position_ids"].repeat_interleave(self.config.n, dim=0)
        response_position_ids = position_ids[:, -1:] + delta_position_ids

        seq = torch.cat((input_ids, response_ids), dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((position_ids, response_position_ids), dim=-1)
        response_mask = response_loss_mask
        observation_mask = response_attention_mask * (1 - response_mask)

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        raw_reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        valid_response_length = attention_mask[:, input_ids.size(-1):].sum(dim=-1)
        for i in range(len(scores)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
            raw_reward_tensor[i, valid_response_length[i].item() - 1] = raw_scores[i]

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask,
                "observation_mask": observation_mask,
                "scores": reward_tensor,
                "task_rounds": torch.ones(batch_size, dtype=torch.float32, device=input_ids.device),
                "task_scores": reward_tensor,
                "task_raw_scores": raw_reward_tensor,
                "task_dones": torch.tensor(task_dones, dtype=torch.float32).to(input_ids.device),
            },
            batch_size=batch_size,
        )

        non_tensor_batch = {
            "generated_text": np.array(decoded_responses, dtype=object),
            "ground_truth": np.array(repeated_targets, dtype=object),
        }

        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
