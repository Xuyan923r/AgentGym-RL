from typing import List

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask
from verl.workers.rollout.base import BaseRollout


class ALFWorldRWMLHFRollout(BaseRollout):
    """HF rollout for offline ALFWorld RWML triplet generation."""

    def __init__(self, module: nn.Module, config, tokenizer, reward_scorer):
        super().__init__()
        self.config = config
        self.module = module
        self.tokenizer = tokenizer
        self.reward_scorer = reward_scorer

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]
        rollout_n = int(getattr(self.config, "n", 1) or 1)
        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        if rollout_n > 1:
            idx = idx.repeat_interleave(rollout_n, dim=0)
            attention_mask = attention_mask.repeat_interleave(rollout_n, dim=0)
            position_ids = position_ids.repeat_interleave(rollout_n, dim=0)
        expanded_batch_size = idx.size(0)

        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = prompts.meta_info.get("top_k", self.config.get("top_k", 0))
        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)
        temperature = prompts.meta_info.get("temperature", self.config.temperature)

        generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

        self.module.eval()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        else:
            class _NullCtx:
                def __enter__(self):
                    return None
                def __exit__(self, exc_type, exc_val, exc_tb):
                    return False
            param_ctx = _NullCtx()

        with param_ctx:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.module.generate(
                    input_ids=idx,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    output_scores=False,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
        seq = output.sequences

        target_length = prompt_length + response_length
        if seq.shape[1] < target_length:
            pad = torch.full(
                size=(expanded_batch_size, target_length - seq.shape[1]),
                fill_value=pad_token_id,
                dtype=seq.dtype,
                device=seq.device,
            )
            seq = torch.cat((seq, pad), dim=1)

        prompt = seq[:, :prompt_length]
        response = seq[:, prompt_length:]

        delta_position_id = torch.arange(1, response.size(1) + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(expanded_batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        full_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        response_mask = response_attention_mask.clone()
        reward_tensor = torch.zeros_like(response, dtype=torch.float32)
        raw_reward_tensor = torch.zeros_like(response, dtype=torch.float32)

        valid_lengths = response_attention_mask.sum(dim=-1)
        decoded_responses: List[str] = []
        references: List[str] = []
        repeated_targets: List[str] = []
        for target in prompts.non_tensor_batch["rwml_target_next_observation"]:
            repeated_targets.extend([target] * rollout_n)

        for i in range(expanded_batch_size):
            valid_len = int(valid_lengths[i].item())
            valid_ids = response[i, :valid_len]
            decoded_responses.append(self.tokenizer.decode(valid_ids, skip_special_tokens=True))
            references.append(repeated_targets[i])
        scores = self.reward_scorer.score_texts(decoded_responses, references)

        for i, score in enumerate(scores):
            valid_len = int(valid_lengths[i].item())
            if valid_len <= 0:
                continue
            reward_tensor[i, valid_len - 1] = score
            raw_reward_tensor[i, valid_len - 1] = score

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": full_attention_mask,
                "position_ids": full_position_ids,
                "response_mask": response_mask,
                "scores": reward_tensor,
                "task_scores": reward_tensor,
                "task_raw_scores": raw_reward_tensor,
                "task_rounds": torch.ones(expanded_batch_size, dtype=torch.float32, device=seq.device),
                "task_dones": torch.ones(expanded_batch_size, dtype=torch.float32, device=seq.device),
            },
            batch_size=expanded_batch_size,
        )

        non_tensors = {
            "generated_text": np.array(decoded_responses, dtype=object),
            "ground_truth": np.array(references, dtype=object),
        }

        self.module.train()
        torch.cuda.empty_cache()
        return DataProto(batch=batch, non_tensor_batch=non_tensors, meta_info={})
