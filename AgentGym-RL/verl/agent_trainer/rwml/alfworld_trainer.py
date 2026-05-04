from copy import deepcopy
from typing import Dict

import numpy as np
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from verl import DataProto
from verl.agent_trainer.ppo import core_algos
from verl.agent_trainer.ppo.ray_trainer import (
    FixedRoundsScheduler,
    RayPPOTrainer,
    StepRoundsScheduler,
    _timer,
    compute_data_metrics,
    compute_timing_metrics,
    compute_advantage,
    reduce_metrics,
)
from verl.utils.agent_dataset.rl_dataset import collate_fn
from verl.utils.rwml.alfworld_triplet_dataset import (
    ALFWorldRWMLTripletDataset,
    collate_triplet_fn,
)


class ALFWorldRWMLTrainer(RayPPOTrainer):
    """Offline RWML trainer for ALFWorld using triplet data and embedding reward."""

    def _validate_config(self):
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        )
        print("[validate_config] RWML configuration checks passed.")

    def _create_dataloader(self):
        self.train_dataset = ALFWorldRWMLTripletDataset(
            data_file=self.config.data.train_file,
            tokenizer=self.tokenizer,
            data_config=self.config.data,
            rollout_config=self.config.actor_rollout_ref.rollout,
        )

        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            drop_last=True,
            collate_fn=collate_triplet_fn,
            sampler=sampler,
        )
        assert len(self.train_dataloader) >= 1

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps

        if self.config.algorithm.rounds_ctrl.type == "fixed":
            self.rounds_scheduler = FixedRoundsScheduler(rounds=self.config.algorithm.rounds_ctrl.rounds)
        elif self.config.algorithm.rounds_ctrl.type == "scaling_inter_stepwise":
            self.rounds_scheduler = StepRoundsScheduler(
                steps_scaling_inter=self.config.algorithm.rounds_ctrl.steps_scaling_inter,
                rounds_ls=self.config.algorithm.rounds_ctrl.rounds,
            )
        else:
            raise NotImplementedError

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        if self.config.trainer.storage_mode == "aistudio":
            self._save_checkpoint()
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: Dict[str, float] = {}
                timing_raw: Dict[str, float] = {}
                batch = DataProto.from_single_dict(batch_dict)

                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=[
                        "item_id",
                        "raw_prompt",
                        "reward_model",
                        "rwml_target_next_observation",
                        "rwml_task_name",
                        "rwml_action_text",
                        "rwml_turn_index",
                        "rwml_metadata",
                        "data_source",
                    ],
                )
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch.meta_info["max_rounds"] = self.rounds_scheduler.get_rounds()
                metrics.update({"max_rounds": self.rounds_scheduler.get_rounds()})

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(i) for i in range(len(batch.batch))],
                        dtype=object,
                    )
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with _timer("adv", timing_raw):
                        batch.batch["token_level_scores"] = batch.batch["scores"]
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )

                    with _timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1
                self.rounds_scheduler.step()

                if self.global_steps >= self.total_training_steps:
                    if self.config.trainer.save_freq > 0 and (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                    return
