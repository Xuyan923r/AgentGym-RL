# Copyright 2024 Bytedance Ltd. and/or its affiliates

import hydra
import ray

from verl.agent_trainer.rwml.alfworld_trainer import ALFWorldRWMLTrainer


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_rwml(config)


def run_rwml(config):
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})
    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    from pprint import pprint

    from omegaconf import OmegaConf

    from verl.agent_trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.workers.agent_fsdp_workers import ActorRolloutRefWorker

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = ALFWorldRWMLTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
