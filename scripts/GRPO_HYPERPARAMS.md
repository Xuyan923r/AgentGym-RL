# AgentGym-RL GRPO Hyperparameters (ALFWorld / SciWorld / WebShop)

> Export time (UTC): 2026-04-18  
> Exported from this repo's scripts + historical run logs.

## 1) Sources

- SciWorld script defaults: `scripts/run_sciworld_grpo_train.sh`
- WebShop script defaults: `scripts/run_webshop_grpo_train.sh`
- ALFWorld score run (actual config dump):
  - `runlogs/ALFWORLD_GRPO_Qwen25_7B_SCORE_LEN6000_BS8_20260417_080919/train_restart_20260417_135819.log`
- ALFWorld ORM run (actual override list):
  - `runlogs/ALFWORLD_GRPO_Qwen25_7B_ORM_LEN6000_BS16_0123_20260417_104226/train.log`

---

## 2) Common GRPO Core (all three tasks)

These are the common algorithm/training knobs used in current scripts/runs:

- Trainer entry: `python -m verl.agent_trainer.main_ppo`
- `algorithm.adv_estimator=grpo`
- PPO/actor:
  - `actor_rollout_ref.actor.use_kl_loss=True`
  - `actor_rollout_ref.actor.kl_loss_type=low_var_kl`
  - `actor_rollout_ref.actor.optim.lr=1e-6`
  - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`
- Rollout backend:
  - `actor_rollout_ref.rollout.name=vllm`
  - `actor_rollout_ref.rollout.dtype=bfloat16`
  - `actor_rollout_ref.rollout.enforce_eager=True`
  - `actor_rollout_ref.rollout.free_cache_engine=True`
  - `actor_rollout_ref.rollout.load_format=dummy_dtensor`
  - `actor_rollout_ref.rollout.enable_chunked_prefill=True`
  - `actor_rollout_ref.rollout.tensor_model_parallel_size=1`
  - `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1`
  - `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1`
  - `actor_rollout_ref.rollout.max_num_batched_tokens=16384` (in training scripts/runs)
  - `actor_rollout_ref.rollout.max_num_seqs=128` (in training scripts/runs)
- KL controller:
  - `algorithm.kl_ctrl.kl_coef=0.001`
- WMC/ERC block in scripts (usually disabled):
  - `wmc_erc.enable=False`
  - `wmc_erc.mu_base=1.0`
  - `wmc_erc.mu_exp=2.0`
  - `wmc_erc.eta_wm=3.0`
  - `wmc_erc.lambda_wm=1.0`
  - `wmc_erc.clipping_type=global`
  - `wmc_erc.clipping_method=mask`
  - `wmc_erc.momentum=0.9`

---

## 3) ALFWorld

Note: current repo does not contain a dedicated `run_alfworld_grpo_train.sh` tracked script; the settings below are from your actual ALFWorld training runs.

### 3.1 ALFWorld Score mode (actual run)

Run ID:
- `ALFWORLD_GRPO_Qwen25_7B_SCORE_LEN6000_BS8_20260417_080919`

Resolved key parameters (from config dump in `train_restart_20260417_135819.log`):

- Task/env:
  - `actor_rollout_ref.agentgym.task_name=alfworld`
  - `actor_rollout_ref.agentgym.env_addr=http://127.0.0.1:36001`
  - `actor_rollout_ref.agentgym.timeout=600`
  - `actor_rollout_ref.agentgym.max_rounds=10` (agentgym block)
  - `algorithm.rounds_ctrl.type=fixed`
  - `algorithm.rounds_ctrl.rounds=30`
- Data:
  - `data.train_file=/idfsdata/yexuyan/AgentGym-RL/AgentItemId/alfworld_train.json`
  - `data.train_batch_size=8`
  - `data.max_prompt_length=1024`
  - `data.max_response_length=6000`
- Model:
  - `actor_rollout_ref.model.path=/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-7B-Instruct`
- PPO/optimization:
  - `actor_rollout_ref.actor.ppo_epochs=2`
  - `actor_rollout_ref.actor.ppo_mini_batch_size=2`
  - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`
  - `actor_rollout_ref.actor.optim.lr=1e-6`
  - `actor_rollout_ref.actor.kl_loss_coef=0.001`
  - `actor_rollout_ref.actor.world_model_coeff=0.0`
- Rollout:
  - `actor_rollout_ref.rollout.n=8`
  - `actor_rollout_ref.rollout.gpu_memory_utilization=0.70`
  - `actor_rollout_ref.rollout.max_model_len=16384`
  - `actor_rollout_ref.rollout.max_tokens=200`
  - `actor_rollout_ref.rollout.reward_mode=score`
  - `actor_rollout_ref.rollout.orm_success_score=100.0`
  - `+actor_rollout_ref.rollout.pad_to_max_response_length=True`
- Trainer/checkpoint:
  - `trainer.project_name=ALFWorld`
  - `trainer.experiment_name=ALFWORLD_GRPO_Qwen25_7B_SCORE_LEN6000_BS8_20260417_080919`
  - `trainer.default_local_dir=/idfsdata/yexuyan/AgentGym-RL/checkpoints/ALFWORLD_GRPO_Qwen25_7B_SCORE_LEN6000_BS8_20260417_080919`
  - `trainer.save_freq=50`
  - `trainer.max_local_ckpt_to_keep=6`
  - `trainer.remove_previous_ckpt_in_save=False`
  - `trainer.total_epochs=2`
  - `trainer.resume_mode=auto`
  - `trainer.resume_from_path=False`
  - `trainer.nnodes=1`
  - `trainer.n_gpus_per_node=4`

### 3.2 ALFWorld ORM mode (actual run)

Run ID:
- `ALFWORLD_GRPO_Qwen25_7B_ORM_LEN6000_BS16_0123_20260417_104226`

Resolved key parameters (from override list in log):

- Task/env:
  - `actor_rollout_ref.agentgym.task_name=alfworld`
  - `actor_rollout_ref.agentgym.env_addr=http://127.0.0.1:36016`
  - `actor_rollout_ref.agentgym.timeout=600`
  - `algorithm.rounds_ctrl.type=fixed`
  - `algorithm.rounds_ctrl.rounds=20`
- Data:
  - `data.train_file=/idfsdata/yexuyan/AgentGym-RL/AgentItemId/alfworld_train.json`
  - `data.train_batch_size=16`
  - `data.max_prompt_length=1024`
  - `data.max_response_length=6000`
- Model:
  - `actor_rollout_ref.model.path=/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-7B-Instruct`
- PPO/optimization:
  - `actor_rollout_ref.actor.ppo_epochs=2`
  - `actor_rollout_ref.actor.ppo_mini_batch_size=8`
  - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`
  - `actor_rollout_ref.actor.optim.lr=1e-6`
  - `actor_rollout_ref.actor.kl_loss_coef=0.001`
  - `actor_rollout_ref.actor.world_model_coeff=0.0`
- Rollout:
  - `actor_rollout_ref.rollout.n=8`
  - `actor_rollout_ref.rollout.gpu_memory_utilization=0.60`
  - `actor_rollout_ref.rollout.max_model_len=32768`
  - `actor_rollout_ref.rollout.max_tokens=200`
  - `actor_rollout_ref.rollout.reward_mode=orm_binary`
  - `actor_rollout_ref.rollout.orm_success_score=100`
  - `+actor_rollout_ref.rollout.pad_to_max_response_length=True`
- Trainer/checkpoint:
  - `trainer.project_name=ALFWorld`
  - `trainer.experiment_name=ALFWORLD_GRPO_Qwen25_7B_ORM_LEN6000_BS16_0123_20260417_104226`
  - `trainer.default_local_dir=/idfsdata/yexuyan/AgentGym-RL/checkpoints/ALFWORLD_GRPO_Qwen25_7B_ORM_LEN6000_BS16_0123_20260417_104226`
  - `trainer.save_freq=50`
  - `trainer.max_local_ckpt_to_keep=6`
  - `trainer.remove_previous_ckpt_in_save=False`
  - `trainer.total_epochs=2`
  - `trainer.resume_mode=auto`
  - `trainer.resume_from_path=False`
  - `trainer.nnodes=1`
  - `trainer.n_gpus_per_node=4`

---

## 4) SciWorld (script default template)

Source script:
- `scripts/run_sciworld_grpo_train.sh`

### 4.1 Runtime/env defaults

- `MODEL_PATH=/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-3B-Instruct`
- `ENV_ADDR=http://127.0.0.1:36005`
- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- `WANDB_MODE=offline`
- `PROJECT_NAME=agentgym-sciworld`
- `VLLM_ATTENTION_BACKEND=FLASH_ATTN`

### 4.2 Hyperparameters

- Algorithm/task:
  - `algorithm.adv_estimator=grpo`
  - `algorithm.rounds_ctrl.type=fixed`
  - `algorithm.rounds_ctrl.rounds=20`
  - `actor_rollout_ref.agentgym.task_name=sciworld`
  - `actor_rollout_ref.agentgym.timeout=600`
- Data:
  - `data.train_file=/idfsdata/yexuyan/AgentGym-RL/AgentItemId/sciworld_train.json`
  - `data.train_batch_size=16`
  - `data.max_prompt_length=1024`
  - `data.max_response_length=4096`
- PPO/optimization:
  - `actor_rollout_ref.actor.ppo_epochs=2`
  - `actor_rollout_ref.actor.ppo_mini_batch_size=8`
  - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`
  - `actor_rollout_ref.actor.optim.lr=1e-6`
  - `actor_rollout_ref.actor.kl_loss_coef=0.001`
- Rollout:
  - `actor_rollout_ref.rollout.n=8`
  - `actor_rollout_ref.rollout.gpu_memory_utilization=0.70`
  - `actor_rollout_ref.rollout.max_model_len=16384`
  - `actor_rollout_ref.rollout.max_tokens=200`
  - `actor_rollout_ref.rollout.reward_mode=score` (default)
  - `actor_rollout_ref.rollout.orm_success_score=100.0`
  - `+actor_rollout_ref.rollout.pad_to_max_response_length=True` (default)
- Trainer/checkpoint:
  - `trainer.save_freq=50`
  - `trainer.max_local_ckpt_to_keep=6`
  - `trainer.remove_previous_ckpt_in_save=False`
  - `trainer.total_epochs=2`
  - `trainer.resume_mode=auto`
  - `trainer.resume_from_path=False`
  - `trainer.nnodes=1`
  - `trainer.n_gpus_per_node=<len(CUDA_VISIBLE_DEVICES)>`

---

## 5) WebShop (script default template)

Source script:
- `scripts/run_webshop_grpo_train.sh`

### 5.1 Runtime/env defaults

- `MODEL_PATH=/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-3B-Instruct`
- `ENV_ADDR=http://127.0.0.1:8013`
- `CUDA_VISIBLE_DEVICES=4,5,6,7`
- `WANDB_MODE=offline`
- `WANDB_BASE_URL=https://api.wandb.ai`
- `PROJECT_NAME=agentgym-webshop`
- `VLLM_ATTENTION_BACKEND=FLASH_ATTN`

### 5.2 Hyperparameters

- Algorithm/task:
  - `algorithm.adv_estimator=grpo`
  - `algorithm.rounds_ctrl.type=fixed`
  - `algorithm.rounds_ctrl.rounds=15`
  - `actor_rollout_ref.agentgym.task_name=webshop`
  - `actor_rollout_ref.agentgym.timeout=2400`
- Data:
  - `data.train_file=/idfsdata/yexuyan/AgentGym-RL/AgentItemId/train/webshop_train.json`
  - `data.train_batch_size=16`
  - `data.max_prompt_length=768`
  - `data.max_response_length=8192`
- PPO/optimization:
  - `actor_rollout_ref.actor.ppo_epochs=2`
  - `actor_rollout_ref.actor.ppo_mini_batch_size=8`
  - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1`
  - `actor_rollout_ref.actor.optim.lr=1e-6`
  - `actor_rollout_ref.actor.kl_loss_coef=0.001`
- Rollout:
  - `actor_rollout_ref.rollout.n=8`
  - `actor_rollout_ref.rollout.gpu_memory_utilization=0.60`
  - `actor_rollout_ref.rollout.max_model_len=16384`
  - `actor_rollout_ref.rollout.max_tokens=256`
  - `actor_rollout_ref.rollout.reward_mode=score` (default)
  - `actor_rollout_ref.rollout.orm_success_score=100.0` (default)
- Trainer/checkpoint:
  - `trainer.save_freq=200`
  - `trainer.max_local_ckpt_to_keep=10`
  - `trainer.remove_previous_ckpt_in_save=False`
  - `trainer.total_epochs=2`
  - `trainer.nnodes=1`
  - `trainer.n_gpus_per_node=<len(CUDA_VISIBLE_DEVICES)>`

---

## 6) Step Counting Formula

For current GRPO trainer setup in these runs, effective training steps are usually:

- `steps_per_epoch = ceil(dataset_len / data.train_batch_size)`
- `total_steps = steps_per_epoch * trainer.total_epochs`

Example from ALFWorld score run log:
- `dataset len: 2420`
- `train_batch_size: 8`
- `steps_per_epoch: 302`
- `total_epochs: 2`
- `total_steps: 604`

---

## 7) Notes

- `train_batch_size * rollout_n` must be divisible by visible GPU count (enforced in scripts).
- In this repo's training scripts, `wmc_erc` is present but defaults to disabled.
- For ALFWorld, both `score` and `orm_binary` reward modes have been used in historical runs.
- Current evaluation code now uses task-specific success thresholds (ALFWorld/WebShop/BabyAI/SciWorld differ).
