#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CODE_DIR="${ROOT}/AgentGym-RL"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/models/Qwen2.5-7B-Instruct}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-${ROOT}/models/Qwen3-Embedding-8B}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/AgentItemId/alfworld_rwml_train.jsonl}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
VISIBLE_GPUS="${#GPU_ARRAY[@]}"
NUM_GPUS="${NUM_GPUS:-4}"
if (( VISIBLE_GPUS < NUM_GPUS )); then
  echo "Visible GPUs (${VISIBLE_GPUS}) is smaller than requested NUM_GPUS (${NUM_GPUS})."
  exit 1
fi
if (( NUM_GPUS < 1 )); then
  echo "CUDA_VISIBLE_DEVICES is empty."
  exit 1
fi

if [[ -z "${EMBEDDING_MODEL_PATH}" ]]; then
  echo "EMBEDDING_MODEL_PATH must be set for ALFWorld RWML training."
  exit 1
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "RWML train file not found: ${TRAIN_FILE}"
  exit 1
fi

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-xuyan923r-renmin-university-of-china}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
PROJECT_NAME="${PROJECT_NAME:-ALFWorld-RWML}"

KL_COEF="${KL_COEF:-0.001}"
POLICY_LR="${POLICY_LR:-1e-6}"
# Appendix B.1 ALFWorld RWML: GRPO group size = 8
ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}"
SAVE_FREQ="${SAVE_FREQ:-100}"
REMOVE_PREVIOUS_CKPT_IN_SAVE="${REMOVE_PREVIOUS_CKPT_IN_SAVE:-0}"
MAX_LOCAL_CKPT_TO_KEEP="${MAX_LOCAL_CKPT_TO_KEEP:-6}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-0}"
RWML_THRESHOLD="${RWML_THRESHOLD:-0.2}"
RWML_ROUND_STEP="${RWML_ROUND_STEP:-0.2}"
RWML_EMBED_MAX_LENGTH="${RWML_EMBED_MAX_LENGTH:-1024}"
RWML_RESPONSE_LENGTH="${RWML_RESPONSE_LENGTH:-256}"
RWML_TEMPERATURE="${RWML_TEMPERATURE:-1.0}"

RAY_TMPDIR="${RAY_TMPDIR:-/idfsdata/yexuyan/ra}"
TMPDIR="${TMPDIR:-/idfsdata/yexuyan/te}"
TMP="${TMP:-${TMPDIR}}"
TEMP="${TEMP:-${TMPDIR}}"
HF_HOME="${HF_HOME:-/idfsdata/yexuyan/he}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/idfsdata/yexuyan/xe}"
WANDB_DIR="${WANDB_DIR:-/idfsdata/yexuyan/we}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/.cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/.config}"

EXP_NAME="${EXP_NAME:-ALFWORLD_RWML_$(basename "${MODEL_PATH}")_$(date -u +%Y%m%d_%H%M%S)}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/${EXP_NAME}}"
RUN_DIR="${RUN_DIR:-${ROOT}/runlogs/${EXP_NAME}}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/train.log}"

mkdir -p \
  "${CKPT_DIR}" "${RUN_DIR}" \
  "${RAY_TMPDIR}" "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" \
  "${XDG_CACHE_HOME}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

source "${CONDA_SH}"
set +u
conda activate "${TRAIN_ENV}"
set -u

REMOVE_PREVIOUS_CKPT_IN_SAVE_VALUE="False"
if [[ "${REMOVE_PREVIOUS_CKPT_IN_SAVE}" == "1" ]]; then
  REMOVE_PREVIOUS_CKPT_IN_SAVE_VALUE="True"
fi

RESUME_FROM_PATH_VALUE="False"
if [[ "${RESUME_FROM_PATH}" == "1" ]]; then
  RESUME_FROM_PATH_VALUE="True"
fi

TOTAL_TRAINING_STEPS_ARGS=()
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  TOTAL_TRAINING_STEPS_ARGS+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

cd "${TRAIN_CODE_DIR}"
exec env \
  -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  NO_PROXY="${NO_PROXY}" \
  no_proxy="${no_proxy}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  RAY_TMPDIR="${RAY_TMPDIR}" \
  TMPDIR="${TMPDIR}" \
  TMP="${TMP}" \
  TEMP="${TEMP}" \
  HF_HOME="${HF_HOME}" \
  TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
  WANDB_DIR="${WANDB_DIR}" \
  WANDB_CACHE_DIR="${WANDB_CACHE_DIR}" \
  WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR}" \
  VLLM_USE_MODELSCOPE=0 \
  HYDRA_FULL_ERROR=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_BASE_URL="${WANDB_BASE_URL}" \
  python -m verl.agent_trainer.main_rwml_alfworld \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=1 \
    data.train_file="${TRAIN_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.return_raw_chat=True \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
    actor_rollout_ref.actor.optim.lr="${POLICY_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.name=alfworld_rwml_vllm \
    actor_rollout_ref.rollout.temperature="${RWML_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.response_length="${RWML_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.max_tokens="${RWML_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.rollout.rwml_reward.embedding_model_path="${EMBEDDING_MODEL_PATH}" \
    +actor_rollout_ref.rollout.rwml_reward.device="cuda:4" \
    +actor_rollout_ref.rollout.rwml_reward.threshold="${RWML_THRESHOLD}" \
    +actor_rollout_ref.rollout.rwml_reward.round_step="${RWML_ROUND_STEP}" \
    +actor_rollout_ref.rollout.rwml_reward.max_length="${RWML_EMBED_MAX_LENGTH}" \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.remove_previous_ckpt_in_save="${REMOVE_PREVIOUS_CKPT_IN_SAVE_VALUE}" \
    trainer.max_local_ckpt_to_keep="${MAX_LOCAL_CKPT_TO_KEEP}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.resume_from_path="${RESUME_FROM_PATH_VALUE}" \
    "${TOTAL_TRAINING_STEPS_ARGS[@]}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
  2>&1 | tee "${LOG_PATH}"
