#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CODE_DIR="${ROOT}/AgentGym-RL"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"
MODEL_PATH="${MODEL_PATH:-/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-3B-Instruct}"
TASK_NAME="alfworld"

ENV_ADDR="${ENV_ADDR:-http://127.0.0.1:36001}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"
if (( NUM_GPUS < 1 )); then
  echo "CUDA_VISIBLE_DEVICES is empty."
  exit 1
fi

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-xuyan923r-renmin-university-of-china}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
PROJECT_NAME="${PROJECT_NAME:-ALFWorld}"
HOME="${HOME:-${ROOT}/runtime/alfworld_train_home}"

# Keep defaults aligned with the historical ALFWorld GRPO score run,
# except this script defaults to the 3B model with wm loss enabled.
KL_COEF="${KL_COEF:-0.001}"
POLICY_LR="${POLICY_LR:-1e-6}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
MAX_ROUNDS="${MAX_ROUNDS:-30}"
AGENT_MAX_ROUNDS="${AGENT_MAX_ROUNDS:-10}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-6000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-200}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}"
REWARD_MODE="${REWARD_MODE:-score}"
ORM_SUCCESS_SCORE="${ORM_SUCCESS_SCORE:-100.0}"
PAD_TO_MAX_RESPONSE_LENGTH="${PAD_TO_MAX_RESPONSE_LENGTH:-1}"
SAVE_FREQ="${SAVE_FREQ:-50}"
REMOVE_PREVIOUS_CKPT_IN_SAVE="${REMOVE_PREVIOUS_CKPT_IN_SAVE:-0}"
MAX_LOCAL_CKPT_TO_KEEP="${MAX_LOCAL_CKPT_TO_KEEP:-6}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-0}"

ENABLE_WMC="${ENABLE_WMC:-1}"
WMC_COEFF="${WMC_COEFF:-0.001}"
ENABLE_ERC="${ENABLE_ERC:-0}"
ERC_MU_BASE="${ERC_MU_BASE:-1.0}"
ERC_MU_EXP="${ERC_MU_EXP:-2.0}"
ERC_ETA_WM="${ERC_ETA_WM:-3.0}"
ERC_LAMBDA_WM="${ERC_LAMBDA_WM:-1.0}"
ERC_CLIPPING_TYPE="${ERC_CLIPPING_TYPE:-global}"
ERC_CLIPPING_METHOD="${ERC_CLIPPING_METHOD:-mask}"
ERC_MOMENTUM="${ERC_MOMENTUM:-0.9}"

# Keep all runtime artifacts on /idfsdata.
RAY_TMPDIR="${RAY_TMPDIR:-/idfsdata/yexuyan/ra}"
TMPDIR="${TMPDIR:-/idfsdata/yexuyan/te}"
TMP="${TMP:-${TMPDIR}}"
TEMP="${TEMP:-${TMPDIR}}"
HF_HOME="${HF_HOME:-/idfsdata/yexuyan/he}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/idfsdata/yexuyan/xe}"
WANDB_DIR="${WANDB_DIR:-/idfsdata/yexuyan/we}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/.cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/.config}"
WANDB_DATA_DIR="${WANDB_DATA_DIR:-${WANDB_DIR}/.local/share}"

EXP_NAME="${EXP_NAME:-ALFWORLD_GRPO_Qwen25_3B_SCORE_WM_LEN6000_BS8_0123_$(date -u +%Y%m%d_%H%M%S)}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/${EXP_NAME}}"
RUN_DIR="${RUN_DIR:-${ROOT}/runlogs/${EXP_NAME}}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${RUN_DIR}/rollout_logs}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/AgentItemId/alfworld_train.json}"
LOG_PATH="${LOG_PATH:-}"

mkdir -p \
  "${HOME}" "${CKPT_DIR}" "${RUN_DIR}" "${ROLLOUT_LOG_DIR}" \
  "${RAY_TMPDIR}" "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" \
  "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"

if [[ -n "${LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${LOG_PATH}")"
  exec >"${LOG_PATH}" 2>&1
fi

REAL_TRAIN_BATCH_SIZE=$(( TRAIN_BATCH_SIZE * ROLLOUT_N ))
if (( REAL_TRAIN_BATCH_SIZE % NUM_GPUS != 0 )); then
  echo "train_batch_size * rollout_n must be divisible by number of visible GPUs."
  echo "train_batch_size=${TRAIN_BATCH_SIZE}, rollout_n=${ROLLOUT_N}, num_gpus=${NUM_GPUS}"
  exit 1
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "ALFWorld train file not found: ${TRAIN_FILE}"
  exit 1
fi

source "${CONDA_SH}"
set +u
conda activate "${TRAIN_ENV}"
set -u

WMC_COEFF_VALUE="0.0"
if [[ "${ENABLE_WMC}" == "1" ]]; then
  WMC_COEFF_VALUE="${WMC_COEFF}"
fi

ERC_ENABLE_VALUE="False"
if [[ "${ENABLE_ERC}" == "1" ]]; then
  ERC_ENABLE_VALUE="True"
fi

REMOVE_PREVIOUS_CKPT_IN_SAVE_VALUE="False"
if [[ "${REMOVE_PREVIOUS_CKPT_IN_SAVE}" == "1" ]]; then
  REMOVE_PREVIOUS_CKPT_IN_SAVE_VALUE="True"
fi

RESUME_FROM_PATH_VALUE="False"
if [[ "${RESUME_FROM_PATH}" == "1" ]]; then
  RESUME_FROM_PATH_VALUE="True"
fi

PAD_TO_MAX_RESPONSE_LENGTH_VALUE="True"
if [[ "${PAD_TO_MAX_RESPONSE_LENGTH}" == "0" ]]; then
  PAD_TO_MAX_RESPONSE_LENGTH_VALUE="False"
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
  HOME="${HOME}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  RAY_TMPDIR="${RAY_TMPDIR}" \
  TMPDIR="${TMPDIR}" \
  TMP="${TMP}" \
  TEMP="${TEMP}" \
  HF_HOME="${HF_HOME}" \
  TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
  WANDB_DIR="${WANDB_DIR}" \
  WANDB_CACHE_DIR="${WANDB_CACHE_DIR}" \
  WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR}" \
  WANDB_DATA_DIR="${WANDB_DATA_DIR}" \
  VLLM_USE_MODELSCOPE=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  HYDRA_FULL_ERROR=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_BASE_URL="${WANDB_BASE_URL}" \
  python -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds="${MAX_ROUNDS}" \
    data.train_file="${TRAIN_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.agentgym.task_name="${TASK_NAME}" \
    actor_rollout_ref.agentgym.env_addr="${ENV_ADDR}" \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.agentgym.max_rounds="${AGENT_MAX_ROUNDS}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.world_model_coeff="${WMC_COEFF_VALUE}" \
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
    actor_rollout_ref.actor.optim.lr="${POLICY_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_tokens="${MAX_TOKENS_PER_TURN}" \
    +actor_rollout_ref.rollout.reward_mode="${REWARD_MODE}" \
    +actor_rollout_ref.rollout.orm_success_score="${ORM_SUCCESS_SCORE}" \
    +actor_rollout_ref.rollout.pad_to_max_response_length="${PAD_TO_MAX_RESPONSE_LENGTH_VALUE}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.rollout_log_dir="${ROLLOUT_LOG_DIR}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    wmc_erc.enable="${ERC_ENABLE_VALUE}" \
    wmc_erc.mu_base="${ERC_MU_BASE}" \
    wmc_erc.mu_exp="${ERC_MU_EXP}" \
    wmc_erc.eta_wm="${ERC_ETA_WM}" \
    wmc_erc.lambda_wm="${ERC_LAMBDA_WM}" \
    wmc_erc.clipping_type="${ERC_CLIPPING_TYPE}" \
    wmc_erc.clipping_method="${ERC_CLIPPING_METHOD}" \
    wmc_erc.momentum="${ERC_MOMENTUM}" \
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
    trainer.n_gpus_per_node="${NUM_GPUS}"
