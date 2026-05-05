#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_CODE_DIR="${ROOT}/AgentGym-RL"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"

MODEL_PATH="${MODEL_PATH:-${ROOT}/models/Qwen2.5-7B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/AgentItemId/alfworld_wm_sft_train.json}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"
if (( NUM_GPUS < 1 )); then
  echo "CUDA_VISIBLE_DEVICES is empty."
  exit 1
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "WM SFT train file not found: ${TRAIN_FILE}"
  exit 1
fi

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-xuyan923r-renmin-university-of-china}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
PROJECT_NAME="${PROJECT_NAME:-ALFWorld-WM-SFT}"

# Faster 4-GPU defaults tuned for throughput.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-8}"
MAX_LENGTH="${MAX_LENGTH:-3072}"
LR="${LR:-2e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29541}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
VERL_SFT_LOGGING_LEVEL="${VERL_SFT_LOGGING_LEVEL:-WARN}"

TMPDIR="${TMPDIR:-/idfsdata/yexuyan/te}"
TMP="${TMP:-${TMPDIR}}"
TEMP="${TEMP:-${TMPDIR}}"
HF_HOME="${HF_HOME:-/idfsdata/yexuyan/he}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/idfsdata/yexuyan/xe}"
WANDB_DIR="${WANDB_DIR:-/idfsdata/yexuyan/we}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/.cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/.config}"

EXP_NAME="${EXP_NAME:-ALFWORLD_WM_SFT_FAST4_$(basename "${MODEL_PATH}")_$(date -u +%Y%m%d_%H%M%S)}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/${EXP_NAME}}"
RUN_DIR="${RUN_DIR:-${ROOT}/runlogs/${EXP_NAME}}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/train.log}"

mkdir -p "${CKPT_DIR}" "${RUN_DIR}" "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

source "${CONDA_SH}"
set +u
conda activate "${TRAIN_ENV}"
set -u

cd "${TRAIN_CODE_DIR}"
exec env \
  -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  TMPDIR="${TMPDIR}" \
  TMP="${TMP}" \
  TEMP="${TEMP}" \
  HF_HOME="${HF_HOME}" \
  TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
  WANDB_DIR="${WANDB_DIR}" \
  WANDB_CACHE_DIR="${WANDB_CACHE_DIR}" \
  WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_BASE_URL="${WANDB_BASE_URL}" \
  VERL_SFT_LOGGING_LEVEL="${VERL_SFT_LOGGING_LEVEL}" \
  torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" --master_port="${MAIN_PROCESS_PORT}" \
    -m verl.agent_trainer.fsdp_sft_trainer \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.train_files="${TRAIN_FILE}" \
    data.max_length="${MAX_LENGTH}" \
    model.partial_pretrain="${MODEL_PATH}" \
    model.enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING}" \
    optim.lr="${LR}" \
    ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
    use_remove_padding="${USE_REMOVE_PADDING}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.logger="['console','wandb']" \
  2>&1 | tee "${LOG_PATH}"
