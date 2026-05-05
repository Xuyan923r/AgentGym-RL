#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
MODEL_PATH="${MODEL_PATH:-/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-7B-Instruct}"
TRIPLET_FILE="${TRIPLET_FILE:-/idfsdata/yexuyan/AgentGym-RL/triplet_train_40k.jsonl}"
WM_SFT_FILE="${WM_SFT_FILE:-/idfsdata/yexuyan/AgentGym-RL/alfworld_wm_sft_train_40k.json}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-xuyan923r-renmin-university-of-china}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
PROJECT_NAME="${PROJECT_NAME:-ALFWorld-WM-SFT}"

# Faster 4-GPU defaults.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-8}"
MAX_LENGTH="${MAX_LENGTH:-3072}"
LR="${LR:-2e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29541}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-True}"

TMPDIR="${TMPDIR:-/idfsdata/yexuyan/te}"
TMP="${TMP:-${TMPDIR}}"
TEMP="${TEMP:-${TMPDIR}}"
HF_HOME="${HF_HOME:-/idfsdata/yexuyan/he}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/idfsdata/yexuyan/xe}"
WANDB_DIR="${WANDB_DIR:-/idfsdata/yexuyan/we}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/.cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/.config}"

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-ALFWORLD_WM_SFT_FAST4_Qwen25_7B_${RUN_TS}}"
PREP_SESSION="${PREP_SESSION:-alfworld_wm_sft_fast4_prep_${RUN_TS}}"
TRAIN_SESSION="${TRAIN_SESSION:-alfworld_wm_sft_fast4_${RUN_TS}}"
PREP_LOG="${ROOT}/runlogs/${EXP_NAME}/prep.log"
TRAIN_LOG="${ROOT}/runlogs/${EXP_NAME}/train.log"

mkdir -p "${ROOT}/runlogs/${EXP_NAME}" "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

if [[ ! -f "${TRIPLET_FILE}" ]]; then
  echo "Triplet file not found: ${TRIPLET_FILE}"
  exit 1
fi
if tmux has-session -t "${PREP_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${PREP_SESSION}"
  exit 1
fi
if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${TRAIN_SESSION}"
  exit 1
fi

tmux new-session -d -s "${PREP_SESSION}" \
  "cd ${ROOT} && python3 ${ROOT}/scripts/build_alfworld_wm_sft_data.py --input ${TRIPLET_FILE} --output ${WM_SFT_FILE} 2>&1 | tee ${PREP_LOG}"

for _ in $(seq 1 120); do
  if [[ -f "${WM_SFT_FILE}" ]]; then
    break
  fi
  sleep 2
done

if [[ ! -f "${WM_SFT_FILE}" ]]; then
  echo "WM SFT data was not created: ${WM_SFT_FILE}"
  exit 1
fi

tmux new-session -d -s "${TRAIN_SESSION}" \
  "cd ${ROOT} && CONDA_SH=${CONDA_SH} TRAIN_ENV=${TRAIN_ENV} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MODEL_PATH=${MODEL_PATH} TRAIN_FILE=${WM_SFT_FILE} WANDB_MODE=${WANDB_MODE} WANDB_ENTITY=${WANDB_ENTITY} WANDB_BASE_URL=${WANDB_BASE_URL} PROJECT_NAME=${PROJECT_NAME} TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU} MAX_LENGTH=${MAX_LENGTH} LR=${LR} TOTAL_EPOCHS=${TOTAL_EPOCHS} MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT} ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE} USE_REMOVE_PADDING=${USE_REMOVE_PADDING} ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING} TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} WANDB_DIR=${WANDB_DIR} WANDB_CACHE_DIR=${WANDB_CACHE_DIR} WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} EXP_NAME=${EXP_NAME} LOG_PATH=${TRAIN_LOG} bash ${ROOT}/scripts/run_alfworld_wm_sft_train_fast4.sh"

echo "Prep tmux session: ${PREP_SESSION}"
echo "Training tmux session: ${TRAIN_SESSION}"
echo "Prep log: ${PREP_LOG}"
echo "Training log: ${TRAIN_LOG}"
echo "WM SFT file: ${WM_SFT_FILE}"
