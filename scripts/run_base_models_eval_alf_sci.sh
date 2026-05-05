#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

ALF_ENV_PORT="${ALF_ENV_PORT:-36111}"
SCI_ENV_PORT="${SCI_ENV_PORT:-36105}"
ALF_ENV_ADDR="${ALF_ENV_ADDR:-http://127.0.0.1:${ALF_ENV_PORT}}"
SCI_ENV_ADDR="${SCI_ENV_ADDR:-http://127.0.0.1:${SCI_ENV_PORT}}"

RUN_TAG="${RUN_TAG:-base_eval_0123_$(date -u +%Y%m%d_%H%M%S)}"
RUNTIME_DIR="${RUNTIME_DIR:-${ROOT}/runtime/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${RUNTIME_DIR}/logs}"
RESULTS_DIR="${RESULTS_DIR:-${RUNTIME_DIR}/results}"

HOME_DIR="${HOME_DIR:-${RUNTIME_DIR}/home}"
TMPDIR="${TMPDIR:-${RUNTIME_DIR}/tmp}"
TMP="${TMP:-${TMPDIR}}"
TEMP="${TEMP:-${TMPDIR}}"
HF_HOME="${HF_HOME:-${RUNTIME_DIR}/hf}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_DIR}/xdg}"
WANDB_DIR="${WANDB_DIR:-${RUNTIME_DIR}/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/.cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/.config}"
RAY_REAL_DIR="${RAY_REAL_DIR:-${RUNTIME_DIR}/ray}"
RAY_SHORT_LINK="${RAY_SHORT_LINK:-/tmp/rb_eval_${PPID}}"
RAY_TMPDIR="${RAY_TMPDIR:-${RAY_SHORT_LINK}}"

ALF_RESULTS_FILE="${ALF_RESULTS_FILE:-${RESULTS_DIR}/FinalResults_ALFWorld_base_3b7b.jsonl}"
SCI_RESULTS_FILE="${SCI_RESULTS_FILE:-${RESULTS_DIR}/FinalResults_SciWorld_base_3b7b.jsonl}"

BASE_3B_MODEL_PATH="${BASE_3B_MODEL_PATH:-/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-3B-Instruct}"
BASE_7B_MODEL_PATH="${BASE_7B_MODEL_PATH:-/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-7B-Instruct}"
EMPTY_CKPT_3B_DIR="${EMPTY_CKPT_3B_DIR:-${RUNTIME_DIR}/empty_ckpts/3b}"
EMPTY_CKPT_7B_DIR="${EMPTY_CKPT_7B_DIR:-${RUNTIME_DIR}/empty_ckpts/7b}"

mkdir -p \
  "${RUNTIME_DIR}" \
  "${LOG_DIR}" \
  "${RESULTS_DIR}" \
  "${HOME_DIR}" \
  "${TMPDIR}" \
  "${HF_HOME}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${RAY_REAL_DIR}" \
  "${EMPTY_CKPT_3B_DIR}" \
  "${EMPTY_CKPT_7B_DIR}"

ln -sfn "${RAY_REAL_DIR}" "${RAY_SHORT_LINK}"

wait_ready() {
  local addr="$1"
  local name="$2"
  for _ in $(seq 1 180); do
    if curl --noproxy '*' -sf "${addr}/" >/dev/null; then
      echo "${name} ready: ${addr}"
      return 0
    fi
    sleep 2
  done
  echo "${name} is not ready: ${addr}" >&2
  return 1
}

echo "Runtime dir: ${RUNTIME_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "ALFWorld env: ${ALF_ENV_ADDR}"
echo "SciWorld env: ${SCI_ENV_ADDR}"

wait_ready "${ALF_ENV_ADDR}" "ALFWorld service"
wait_ready "${SCI_ENV_ADDR}" "SciWorld service"

export HOME="${HOME_DIR}"
export CONDA_SH
export TRAIN_ENV
export CUDA_VISIBLE_DEVICES
export TMPDIR TMP TEMP
export HF_HOME TRANSFORMERS_CACHE HF_DATASETS_CACHE XDG_CACHE_HOME
export WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR
export RAY_TMPDIR

echo "=== Running ALFWorld base model evals (3B + 7B) ==="
ENV_ADDR="${ALF_ENV_ADDR}" \
RESULTS_FILE="${ALF_RESULTS_FILE}" \
OVERWRITE_RESULTS=1 \
SKIP_DONE_MODELS=0 \
BASE_3B_MODEL_PATH="${BASE_3B_MODEL_PATH}" \
BASE_7B_MODEL_PATH="${BASE_7B_MODEL_PATH}" \
CKPT_ROOT_3B="${EMPTY_CKPT_3B_DIR}" \
CKPT_ROOT_7B="${EMPTY_CKPT_7B_DIR}" \
bash "${ROOT}/scripts/batch_eval_alfworld_ckpts.sh" \
  | tee "${LOG_DIR}/eval_alfworld_base_models.log"

echo "=== Running SciWorld base model evals (3B) ==="
ENV_ADDR="${SCI_ENV_ADDR}" \
RESULTS_FILE="${SCI_RESULTS_FILE}" \
OVERWRITE_RESULTS=1 \
SKIP_DONE_MODELS=0 \
INCLUDE_BASE_MODEL=1 \
BASE_MODEL_LABEL="base_3b" \
BASE_MODEL_PATH="${BASE_3B_MODEL_PATH}" \
CKPT_STEPS_STR="" \
bash "${ROOT}/scripts/batch_eval_sciworld_ckpts.sh" \
  | tee "${LOG_DIR}/eval_sciworld_base_3b.log"

echo "=== Running SciWorld base model evals (7B) ==="
ENV_ADDR="${SCI_ENV_ADDR}" \
RESULTS_FILE="${SCI_RESULTS_FILE}" \
OVERWRITE_RESULTS=0 \
SKIP_DONE_MODELS=0 \
INCLUDE_BASE_MODEL=1 \
BASE_MODEL_LABEL="base_7b" \
BASE_MODEL_PATH="${BASE_7B_MODEL_PATH}" \
CKPT_STEPS_STR="" \
bash "${ROOT}/scripts/batch_eval_sciworld_ckpts.sh" \
  | tee "${LOG_DIR}/eval_sciworld_base_7b.log"

echo "All evaluations finished."
echo "ALFWorld results: ${ALF_RESULTS_FILE}"
echo "SciWorld results: ${SCI_RESULTS_FILE}"
