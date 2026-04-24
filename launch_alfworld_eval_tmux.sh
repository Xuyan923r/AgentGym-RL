#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PORT="${ENV_PORT:-36001}"
ENV_ADDR="http://127.0.0.1:${ENV_PORT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
RESULTS_FILE="${RESULTS_FILE:-${ROOT}/FinalResults_ALFWorld.jsonl}"
OVERWRITE_RESULTS="${OVERWRITE_RESULTS:-1}"
SKIP_DONE_MODELS="${SKIP_DONE_MODELS:-0}"
MAX_ROUND="${MAX_ROUND:-30}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-38877}"
ALFWORLD_DATA="${ALFWORLD_DATA:-/idfsdata/yexuyan/alfworld_data}"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
ALFWORLD_ENV="${ALFWORLD_ENV:-/idfsdata/yexuyan/conda_envs/agentenv-alfworld}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"

BASE_3B_MODEL_PATH="${BASE_3B_MODEL_PATH:-/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-3B-Instruct}"
BASE_7B_MODEL_PATH="${BASE_7B_MODEL_PATH:-/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-7B-Instruct}"
CKPT_ROOT_3B="${CKPT_ROOT_3B:-/idfsdata/yexuyan/AgentGym-RL/checkpoints/ALFWORLD_GRPO_3B_SCORE_20260417_143823}"
CKPT_ROOT_7B="${CKPT_ROOT_7B:-/idfsdata/yexuyan/AgentGym-RL/checkpoints/ALFWORLD_GRPO_Qwen25_7B_SCORE_LEN6000_BS8_20260417_080919}"

# Use short runtime paths to avoid unix socket path-length issues.
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
ENV_SESSION="${ENV_SESSION:-alfworld_env_${ENV_PORT}_${RUN_TS}}"
EVAL_SESSION="${EVAL_SESSION:-alfworld_eval_${RUN_TS}}"
ENV_LOG="${ROOT}/runlogs/${ENV_SESSION}.log"
EVAL_LOG="${ROOT}/runlogs/${EVAL_SESSION}.log"

mkdir -p "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${ALFWORLD_DATA}" "${ROOT}/runlogs"

if tmux has-session -t "${ENV_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${ENV_SESSION}"
  exit 1
fi
if tmux has-session -t "${EVAL_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${EVAL_SESSION}"
  exit 1
fi

tmux new-session -d -s "${ENV_SESSION}" \
  "cd ${ROOT} && CONDA_SH=${CONDA_SH} ALFWORLD_ENV=${ALFWORLD_ENV} HOST=127.0.0.1 PORT=${ENV_PORT} LOG_PATH=${ENV_LOG} ALFWORLD_DATA=${ALFWORLD_DATA} TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} bash ${ROOT}/scripts/run_alfworld_env_service.sh"

for _ in $(seq 1 120); do
  if curl --noproxy '*' -sf "${ENV_ADDR}/" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl --noproxy '*' -sf "${ENV_ADDR}/" >/dev/null; then
  echo "ALFWorld service did not become healthy on ${ENV_ADDR}"
  exit 1
fi

tmux new-session -d -s "${EVAL_SESSION}" \
  "cd ${ROOT} && CONDA_SH=${CONDA_SH} TRAIN_ENV=${TRAIN_ENV} ENV_ADDR=${ENV_ADDR} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} RESULTS_FILE=${RESULTS_FILE} OVERWRITE_RESULTS=${OVERWRITE_RESULTS} SKIP_DONE_MODELS=${SKIP_DONE_MODELS} MAX_ROUND=${MAX_ROUND} EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE} MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT} BASE_3B_MODEL_PATH=${BASE_3B_MODEL_PATH} BASE_7B_MODEL_PATH=${BASE_7B_MODEL_PATH} CKPT_ROOT_3B=${CKPT_ROOT_3B} CKPT_ROOT_7B=${CKPT_ROOT_7B} TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} WANDB_DIR=${WANDB_DIR} WANDB_CACHE_DIR=${WANDB_CACHE_DIR} WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} bash ${ROOT}/scripts/batch_eval_alfworld_ckpts.sh 2>&1 | tee -a ${EVAL_LOG}"

echo "Environment tmux session: ${ENV_SESSION}"
echo "Evaluation tmux session: ${EVAL_SESSION}"
echo "ALFWorld service: ${ENV_ADDR}"
echo "Environment log: ${ENV_LOG}"
echo "Evaluation log: ${EVAL_LOG}"
echo "Results file: ${RESULTS_FILE}"
