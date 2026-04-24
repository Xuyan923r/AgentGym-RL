#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PORT="${ENV_PORT:-36005}"
ENV_ADDR="http://127.0.0.1:${ENV_PORT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/models/Qwen2.5-7B-Instruct}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
PROJECT_NAME="${PROJECT_NAME:-agentgym-babyai}"
REWARD_MODE="${REWARD_MODE:-score}"
ORM_SUCCESS_SCORE="${ORM_SUCCESS_SCORE:-1.0}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
SAVE_FREQ="${SAVE_FREQ:-25}"
RUNTIME_BASE="${RUNTIME_BASE:-/idfsdata/yexuyan/rb}"
RAY_TMPDIR="${RAY_TMPDIR:-${RUNTIME_BASE}/ray_tmp}"
TMPDIR="${TMPDIR:-${RUNTIME_BASE}/tmp}"
TMP="${TMP:-${TMPDIR}}"
TEMP="${TEMP:-${TMPDIR}}"
HF_HOME="${HF_HOME:-${RUNTIME_BASE}/hf}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_BASE}/xdg_cache}"
WANDB_DIR="${WANDB_DIR:-${RUNTIME_BASE}/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/.cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/.config}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/idfsdata/yexuyan/babyai_mpl}"

BASE_MODEL_NAME="$(basename "${MODEL_PATH}")"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-BABYAI_GRPO_7B_SCORE_${RUN_TS}}"

ENV_SESSION="${ENV_SESSION:-babyai_env_${ENV_PORT}_${RUN_TS}}"
TRAIN_SESSION="${TRAIN_SESSION:-babyai_grpo_score_0123_${RUN_TS}}"
ENV_LOG="${ROOT}/runlogs/${ENV_SESSION}.log"
TRAIN_LOG="${ROOT}/runlogs/${EXP_NAME}/train.log"

mkdir -p \
  "${ROOT}/runlogs/${EXP_NAME}" \
  "${RAY_TMPDIR}" \
  "${TMPDIR}" \
  "${HF_HOME}" \
  "${TRANSFORMERS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${MPLCONFIGDIR}"

if tmux has-session -t "${ENV_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${ENV_SESSION}"
  exit 1
fi
if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${TRAIN_SESSION}"
  exit 1
fi

tmux new-session -d -s "${ENV_SESSION}" \
  "cd ${ROOT} && TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} XDG_CACHE_HOME=${XDG_CACHE_HOME} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} MPLCONFIGDIR=${MPLCONFIGDIR} HOST=127.0.0.1 PORT=${ENV_PORT} LOG_PATH=${ENV_LOG} bash ${ROOT}/scripts/run_babyai_env_service.sh"

for _ in $(seq 1 120); do
  if curl --noproxy '*' -sf "${ENV_ADDR}/" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl --noproxy '*' -sf "${ENV_ADDR}/" >/dev/null; then
  echo "BabyAI service did not become healthy on ${ENV_ADDR}"
  exit 1
fi

WARMUP_ID="$(curl --noproxy '*' --max-time 60 -sS -X POST "${ENV_ADDR}/create" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
curl --noproxy '*' -sS -X POST "${ENV_ADDR}/reset" \
  -H 'Content-Type: application/json' \
  -d "{\"id\": ${WARMUP_ID}, \"data_idx\": 0}" >/dev/null
curl --noproxy '*' -sS -X POST "${ENV_ADDR}/close" \
  -H 'Content-Type: application/json' \
  -d "{\"id\": ${WARMUP_ID}}" >/dev/null || true

tmux new-session -d -s "${TRAIN_SESSION}" \
  "cd ${ROOT} && TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} RAY_TMPDIR=${RAY_TMPDIR} XDG_CACHE_HOME=${XDG_CACHE_HOME} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} WANDB_DIR=${WANDB_DIR} WANDB_CACHE_DIR=${WANDB_CACHE_DIR} WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ENV_ADDR=${ENV_ADDR} MODEL_PATH=${MODEL_PATH} WANDB_MODE=${WANDB_MODE} WANDB_BASE_URL=${WANDB_BASE_URL} WANDB_ENTITY=${WANDB_ENTITY} PROJECT_NAME=${PROJECT_NAME} ROLLOUT_N=${ROLLOUT_N} TOTAL_EPOCHS=${TOTAL_EPOCHS} SAVE_FREQ=${SAVE_FREQ} REWARD_MODE=${REWARD_MODE} ORM_SUCCESS_SCORE=${ORM_SUCCESS_SCORE} EXP_NAME=${EXP_NAME} LOG_PATH=${TRAIN_LOG} bash ${ROOT}/scripts/run_babyai_grpo_train.sh"

echo "Environment tmux session: ${ENV_SESSION}"
echo "Training tmux session: ${TRAIN_SESSION}"
echo "BabyAI service: ${ENV_ADDR}"
echo "Environment log: ${ENV_LOG}"
echo "Training log: ${TRAIN_LOG}"
echo "Checkpoint dir: ${ROOT}/checkpoints/${EXP_NAME}"
echo "Warmup env id: ${WARMUP_ID}"
