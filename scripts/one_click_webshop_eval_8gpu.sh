#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/idfsdata/yexuyan/AgentGym-RL}"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
WEBSHOP_ENV="${WEBSHOP_ENV:-/idfsdata/yexuyan/conda_envs/agentenv-webshop}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ENV_PORT="${ENV_PORT:-8015}"
ENV_ADDR="${ENV_ADDR:-http://127.0.0.1:${ENV_PORT}}"

BASE_MODEL_3B="${BASE_MODEL_3B:-${ROOT}/models/Qwen2.5-3B-Instruct}"
BASE_MODEL_7B="${BASE_MODEL_7B:-${ROOT}/models/Qwen2.5-7B-Instruct}"
CKPT_ROOT_3B="${CKPT_ROOT_3B:-${ROOT}/checkpoints/WEBSHOP_GRPO_3B_SCORE_SF50_20260418_171148}"
CKPT_ROOT_7B="${CKPT_ROOT_7B:-${ROOT}/checkpoints/WEBSHOP_GRPO_7B_SCORE_20260418_055839}"

INCLUDE_BASE_MODELS="${INCLUDE_BASE_MODELS:-1}"
OVERWRITE_RESULTS="${OVERWRITE_RESULTS:-1}"
SKIP_DONE_MODELS="${SKIP_DONE_MODELS:-0}"
EVAL_BATCH_SIZE_3B="${EVAL_BATCH_SIZE_3B:-160}"
EVAL_BATCH_SIZE_7B="${EVAL_BATCH_SIZE_7B:-96}"

MAX_ROUNDS="${MAX_ROUNDS:-15}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-2400}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-768}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-256}"
N_SAMPLES="${N_SAMPLES:-1}"

# Keep runtime/cache paths short and on /idfsdata to avoid AF_UNIX path limit and /tmp writes.
RUNTIME_BASE="${RUNTIME_BASE:-/idfsdata/yexuyan/r}"
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

mkdir -p \
  "${RUNTIME_BASE}" \
  "${RAY_TMPDIR}" \
  "${TMPDIR}" \
  "${HF_HOME}" \
  "${TRANSFORMERS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${ROOT}/runlogs"

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
ENV_SESSION="${ENV_SESSION:-webshop_env_eval_${ENV_PORT}_${RUN_TS}}"
EVAL_SESSION="${EVAL_SESSION:-webshop_eval_8gpu_${RUN_TS}}"
ENV_LOG="${ENV_LOG:-${ROOT}/runlogs/${ENV_SESSION}.log}"
EVAL_LOG="${EVAL_LOG:-${ROOT}/runlogs/${EVAL_SESSION}.log}"
RESULTS_FILE="${RESULTS_FILE:-${ROOT}/FinalResults_WebShop_3B7B_8GPU_${RUN_TS}.jsonl}"

if tmux has-session -t "${ENV_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${ENV_SESSION}" >&2
  exit 1
fi
if tmux has-session -t "${EVAL_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${EVAL_SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${ENV_SESSION}" \
  "cd ${ROOT} && \
   CONDA_SH=${CONDA_SH} WEBSHOP_ENV=${WEBSHOP_ENV} HOST=127.0.0.1 PORT=${ENV_PORT} \
   WEBSHOP_DATASET_SIZE=all WEBSHOP_GOAL_SOURCE=human WEBSHOP_HUMAN_GOAL_MODE=official WEBSHOP_GOAL_SPLIT=test \
   RAY_TMPDIR=${RAY_TMPDIR} TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} \
   WANDB_DIR=${WANDB_DIR} WANDB_CACHE_DIR=${WANDB_CACHE_DIR} WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} \
   LOG_PATH=${ENV_LOG} bash ${ROOT}/scripts/run_webshop_env_service.sh"

for _ in $(seq 1 180); do
  if curl --noproxy '*' -sf "${ENV_ADDR}/" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl --noproxy '*' -sf "${ENV_ADDR}/" >/dev/null; then
  echo "WebShop service did not become healthy on ${ENV_ADDR}" >&2
  exit 1
fi

# Warmup once to build env caches before evaluation.
WARMUP_ID="$(curl --noproxy '*' --max-time 1800 -sS -X POST "${ENV_ADDR}/create")"
curl --noproxy '*' -sS -X POST "${ENV_ADDR}/close" \
  -H 'Content-Type: application/json' \
  -d "{\"env_idx\": ${WARMUP_ID}}" >/dev/null || true

tmux new-session -d -s "${EVAL_SESSION}" \
  "cd ${ROOT} && \
   ROOT=${ROOT} CONDA_SH=${CONDA_SH} TRAIN_ENV=${TRAIN_ENV} ENV_ADDR=${ENV_ADDR} \
   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} RESULTS_FILE=${RESULTS_FILE} \
   OVERWRITE_RESULTS=${OVERWRITE_RESULTS} SKIP_DONE_MODELS=${SKIP_DONE_MODELS} INCLUDE_BASE_MODELS=${INCLUDE_BASE_MODELS} \
   BASE_MODEL_3B=${BASE_MODEL_3B} BASE_MODEL_7B=${BASE_MODEL_7B} \
   CKPT_ROOT_3B=${CKPT_ROOT_3B} CKPT_ROOT_7B=${CKPT_ROOT_7B} \
   EVAL_BATCH_SIZE_3B=${EVAL_BATCH_SIZE_3B} EVAL_BATCH_SIZE_7B=${EVAL_BATCH_SIZE_7B} \
   MAX_ROUNDS=${MAX_ROUNDS} TIMEOUT_SECONDS=${TIMEOUT_SECONDS} \
   MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH} \
   MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_TOKENS_PER_TURN=${MAX_TOKENS_PER_TURN} N_SAMPLES=${N_SAMPLES} \
   RAY_TMPDIR=${RAY_TMPDIR} TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} \
   WANDB_DIR=${WANDB_DIR} WANDB_CACHE_DIR=${WANDB_CACHE_DIR} WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} \
   bash ${ROOT}/scripts/batch_eval_webshop_ckpts.sh 2>&1 | tee -a ${EVAL_LOG}"

echo "Started WebShop env session: ${ENV_SESSION}"
echo "Started WebShop eval session: ${EVAL_SESSION}"
echo "WebShop env addr: ${ENV_ADDR}"
echo "Warmup env id: ${WARMUP_ID}"
echo "Env log: ${ENV_LOG}"
echo "Eval log: ${EVAL_LOG}"
echo "Results file: ${RESULTS_FILE}"

