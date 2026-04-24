#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/idfsdata/yexuyan/AgentGym-RL}"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
SCIWORLD_ENV="${SCIWORLD_ENV:-/idfsdata/yexuyan/conda_envs/agentenv-sciworld}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"

ENV_PORT="${ENV_PORT:-36005}"
ENV_ADDR="${ENV_ADDR:-http://127.0.0.1:${ENV_PORT}}"
ENV_SESSION="${ENV_SESSION:-sciworld_env_eval_${ENV_PORT}}"
EVAL_SESSION="${EVAL_SESSION:-sciworld_eval_topics_20260415_063900}"

ENV_LOG="${ENV_LOG:-${ROOT}/runlogs/sciworld_env_eval_${ENV_PORT}.log}"
EVAL_LOG="${EVAL_LOG:-${ROOT}/runlogs/${EVAL_SESSION}.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${ROOT}/runlogs/watchdog_${EVAL_SESSION}.log}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
RESULTS_FILE="${RESULTS_FILE:-${ROOT}/FinalResults_Sciworld.jsonl}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-3B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/checkpoints/GRPO-3B-0414}"
CKPT_STEPS_STR="${CKPT_STEPS_STR:-50 100 150 200}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
INCLUDE_BASE_MODEL="${INCLUDE_BASE_MODEL:-1}"

CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
STALL_SECONDS="${STALL_SECONDS:-900}"

mkdir -p "${ROOT}/runlogs"
touch "${WATCHDOG_LOG}"

log() {
  local msg="$1"
  local ts
  ts="$(date -u '+%F %T UTC')"
  printf '[%s] %s\n' "${ts}" "${msg}" | tee -a "${WATCHDOG_LOG}"
}

env_healthy() {
  curl --noproxy '*' -sf "${ENV_ADDR}/" >/dev/null
}

eval_completed() {
  if [[ ! -f "${EVAL_LOG}" ]]; then
    return 1
  fi
  if command -v rg >/dev/null 2>&1; then
    rg -q "All SciWorld checkpoint evaluations completed." "${EVAL_LOG}"
  else
    grep -q "All SciWorld checkpoint evaluations completed." "${EVAL_LOG}"
  fi
}

eval_session_alive() {
  tmux has-session -t "${EVAL_SESSION}" 2>/dev/null
}

log_is_stale() {
  [[ -f "${EVAL_LOG}" ]] || return 1
  local now mtime age
  now="$(date +%s)"
  mtime="$(stat -c '%Y' "${EVAL_LOG}")"
  age="$((now - mtime))"
  [[ "${age}" -gt "${STALL_SECONDS}" ]]
}

start_env_session() {
  if tmux has-session -t "${ENV_SESSION}" 2>/dev/null; then
    tmux kill-session -t "${ENV_SESSION}" || true
    sleep 2
  fi

  log "Starting SciWorld env session ${ENV_SESSION} on ${ENV_ADDR}"
  tmux new-session -d -s "${ENV_SESSION}" \
    "cd ${ROOT} && CONDA_SH=${CONDA_SH} SCIWORLD_ENV=${SCIWORLD_ENV} HOST=127.0.0.1 PORT=${ENV_PORT} LOG_PATH=${ENV_LOG} bash ${ROOT}/scripts/run_sciworld_env_service.sh"

  for _ in $(seq 1 120); do
    if env_healthy; then
      log "SciWorld env healthy at ${ENV_ADDR}"
      return 0
    fi
    sleep 2
  done

  log "SciWorld env failed health check after restart."
  return 1
}

start_eval_session() {
  if tmux has-session -t "${EVAL_SESSION}" 2>/dev/null; then
    tmux kill-session -t "${EVAL_SESSION}" || true
    sleep 2
  fi

  log "Starting eval session ${EVAL_SESSION} (resume mode, no overwrite)"
  tmux new-session -d -s "${EVAL_SESSION}" \
    "cd ${ROOT} && CONDA_SH=${CONDA_SH} TRAIN_ENV=${TRAIN_ENV} ENV_ADDR=${ENV_ADDR} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} RESULTS_FILE=${RESULTS_FILE} OVERWRITE_RESULTS=0 SKIP_DONE_MODELS=1 BASE_MODEL_PATH=${BASE_MODEL_PATH} CKPT_ROOT=${CKPT_ROOT} CKPT_STEPS_STR='${CKPT_STEPS_STR}' EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE} INCLUDE_BASE_MODEL=${INCLUDE_BASE_MODEL} bash ${ROOT}/scripts/batch_eval_sciworld_ckpts.sh 2>&1 | tee -a ${EVAL_LOG}"
}

log "Watchdog online. env_session=${ENV_SESSION}, eval_session=${EVAL_SESSION}, eval_log=${EVAL_LOG}"

if ! env_healthy; then
  log "SciWorld env not healthy at startup; attempting restart."
  start_env_session
fi

if ! eval_session_alive; then
  log "Eval session absent at startup; launching now."
  start_eval_session
fi

while true; do
  if eval_completed; then
    log "Evaluation completed successfully. Watchdog exiting."
    exit 0
  fi

  if ! env_healthy; then
    log "Env health check failed; restarting env and eval session."
    start_env_session
    start_eval_session
    sleep "${CHECK_INTERVAL_SECONDS}"
    continue
  fi

  if ! eval_session_alive; then
    log "Eval session not found; restarting eval session."
    start_eval_session
    sleep "${CHECK_INTERVAL_SECONDS}"
    continue
  fi

  if log_is_stale; then
    log "Eval log stale for more than ${STALL_SECONDS}s; restarting eval session."
    start_eval_session
    sleep "${CHECK_INTERVAL_SECONDS}"
    continue
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done
