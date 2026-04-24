#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/idfsdata/yexuyan/AgentGym-RL}"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
ALFWORLD_ENV="${ALFWORLD_ENV:-/idfsdata/yexuyan/conda_envs/agentenv-alfworld}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"
ALFWORLD_DATA="${ALFWORLD_DATA:-/idfsdata/yexuyan/alfworld_data}"

ENV_PORT="${ENV_PORT:-36018}"
ENV_ADDR="${ENV_ADDR:-http://127.0.0.1:${ENV_PORT}}"
ENV_SESSION="${ENV_SESSION:-alfworld_env_${ENV_PORT}}"
TRAIN_SESSION="${TRAIN_SESSION:-alfworld_ppo_${ENV_PORT}}"
EXP_NAME="${EXP_NAME:-ALFWORLD_PPO_7B_SCORE_AUTO_4567}"

MODEL_PATH="${MODEL_PATH:-/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-7B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ANONYMOUS="${WANDB_ANONYMOUS:-allow}"
WANDB_ENTITY="${WANDB_ENTITY:-xuyan923r-renmin-university-of-china}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
PROJECT_NAME="${PROJECT_NAME:-ALFWorld}"

KL_COEF="${KL_COEF:-0.001}"
POLICY_LR="${POLICY_LR:-1e-6}"
CRITIC_LR="${CRITIC_LR:-1e-5}"
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
PAD_TO_MAX_RESPONSE_LENGTH="${PAD_TO_MAX_RESPONSE_LENGTH:-0}"
SAVE_FREQ="${SAVE_FREQ:-50}"
REMOVE_PREVIOUS_CKPT_IN_SAVE="${REMOVE_PREVIOUS_CKPT_IN_SAVE:-0}"
MAX_LOCAL_CKPT_TO_KEEP="${MAX_LOCAL_CKPT_TO_KEEP:-6}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-0}"
ENABLE_WMC="${ENABLE_WMC:-0}"
WMC_COEFF="${WMC_COEFF:-0.001}"
ENABLE_ERC="${ENABLE_ERC:-0}"
ERC_MU_BASE="${ERC_MU_BASE:-1.0}"
ERC_MU_EXP="${ERC_MU_EXP:-2.0}"
ERC_ETA_WM="${ERC_ETA_WM:-3.0}"
ERC_LAMBDA_WM="${ERC_LAMBDA_WM:-1.0}"
ERC_CLIPPING_TYPE="${ERC_CLIPPING_TYPE:-global}"
ERC_CLIPPING_METHOD="${ERC_CLIPPING_METHOD:-mask}"
ERC_MOMENTUM="${ERC_MOMENTUM:-0.9}"

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

RUN_DIR="${RUN_DIR:-${ROOT}/runlogs/${EXP_NAME}}"
TRAIN_LOG="${TRAIN_LOG:-${RUN_DIR}/train.log}"
ENV_LOG="${ENV_LOG:-${ROOT}/runlogs/${ENV_SESSION}.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${ROOT}/runlogs/watchdog_${TRAIN_SESSION}.log}"

CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-45}"
STALL_SECONDS="${STALL_SECONDS:-1200}"
MIN_RESTART_INTERVAL_SECONDS="${MIN_RESTART_INTERVAL_SECONDS:-30}"

mkdir -p \
  "${RUN_DIR}" \
  "$(dirname "${ENV_LOG}")" \
  "$(dirname "${WATCHDOG_LOG}")" \
  "${ALFWORLD_DATA}" \
  "${RAY_TMPDIR}" \
  "${TMPDIR}" \
  "${HF_HOME}" \
  "${TRANSFORMERS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}"

touch "${WATCHDOG_LOG}"

CURRENT_ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
CURRENT_PAD_TO_MAX_RESPONSE_LENGTH="${PAD_TO_MAX_RESPONSE_LENGTH}"
LAST_RESTART_TS=0

log() {
  local msg="$1"
  local ts
  ts="$(date -u '+%F %T UTC')"
  printf '[%s] %s\n' "${ts}" "${msg}" | tee -a "${WATCHDOG_LOG}"
}

env_healthy() {
  curl --noproxy '*' -sf "${ENV_ADDR}/" >/dev/null
}

session_alive() {
  local name="$1"
  tmux has-session -t "${name}" 2>/dev/null
}

log_stale() {
  [[ -f "${TRAIN_LOG}" ]] || return 1
  local now mtime age
  now="$(date +%s)"
  mtime="$(stat -c '%Y' "${TRAIN_LOG}")"
  age="$((now - mtime))"
  [[ "${age}" -gt "${STALL_SECONDS}" ]]
}

failure_reason() {
  [[ -f "${TRAIN_LOG}" ]] || return 1
  local chunk
  chunk="$(tail -n 240 "${TRAIN_LOG}" 2>/dev/null || true)"
  if printf '%s' "${chunk}" | grep -Eqi 'CUDA out of memory|OOM|CUBLAS_STATUS_ALLOC_FAILED|out of memory'; then
    echo "oom"
    return 0
  fi
  if printf '%s' "${chunk}" | grep -Eqi 'The size of tensor a .* must match the size of tensor b|compute_gae_advantage_return'; then
    echo "shape_mismatch"
    return 0
  fi
  if printf '%s' "${chunk}" | grep -Eqi 'RayTaskError|Traceback|RuntimeError|Error executing job'; then
    echo "runtime_error"
    return 0
  fi
  return 1
}

adjust_after_failure() {
  local reason="$1"
  if [[ "${reason}" == "shape_mismatch" ]]; then
    CURRENT_PAD_TO_MAX_RESPONSE_LENGTH="0"
    log "Auto-fix: force PAD_TO_MAX_RESPONSE_LENGTH=0 due to PPO length mismatch."
    return
  fi
  if [[ "${reason}" == "oom" ]]; then
    local next_util
    next_util="$(awk -v x="${CURRENT_ROLLOUT_GPU_MEMORY_UTILIZATION}" 'BEGIN { y=x-0.05; if (y < 0.55) y=0.55; printf "%.2f", y }')"
    if [[ "${next_util}" != "${CURRENT_ROLLOUT_GPU_MEMORY_UTILIZATION}" ]]; then
      CURRENT_ROLLOUT_GPU_MEMORY_UTILIZATION="${next_util}"
      log "Auto-fix: reduce ROLLOUT_GPU_MEMORY_UTILIZATION to ${CURRENT_ROLLOUT_GPU_MEMORY_UTILIZATION} after OOM."
    fi
  fi
}

respect_restart_interval() {
  local now
  now="$(date +%s)"
  if (( now - LAST_RESTART_TS < MIN_RESTART_INTERVAL_SECONDS )); then
    local wait_s=$((MIN_RESTART_INTERVAL_SECONDS - (now - LAST_RESTART_TS)))
    sleep "${wait_s}"
  fi
  LAST_RESTART_TS="$(date +%s)"
}

start_env_session() {
  if session_alive "${ENV_SESSION}"; then
    tmux kill-session -t "${ENV_SESSION}" || true
    sleep 2
  fi
  log "Starting ALFWorld env session ${ENV_SESSION} at ${ENV_ADDR}"
  tmux new-session -d -s "${ENV_SESSION}" \
    "cd ${ROOT} && CONDA_SH=${CONDA_SH} ALFWORLD_ENV=${ALFWORLD_ENV} HOST=127.0.0.1 PORT=${ENV_PORT} LOG_PATH=${ENV_LOG} ALFWORLD_DATA=${ALFWORLD_DATA} TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} bash ${ROOT}/scripts/run_alfworld_env_service.sh"
  for _ in $(seq 1 150); do
    if env_healthy; then
      log "ALFWorld env healthy at ${ENV_ADDR}"
      return 0
    fi
    sleep 2
  done
  log "ALFWorld env failed health check after restart."
  return 1
}

start_train_session() {
  if session_alive "${TRAIN_SESSION}"; then
    tmux kill-session -t "${TRAIN_SESSION}" || true
    sleep 2
  fi
  mkdir -p "${RUN_DIR}"
  touch "${TRAIN_LOG}"
  log "Starting train session ${TRAIN_SESSION} (exp=${EXP_NAME}, util=${CURRENT_ROLLOUT_GPU_MEMORY_UTILIZATION}, pad=${CURRENT_PAD_TO_MAX_RESPONSE_LENGTH})"
  tmux new-session -d -s "${TRAIN_SESSION}" \
    "cd ${ROOT} && CONDA_SH=${CONDA_SH} TRAIN_ENV=${TRAIN_ENV} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ENV_ADDR=${ENV_ADDR} MODEL_PATH=${MODEL_PATH} WANDB_MODE=${WANDB_MODE} WANDB_ANONYMOUS=${WANDB_ANONYMOUS} WANDB_ENTITY=${WANDB_ENTITY} WANDB_BASE_URL=${WANDB_BASE_URL} PROJECT_NAME=${PROJECT_NAME} KL_COEF=${KL_COEF} POLICY_LR=${POLICY_LR} CRITIC_LR=${CRITIC_LR} ROLLOUT_N=${ROLLOUT_N} TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU} PPO_EPOCHS=${PPO_EPOCHS} TOTAL_EPOCHS=${TOTAL_EPOCHS} TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} MAX_ROUNDS=${MAX_ROUNDS} AGENT_MAX_ROUNDS=${AGENT_MAX_ROUNDS} MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH} MAX_MODEL_LEN=${MAX_MODEL_LEN} MAX_TOKENS_PER_TURN=${MAX_TOKENS_PER_TURN} ROLLOUT_GPU_MEMORY_UTILIZATION=${CURRENT_ROLLOUT_GPU_MEMORY_UTILIZATION} REWARD_MODE=${REWARD_MODE} ORM_SUCCESS_SCORE=${ORM_SUCCESS_SCORE} PAD_TO_MAX_RESPONSE_LENGTH=${CURRENT_PAD_TO_MAX_RESPONSE_LENGTH} SAVE_FREQ=${SAVE_FREQ} REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE} MAX_LOCAL_CKPT_TO_KEEP=${MAX_LOCAL_CKPT_TO_KEEP} RESUME_MODE=${RESUME_MODE} RESUME_FROM_PATH=${RESUME_FROM_PATH} ENABLE_WMC=${ENABLE_WMC} WMC_COEFF=${WMC_COEFF} ENABLE_ERC=${ENABLE_ERC} ERC_MU_BASE=${ERC_MU_BASE} ERC_MU_EXP=${ERC_MU_EXP} ERC_ETA_WM=${ERC_ETA_WM} ERC_LAMBDA_WM=${ERC_LAMBDA_WM} ERC_CLIPPING_TYPE=${ERC_CLIPPING_TYPE} ERC_CLIPPING_METHOD=${ERC_CLIPPING_METHOD} ERC_MOMENTUM=${ERC_MOMENTUM} RAY_TMPDIR=${RAY_TMPDIR} TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} WANDB_DIR=${WANDB_DIR} WANDB_CACHE_DIR=${WANDB_CACHE_DIR} WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} EXP_NAME=${EXP_NAME} LOG_PATH=${TRAIN_LOG} bash ${ROOT}/scripts/run_alfworld_ppo_train.sh"
}

restart_train() {
  local reason="$1"
  respect_restart_interval
  if [[ -n "${reason}" ]]; then
    log "Triggering train restart due to ${reason}."
    adjust_after_failure "${reason}"
  else
    log "Triggering train restart."
  fi
  if ! env_healthy; then
    log "Env unhealthy before train restart; restarting env first."
    start_env_session
  fi
  start_train_session
}

log "Watchdog online. env_session=${ENV_SESSION}, train_session=${TRAIN_SESSION}, env_addr=${ENV_ADDR}, train_log=${TRAIN_LOG}"
log "This watchdog only manages ALFWorld sessions and does not touch WebShop sessions."

if ! env_healthy; then
  log "Env not healthy at startup, starting env."
  start_env_session
fi

if ! session_alive "${TRAIN_SESSION}"; then
  log "Train session absent at startup, starting train."
  restart_train ""
fi

while true; do
  if ! env_healthy; then
    log "Env health check failed."
    respect_restart_interval
    start_env_session
    start_train_session
    sleep "${CHECK_INTERVAL_SECONDS}"
    continue
  fi

  if ! session_alive "${TRAIN_SESSION}"; then
    local_reason=""
    if reason="$(failure_reason)"; then
      local_reason="${reason}"
    fi
    if [[ -n "${local_reason}" ]]; then
      restart_train "${local_reason}"
    else
      restart_train "session_missing"
    fi
    sleep "${CHECK_INTERVAL_SECONDS}"
    continue
  fi

  if reason="$(failure_reason)"; then
    restart_train "${reason}"
    sleep "${CHECK_INTERVAL_SECONDS}"
    continue
  fi

  if log_stale; then
    restart_train "log_stale"
    sleep "${CHECK_INTERVAL_SECONDS}"
    continue
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done
