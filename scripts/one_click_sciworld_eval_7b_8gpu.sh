#!/usr/bin/env bash

set -euo pipefail

ROOT="/idfsdata/yexuyan/AgentGym-RL"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
SCIWORLD_ENV="${SCIWORLD_ENV:-/idfsdata/yexuyan/conda_envs/agentenv-sciworld}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ENV_PORT="${ENV_PORT:-36016}"
ENV_ADDR="${ENV_ADDR:-http://127.0.0.1:${ENV_PORT}}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-7B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-/idfsdata/yexuyan/AgentGym-RL/checkpoints/GRPO-7B-SCORE-0415-SciWorld}"
CKPT_STEPS_STR="${CKPT_STEPS_STR:-}"

INCLUDE_BASE_MODEL="${INCLUDE_BASE_MODEL:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
OVERWRITE_RESULTS="${OVERWRITE_RESULTS:-0}"
SKIP_DONE_MODELS="${SKIP_DONE_MODELS:-1}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
STALL_SECONDS="${STALL_SECONDS:-1800}"

RESULTS_FILE="${RESULTS_FILE:-${ROOT}/FinalResults_Sciworld_7B.jsonl}"

TMPDIR="${TMPDIR:-/idfsdata/yexuyan/te}"
TMP="${TMP:-${TMPDIR}}"
TEMP="${TEMP:-${TMPDIR}}"
HF_HOME="${HF_HOME:-/idfsdata/yexuyan/he}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/idfsdata/yexuyan/xe}"
WANDB_DIR="${WANDB_DIR:-/idfsdata/yexuyan/we}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/.cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/.config}"
RAY_TMPDIR="${RAY_TMPDIR:-/idfsdata/yexuyan/ra}"

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
WATCHDOG_SESSION="${WATCHDOG_SESSION:-sciworld_eval_watchdog_7b_${RUN_TS}}"
EVAL_SESSION="${EVAL_SESSION:-sciworld_eval_7b_${RUN_TS}}"
ENV_SESSION="${ENV_SESSION:-sciworld_env_eval_${ENV_PORT}_${RUN_TS}}"
ENV_LOG="${ENV_LOG:-${ROOT}/runlogs/${ENV_SESSION}.log}"
EVAL_LOG="${EVAL_LOG:-${ROOT}/runlogs/${EVAL_SESSION}.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${ROOT}/runlogs/${WATCHDOG_SESSION}.log}"
FORCE_RESTART="${FORCE_RESTART:-1}"

if [[ ! -d "${CKPT_ROOT}" ]]; then
  echo "CKPT_ROOT not found: ${CKPT_ROOT}" >&2
  exit 1
fi

if [[ -z "${CKPT_STEPS_STR}" ]]; then
  mapfile -t _ckpt_dirs < <(find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' | sort -V)
  _steps=()
  for _dir in "${_ckpt_dirs[@]}"; do
    _name="$(basename "${_dir}")"
    _steps+=("${_name#global_step_}")
  done
  CKPT_STEPS_STR="${_steps[*]:-}"
fi

if [[ "${INCLUDE_BASE_MODEL}" != "1" && -z "${CKPT_STEPS_STR}" ]]; then
  echo "No models to evaluate: INCLUDE_BASE_MODEL=${INCLUDE_BASE_MODEL}, CKPT_STEPS_STR is empty." >&2
  exit 1
fi

mkdir -p \
  "${ROOT}/runlogs" \
  "$(dirname "${RESULTS_FILE}")" \
  "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" \
  "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${RAY_TMPDIR}"

cd "${ROOT}"

if tmux has-session -t "${WATCHDOG_SESSION}" 2>/dev/null; then
  if [[ "${FORCE_RESTART}" == "1" ]]; then
    tmux kill-session -t "${WATCHDOG_SESSION}" || true
  else
    echo "tmux session already exists: ${WATCHDOG_SESSION}" >&2
    exit 1
  fi
fi

tmux new-session -d -s "${WATCHDOG_SESSION}" \
  "cd ${ROOT} && \
   TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} \
   HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} \
   WANDB_DIR=${WANDB_DIR} WANDB_CACHE_DIR=${WANDB_CACHE_DIR} WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} \
   RAY_TMPDIR=${RAY_TMPDIR} \
   CONDA_SH=${CONDA_SH} SCIWORLD_ENV=${SCIWORLD_ENV} TRAIN_ENV=${TRAIN_ENV} \
   ENV_PORT=${ENV_PORT} ENV_ADDR=${ENV_ADDR} ENV_SESSION=${ENV_SESSION} EVAL_SESSION=${EVAL_SESSION} \
   ENV_LOG=${ENV_LOG} EVAL_LOG=${EVAL_LOG} WATCHDOG_LOG=${WATCHDOG_LOG} \
   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
   RESULTS_FILE=${RESULTS_FILE} OVERWRITE_RESULTS=${OVERWRITE_RESULTS} SKIP_DONE_MODELS=${SKIP_DONE_MODELS} \
   BASE_MODEL_PATH=${BASE_MODEL_PATH} CKPT_ROOT=${CKPT_ROOT} CKPT_STEPS_STR='${CKPT_STEPS_STR}' \
   INCLUDE_BASE_MODEL=${INCLUDE_BASE_MODEL} EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE} \
   CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS} STALL_SECONDS=${STALL_SECONDS} \
   bash ${ROOT}/scripts/watch_sciworld_eval.sh 2>&1 | tee -a ${WATCHDOG_LOG}"

echo "Started SciWorld 7B eval watchdog in tmux: ${WATCHDOG_SESSION}"
echo "Env session: ${ENV_SESSION}"
echo "Eval session: ${EVAL_SESSION}"
echo "Env addr: ${ENV_ADDR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Base model: ${BASE_MODEL_PATH}"
echo "Checkpoint root: ${CKPT_ROOT}"
echo "Checkpoint steps: ${CKPT_STEPS_STR:-<none>}"
echo "Results file: ${RESULTS_FILE}"
echo "Watchdog log: ${WATCHDOG_LOG}"
echo "Eval log: ${EVAL_LOG}"
