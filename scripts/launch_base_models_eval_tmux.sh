#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
ALFWORLD_ENV="${ALFWORLD_ENV:-/idfsdata/yexuyan/conda_envs/agentenv-alfworld}"
SCIWORLD_ENV="${SCIWORLD_ENV:-/idfsdata/yexuyan/conda_envs/agentenv-sciworld}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-base_eval_0123_${RUN_TS}}"
RUNTIME_DIR="${RUNTIME_DIR:-${ROOT}/runtime/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${RUNTIME_DIR}/logs}"
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
RAY_SHORT_LINK="${RAY_SHORT_LINK:-/tmp/rb${RUN_TS:9:6}}"
RAY_TMPDIR="${RAY_TMPDIR:-${RAY_SHORT_LINK}}"

ALF_ENV_PORT="${ALF_ENV_PORT:-36111}"
SCI_ENV_PORT="${SCI_ENV_PORT:-36105}"
ALF_ENV_ADDR="http://127.0.0.1:${ALF_ENV_PORT}"
SCI_ENV_ADDR="http://127.0.0.1:${SCI_ENV_PORT}"

ALF_ENV_SESSION="${ALF_ENV_SESSION:-alf_eval_env_${RUN_TS}}"
SCI_ENV_SESSION="${SCI_ENV_SESSION:-sci_eval_env_${RUN_TS}}"
EVAL_SESSION="${EVAL_SESSION:-base_eval_${RUN_TS}}"

ALF_ENV_LOG="${ALF_ENV_LOG:-${LOG_DIR}/alfworld_env.log}"
SCI_ENV_LOG="${SCI_ENV_LOG:-${LOG_DIR}/sciworld_env.log}"
EVAL_LOG="${EVAL_LOG:-${LOG_DIR}/eval_master.log}"

mkdir -p \
  "${RUNTIME_DIR}" \
  "${LOG_DIR}" \
  "${HOME_DIR}" \
  "${TMPDIR}" \
  "${HF_HOME}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${RAY_REAL_DIR}"

ln -sfn "${RAY_REAL_DIR}" "${RAY_SHORT_LINK}"

for s in "${ALF_ENV_SESSION}" "${SCI_ENV_SESSION}" "${EVAL_SESSION}"; do
  if tmux has-session -t "${s}" 2>/dev/null; then
    echo "tmux session already exists: ${s}" >&2
    exit 1
  fi
done

tmux new-session -d -s "${ALF_ENV_SESSION}" \
  "cd ${ROOT} && HOME=${HOME_DIR} CONDA_SH=${CONDA_SH} ALFWORLD_ENV=${ALFWORLD_ENV} HOST=127.0.0.1 PORT=${ALF_ENV_PORT} LOG_PATH=${ALF_ENV_LOG} TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} HF_DATASETS_CACHE=${HF_DATASETS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} bash ${ROOT}/scripts/run_alfworld_env_service.sh"

tmux new-session -d -s "${SCI_ENV_SESSION}" \
  "cd ${ROOT} && HOME=${HOME_DIR} CONDA_SH=${CONDA_SH} SCIWORLD_ENV=${SCIWORLD_ENV} HOST=127.0.0.1 PORT=${SCI_ENV_PORT} LOG_PATH=${SCI_ENV_LOG} bash ${ROOT}/scripts/run_sciworld_env_service.sh"

tmux new-session -d -s "${EVAL_SESSION}" \
  "cd ${ROOT} && HOME=${HOME_DIR} CONDA_SH=${CONDA_SH} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} RUN_TAG=${RUN_TAG} RUNTIME_DIR=${RUNTIME_DIR} ALF_ENV_PORT=${ALF_ENV_PORT} SCI_ENV_PORT=${SCI_ENV_PORT} ALF_ENV_ADDR=${ALF_ENV_ADDR} SCI_ENV_ADDR=${SCI_ENV_ADDR} TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} HF_DATASETS_CACHE=${HF_DATASETS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} WANDB_DIR=${WANDB_DIR} WANDB_CACHE_DIR=${WANDB_CACHE_DIR} WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} RAY_REAL_DIR=${RAY_REAL_DIR} RAY_SHORT_LINK=${RAY_SHORT_LINK} RAY_TMPDIR=${RAY_TMPDIR} bash ${ROOT}/scripts/run_base_models_eval_alf_sci.sh > ${EVAL_LOG} 2>&1"

echo "ALFWorld env session: ${ALF_ENV_SESSION}"
echo "SciWorld env session: ${SCI_ENV_SESSION}"
echo "Eval session: ${EVAL_SESSION}"
echo "Runtime dir: ${RUNTIME_DIR}"
echo "ALFWorld env addr: ${ALF_ENV_ADDR}"
echo "SciWorld env addr: ${SCI_ENV_ADDR}"
echo "Env logs: ${ALF_ENV_LOG} / ${SCI_ENV_LOG}"
echo "Eval log: ${EVAL_LOG}"
echo "Ray tmp link: ${RAY_SHORT_LINK} -> ${RAY_REAL_DIR}"
