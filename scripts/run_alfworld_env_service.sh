#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
ALFWORLD_ENV="${ALFWORLD_ENV:-/idfsdata/yexuyan/conda_envs/agentenv-alfworld}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-36001}"
LOG_PATH="${LOG_PATH:-}"
ALFWORLD_DATA="${ALFWORLD_DATA:-/idfsdata/yexuyan/alfworld_data}"
HOME="${HOME:-${ROOT}/runtime/alfworld_env_home}"

TMPDIR="${TMPDIR:-/idfsdata/yexuyan/te}"
TMP="${TMP:-${TMPDIR}}"
TEMP="${TEMP:-${TMPDIR}}"
HF_HOME="${HF_HOME:-/idfsdata/yexuyan/he}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/idfsdata/yexuyan/xe}"

mkdir -p "${ALFWORLD_DATA}" "${HOME}" "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}"

if [[ -n "${LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${LOG_PATH}")"
  exec >"${LOG_PATH}" 2>&1
fi

source "${CONDA_SH}"
set +u
conda activate "${ALFWORLD_ENV}"
set -u

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

cd "${ROOT}"
exec env \
  -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  NO_PROXY="${NO_PROXY}" \
  no_proxy="${no_proxy}" \
  HOME="${HOME}" \
  ALFWORLD_DATA="${ALFWORLD_DATA}" \
  TMPDIR="${TMPDIR}" \
  TMP="${TMP}" \
  TEMP="${TEMP}" \
  HF_HOME="${HF_HOME}" \
  TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
  alfworld --host "${HOST}" --port "${PORT}"
