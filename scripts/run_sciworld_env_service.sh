#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
SCIWORLD_ENV="${SCIWORLD_ENV:-/idfsdata/yexuyan/conda_envs/agentenv-sciworld}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-36005}"
LOG_PATH="${LOG_PATH:-}"

if [[ -n "${LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${LOG_PATH}")"
  exec >"${LOG_PATH}" 2>&1
fi

source "${CONDA_SH}"
set +u
conda activate "${SCIWORLD_ENV}"
set -u

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

cd "${ROOT}"
exec env \
  -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  NO_PROXY="${NO_PROXY}" \
  no_proxy="${no_proxy}" \
  sciworld --host "${HOST}" --port "${PORT}"
