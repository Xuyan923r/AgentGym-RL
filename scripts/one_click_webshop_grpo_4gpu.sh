#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 4-GPU default; can be overridden by external env.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Conda/env defaults.
export CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
export TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"
export WEBSHOP_ENV="${WEBSHOP_ENV:-/idfsdata/yexuyan/conda_envs/agentenv-webshop}"

# Model and run defaults.
export MODEL_PATH="${MODEL_PATH:-/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-7B-Instruct}"
export EXP_NAME="${EXP_NAME:-WEBSHOP_GRPO_4GPU_$(date -u +%Y%m%d_%H%M%S)}"
export ENV_PORT="${ENV_PORT:-8013}"

# W&B defaults (online, target project URL).
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-xuyan923r-renmin-university-of-china}"
export PROJECT_NAME="${PROJECT_NAME:-agentgym-webshop}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"

# Keep writable caches/logs on idfsdata.
export RAY_TMPDIR="${RAY_TMPDIR:-/idfsdata/yexuyan/ray_tmp}"
export TMPDIR="${TMPDIR:-/idfsdata/yexuyan/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export HF_HOME="${HF_HOME:-/idfsdata/yexuyan/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/idfsdata/yexuyan/hf/hub}"
export WANDB_DIR="${WANDB_DIR:-/idfsdata/yexuyan/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/idfsdata/yexuyan/wandb/.cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/idfsdata/yexuyan/wandb/.config}"

mkdir -p \
  "${RAY_TMPDIR}" \
  "${TMPDIR}" \
  "${HF_HOME}" \
  "${TRANSFORMERS_CACHE}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}"

cd "${ROOT}"
exec bash "${ROOT}/run_webshop_grpo_background.sh"

