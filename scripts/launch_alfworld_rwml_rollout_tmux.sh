#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PORT="${ENV_PORT:-36051}"
ENV_ADDR="http://127.0.0.1:${ENV_PORT}"

CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
ALFWORLD_ENV="${ALFWORLD_ENV:-/idfsdata/yexuyan/conda_envs/agentenv-alfworld}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"
ALFWORLD_DATA="${ALFWORLD_DATA:-/idfsdata/yexuyan/alfworld_data}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra CUDA_DEVICE_LIST <<< "${CUDA_VISIBLE_DEVICES}"
N_GPUS="${N_GPUS:-${#CUDA_DEVICE_LIST[@]}}"
MODEL_PATH="${MODEL_PATH:-/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-7B-Instruct}"
INPUT_FILE="${INPUT_FILE:-/idfsdata/yexuyan/AgentGym-RL/AgentItemId/alfworld_train.json}"
N_ROLLOUTS_PER_TASK="${N_ROLLOUTS_PER_TASK:-3}"
MAX_ROUNDS="${MAX_ROUNDS:-30}"
BATCH_SIZE="${BATCH_SIZE:-24}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-38941}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-200}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-96}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-24576}"
RESUME_FROM_RUN_DIR="${RESUME_FROM_RUN_DIR:-}"
APPEND_TO_RUN_DIR="${APPEND_TO_RUN_DIR:-}"

TMPDIR="${TMPDIR:-${ROOT}/.cache/tmp}"
TMP="${TMP:-${TMPDIR}}"
TEMP="${TEMP:-${TMPDIR}}"
ALFWORLD_TMPDIR="${ALFWORLD_TMPDIR:-${ROOT}/.cache/alfworld_tmp}"
HF_HOME="${HF_HOME:-${ROOT}/.cache/hf}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT}/.cache/xdg}"
RAY_TMPDIR="${RAY_TMPDIR:-${ROOT}/.cache/ray}"
TMUX_TMPDIR="${TMUX_TMPDIR:-${ROOT}/.cache/tmux}"
TMUX_SOCKET="${TMUX_SOCKET:-${TMUX_TMPDIR}/alfworld_rwml.sock}"

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-alfworld_rwml_rollout_${RUN_TS}}"
ROLL_DIR="${ROLL_DIR:-${ROOT}/runlogs/${RUN_NAME}}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${ROLL_DIR}/executer_logs}"
ROLL_JSON="${ROLL_JSON:-${ROLL_DIR}/rollouts.json}"
TRIPLET_FILE="${TRIPLET_FILE:-${ROOT}/AgentItemId/alfworld_rwml_train.jsonl}"

ENV_SESSION="${ENV_SESSION:-alfworld_rwml_env_${ENV_PORT}_${RUN_TS}}"
ROLL_SESSION="${ROLL_SESSION:-alfworld_rwml_rollout_${RUN_TS}}"
POST_SESSION="${POST_SESSION:-alfworld_rwml_post_${RUN_TS}}"
ENV_LOG="${ROLL_DIR}/env.log"
ROLL_LOG="${ROLL_DIR}/rollout.log"
POST_LOG="${ROLL_DIR}/post.log"

mkdir -p "${ROLL_DIR}" "${ROLLOUT_LOG_DIR}" "${ALFWORLD_DATA}" "${TMPDIR}" "${ALFWORLD_TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" "${RAY_TMPDIR}" "${TMUX_TMPDIR}" "$(dirname "${TRIPLET_FILE}")"

START_BATCH=0
RESUME_INPUT_FILE="${INPUT_FILE}"
if [[ -n "${APPEND_TO_RUN_DIR}" ]]; then
  ROLL_DIR="${APPEND_TO_RUN_DIR}"
  ROLLOUT_LOG_DIR="${ROLL_DIR}/executer_logs"
  ROLL_JSON="${ROLL_DIR}/rollouts.json"
  ENV_LOG="${ROLL_DIR}/env.log"
  ROLL_LOG="${ROLL_DIR}/rollout_resume_$(date -u +%Y%m%d_%H%M%S).log"
  POST_LOG="${ROLL_DIR}/post_resume_$(date -u +%Y%m%d_%H%M%S).log"
  mkdir -p "${ROLL_DIR}" "${ROLLOUT_LOG_DIR}"
fi
if [[ -n "${RESUME_FROM_RUN_DIR}" ]]; then
  LAST_BATCH="$(find "${RESUME_FROM_RUN_DIR}/executer_logs" -maxdepth 1 -type d -name 'steptest_batch_*' 2>/dev/null | sed 's#.*steptest_batch_##' | sort -n | tail -n 1)"
  if [[ -n "${LAST_BATCH}" ]]; then
    START_BATCH=$(( LAST_BATCH + 1 ))
    START_INDEX="$(
      python3 - "${RESUME_FROM_RUN_DIR}/executer_logs" <<'PY'
import glob
import json
import os
import re
import sys

log_dir = sys.argv[1]
seen = set()
pattern = re.compile(r"(\d+)$")
for path in glob.glob(os.path.join(log_dir, "steptest_batch_*", "*.json")):
    try:
        rows = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(rows, list):
        rows = [rows]
    for row in rows:
        item_id = row.get("item_id") if isinstance(row, dict) else None
        match = pattern.search(str(item_id))
        if match:
            seen.add(int(match.group(1)))
print((max(seen) + 1) if seen else 0)
PY
    )"
    RESUME_INPUT_FILE="${ROLL_DIR}/resume_input_from_batch_${START_BATCH}.json"
    python3 - "${INPUT_FILE}" "${RESUME_INPUT_FILE}" "${START_INDEX}" <<'PY'
import json
import sys
src, dst, start_idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = json.load(open(src, "r", encoding="utf-8"))
rows = rows[start_idx:]
with open(dst, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=True)
print(f"Wrote {len(rows)} remaining items to {dst}")
PY
  fi
fi

if tmux -S "${TMUX_SOCKET}" has-session -t "${ENV_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${ENV_SESSION}"
  exit 1
fi
if tmux -S "${TMUX_SOCKET}" has-session -t "${ROLL_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${ROLL_SESSION}"
  exit 1
fi
if tmux -S "${TMUX_SOCKET}" has-session -t "${POST_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${POST_SESSION}"
  exit 1
fi

tmux -S "${TMUX_SOCKET}" new-session -d -s "${ENV_SESSION}" \
  "cd ${ROOT} && CONDA_SH=${CONDA_SH} ALFWORLD_ENV=${ALFWORLD_ENV} HOST=127.0.0.1 PORT=${ENV_PORT} LOG_PATH=${ENV_LOG} ALFWORLD_DATA=${ALFWORLD_DATA} TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} ALFWORLD_TMPDIR=${ALFWORLD_TMPDIR} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} RAY_TMPDIR=${RAY_TMPDIR} bash ${ROOT}/scripts/run_alfworld_env_service.sh"

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

tmux -S "${TMUX_SOCKET}" new-session -d -s "${ROLL_SESSION}" \
  "bash -lc 'source ${CONDA_SH} && conda activate ${TRAIN_ENV} && cd ${ROOT}/AgentGym-RL && env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} ALFWORLD_TMPDIR=${ALFWORLD_TMPDIR} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} RAY_TMPDIR=${RAY_TMPDIR} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m verl.agent_trainer.main_generation +data.input_file=${RESUME_INPUT_FILE} +data.start_batch_idx=${START_BATCH} data.path=${ROOT}/AgentEval/alfworld data.max_prompt_length=${MAX_PROMPT_LENGTH} data.max_response_length=${MAX_RESPONSE_LENGTH} data.n_samples=${N_ROLLOUTS_PER_TASK} data.batch_size=${BATCH_SIZE} agentgym.task_name=alfworld agentgym.env_addr=${ENV_ADDR} agentgym.max_rounds=${MAX_ROUNDS} agentgym.timeout=2400 model.path=${MODEL_PATH} rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} rollout.temperature=1 rollout.max_model_len=${MAX_MODEL_LEN} rollout.max_tokens=${MAX_TOKENS_PER_TURN} rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} rollout.tensor_model_parallel_size=1 rollout.rollout_log_dir=${ROLLOUT_LOG_DIR} trainer.nnodes=1 trainer.n_gpus_per_node=${N_GPUS} 2>&1 | tee ${ROLL_LOG}'"

tmux -S "${TMUX_SOCKET}" new-session -d -s "${POST_SESSION}" \
  "bash -lc 'while tmux -S ${TMUX_SOCKET} has-session -t ${ROLL_SESSION} 2>/dev/null; do sleep 10; done; cd ${ROOT} && python3 ${ROOT}/scripts/generate_alfworld_rwml_rollouts.py --rollout-log-dir ${ROLLOUT_LOG_DIR} --output ${ROLL_JSON} 2>&1 | tee ${POST_LOG}; python3 ${ROOT}/scripts/build_alfworld_rwml_triplets.py --input ${ROLL_JSON} --output ${TRIPLET_FILE} --rollouts-per-task ${N_ROLLOUTS_PER_TASK} 2>&1 | tee -a ${POST_LOG}'"

echo "ALFWorld train set size: 2420"
echo "Environment tmux session: ${ENV_SESSION}"
echo "Rollout tmux session: ${ROLL_SESSION}"
echo "Postprocess tmux session: ${POST_SESSION}"
echo "ALFWorld service: ${ENV_ADDR}"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES}"
echo "GPUs per node: ${N_GPUS}"
echo "tmux socket: ${TMUX_SOCKET}"
echo "Resume from batch: ${START_BATCH}"
echo "Resume from item index: ${START_INDEX}"
echo "Resume input file: ${RESUME_INPUT_FILE}"
echo "Rollout log dir: ${ROLLOUT_LOG_DIR}"
echo "Rollout JSON: ${ROLL_JSON}"
echo "Triplet file target: ${TRIPLET_FILE}"
