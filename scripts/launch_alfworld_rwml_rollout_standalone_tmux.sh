#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
ALFWORLD_ENV="${ALFWORLD_ENV:-/idfsdata/yexuyan/conda_envs/agentenv-alfworld}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"
ALFWORLD_DATA="${ALFWORLD_DATA:-/idfsdata/yexuyan/alfworld_data}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra CUDA_DEVICE_LIST <<< "${CUDA_VISIBLE_DEVICES}"
N_GPUS="${N_GPUS:-${#CUDA_DEVICE_LIST[@]}}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-2}"
TP_SIZE_PER_WORKER="${TP_SIZE_PER_WORKER:-4}"

ENV_PORT_BASE="${ENV_PORT_BASE:-36051}"
ENV_SERVER_COUNT="${ENV_SERVER_COUNT:-1}"
MODEL_PATH="${MODEL_PATH:-/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-7B-Instruct}"
INPUT_FILE="${INPUT_FILE:-/idfsdata/yexuyan/AgentGym-RL/AgentItemId/alfworld_train.json}"
BATCH_SIZE="${BATCH_SIZE:-24}"
N_SAMPLES="${N_SAMPLES:-3}"
MAX_ROUNDS="${MAX_ROUNDS:-30}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-200}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-96}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-24576}"
BATCH_INDEX_BASE="${BATCH_INDEX_BASE:-0}"
RESUME_FROM_RUN_DIR="${RESUME_FROM_RUN_DIR:-}"
APPEND_TO_RUN_DIR="${APPEND_TO_RUN_DIR:-}"

TMPDIR="${TMPDIR:-${ROOT}/.cache/tmp}"
TMP="${TMP:-${TMPDIR}}"
TEMP="${TEMP:-${TMPDIR}}"
ALFWORLD_TMPDIR="${ALFWORLD_TMPDIR:-${ROOT}/.cache/alfworld_tmp}"
HF_HOME="${HF_HOME:-${ROOT}/.cache/hf}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT}/.cache/xdg}"
TMUX_TMPDIR="${TMUX_TMPDIR:-${ROOT}/.cache/tmux}"
TMUX_SOCKET="${TMUX_SOCKET:-${TMUX_TMPDIR}/alfworld_rwml_standalone.sock}"

RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-alfworld_rwml_rollout_standalone_${RUN_TS}}"
ROLL_DIR="${ROLL_DIR:-${ROOT}/runlogs/${RUN_NAME}}"
ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-${ROLL_DIR}/executer_logs}"
ROLL_JSON="${ROLL_JSON:-${ROLL_DIR}/rollouts.json}"
TRIPLET_FILE="${TRIPLET_FILE:-${ROOT}/AgentItemId/alfworld_rwml_train.jsonl}"
ROLL_LOG="${ROLL_DIR}/rollout.log"
POST_LOG="${ROLL_DIR}/post.log"

mkdir -p "${ROLL_DIR}" "${ROLLOUT_LOG_DIR}" "${ALFWORLD_DATA}" "${TMPDIR}" "${ALFWORLD_TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" "${TMUX_TMPDIR}" "$(dirname "${TRIPLET_FILE}")"

START_BATCH=0
START_INDEX=0
RESUME_INPUT_FILE="${INPUT_FILE}"

if [[ -n "${APPEND_TO_RUN_DIR}" ]]; then
  ROLL_DIR="${APPEND_TO_RUN_DIR}"
  ROLLOUT_LOG_DIR="${ROLL_DIR}/executer_logs"
  ROLL_JSON="${ROLL_DIR}/rollouts.json"
  ROLL_LOG="${ROLL_DIR}/rollout_standalone_resume_$(date -u +%Y%m%d_%H%M%S).log"
  POST_LOG="${ROLL_DIR}/post_standalone_resume_$(date -u +%Y%m%d_%H%M%S).log"
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

ENV_ADDRS=()
ENV_SESSIONS=()
for idx in $(seq 0 $((ENV_SERVER_COUNT - 1))); do
  port=$((ENV_PORT_BASE + idx))
  ENV_ADDRS+=("http://127.0.0.1:${port}")
  ENV_SESSIONS+=("alfworld_rwml_standalone_env_${port}_${RUN_TS}")
done
ENV_ADDRS_JOINED="$(IFS=,; echo "${ENV_ADDRS[*]}")"
ROLL_SESSIONS=()
POST_SESSION="alfworld_rwml_standalone_post_${RUN_TS}"

for worker_idx in $(seq 0 $((ROLLOUT_WORKERS - 1))); do
  ROLL_SESSIONS+=("alfworld_rwml_standalone_rollout_${worker_idx}_${RUN_TS}")
done

for session in "${ENV_SESSIONS[@]}" "${ROLL_SESSIONS[@]}" "${POST_SESSION}"; do
  if tmux -S "${TMUX_SOCKET}" has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}"
    exit 1
  fi
done

for idx in "${!ENV_ADDRS[@]}"; do
  port=$((ENV_PORT_BASE + idx))
  env_log="${ROLL_DIR}/env_${port}.log"
  tmux -S "${TMUX_SOCKET}" new-session -d -s "${ENV_SESSIONS[$idx]}" \
    "cd ${ROOT} && CONDA_SH=${CONDA_SH} ALFWORLD_ENV=${ALFWORLD_ENV} HOST=127.0.0.1 PORT=${port} LOG_PATH=${env_log} ALFWORLD_DATA=${ALFWORLD_DATA} TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} ALFWORLD_TMPDIR=${ALFWORLD_TMPDIR} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} bash ${ROOT}/scripts/run_alfworld_env_service.sh"
done

for addr in "${ENV_ADDRS[@]}"; do
  for _ in $(seq 1 120); do
    if curl --noproxy '*' -sf "${addr}/" >/dev/null; then
      break
    fi
    sleep 2
  done
  if ! curl --noproxy '*' -sf "${addr}/" >/dev/null; then
    echo "ALFWorld service did not become healthy on ${addr}"
    exit 1
  fi
done

ROLL_SESSION_NAMES="$(IFS=,; echo "${ROLL_SESSIONS[*]}")"

for worker_idx in $(seq 0 $((ROLLOUT_WORKERS - 1))); do
  start_gpu=$(( worker_idx * TP_SIZE_PER_WORKER ))
  end_gpu=$(( start_gpu + TP_SIZE_PER_WORKER - 1 ))
  gpu_slice="$(IFS=,; echo "${CUDA_DEVICE_LIST[*]:${start_gpu}:${TP_SIZE_PER_WORKER}}")"
  worker_log="${ROLL_DIR}/rollout_standalone_resume_${RUN_TS}_worker${worker_idx}.log"
  tmux -S "${TMUX_SOCKET}" new-session -d -s "${ROLL_SESSIONS[$worker_idx]}" \
    "bash -lc 'source ${CONDA_SH} && conda activate ${TRAIN_ENV} && cd ${ROOT} && env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost CUDA_VISIBLE_DEVICES=${gpu_slice} TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} ALFWORLD_TMPDIR=${ALFWORLD_TMPDIR} HF_HOME=${HF_HOME} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} XDG_CACHE_HOME=${XDG_CACHE_HOME} python3 ${ROOT}/scripts/standalone_alfworld_rwml_rollout.py --model-path ${MODEL_PATH} --input-file ${RESUME_INPUT_FILE} --rollout-log-dir ${ROLLOUT_LOG_DIR} --env-addrs ${ENV_ADDRS_JOINED} --batch-size ${BATCH_SIZE} --n-samples ${N_SAMPLES} --max-rounds ${MAX_ROUNDS} --max-prompt-length ${MAX_PROMPT_LENGTH} --max-response-length ${MAX_RESPONSE_LENGTH} --max-model-len ${MAX_MODEL_LEN} --max-tokens-per-turn ${MAX_TOKENS_PER_TURN} --gpu-memory-utilization ${ROLLOUT_GPU_MEMORY_UTILIZATION} --max-num-seqs ${ROLLOUT_MAX_NUM_SEQS} --max-num-batched-tokens ${ROLLOUT_MAX_NUM_BATCHED_TOKENS} --resume-batch-idx ${START_BATCH} --batch-index-base ${BATCH_INDEX_BASE} --tensor-parallel-size ${TP_SIZE_PER_WORKER} --batch-stride ${ROLLOUT_WORKERS} --batch-offset ${worker_idx} 2>&1 | tee ${worker_log}'"
done

tmux -S "${TMUX_SOCKET}" new-session -d -s "${POST_SESSION}" \
  "bash -lc 'ROLL_SESSIONS_CSV=${ROLL_SESSION_NAMES}; while true; do active=0; IFS=, read -ra SESSIONS <<< \"\$ROLL_SESSIONS_CSV\"; for s in \"\${SESSIONS[@]}\"; do if tmux -S ${TMUX_SOCKET} has-session -t \"\$s\" 2>/dev/null; then active=1; fi; done; if [ \"\$active\" -eq 0 ]; then break; fi; sleep 10; done; cd ${ROOT} && python3 ${ROOT}/scripts/generate_alfworld_rwml_rollouts.py --rollout-log-dir ${ROLLOUT_LOG_DIR} --output ${ROLL_JSON} 2>&1 | tee ${POST_LOG}; python3 ${ROOT}/scripts/build_alfworld_rwml_triplets.py --input ${ROLL_JSON} --output ${TRIPLET_FILE} --rollouts-per-task ${N_SAMPLES} 2>&1 | tee -a ${POST_LOG}'"

echo "ALFWorld train set size: 2420"
echo "Environment tmux sessions: ${ENV_SESSIONS[*]}"
echo "Rollout tmux sessions: ${ROLL_SESSIONS[*]}"
echo "Postprocess tmux session: ${POST_SESSION}"
echo "ALFWorld services: ${ENV_ADDRS_JOINED}"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES}"
echo "GPUs per node: ${N_GPUS}"
echo "Standalone workers: ${ROLLOUT_WORKERS}"
echo "TP size per worker: ${TP_SIZE_PER_WORKER}"
echo "tmux socket: ${TMUX_SOCKET}"
echo "Resume from batch: ${START_BATCH}"
echo "Batch index base: ${BATCH_INDEX_BASE}"
echo "Resume from item index: ${START_INDEX}"
echo "Resume input file: ${RESUME_INPUT_FILE}"
echo "Rollout log dir: ${ROLLOUT_LOG_DIR}"
echo "Rollout JSON: ${ROLL_JSON}"
echo "Triplet file target: ${TRIPLET_FILE}"
