#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/idfsdata/yexuyan/AgentGym-RL}"
CODE_DIR="${CODE_DIR:-${ROOT}/AgentGym-RL}"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"

ENV_ADDR="${ENV_ADDR:-http://127.0.0.1:36007}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/checkpoints/BABYAI_GRPO_7B_SCORE_20260421_101616}"
RESULTS_FILE="${RESULTS_FILE:-${ROOT}/FinalResults_BabyAI.jsonl}"
OVERWRITE_RESULTS="${OVERWRITE_RESULTS:-0}"
SKIP_DONE_MODELS="${SKIP_DONE_MODELS:-1}"

MAX_ROUND="${MAX_ROUND:-20}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS_PER_TURN="${MAX_TOKENS_PER_TURN:-200}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-600}"
N_SAMPLES="${N_SAMPLES:-1}"

EVAL_DATA_DIR="${EVAL_DATA_DIR:-${ROOT}/AgentEval/babyai}"

RUNTIME_BASE="${RUNTIME_BASE:-/idfsdata/yexuyan/rb}"
TMPDIR="${TMPDIR:-${RUNTIME_BASE}/tmp}"
TMP="${TMP:-${TMPDIR}}"
TEMP="${TEMP:-${TMPDIR}}"
HF_HOME="${HF_HOME:-${RUNTIME_BASE}/hf}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_BASE}/xdg_cache}"
WANDB_DIR="${WANDB_DIR:-${RUNTIME_BASE}/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/.cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/.config}"
RAY_TMPDIR="${RAY_TMPDIR:-${RUNTIME_BASE}/ray_tmp}"

mkdir -p \
  "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" \
  "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${RAY_TMPDIR}" \
  "$(dirname "${RESULTS_FILE}")" "${EVAL_DATA_DIR}"

if [[ "${OVERWRITE_RESULTS}" == "1" ]]; then
  : > "${RESULTS_FILE}"
fi

source "${CONDA_SH}"
set +u
conda activate "${TRAIN_ENV}"
set -u

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"
if (( NUM_GPUS < 1 )); then
  echo "CUDA_VISIBLE_DEVICES is empty." >&2
  exit 1
fi

if [[ ! -d "${CKPT_ROOT}" ]]; then
  echo "Checkpoint root not found: ${CKPT_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${EVAL_DATA_DIR}/babyai_test.json" ]]; then
  echo "BabyAI eval file not found: ${EVAL_DATA_DIR}/babyai_test.json" >&2
  exit 1
fi

should_skip_model() {
  local model_label="$1"
  local results_file="$2"

  if [[ "${SKIP_DONE_MODELS}" != "1" ]]; then
    return 1
  fi
  if [[ ! -s "${results_file}" ]]; then
    return 1
  fi

  python3 - "${results_file}" "${model_label}" <<'PY'
import json
import sys

path, target = sys.argv[1], sys.argv[2]
for raw in open(path, "r", encoding="utf-8"):
    line = raw.strip()
    if not line:
        continue
    try:
        row = json.loads(line)
    except Exception:
        continue
    if row.get("model_label") == target:
        print("1")
        break
else:
    print("0")
PY
}

has_hf_weights() {
  local model_path="$1"
  [[ -f "${model_path}/model.safetensors.index.json" || -f "${model_path}/model.safetensors" || -f "${model_path}/pytorch_model.bin" ]]
}

is_oom_log() {
  local log_file="$1"
  grep -Eiq \
    "cuda out of memory|outofmemoryerror|no available memory for the cache blocks|oom|allocation failed|device out of memory" \
    "${log_file}"
}

run_one_model_with_fallback() {
  local model_label="$1"
  local model_path="$2"
  local log_path="$3"
  local run_dir="$4"

  local -a candidates=(
    "64:0.86:256:32768"
    "48:0.84:192:24576"
    "32:0.82:160:20480"
    "24:0.80:128:16384"
    "16:0.78:96:12288"
  )

  local ok=0
  local attempt=0
  for c in "${candidates[@]}"; do
    attempt=$((attempt + 1))
    IFS=':' read -r batch_size gpu_util max_num_seqs max_num_batched_tokens <<< "${c}"
    local attempt_log="${run_dir}/eval_attempt_${attempt}.log"

    {
      echo "===== ATTEMPT ${attempt} model=${model_label} ====="
      echo "batch_size=${batch_size} gpu_util=${gpu_util} max_num_seqs=${max_num_seqs} max_num_batched_tokens=${max_num_batched_tokens}"
    } | tee -a "${log_path}"

    if (
      cd "${CODE_DIR}"
      exec env \
        -u http_proxy -u https_proxy -u all_proxy \
        -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        NO_PROXY="${NO_PROXY}" \
        no_proxy="${no_proxy}" \
        TMPDIR="${TMPDIR}" \
        TMP="${TMP}" \
        TEMP="${TEMP}" \
        HF_HOME="${HF_HOME}" \
        TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
        XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
        WANDB_DIR="${WANDB_DIR}" \
        WANDB_CACHE_DIR="${WANDB_CACHE_DIR}" \
        WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR}" \
        RAY_TMPDIR="${RAY_TMPDIR}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        VLLM_USE_MODELSCOPE=0 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        VLLM_ATTENTION_BACKEND=XFORMERS \
        HYDRA_FULL_ERROR=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python -m verl.agent_trainer.main_generation \
          data.path="${EVAL_DATA_DIR}" \
          data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
          data.max_response_length="${MAX_RESPONSE_LENGTH}" \
          data.n_samples="${N_SAMPLES}" \
          data.batch_size="${batch_size}" \
          agentgym.task_name=babyai \
          agentgym.env_addr="${ENV_ADDR}" \
          agentgym.max_rounds="${MAX_ROUND}" \
          agentgym.timeout="${AGENT_TIMEOUT}" \
          model.path="${model_path}" \
          rollout.gpu_memory_utilization="${gpu_util}" \
          rollout.temperature=1 \
          rollout.max_model_len="${MAX_MODEL_LEN}" \
          rollout.max_tokens="${MAX_TOKENS_PER_TURN}" \
          rollout.max_num_seqs="${max_num_seqs}" \
          rollout.max_num_batched_tokens="${max_num_batched_tokens}" \
          rollout.tensor_model_parallel_size=1 \
          rollout.rollout_log_dir="${run_dir}/executer_logs" \
          trainer.nnodes=1 \
          trainer.n_gpus_per_node="${NUM_GPUS}"
    ) 2>&1 | tee "${attempt_log}" | tee -a "${log_path}"; then
      ok=1
      break
    else
      if is_oom_log "${attempt_log}"; then
        echo "OOM detected on attempt ${attempt}, fallback to lower concurrency..." | tee -a "${log_path}"
        continue
      fi
      echo "Non-OOM failure on attempt ${attempt}, aborting model ${model_label}." | tee -a "${log_path}"
      return 1
    fi
  done

  if [[ "${ok}" != "1" ]]; then
    echo "All fallback attempts failed for ${model_label}." | tee -a "${log_path}"
    return 1
  fi
}

MODEL_LABELS=()
MODEL_PATHS=()

mapfile -t CKPTS < <(find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' | sort -V)
for ckpt_dir in "${CKPTS[@]}"; do
  step_name="$(basename "${ckpt_dir}")"
  MODEL_LABELS+=("${step_name}")
  MODEL_PATHS+=("${ckpt_dir}/actor/huggingface")
done

if (( ${#MODEL_LABELS[@]} == 0 )); then
  echo "No global_step_* checkpoints found under ${CKPT_ROOT}" >&2
  exit 1
fi

echo "Found ${#MODEL_LABELS[@]} BabyAI checkpoints under ${CKPT_ROOT}"
echo "Results file: ${RESULTS_FILE}"

for idx in "${!MODEL_LABELS[@]}"; do
  model_label="${MODEL_LABELS[$idx]}"
  model_path="${MODEL_PATHS[$idx]}"

  if [[ "$(should_skip_model "${model_label}" "${RESULTS_FILE}")" == "1" ]]; then
    echo "===== EVAL SKIP model=${model_label} ====="
    continue
  fi

  if [[ ! -d "${model_path}" ]]; then
    echo "Missing model directory for ${model_label}: ${model_path}" >&2
    continue
  fi
  if ! has_hf_weights "${model_path}"; then
    echo "Missing HF weights for ${model_label}: ${model_path}" >&2
    continue
  fi

  run_name="eval_babyai_${model_label}_$(date -u +%Y%m%d_%H%M%S)"
  run_dir="${ROOT}/runlogs/${run_name}"
  log_path="${run_dir}/eval.log"
  metrics_json_path="${run_dir}/metrics.json"
  mkdir -p "${run_dir}"

  echo "===== EVAL START model=${model_label} =====" | tee -a "${log_path}"
  run_one_model_with_fallback "${model_label}" "${model_path}" "${log_path}" "${run_dir}"

  python3 - "${log_path}" "${metrics_json_path}" <<'PY'
import json
import sys

log_path, out_path = sys.argv[1], sys.argv[2]
metrics_line = None
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "METRICS_JSON:" in line:
            metrics_line = line.split("METRICS_JSON:", 1)[1].strip()
if metrics_line is None:
    raise SystemExit(f"Failed to find METRICS_JSON in log: {log_path}")
metrics = json.loads(metrics_line)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=True, sort_keys=True)
PY

  read -r score succ pass_rate <<<"$(python3 - "${metrics_json_path}" <<'PY'
import json
import sys
metrics = json.load(open(sys.argv[1], "r", encoding="utf-8"))
overall = metrics.get("overall", {})
print(
    f"{float(overall.get('score', 0.0))} "
    f"{float(overall.get('succ', 0.0))} "
    f"{float(overall.get('pass', 0.0))}"
)
PY
)"
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 - "${RESULTS_FILE}" "${ts}" "${model_label}" "${model_path}" "${score}" "${succ}" "${pass_rate}" "${metrics_json_path}" "${log_path}" <<'PY'
import json
import sys

out_path, ts, label, model_path, score, succ, pass_rate, metrics_path, log_path = sys.argv[1:]
metrics = json.load(open(metrics_path, "r", encoding="utf-8"))
row = {
    "timestamp": ts,
    "task": "babyai",
    "split": "test",
    "model_label": label,
    "model_path": model_path,
    "score": float(score),
    "succ": float(succ),
    "pass": float(pass_rate),
    "metrics": metrics,
    "log_path": log_path,
}
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
PY

  echo "===== EVAL DONE model=${model_label} score=${score} succ=${succ} pass=${pass_rate} =====" | tee -a "${log_path}"
done

echo "All BabyAI checkpoint evaluations completed."
echo "Results file: ${RESULTS_FILE}"
python3 - "${RESULTS_FILE}" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], "r", encoding="utf-8") if line.strip()]
print("===== SUMMARY =====")
for row in rows:
    print(
        f"{row.get('model_label', 'unknown'):>20} | "
        f"Score={float(row.get('score', 0.0)):.4f} | "
        f"Succ={float(row.get('succ', 0.0)):.4f} | "
        f"Pass={float(row.get('pass', 0.0)):.4f}"
    )
PY
