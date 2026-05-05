#!/usr/bin/env bash

set -euo pipefail

ROOT="/idfsdata/yexuyan/AgentGym-RL"
CODE_DIR="${ROOT}/AgentGym-RL"
CONDA_SH="${CONDA_SH:-/home/yexuyan/miniconda3/etc/profile.d/conda.sh}"
TRAIN_ENV="${TRAIN_ENV:-/idfsdata/yexuyan/conda_envs/agentgym-rl-webshop}"

ENV_ADDR="${ENV_ADDR:-http://127.0.0.1:36005}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
RESULTS_FILE="${RESULTS_FILE:-${ROOT}/FinalResults_Sciworld.jsonl}"
OVERWRITE_RESULTS="${OVERWRITE_RESULTS:-1}"
SKIP_DONE_MODELS="${SKIP_DONE_MODELS:-1}"
INCLUDE_BASE_MODEL="${INCLUDE_BASE_MODEL:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.95}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-32768}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/idfsdata/yexuyan/AgentGym-RL/models/Qwen2.5-3B-Instruct}"
BASE_MODEL_LABEL="${BASE_MODEL_LABEL:-base_model}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/checkpoints/GRPO-3B-0414}"
CKPT_STEPS_STR="${CKPT_STEPS_STR-50 100 150 200}"

if [[ "${OVERWRITE_RESULTS}" == "1" ]]; then
  : > "${RESULTS_FILE}"
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

  python - "${results_file}" "${model_label}" <<'PY'
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

source "${CONDA_SH}"
set +u
conda activate "${TRAIN_ENV}"
set -u

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_ARRAY[@]}"
if (( NUM_GPUS < 1 )); then
  echo "CUDA_VISIBLE_DEVICES is empty."
  exit 1
fi

# main_generation requires effective batch size divisible by dp_size (num GPUs).
# Auto-adjust to avoid immediate assertion failures (e.g. batch_size=1 with 8 GPUs).
if ! [[ "${EVAL_BATCH_SIZE}" =~ ^[0-9]+$ ]]; then
  echo "Invalid EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}, fallback to ${NUM_GPUS}"
  EVAL_BATCH_SIZE="${NUM_GPUS}"
fi
if (( EVAL_BATCH_SIZE < NUM_GPUS )); then
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE} < NUM_GPUS=${NUM_GPUS}, bump to ${NUM_GPUS}"
  EVAL_BATCH_SIZE="${NUM_GPUS}"
fi
if (( EVAL_BATCH_SIZE % NUM_GPUS != 0 )); then
  ADJUSTED_BATCH_SIZE="$(( (EVAL_BATCH_SIZE / NUM_GPUS + 1) * NUM_GPUS ))"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE} is not divisible by NUM_GPUS=${NUM_GPUS}, adjust to ${ADJUSTED_BATCH_SIZE}"
  EVAL_BATCH_SIZE="${ADJUSTED_BATCH_SIZE}"
fi
echo "Using EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}, NUM_GPUS=${NUM_GPUS}"
echo "Using rollout config: gpu_util=${ROLLOUT_GPU_MEMORY_UTILIZATION}, max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}, max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}, max_model_len=${ROLLOUT_MAX_MODEL_LEN}"

read -r -a CKPT_STEPS <<< "${CKPT_STEPS_STR}"

MODEL_LABELS=()
MODEL_PATHS=()

if [[ "${INCLUDE_BASE_MODEL}" == "1" ]]; then
  MODEL_LABELS+=("${BASE_MODEL_LABEL}")
  MODEL_PATHS+=("${BASE_MODEL_PATH}")
fi

for step in "${CKPT_STEPS[@]}"; do
  MODEL_LABELS+=("global_step_${step}")
  MODEL_PATHS+=("${CKPT_ROOT}/global_step_${step}/actor/huggingface")
done

for idx in "${!MODEL_LABELS[@]}"; do
  model_label="${MODEL_LABELS[$idx]}"
  model_path="${MODEL_PATHS[$idx]}"

  if [[ "$(should_skip_model "${model_label}" "${RESULTS_FILE}")" == "1" ]]; then
    echo "===== EVAL SKIP model=${model_label} (already exists in ${RESULTS_FILE}) ====="
    continue
  fi

  run_name="eval_sciworld_${model_label}_$(date -u +%Y%m%d_%H%M%S)"
  log_dir="${ROOT}/runlogs/${run_name}"
  log_path="${log_dir}/eval.log"
  metrics_json_path="${log_dir}/metrics.json"

  mkdir -p "${log_dir}"

  if [[ ! -d "${model_path}" ]]; then
    echo "Missing model directory for ${model_label}: ${model_path}" >&2
    exit 1
  fi
  if [[ ! -f "${model_path}/model.safetensors.index.json" && ! -f "${model_path}/model.safetensors" && ! -f "${model_path}/pytorch_model.bin" ]]; then
    echo "Missing model weights for ${model_label}: ${model_path}" >&2
    exit 1
  fi

  echo "===== EVAL START model=${model_label} ====="
  echo "batch_size=${EVAL_BATCH_SIZE} gpu_util=${ROLLOUT_GPU_MEMORY_UTILIZATION} max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
  (
    cd "${CODE_DIR}"
    exec env \
      -u http_proxy -u https_proxy -u all_proxy \
      -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      NO_PROXY="${NO_PROXY}" \
      no_proxy="${no_proxy}" \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      VLLM_USE_MODELSCOPE=0 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      VLLM_ATTENTION_BACKEND=XFORMERS \
      HYDRA_FULL_ERROR=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python -m verl.agent_trainer.main_generation \
        data.path="${ROOT}/AgentEval/sciworld" \
        data.max_prompt_length=1024 \
        data.max_response_length=8192 \
        data.n_samples=1 \
        data.batch_size="${EVAL_BATCH_SIZE}" \
        agentgym.task_name=sciworld \
        agentgym.env_addr="${ENV_ADDR}" \
        agentgym.max_rounds=30 \
        agentgym.timeout=500 \
        model.path="${model_path}" \
        rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
        rollout.temperature=1 \
        rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
        rollout.max_tokens=200 \
        rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
        rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
        rollout.tensor_model_parallel_size=1 \
        rollout.rollout_log_dir="${log_dir}/executer_logs" \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node="${NUM_GPUS}"
  ) 2>&1 | tee "${log_path}"

  python - "${log_path}" "${metrics_json_path}" <<'PY'
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

  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  read -r overall_score overall_succ overall_pass <<< "$(python - "${metrics_json_path}" <<'PY'
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
  python - "${metrics_json_path}" "${RESULTS_FILE}" "${ts}" "${model_label}" "${model_path}" "${log_path}" <<'PY'
import json
import sys

metrics_path, results_path, ts, model_label, model_path, log_path = sys.argv[1:7]
metrics = json.load(open(metrics_path, "r", encoding="utf-8"))
per_topic = metrics.get("per_topic", {})
topic_scores = {
    topic: float(values.get("score", 0.0))
    for topic, values in per_topic.items()
}
row = {
    "timestamp": ts,
    "task": "sciworld",
    "split": "test",
    "model_label": model_label,
    "model_path": model_path,
    "score": float(metrics.get("overall", {}).get("score", 0.0)),
    "succ": float(metrics.get("overall", {}).get("succ", 0.0)),
    "pass": float(metrics.get("overall", {}).get("pass", 0.0)),
    "topic_scores": topic_scores,
    "topic_metrics": per_topic,
    "metrics": metrics,
    "log_path": log_path,
}
with open(results_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=True) + "\n")
PY

  python - "${metrics_json_path}" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], "r", encoding="utf-8"))
for topic, vals in metrics.get("per_topic", {}).items():
    print(
        f"TOPIC {topic}: "
        f"Score={float(vals.get('score', 0.0)):.3f} "
        f"Succ={float(vals.get('succ', 0.0)):.3f} "
        f"Pass={float(vals.get('pass', 0.0)):.3f} "
        f"Count={int(vals.get('count', 0))}"
    )
PY

  echo "===== EVAL DONE model=${model_label} score=${overall_score} succ=${overall_succ} pass=${overall_pass} ====="
done

echo "All SciWorld checkpoint evaluations completed."
echo "Results file: ${RESULTS_FILE}"
python - "${RESULTS_FILE}" <<'PY'
import json
import sys

path = sys.argv[1]
rows = [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]
print("===== SUMMARY =====")
for row in rows:
    print(
        f"{row.get('model_label', 'unknown'):>16} | "
        f"Score={float(row.get('score', 0.0)):.3f} | "
        f"Succ={float(row.get('succ', 0.0)):.3f} | "
        f"Pass={float(row.get('pass', 0.0)):.3f}"
    )
PY
