#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _normalize_messages(messages):
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", ""))
        if not role:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def extract_triplets_from_record(record: Dict) -> List[Dict]:
    item_id = str(record.get("item_id", "alfworld_unknown"))
    messages = _normalize_messages(record.get("conversations", []))
    triplets: List[Dict] = []
    last_assistant_idx = None
    for idx, msg in enumerate(messages):
        if msg["role"] == "assistant":
            last_assistant_idx = idx
            continue
        if msg["role"] != "user":
            continue
        if last_assistant_idx is None:
            continue
        history = messages[: last_assistant_idx + 1]
        action_text = history[-1]["content"]
        target_next_observation = msg["content"]
        triplets.append(
            {
                "item_id": item_id,
                "task_name": str(item_id).split("_")[0],
                "turn_index": len(triplets),
                "action_text": action_text,
                "messages": history,
                "target_next_observation": target_next_observation,
            }
        )
    return triplets


def main():
    parser = argparse.ArgumentParser(description="Build ALFWorld RWML triplets from rollout logs.")
    parser.add_argument("--input", required=True, help="Path to a rollout log JSON or JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument(
        "--rollouts-per-task",
        type=int,
        default=3,
        help="Appendix B.1 default for ALFWorld data collection: rollout trajectories per training task N=3. Metadata only.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    if input_path.suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    else:
        with input_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            records.extend(payload)
        else:
            raise ValueError("Expected list payload in JSON input")

    num_triplets = 0
    with output_path.open("w", encoding="utf-8") as out:
        for record in records:
            for triplet in extract_triplets_from_record(record):
                triplet["rollouts_per_task"] = int(args.rollouts_per_task)
                out.write(json.dumps(triplet, ensure_ascii=True) + "\n")
                num_triplets += 1

    print(f"Wrote {num_triplets} triplets to {output_path}")


if __name__ == "__main__":
    main()
