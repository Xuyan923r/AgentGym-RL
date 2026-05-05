#!/usr/bin/env python3

import argparse
import collections
import glob
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build supplement input so every ALFWorld item reaches N rollouts.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--executer-log-dir", required=True)
    parser.add_argument("--target-rollouts", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_rows = json.load(open(args.train_file, "r", encoding="utf-8"))
    counts = collections.Counter()

    for path in glob.glob(str(Path(args.executer_log_dir) / "steptest_batch_*" / "*.json")):
        rows = json.load(open(path, "r", encoding="utf-8"))
        for row in rows:
            counts[int(row["item_id"])] += 1

    supplement = []
    for row in train_rows:
        item_id = str(row["item_id"])
        idx = int(item_id.split("_")[-1])
        need = max(0, int(args.target_rollouts) - counts[idx])
        for _ in range(need):
            supplement.append({"item_id": item_id})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(supplement, f, ensure_ascii=True)

    print(f"Wrote {len(supplement)} supplement rollout requests to {output_path}")


if __name__ == "__main__":
    main()
