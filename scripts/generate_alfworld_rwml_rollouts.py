#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Collect ALFWorld rollout logs into one JSON file for RWML triplet building.")
    parser.add_argument("--rollout-log-dir", required=True, help="Directory containing step*/rank.json rollout logs.")
    parser.add_argument("--output", required=True, help="Output JSON file.")
    args = parser.parse_args()

    rollout_log_dir = Path(args.rollout_log_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    for step_dir in sorted(rollout_log_dir.glob("step*")):
        for rank_file in sorted(step_dir.glob("*.json")):
            try:
                payload = json.load(open(rank_file, "r", encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, list):
                all_records.extend(payload)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=True)
    print(f"Wrote {len(all_records)} rollout records to {output_path}")


if __name__ == "__main__":
    main()
