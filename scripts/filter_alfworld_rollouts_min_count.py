#!/usr/bin/env python3

import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Filter ALFWorld rollout records by minimum rollout count per item_id.")
    parser.add_argument("--input", required=True, help="Input rollout JSON file.")
    parser.add_argument("--output", required=True, help="Output filtered rollout JSON file.")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum number of rollout records required per item_id.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = json.load(open(input_path, "r", encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Expected list payload in input JSON.")

    counts = collections.Counter()
    for record in records:
        counts[int(record["item_id"])] += 1

    kept = [record for record in records if counts[int(record["item_id"])] >= args.min_count]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=True)

    print(f"Input records: {len(records)}")
    print(f"Kept records: {len(kept)}")
    print(f"Dropped records: {len(records) - len(kept)}")
    print(f"Unique kept item_ids: {len({int(r['item_id']) for r in kept})}")
    print(f"Wrote filtered rollout JSON to {output_path}")


if __name__ == "__main__":
    main()
