#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an expert agent operating in the ALFRED Embodied Environment."
)


def main():
    parser = argparse.ArgumentParser(description="Convert ALFWorld RWML triplets into WM SFT training data.")
    parser.add_argument("--input", required=True, help="RWML triplet JSONL file.")
    parser.add_argument("--output", required=True, help="WM SFT JSON file.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            history = row["messages"]
            if not history:
                continue
            current_state = ""
            potential_action = ""
            for msg in history:
                if msg["role"] == "user":
                    current_state = msg["content"]
                elif msg["role"] == "assistant":
                    potential_action = msg["content"]
            prompt = (
                f"{SYSTEM_PROMPT}\n"
                f"Your current observation is:\n{current_state}\n\n"
                f"Potential action:\n{potential_action}\n\n"
                "Now your task is to predict the immediate next observation after executing the potential action above. "
                "Directly present your final prediction of the next observation within <next_state> </next_state> tags. "
                "DO NOT generate anything else."
            )
            target = f"<think> </think>\n<next_state>{row['target_next_observation']}</next_state>"
            rows.append(
                {
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": target},
                    ]
                }
            )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=True)
    print(f"Wrote {len(rows)} WM SFT rows to {output_path}")


if __name__ == "__main__":
    main()
