#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = REPO_ROOT / "AgentGym-RL"
AGENTENV_ROOT = REPO_ROOT / "AgentGym"
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))
if str(AGENTENV_ROOT / "agentenv") not in sys.path:
    sys.path.insert(0, str(AGENTENV_ROOT / "agentenv"))

from vllm import LLM, SamplingParams
from agentenv.envs.alfworld import AlfWorldEnvClient
from verl.workers.rollout.schemas import Message, RolloutHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone ALFWorld RWML rollout without verl/Ray.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--rollout-log-dir", required=True)
    parser.add_argument("--env-addrs", required=True, help="Comma-separated ALFWorld server URLs.")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens-per-turn", type=int, default=200)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-num-seqs", type=int, default=96)
    parser.add_argument("--max-num-batched-tokens", type=int, default=24576)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--resume-batch-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--send-interval", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-stride", type=int, default=1)
    parser.add_argument("--batch-offset", type=int, default=0)
    parser.add_argument("--batch-index-base", type=int, default=0)
    return parser.parse_args()


def load_item_ids(input_file: Path) -> List[str]:
    rows = json.load(open(input_file, "r", encoding="utf-8"))
    return [str(row["item_id"]) for row in rows]


def build_initial_prompt(
    env_client: AlfWorldEnvClient,
    tokenizer,
    item_id_str: str,
    max_prompt_length: int,
    max_response_length: int,
    max_model_len: int,
) -> RolloutHandler:
    raw_prompt = [
        {"role": "user", "content": env_client.conversation_start[0]["value"]},
        {"role": "assistant", "content": env_client.conversation_start[1]["value"]},
    ]
    prompt_with_chat_template = (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        + env_client.conversation_start[0]["value"]
        + "<|im_end|>\n"
        "<|im_start|>assistant\n"
        + env_client.conversation_start[1]["value"]
        + "<|im_end|>"
    )
    prompt_ids = tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]
    attention_mask = [1] * len(prompt_ids)
    position_ids = list(range(len(prompt_ids)))
    return RolloutHandler(
        messages=[Message(role=msg["role"], content=msg["content"]) for msg in raw_prompt],
        task_name=item_id_str.split("_")[0],
        item_id=int(item_id_str.split("_")[-1]),
        score=0.0,
        done=False,
        input_ids=list(prompt_ids),
        prompt_ids=list(prompt_ids),
        response_ids=[],
        attention_mask=list(attention_mask),
        prompt_attention_mask=list(attention_mask),
        response_attention_mask=[],
        position_ids=list(position_ids),
        prompt_position_ids=list(position_ids),
        response_position_ids=[],
        loss_mask=[0] * len(prompt_ids),
        prompt_loss_mask=[0] * len(prompt_ids),
        response_loss_mask=[],
        observation_mask=[0] * len(prompt_ids),
        prompt_observation_mask=[0] * len(prompt_ids),
        response_observation_mask=[],
        max_response_len=max_response_length,
        max_model_len=min(max_model_len, max_prompt_length + max_response_length),
    )


def write_step_log(step_dir: Path, rank_idx: int, payload: List[Dict[str, Any]]) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    with (step_dir / f"{rank_idx}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=4)


def main() -> None:
    global args
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rollout_log_dir = Path(args.rollout_log_dir)
    rollout_log_dir.mkdir(parents=True, exist_ok=True)

    env_addrs = [addr.strip() for addr in args.env_addrs.split(",") if addr.strip()]
    if not env_addrs:
        raise ValueError("No env server address was provided.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_log_stats=True,
        enable_chunked_prefill=True,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens_per_turn,
        logprobs=1,
        detokenize=False,
    )

    item_ids = load_item_ids(Path(args.input_file))
    total_batches = math.ceil(len(item_ids) / args.batch_size)
    print(f"Loaded {len(item_ids)} items from {args.input_file}; total batches={total_batches}")

    env_clients = [
        AlfWorldEnvClient(env_server_base=env_addrs[i % len(env_addrs)], data_len=1, timeout=2400)
        for i in range(args.batch_size * args.n_samples)
    ]

    try:
        for local_batch_idx, start in enumerate(range(0, len(item_ids), args.batch_size)):
            global_batch_idx = args.batch_index_base + args.resume_batch_idx + local_batch_idx
            if global_batch_idx % args.batch_stride != args.batch_offset:
                continue
            step_dir = rollout_log_dir / f"steptest_batch_{global_batch_idx}"
            if step_dir.exists():
                print(f"[skip] batch {global_batch_idx} already exists")
                continue

            batch_item_ids = item_ids[start : start + args.batch_size]
            handlers: List[RolloutHandler] = []
            client_indices: List[int] = []

            for item_offset, item_id_str in enumerate(batch_item_ids):
                for sample_idx in range(args.n_samples):
                    client_idx = item_offset * args.n_samples + sample_idx
                    env_client = env_clients[client_idx]
                    handler = build_initial_prompt(
                        env_client=env_client,
                        tokenizer=tokenizer,
                        item_id_str=item_id_str,
                        max_prompt_length=args.max_prompt_length,
                        max_response_length=args.max_response_length,
                        max_model_len=args.max_model_len,
                    )
                    try:
                        env_client.reset(handler.item_id)
                        handler.add_user_message(tokenizer, env_client.observe())
                    except Exception as exc:
                        print(f"[reset-error] batch={global_batch_idx} item={handler.item_id}: {exc}")
                        handler.done = True
                        handler.score = 0.0
                    handlers.append(handler)
                    client_indices.append(client_idx)

            for round_idx in range(args.max_rounds):
                active = [(idx, handler) for idx, handler in enumerate(handlers) if not handler.done]
                if not active:
                    break
                print(
                    f"Rounds {round_idx + 1}/{args.max_rounds} | Active trajectories: {len(active)} | batch {global_batch_idx}/{args.resume_batch_idx + total_batches - 1}",
                    flush=True,
                )
                prompt_token_ids = [
                    handler.get_generation_prompt(tokenizer) for _, handler in active
                ]
                outputs = llm.generate(
                    prompts=None,
                    prompt_token_ids=prompt_token_ids,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

                def step_one(pair):
                    output_idx, handler_idx = pair
                    handler = handlers[handler_idx]
                    env_client = env_clients[client_indices[handler_idx]]
                    content = tokenizer.decode(outputs[output_idx].outputs[0].token_ids, skip_special_tokens=True)
                    handler.add_assistant_message(tokenizer, content)
                    try:
                        step_output = env_client.step(content)
                        handler.score = float(step_output.reward)
                        handler.done = bool(step_output.done)
                        handler.add_user_message(tokenizer, step_output.state)
                    except Exception as exc:
                        print(f"[step-error] batch={global_batch_idx} item={handler.item_id}: {exc}")
                        handler.score = 0.0
                        handler.done = True

                with ThreadPoolExecutor(max_workers=max(1, len(active))) as executor:
                    list(executor.map(step_one, [(i, handler_idx) for i, (handler_idx, _) in enumerate(active)]))

                if args.send_interval > 0:
                    time.sleep(args.send_interval)

            shard_payloads: List[List[Dict[str, Any]]] = [[] for _ in range(args.tensor_parallel_size)]
            for idx, handler in enumerate(handlers):
                rank_idx = idx % args.tensor_parallel_size
                shard_payloads[rank_idx].append(
                    {
                        "item_id": handler.item_id,
                        "conversations": [msg.to_dict() for msg in handler.messages],
                        "reward": float(handler.score),
                        "raw_score": float(handler.score),
                        "done": bool(handler.done),
                    }
                )
            for rank_idx, payload in enumerate(shard_payloads):
                if payload:
                    write_step_log(step_dir, rank_idx, payload)
            print(f"[done] wrote batch {global_batch_idx} to {step_dir}", flush=True)
    finally:
        for client in env_clients:
            try:
                client.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
