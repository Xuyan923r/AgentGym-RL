# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
from collections import defaultdict
import json
import math
import ray
import numpy as np
import hydra
import verl.utils.torch_functional as verl_F
import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.agent_fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.agentgym.client import init_env_client


SCIWORLD_TASK_TO_TOPIC = {
    "boil": "Matter",
    "melt": "Matter",
    "freeze": "Matter",
    "change-the-state-of-matter-of": "Matter",
    "use-thermometer": "Measurement",
    "measure-melting-point-known-substance": "Measurement",
    "measure-melting-point-unknown-substance": "Measurement",
    "power-component": "Electricity",
    "power-component-renewable-vs-nonrenewable-energy": "Electricity",
    "test-conductivity": "Electricity",
    "test-conductivity-of-unknown-substances": "Electricity",
    "find-living-thing": "Classification",
    "find-non-living-thing": "Classification",
    "find-plant": "Classification",
    "find-animal": "Classification",
    "grow-plant": "Biology",
    "grow-fruit": "Biology",
    "chemistry-mix": "Chemistry",
    "chemistry-mix-paint-secondary-color": "Chemistry",
    "chemistry-mix-paint-tertiary-color": "Chemistry",
    "lifespan-longest-lived": "Biology",
    "lifespan-shortest-lived": "Biology",
    "lifespan-longest-lived-then-shortest-lived": "Biology",
    "identify-life-stages-1": "Biology",
    "identify-life-stages-2": "Biology",
    "inclined-plane-determine-angle": "Forces",
    "inclined-plane-friction-named-surfaces": "Forces",
    "inclined-plane-friction-unnamed-surfaces": "Forces",
    "mendelian-genetics-known-plant": "Biology",
    "mendelian-genetics-unknown-plant": "Biology",
}

SCIWORLD_TOPIC_ALIASES = {
    "Biology": "Bio.",
    "Chemistry": "Chem.",
    "Classification": "Class.",
    "Electricity": "Elec.",
    "Matter": "Matt.",
    "Measurement": "Meas",
    "Forces": "Forces",
}

SCIWORLD_TOPIC_ORDER = {
    "Bio.": 0,
    "Chem.": 1,
    "Class.": 2,
    "Elec.": 3,
    "Matt.": 4,
    "Meas": 5,
    "Forces": 6,
}

ALFWORLD_TASK_PREFIX_TO_TOPIC = [
    ("pick_and_place_simple-", "Pick"),
    ("look_at_obj_in_light-", "Look"),
    ("pick_clean_then_place_in_recep-", "Clean"),
    ("pick_heat_then_place_in_recep-", "Heat"),
    ("pick_cool_then_place_in_recep-", "Cool"),
    ("pick_two_obj_and_place-", "Pick2"),
]

ALFWORLD_TOPIC_ORDER = {
    "Pick": 0,
    "Look": 1,
    "Clean": 2,
    "Heat": 3,
    "Cool": 4,
    "Pick2": 5,
}


def _normalize_sciworld_topic(topic):
    return SCIWORLD_TOPIC_ALIASES.get(topic, topic or "Unknown")


def _normalize_alfworld_topic(task_type):
    task_type = str(task_type or "")
    for prefix, topic in ALFWORLD_TASK_PREFIX_TO_TOPIC:
        if task_type.startswith(prefix):
            return topic
    return "Unknown"


def _topic_sort_key(task_name_lower, topic):
    if task_name_lower == "alfworld":
        return (ALFWORLD_TOPIC_ORDER.get(topic, 10**6), topic)
    if task_name_lower == "sciworld":
        return (SCIWORLD_TOPIC_ORDER.get(topic, 10**6), topic)
    return (10**6, topic)


def _extract_item_index(item_id):
    try:
        return int(str(item_id).split("_")[-1])
    except Exception:
        return None


def _build_item_metadata(env_client, item_ids):
    metadata = {}
    total = len(item_ids)
    for idx, item_id in enumerate(item_ids):
        item_index = _extract_item_index(item_id)
        task_name = "unknown_task"
        topic = "Unknown"
        if item_index is not None:
            try:
                reset_info = env_client.reset(item_index)
                task_name = reset_info.get("task_name", "unknown_task")
                topic = SCIWORLD_TASK_TO_TOPIC.get(task_name, "Unknown")
            except Exception as e:
                print(f"Failed to resolve topic for item_id={item_id}: {e}")
        metadata[item_id] = {"task_name": task_name, "topic": topic}
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"Resolved item metadata: {idx + 1}/{total}")
    return metadata


def _build_alfworld_item_metadata(item_ids, mappings_path):
    with open(mappings_path, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    mapping_by_item_id = {}
    for row in mappings:
        item_id = f"alfworld_{int(row['item_id'])}"
        task_type = row.get("task_type", "")
        mapping_by_item_id[item_id] = {
            "task_name": task_type,
            "topic": _normalize_alfworld_topic(task_type),
        }

    metadata = {}
    for item_id in item_ids:
        metadata[item_id] = mapping_by_item_id.get(
            item_id,
            {"task_name": "unknown_task", "topic": "Unknown"},
        )
    return metadata


def _aggregate_metrics(score_np, done_np, success_score=100.0):
    # Strict success:
    # - If success_score is None: done=True is enough.
    # - Else: done=True and score reaches success_score.
    if success_score is None:
        success_mask = done_np > 0
    else:
        success_mask = np.logical_and(
            done_np > 0,
            np.isclose(score_np, float(success_score), rtol=0.0, atol=1e-6),
        )
    return {
        "score": float(np.mean(score_np)),
        "pass": float(np.mean(np.max(score_np, axis=-1) > 0)),
        "succ": float(np.mean(np.max(success_mask, axis=-1) > 0)),
        "count": int(score_np.shape[0]),
    }


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. By default we read {data.path}/{task_name}_test.json for evaluation,
    # but callers can override with data.input_file to reuse the same rollout path on
    # arbitrary splits such as ALFWorld train ids for RWML data collection.
    input_file = getattr(config.data, "input_file", None)
    if input_file is None:
        input_file = os.path.join(config.data.path, f"{config.agentgym.task_name}_test.json")
    dataset = pd.read_json(input_file)
    item_ids = dataset[config.data.prompt_key].tolist()

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
    num_batch = math.ceil(total_samples / config_batch_size)
    start_batch_idx = int(getattr(config.data, "start_batch_idx", 0) or 0)
    score_lst = [[] for _ in range(config.data.n_samples)]
    done_lst = [[] for _ in range(config.data.n_samples)]
    env_client = init_env_client(config.agentgym)
    task_name_lower = str(config.agentgym.task_name).lower()
    if task_name_lower == "sciworld":
        item_metadata = _build_item_metadata(env_client, item_ids)
        for item_id, meta in item_metadata.items():
            meta["topic"] = _normalize_sciworld_topic(meta.get("topic"))
    elif task_name_lower == "alfworld":
        mappings_path = getattr(config.data, "topic_mapping_file", None)
        if mappings_path is None:
            mappings_path = os.path.join(
                config.data.path,
                "alfworld_test_mappings.json",
            )
        item_metadata = _build_alfworld_item_metadata(item_ids, mappings_path)
    else:
        # Avoid expensive per-item reset probing on unrelated tasks.
        item_metadata = {
            item_id: {"task_name": "unknown_task", "topic": "All"}
            for item_id in item_ids
        }

    for batch_idx in range(num_batch):
        display_batch_idx = start_batch_idx + batch_idx
        print(f'[{display_batch_idx+1}/{start_batch_idx + num_batch}] Start to process.')
        start_idx = batch_idx * config_batch_size
        end_idx = min(total_samples, start_idx + config_batch_size)
        batch_item_ids = item_ids[start_idx: end_idx]
        if not batch_item_ids:
            print(f'[{display_batch_idx+1}/{start_batch_idx + num_batch}] Empty batch, skip.')
            continue
        prompt_with_chat_template = ["<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + env_client.conversation_start[0]["value"] + "<|im_end|>\n<|im_start|>assistant\n" + env_client.conversation_start[1]["value"] + "<|im_end|>" for _ in range(len(batch_item_ids))]
        messages = [[{"role": "user", "content": env_client.conversation_start[0]["value"]},
                     {"role": "assistant", "content": env_client.conversation_start[1]["value"]}] for _ in range(len(batch_item_ids))]

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=tokenizer,
                                                                         max_length=config.data.max_prompt_length,
                                                                         pad_token_id=tokenizer.pad_token_id,
                                                                         left_pad=True)
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        data.meta_info['global_steps'] = 'test_batch_' + str(display_batch_idx)
        data.meta_info['max_rounds'] = config.agentgym.max_rounds
        data.non_tensor_batch["item_id"] = np.array(batch_item_ids, dtype=object)
        data.non_tensor_batch["raw_prompt"] = np.array(messages, dtype=object)
        real_batch_size = data.batch['input_ids'].shape[0]
        if real_batch_size % dp_size != 0:
            dummy_data_size = dp_size - real_batch_size % dp_size
            dummy_data = data[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(
                f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
            )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

        print(f'[{display_batch_idx+1}/{start_batch_idx + num_batch}] Start to generate.')

        for i in range(config.data.n_samples):
            output = wg.generate_sequences(data)
            # remove dummy data
            output = output[:real_batch_size]

            score_lst[i].extend(output.batch['task_scores'].sum(dim=-1).tolist())
            if 'task_dones' in output.batch.keys():
                done_lst[i].extend(output.batch['task_dones'].tolist())
            else:
                # Backward compatible fallback if rollout worker does not emit task_dones.
                done_lst[i].extend([0.0] * real_batch_size)

    # convert from (n_samples, n_data) to (n_data, n_samples)
    score_np = np.array(score_lst, dtype=np.float32).transpose(1, 0)
    done_np = np.array(done_lst, dtype=np.float32).transpose(1, 0)
    # Task-specific strict success threshold.
    # - SciWorld: done + score==100
    # - ALFWorld: done + won==1 (score==1)
    # - WebShop: done + reward score==1 (exact purchase match)
    # - BabyAI: done=True
    success_score = 100.0
    if task_name_lower in {"alfworld", "webshop"}:
        success_score = 1.0
    elif task_name_lower == "babyai":
        success_score = None
    overall_metrics = _aggregate_metrics(score_np, done_np, success_score=success_score)

    print("============Total Task Evaluation============")
    print(f"Score@{config.data.n_samples}: {overall_metrics['score']}")
    print(f"Avg@{config.data.n_samples}: {overall_metrics['score']}")
    print(f"Pass@{config.data.n_samples}: {overall_metrics['pass']}")
    if success_score is None:
        print("Succ definition: done=True")
    else:
        print(f"Succ definition: done=True and score={success_score}")
    print(f"Succ@{config.data.n_samples}: {overall_metrics['succ']}")
    print("============Per Topic Evaluation============")

    topic_scores = defaultdict(list)
    topic_dones = defaultdict(list)
    for idx, item_id in enumerate(item_ids):
        topic = item_metadata.get(item_id, {}).get("topic", "Unknown")
        topic_scores[topic].append(score_np[idx].tolist())
        topic_dones[topic].append(done_np[idx].tolist())

    per_topic_metrics = {}
    for topic in sorted(topic_scores.keys(), key=lambda x: _topic_sort_key(task_name_lower, x)):
        topic_score_np = np.array(topic_scores[topic], dtype=np.float32)
        topic_done_np = np.array(topic_dones[topic], dtype=np.float32)
        metrics = _aggregate_metrics(topic_score_np, topic_done_np, success_score=success_score)
        per_topic_metrics[topic] = metrics
        print(f"Topic: {topic}")
        print(f"Score@{config.data.n_samples}: {metrics['score']}")
        if success_score is None:
            print("Succ definition: done=True")
        else:
            print(f"Succ definition: done=True and score={success_score}")
        print(f"Succ@{config.data.n_samples}: {metrics['succ']}")
        print(f"Pass@{config.data.n_samples}: {metrics['pass']}")

    metrics_json = {
        "n_samples": int(config.data.n_samples),
        "overall": overall_metrics,
        "per_topic": per_topic_metrics,
    }
    print("METRICS_JSON:", json.dumps(metrics_json, sort_keys=True))

    try:
        env_client.close()
    except Exception:
        pass



if __name__ == '__main__':
    main()
