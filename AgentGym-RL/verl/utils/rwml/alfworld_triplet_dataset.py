import copy
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import datasets
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def collate_triplet_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", ""))
        if not role:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _extract_conversation_fields(row: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    messages = _normalize_messages(row.get("messages", []))
    target = str(row.get("target_next_observation", ""))
    if not messages:
        raise ValueError("RWML triplet row is missing messages")
    if not target:
        raise ValueError("RWML triplet row is missing target_next_observation")
    return messages, target


class ALFWorldRWMLTripletDataset(Dataset):
    """Offline triplet dataset for ALFWorld RWML stage.

    Each row is expected to contain:
    - ``item_id``: e.g. ``alfworld_123``
    - ``messages``: chat history ending with the last agent action
    - ``target_next_observation``: realized next environment observation
    - optional metadata such as ``task_name``, ``action_text`` and ``turn_index``
    """

    def __init__(
        self,
        data_file: str,
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        rollout_config: DictConfig,
    ):
        self.data_file = copy.deepcopy(data_file)
        self.original_data_file = copy.deepcopy(data_file)
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.rollout_config = rollout_config

        self.max_prompt_length = int(data_config.get("max_prompt_length", 2048))
        self.return_raw_chat = bool(data_config.get("return_raw_chat", True))
        self.truncation = data_config.get("truncation", "error")

        self._read_files()

    def _read_files(self):
        self.dataframe = datasets.load_dataset("json", data_files=self.data_file)["train"]
        logger.info("Loaded ALFWorld RWML dataset with %d rows from %s", len(self.dataframe), self.data_file)

    def resume_dataset_state(self):
        self._read_files()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row_dict: dict = dict(self.dataframe[item])
        messages, target_next_observation = _extract_conversation_fields(row_dict)

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_text,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        item_id = str(row_dict.get("item_id", f"alfworld_{item}"))
        task_name = str(row_dict.get("task_name", item_id.split("_")[0] if "_" in item_id else "alfworld"))

        output = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "item_id": item_id,
            "data_source": task_name,
            "reward_model": {
                "ground_truth": target_next_observation,
            },
            "rwml_target_next_observation": target_next_observation,
            "rwml_task_name": task_name,
            "rwml_action_text": str(row_dict.get("action_text", "")),
            "rwml_turn_index": int(row_dict.get("turn_index", 0)),
            "rwml_metadata": {
                key: row_dict.get(key)
                for key in ["task_name", "action_text", "turn_index", "source_file"]
                if key in row_dict
            },
        }
        if self.return_raw_chat:
            output["raw_prompt"] = messages
        return output
