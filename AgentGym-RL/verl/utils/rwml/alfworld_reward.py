import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def _strip_prediction(text: str) -> str:
    text = str(text or "").strip()
    patterns = [
        r"(?is)final\s+prediction\s*:\s*",
        r"(?is)predicted\s+next\s+observation\s*:\s*",
        r"(?is)next\s+observation\s*:\s*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    return text.strip()


def _round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(value / step) * step


@dataclass
class ALFWorldRWMLRewardConfig:
    embedding_model_path: str
    threshold: float = 0.2
    round_step: float = 0.2
    max_length: int = 1024
    normalize_embeddings: bool = True
    cache_size: int = 8192


class ALFWorldRWMLRewardScorer:
    """Embedding-similarity reward used by ALFWorld RWML."""

    def __init__(self, config: ALFWorldRWMLRewardConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            config.embedding_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else None,
        ).to(device)
        self.model.eval()
        self._cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _embed_batch(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            hidden = outputs[0]
        else:
            raise ValueError("Unsupported embedding model output format")
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        if self.config.normalize_embeddings:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def _lookup_or_encode(self, texts: Sequence[str]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = [None] * len(texts)  # type: ignore[assignment]
        missing_indices: List[int] = []
        missing_texts: List[str] = []
        for idx, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is None:
                missing_indices.append(idx)
                missing_texts.append(text)
            else:
                results[idx] = cached
        if missing_texts:
            embs = self._embed_batch(missing_texts)
            for pos, emb in zip(missing_indices, embs):
                text = texts[pos]
                emb_cpu = emb.detach().cpu()
                if len(self._cache) >= self.config.cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[text] = emb_cpu
                results[pos] = emb_cpu
        return results

    def score_texts(self, predictions: Sequence[str], references: Sequence[str]) -> List[float]:
        normalized_preds = [_strip_prediction(text) for text in predictions]
        normalized_refs = [str(text or "").strip() for text in references]
        pred_embs = torch.stack(self._lookup_or_encode(normalized_preds), dim=0).to(self.device)
        ref_embs = torch.stack(self._lookup_or_encode(normalized_refs), dim=0).to(self.device)
        sims = F.cosine_similarity(pred_embs, ref_embs, dim=-1)
        scores = []
        for sim in sims.tolist():
            if sim < self.config.threshold:
                scores.append(0.0)
            else:
                scores.append(float(_round_to_step(sim, self.config.round_step)))
        return scores

    def score_one(self, prediction: str, reference: str) -> float:
        return self.score_texts([prediction], [reference])[0]


def load_rwml_reward_config(path: str) -> ALFWorldRWMLRewardConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return ALFWorldRWMLRewardConfig(**raw)
