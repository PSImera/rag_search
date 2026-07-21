from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_PASSAGE_PREFIX,
    EMBEDDING_QUERY_PREFIX,
    GPU_BATCH_SIZE,
)


class E5Embeddings:
    """Локальные эмбеддинги через sentence-transformers (multilingual-e5-large).

    Автоматически использует CUDA, если доступна. Токены платного API не тратятся.
    Для e5 добавляются префиксы query:/passage:.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, *, offline: bool = True):
        resolved = self._resolve_model_name(model_name)
        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.model = SentenceTransformer(resolved, trust_remote_code=True)
        try:
            dim = self.model.get_sentence_embedding_dimension()
        except Exception:
            dim = None
        # get_sentence_embedding_dimension() может вернуть None — тогда меряем эмпирически
        self.dim: int = dim if dim is not None else len(self.embed_query("test"))

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        candidate = Path(model_name)
        if candidate.exists():
            return str(candidate)
        if "/" not in model_name:
            return model_name
        org, repo = model_name.split("/", 1)
        snapshot_root = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--{org}--{repo}"
            / "snapshots"
        )
        if snapshot_root.exists():
            snaps = [p for p in snapshot_root.iterdir() if p.is_dir()]
            if snaps:
                snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return str(snaps[0])
        return model_name

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # Передаём большой список целиком: sentence-transformers сортирует по длине,
        # так что внутренние батчи получаются однородными (минимум паддинга).
        vectors = self.model.encode(
            [EMBEDDING_PASSAGE_PREFIX + t for t in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=GPU_BATCH_SIZE,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            EMBEDDING_QUERY_PREFIX + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return cast(np.ndarray, vector.astype(np.float32))
