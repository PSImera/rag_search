from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np

from src.config import STORE_ROOT
from src.store import RawStore


def chunk_id(chat_id: Any, message_id: int) -> int:
    """Идемпотентный int64-id чанка: одно сообщение -> один устойчивый id.

    Повторная индексация того же (chat_id, message_id) даёт тот же id => дублей нет.
    """
    raw = f"{chat_id}:{message_id}".encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)


class VectorStore:
    """FAISS flat-индекс (IndexFlatIP + IndexIDMap2) + docstore + сырое SQLite-хранилище."""

    def __init__(self, collection: str, dim: int | None = None):
        self.collection = collection
        self.root = STORE_ROOT / collection
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root / "index.faiss"
        self.docstore_file = self.root / "docstore.pkl"
        self.meta_file = self.root / "meta.json"
        self.raw = RawStore(self.root / "raw.sqlite")

        self.index: faiss.IndexIDMap2 | None = None
        self.docstore: dict[int, dict[str, Any]] = {}
        self.dim = dim
        self._load()

    def _load(self) -> None:
        if self.index_file.exists() and self.docstore_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with self.docstore_file.open("rb") as f:
                payload = pickle.load(f)
            self.docstore = {int(k): v for k, v in payload["docstore"].items()}
            self.dim = payload.get("dim", self.dim)
        elif self.dim is not None:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    def ensure_index(self, dim: int) -> None:
        if self.index is None:
            self.dim = dim
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def existing_ids(self) -> set[int]:
        return set(self.docstore.keys())

    def add_batch(
        self, ids: list[int], vectors: np.ndarray, metadatas: list[dict[str, Any]]
    ) -> None:
        assert self.index is not None
        ids_arr = np.asarray(ids, dtype=np.int64)
        self.index.add_with_ids(vectors.astype(np.float32), ids_arr)  # type: ignore[call-arg]
        for i, meta in zip(ids, metadatas):
            self.docstore[int(i)] = meta

    def remove_ids(self, ids: list[int]) -> int:
        """Удаляет векторы по id из FAISS-индекса и docstore. GPU не задействуется.

        Возвращает число фактически удалённых. IndexIDMap2 поддерживает remove_ids;
        индекс физически ужимается, ntotal уменьшается.
        """
        if self.index is None or not ids:
            return 0
        present = [int(i) for i in ids if int(i) in self.docstore]
        if not present:
            return 0
        self.index.remove_ids(np.asarray(present, dtype=np.int64))  # type: ignore[call-arg]
        for i in present:
            self.docstore.pop(i, None)
        return len(present)

    def save(self, extra_meta: dict[str, Any] | None = None) -> None:
        assert self.index is not None
        faiss.write_index(self.index, str(self.index_file))
        with self.docstore_file.open("wb") as f:
            pickle.dump({"dim": self.dim, "docstore": self.docstore}, f)
        meta = {
            "collection": self.collection,
            "dim": self.dim,
            "vectors": self.index.ntotal,
        }
        if extra_meta:
            meta.update(extra_meta)
        self.meta_file.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[dict[str, Any]]:
        """Поиск top-k по косинусу. `predicate` — фильтр по метаданным чанка
        (собирается вызывающей стороной из любых полей docstore); None — без фильтра."""
        if self.index is None or self.index.ntotal == 0:
            return []
        total = self.index.ntotal
        qv = np.asarray([query_vector], dtype=np.float32)

        # Итеративное углубление: при фильтрах top-k всего индекса может не содержать
        # нужный источник (например 79 чанков транскрипта среди 300k сообщений).
        fetch_k = k if predicate is None else max(k * 20, 200)
        while True:
            fetch_k = min(fetch_k, total)
            scores, ids = self.index.search(qv, fetch_k)  # type: ignore[call-arg]
            results: list[dict[str, Any]] = []
            for score, doc_id in zip(scores[0], ids[0]):
                if doc_id == -1:
                    continue
                meta = self.docstore.get(int(doc_id))
                if meta is None or (predicate is not None and not predicate(meta)):
                    continue
                results.append({"score": float(score), **meta})
                if len(results) >= k:
                    return results
            if fetch_k >= total:
                return results
            fetch_k *= 10

    def close(self) -> None:
        self.raw.close()
