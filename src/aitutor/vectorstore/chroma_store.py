from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ..config import AppConfig, get_config


@lru_cache(maxsize=1)
def _cached_client(path: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=path)


@lru_cache(maxsize=2)
def _cached_embedding_fn(model_name: str) -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


@dataclass(frozen=True)
class ChromaStore:
    cfg: AppConfig

    @classmethod
    def default(cls) -> "ChromaStore":
        return cls(get_config())

    def _client(self) -> chromadb.PersistentClient:
        self.cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
        return _cached_client(str(self.cfg.chroma_dir))

    def collection(self):
        client = self._client()
        emb_fn = _cached_embedding_fn(self.cfg.embedding_model_name)
        return client.get_or_create_collection(
            name=self.cfg.chroma_collection,
            embedding_function=emb_fn,
            metadata={"hnsw:space": self.cfg.chroma_space},
        )

    def raw_collection(self):
        """
        Collection handle WITHOUT embedding function.

        This is safe for `.get(...)` (reading stored documents/metadatas) even
        when the embedding model cannot be downloaded/initialized.
        """
        client = self._client()
        return client.get_or_create_collection(
            name=self.cfg.chroma_collection,
            embedding_function=None,
            metadata={"hnsw:space": self.cfg.chroma_space},
        )

    def add_texts(self, *, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        if not ids:
            return
        col = self.collection()
        col.add(ids=ids, documents=texts, metadatas=metadatas)

    def query(
        self,
        *,
        query_text: str,
        n_results: int,
        where: dict[str, Any] | None,
    ) -> dict[str, Any]:
        col = self.collection()
        return col.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            # In newer Chroma versions, "ids" is always returned and
            # is not allowed inside `include`, so we only request the
            # extra fields we need.
            include=["documents", "metadatas", "distances"],
        )

    def get(
        self,
        *,
        where: dict[str, Any] | None,
        limit: int,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Fetch stored items without requiring embeddings.
        """
        col = self.raw_collection()
        return col.get(where=where, limit=limit, include=include or ["documents", "metadatas"])

