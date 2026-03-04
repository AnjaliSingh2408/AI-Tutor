from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import AppConfig, get_config
from ..types import Chunk, RetrievedChunk
from ..vectorstore import ChromaStore


def _similarity_from_distance(distance: float) -> float:
    # For Chroma cosine distance: distance = 1 - cosine_similarity
    return 1.0 - float(distance)


@dataclass(frozen=True)
class Retriever:
    cfg: AppConfig
    store: ChromaStore

    @classmethod
    def default(cls) -> "Retriever":
        cfg = get_config()
        return cls(cfg=cfg, store=ChromaStore(cfg))

    def retrieve(
        self,
        *,
        query: str,
        class_: str,
        subject: str,
        chapter: str | None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        # ChromaDB >=0.5 expects a single top-level operator in `where`,
        # so we use an explicit $and of equality conditions.
        filters: list[dict[str, Any]] = [
            {"class": {"$eq": str(class_)}},
            {"subject": {"$eq": str(subject)}},
        ]
        if chapter:
            filters.append({"chapter": {"$eq": str(chapter)}})
        where: dict[str, Any] | None = {"$and": filters} if filters else None

        res = self.store.query(query_text=query, n_results=top_k, where=where)

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: list[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            chunk = Chunk(id=str(cid), text=str(doc), metadata=dict(meta or {}))
            out.append(RetrievedChunk(chunk=chunk, similarity=_similarity_from_distance(dist)))
        out.sort(key=lambda x: x.similarity, reverse=True)
        return out

