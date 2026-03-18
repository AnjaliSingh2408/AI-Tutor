from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import AppConfig, get_config
from ..types import Chunk, RetrievedChunk
from ..vectorstore import ChromaStore
from .bm25_retriever import BM25Retriever
from .reranker import rerank


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

        # 1) Vector retrieval (top 8 candidates)
        vector_k = 8
        res = self.store.query(query_text=query, n_results=vector_k, where=where)

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        chunks: list[Chunk] = []
        vector_results: list[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            chunk = Chunk(id=str(cid), text=str(doc), metadata=dict(meta or {}))
            chunks.append(chunk)
            vector_results.append(
                RetrievedChunk(chunk=chunk, similarity=_similarity_from_distance(dist))
            )

        if not chunks:
            return []

        # 2) BM25 retrieval (top 8 candidates over the same chunk list)
        bm25_results: list[RetrievedChunk] = []
        try:
            bm25_k = 8
            bm25_retriever = BM25Retriever.from_chunks(chunks)
            bm25_results = bm25_retriever.retrieve(query=query, top_k=bm25_k)
        except Exception:
            # If BM25 fails for any reason, fall back to vector-only results.
            bm25_results = []

        # 3) Merge results and 4) deduplicate by chunk id (keep first occurrence)
        merged_by_id: dict[str, RetrievedChunk] = {}
        for rc in vector_results + bm25_results:
            cid = rc.chunk.id
            if cid not in merged_by_id:
                merged_by_id[cid] = rc

        merged = list(merged_by_id.values())

        # 5) Rerank merged candidates with cross-encoder (by text only).
        # If reranking fails, fall back to merged retrieval ordering.
        try:
            candidate_texts = [rc.chunk.text for rc in merged]
            reranked_texts = rerank(query=query, candidates=candidate_texts, top_k=top_k)

            # Map reranked texts back to RetrievedChunk objects, preserving similarity scores.
            remaining = merged.copy()
            out: list[RetrievedChunk] = []
            for text in reranked_texts:
                for idx, rc in enumerate(remaining):
                    if rc.chunk.text == text:
                        out.append(rc)
                        remaining.pop(idx)
                        break

            # 6) Return top_k results in the same format expected by Tutor.answer
            return out
        except Exception:
            # Safe fallback: just return up to top_k merged results (vector + BM25) without reranking.
            merged.sort(key=lambda rc: rc.similarity, reverse=True)
            return merged[:top_k]

