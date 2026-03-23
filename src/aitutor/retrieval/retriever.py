from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig, get_config
from ..types import Chunk, RetrievedChunk
from ..vectorstore import ChromaStore
from .langchain_hybrid import (
    get_dense_retriever,
    get_hybrid_retriever,
    get_sparse_retriever,
)


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

        # Hybrid Retrieval WITHOUT reranker (offline-safe)

        dense_k = max(20, top_k * 4)
        sparse_k = max(20, top_k * 4)
        min_candidates = 20

        dense_retriever = get_dense_retriever(
            cfg=self.cfg, class_=class_, subject=subject, chapter=chapter, k=dense_k
        )

        sparse_retriever = get_sparse_retriever(
            cfg=self.cfg, class_=class_, subject=subject, chapter=chapter, k=sparse_k
        )

        hybrid_retriever = get_hybrid_retriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            min_candidates=min_candidates,
        )

        candidates = hybrid_retriever.invoke(query)
        retrieved_docs = candidates[:15]

        print("[HYBRID-DEBUG] hybrid_top_results")

        out: list[RetrievedChunk] = []

        for i, doc in enumerate(retrieved_docs, start=1):
            meta = doc.metadata or {}
            chunk_id = str(meta.get("chunk_id") or doc.id or "")
            dense_similarity = float(meta.get("dense_similarity") or 0.0)

            cleaned_meta = {
                k: v for k, v in meta.items()
                if k not in {"chunk_id", "dense_similarity"}
            }

            chunk = Chunk(
                id=chunk_id,
                text=doc.page_content,
                metadata=cleaned_meta,
            )

            out.append(
                RetrievedChunk(
                    chunk=chunk,
                    similarity=dense_similarity,
                )
            )

            title = cleaned_meta.get("concept_title") or ""
            pages = ""
            if cleaned_meta.get("page_start") is not None or cleaned_meta.get("page_end") is not None:
                pages = f" pages={cleaned_meta.get('page_start')}–{cleaned_meta.get('page_end')}"

            print(
                f"[HYBRID-DEBUG]  R{i}: id={chunk_id} sim={dense_similarity} concept={title!r}{pages}"
            )

        return out