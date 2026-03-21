from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig, get_config
from ..types import Chunk, RetrievedChunk
from ..vectorstore import ChromaStore
from .langchain_hybrid import (
    get_dense_retriever,
    get_hybrid_retriever,
    get_reranker,
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
        # Requirement targets:
        # Query -> Hybrid Retriever -> Reranker -> Top 5 docs -> LLM
        #
        # We:
        # 1) Retrieve dense (Chroma) + sparse (BM25) candidates
        # 2) Fuse with LangChain `EnsembleRetriever` (RRF)
        # 3) Rerank using HuggingFace cross-encoder via `ContextualCompressionRetriever`
        #
        # Debug prints are inside the HybridCandidatesRetriever.

        # At least top-20 BEFORE reranking.
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
        compression_retriever = get_reranker(base_retriever=hybrid_retriever, top_n=top_k)

        hybrid_docs = hybrid_retriever.invoke(query)
        retrieved_docs = compression_retriever.invoke(query)
        # Failsafe: if reranker gives nothing, fall back to hybrid docs.
        if not retrieved_docs:
            print("[HYBRID-DEBUG] reranker fallback: using hybrid results")
            retrieved_docs = hybrid_docs[:top_k]

        # Keep candidate order ids for "did reranker change order?" debug.
        pre_ids = list(getattr(hybrid_retriever, "_last_candidate_chunk_ids", []))

        # `ContextualCompressionRetriever` returns the reranked top-N docs.
        # For debugging, we compare reranked top ordering vs the original candidate ordering.
        post_ids = [
            str((d.metadata or {}).get("chunk_id")) for d in retrieved_docs if d.metadata
        ]
        if pre_ids and post_ids:
            pre_top = pre_ids[: len(post_ids)]
            if pre_top == post_ids:
                print(
                    "[HYBRID-DEBUG] reranker order check: WARNING (top results order matches pre-rerank)"
                )

        print("[HYBRID-DEBUG] reranked_top_results")
        out: list[RetrievedChunk] = []
        for i, doc in enumerate(retrieved_docs, start=1):
            meta = doc.metadata or {}
            chunk_id = str(meta.get("chunk_id") or doc.id or "")
            dense_similarity = float(meta.get("dense_similarity") or 0.0)

            # Keep metadata but drop our helper fields so `RetrievedChunk.chunk.metadata`
            # stays aligned with your original code's expectations.
            cleaned_meta = {k: v for k, v in meta.items() if k not in {"chunk_id", "dense_similarity"}}
            chunk = Chunk(id=chunk_id, text=doc.page_content, metadata=cleaned_meta)
            out.append(RetrievedChunk(chunk=chunk, similarity=dense_similarity))

            title = cleaned_meta.get("concept_title") or ""
            pages = ""
            if cleaned_meta.get("page_start") is not None or cleaned_meta.get("page_end") is not None:
                pages = f" pages={cleaned_meta.get('page_start')}–{cleaned_meta.get('page_end')}"
            print(
                f"[HYBRID-DEBUG]  R{i}: id={chunk_id} sim={dense_similarity} concept={title!r}{pages}"
            )

        return out

