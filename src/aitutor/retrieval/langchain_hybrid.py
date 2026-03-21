from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever as LCBM25Retriever

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)

from ..config import AppConfig
from ..vectorstore import ChromaStore


def _build_where(*, class_: str, subject: str, chapter: str | None) -> dict[str, Any] | None:
    filters: list[dict[str, Any]] = [
        {"class": {"$eq": str(class_)}},
        {"subject": {"$eq": str(subject)}},
    ]
    if chapter:
        filters.append({"chapter": {"$eq": str(chapter)}})
    return {"$and": filters} if filters else None


from pydantic import ConfigDict


class ChromaDenseRetriever(BaseRetriever):
    """
    Dense retriever backed by the project's existing `ChromaStore`.

    It returns `Document` objects where:
      - `page_content` is the chunk text
      - `metadata['chunk_id']` is the stored chunk id
      - `metadata['dense_similarity']` is derived from cosine distance as (1 - distance)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cfg: AppConfig
    where: dict[str, Any] | None
    k: int
    store: ChromaStore

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        try:
            res = self.store.query(query_text=query, n_results=self.k, where=self.where)
            ids = (res.get("ids") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
        except Exception:
            # If embeddings can't be initialized/downloaded, fall back to returning
            # stored documents fetched via `.get(...)`. We keep the pipeline alive
            # for sparse+reranking, but dense similarity will be 0.0.
            raw = self.store.get(
                where=self.where,
                limit=self.k,
                include=["documents", "metadatas"],
            )
            ids = raw.get("ids") or []
            docs = raw.get("documents") or []
            metas = raw.get("metadatas") or []
            # When dense similarity can't be computed (embeddings unavailable),
            # keep a conservative non-zero similarity proxy so the rest of the
            # pipeline (BM25 + cross-encoder rerank) can still decide usefulness.
            # We later convert cosine distance to similarity via: similarity = 1.0 - distance
            # so we use distance=0.0 here.
            dists = [0.0] * min(len(ids), len(docs), len(metas))

        out: list[Document] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            meta = dict(meta or {})
            # For Chroma cosine distance: distance = 1 - cosine_similarity
            dense_similarity = 1.0 - float(dist)
            chunk_id = str(cid)
            meta["chunk_id"] = chunk_id
            meta["dense_similarity"] = dense_similarity
            out.append(
                Document(
                    id=chunk_id,
                    page_content=str(doc),
                    metadata=meta,
                )
            )
        return out


@lru_cache(maxsize=64)
def _cached_sparse_index(
    *,
    chroma_dir: str,
    class_: str,
    subject: str,
    chapter: str | None,
    embedding_model_name: str,
    chroma_collection: str,
    chroma_space: str,
    candidate_limit: int,
) -> tuple[list[Document], str]:
    """
    Build and cache a BM25 index (LangChain BM25Retriever) per filter.

    Note: BM25Retriever requires an in-memory list of documents, so we cache the
    underlying `Document` list (and rebuild the retriever object).
    """
    # Import here to avoid potential circular imports.
    from ..vectorstore import ChromaStore  # noqa: WPS433

    cfg = AppConfig(
        project_root=ChromaStore.default().cfg.project_root,
        data_dir=ChromaStore.default().cfg.data_dir,
        chroma_dir=ChromaStore.default().cfg.chroma_dir,
        embedding_model_name=embedding_model_name,
        chroma_collection=chroma_collection,
        chroma_space=chroma_space,
    )

    where = _build_where(class_=class_, subject=subject, chapter=chapter)
    store = ChromaStore(cfg)
    raw = store.get(where=where, limit=candidate_limit, include=["documents", "metadatas"])

    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    # `chroma` should always provide ids, but guard just in case.
    if not ids:
        ids = [str(i) for i in range(min(len(docs), len(metas)))]

    documents: list[Document] = []
    for cid, doc, meta in zip(ids, docs, metas):
        meta = dict(meta or {})
        chunk_id = str(cid)
        meta["chunk_id"] = chunk_id
        documents.append(Document(id=chunk_id, page_content=str(doc), metadata=meta))

    where_key = f"class={class_}|subject={subject}|chapter={chapter or ''}"
    return documents, where_key


def get_dense_retriever(
    *,
    cfg: AppConfig,
    class_: str,
    subject: str,
    chapter: str | None,
    k: int,
) -> BaseRetriever:
    store = ChromaStore(cfg)
    where = _build_where(class_=class_, subject=subject, chapter=chapter)
    return ChromaDenseRetriever(cfg=cfg, where=where, k=k, store=store)


def get_sparse_retriever(
    *,
    cfg: AppConfig,
    class_: str,
    subject: str,
    chapter: str | None,
    k: int,
    candidate_limit: int = 5000,
) -> BaseRetriever:
    where = _build_where(class_=class_, subject=subject, chapter=chapter)
    # Fetch & cache BM25 corpus from the same Chroma collection.
    docs, _ = _cached_sparse_index(
        chroma_dir=str(cfg.chroma_dir),
        class_=str(class_),
        subject=str(subject),
        chapter=str(chapter) if chapter else None,
        embedding_model_name=cfg.embedding_model_name,
        chroma_collection=cfg.chroma_collection,
        chroma_space=cfg.chroma_space,
        candidate_limit=candidate_limit,
    )
    bm25 = LCBM25Retriever.from_documents(docs, k=k)
    return bm25


class HybridCandidatesRetriever(BaseRetriever):
    """
    Base retriever that:
      Dense retriever (Chroma) -> top K dense docs
      Sparse retriever (BM25) -> top K sparse docs
      EnsembleRetriever (RRF) -> fused candidates
      (Optional padding) -> ensure at least `min_candidates` before reranking
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dense_retriever: BaseRetriever
    sparse_retriever: BaseRetriever
    ensemble_retriever: EnsembleRetriever
    min_candidates: int
    _last_candidate_chunk_ids: list[str] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.sparse_retriever.invoke(query)

        print("[HYBRID-DEBUG] dense_results (top K)")
        for i, d in enumerate(dense_docs[: self.min_candidates], start=1):
            meta = d.metadata or {}
            title = meta.get("concept_title") or ""
            pages = ""
            if meta.get("page_start") is not None or meta.get("page_end") is not None:
                pages = f" pages={meta.get('page_start')}–{meta.get('page_end')}"
            sim = meta.get("dense_similarity")
            print(f"[HYBRID-DEBUG]  D{i}: id={meta.get('chunk_id')} sim={sim} concept={title!r}{pages}")

        print("[HYBRID-DEBUG] sparse_results (top K)")
        for i, d in enumerate(sparse_docs[: self.min_candidates], start=1):
            meta = d.metadata or {}
            title = meta.get("concept_title") or ""
            pages = ""
            if meta.get("page_start") is not None or meta.get("page_end") is not None:
                pages = f" pages={meta.get('page_start')}–{meta.get('page_end')}"
            print(f"[HYBRID-DEBUG]  S{i}: id={meta.get('chunk_id')} concept={title!r}{pages}")

        candidates = self.ensemble_retriever.invoke(query)
        # `EnsembleRetriever` dedups by `id_key` (if provided) using metadata.
        print(f"[HYBRID-DEBUG] combined_ensemble_candidates_count={len(candidates)}")

        # Ensure we pass at least `min_candidates` documents to the reranker.
        if len(candidates) < self.min_candidates:
            seen: set[str] = {str(d.metadata.get("chunk_id")) for d in candidates if d.metadata}
            padding: list[Document] = []
            for d in dense_docs + sparse_docs:
                cid = str((d.metadata or {}).get("chunk_id"))
                if cid and cid not in seen:
                    seen.add(cid)
                    padding.append(d)
                if len(candidates) + len(padding) >= self.min_candidates:
                    break
            candidates = candidates + padding
            print(
                f"[HYBRID-DEBUG] padded_candidates_to={len(candidates)} (min required={self.min_candidates})"
            )

        self._last_candidate_chunk_ids = [
            str((d.metadata or {}).get("chunk_id")) for d in candidates if d.metadata
        ]
        return candidates


def get_hybrid_retriever(
    *,
    dense_retriever: BaseRetriever,
    sparse_retriever: BaseRetriever,
    min_candidates: int,
) -> HybridCandidatesRetriever:
    # Combine dense + sparse using LangChain's EnsembleRetriever.
    # We set `id_key` so dedup happens by chunk id rather than page_content.
    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5],
        c=60,
        id_key="chunk_id",
    )
    return HybridCandidatesRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        ensemble_retriever=ensemble,
        min_candidates=min_candidates,
    )


def get_reranker(
    *,
    base_retriever: BaseRetriever,
    top_n: int = 5,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> ContextualCompressionRetriever:
    # In restricted environments we may not be able to hit HF again.
    # The weights/tokenizers are typically cached from earlier runs.
    hf_cross_encoder = HuggingFaceCrossEncoder(
        model_name=cross_encoder_model,
        model_kwargs={"device": "cpu", "local_files_only": True},
    )
    reranker = CrossEncoderReranker(model=hf_cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=reranker)

