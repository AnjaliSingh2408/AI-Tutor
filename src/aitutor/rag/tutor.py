from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig, get_config
from ..generation import GroundedLLM
from ..retrieval import Retriever
from ..vectorstore import ChromaStore
from ..retrieval.query_corrector import correct_query


@dataclass
class Tutor:
    cfg: AppConfig
    retriever: Retriever
    llm: GroundedLLM

    # ⭐ Factory method — ALWAYS use this
    @classmethod
    def default(cls) -> "Tutor":
        cfg = get_config()
        store = ChromaStore(cfg)
        retriever = Retriever(cfg=cfg, store=store)
        llm = GroundedLLM(cfg)
        return cls(cfg=cfg, retriever=retriever, llm=llm)

    # ⭐ Main answer function
    def answer(
        self,
        *,
        query: str,
        class_: str,
        subject: str,
        chapter: str | None = None,
        top_k: int = 5,
    ) -> str:

        # 🔥 Save original query
        original_query = query

        # ✅ Step 1 — Strong spelling correction
        corrected_query = correct_query(query)

        # ✅ Step 2 — Retrieve using corrected query
        retrieved = self.retriever.retrieve(
            query=corrected_query,
            class_=class_,
            subject=subject,
            chapter=chapter,
            top_k=top_k,
        )

        # 🔁 Step 3 — Fallback: try original query
        if not retrieved:
            retrieved = self.retriever.retrieve(
                query=original_query,
                class_=class_,
                subject=subject,
                chapter=chapter,
                top_k=top_k,
            )

        # ❌ Step 4 — If still nothing found
        if not retrieved:
            return (
                "I couldn’t find this topic in the selected NCERT content. "
                "Try specifying the chapter or asking in another way."
            )

        # ⭐ Step 5 — Generate grounded answer
        return self.llm.generate(query=corrected_query, retrieved=retrieved)