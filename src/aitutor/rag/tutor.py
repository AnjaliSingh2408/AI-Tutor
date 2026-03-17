from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig, get_config
from ..generation import GroundedLLM
from ..retrieval import Retriever
from ..vectorstore import ChromaStore


@dataclass(frozen=True)
class Tutor:
    cfg: AppConfig
    retriever: Retriever
    llm: GroundedLLM

    @classmethod
    def default(cls) -> "Tutor":
        cfg = get_config()
        return cls(cfg=cfg, retriever=Retriever.default(), llm=GroundedLLM(cfg))

    def __init__(self, cfg: AppConfig | None = None):
        cfg = cfg or get_config()
        object.__setattr__(self, "cfg", cfg)
        object.__setattr__(self, "retriever", Retriever(cfg=cfg, store=ChromaStore(cfg)))
        object.__setattr__(self, "llm", GroundedLLM(cfg))

    def answer(
        self,
        *,
        query: str,
        class_: str,
        subject: str,
        chapter: str | None,
        top_k: int = 5,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        retrieved = self.retriever.retrieve(
            query=query, class_=class_, subject=subject, chapter=chapter, top_k=top_k
        )
        if not retrieved:
            return (
                "I can’t answer this from the NCERT text I have indexed for the selected class/subject/chapter. "
                "Try specifying the chapter or rephrasing your question using NCERT terms.\n\n"
                "Follow-up questions:\n"
                "- Which class/subject and chapter is this from?\n"
                "- Can you paste the exact NCERT line/paragraph you’re asking about?\n"
                "- What exact keyword appears in your textbook for this topic?\n"
            )

        best = retrieved[0].similarity
        if best < self.cfg.min_similarity:
            return (
                "I can’t answer this confidently from the retrieved NCERT content (low retrieval confidence). "
                "Please rephrase your question or specify the exact chapter/topic from NCERT.\n\n"
                "Follow-up questions:\n"
                "- Which chapter number/title are you studying?\n"
                "- What is the exact term/definition used in NCERT for this?\n"
                "- Do you want a definition, explanation, or a solved example from the chapter?\n"
            )

        return self.llm.generate(query=query, retrieved=retrieved, history=history)

