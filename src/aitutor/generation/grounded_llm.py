from __future__ import annotations

from dataclasses import dataclass

from google import genai

from ..config import AppConfig, get_config, get_gemini_api_key
from ..types import RetrievedChunk


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for i, rc in enumerate(chunks, start=1):
        m = rc.chunk.metadata or {}
        title = m.get("concept_title", "Concept")
        source_pdf = m.get("source_pdf", "NCERT PDF")
        chapter = m.get("chapter", "?")
        ps = m.get("page_start", "?")
        pe = m.get("page_end", "?")
        blocks.append(
            "\n".join(
                [
                    f"[Chunk {i}] Concept: {title}",
                    f"Source: {source_pdf} | Chapter: {chapter} | Pages: {ps}-{pe} | Similarity: {rc.similarity:.3f}",
                    "Text:",
                    rc.chunk.text.strip(),
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


@dataclass(frozen=True)
class GroundedLLM:
    cfg: AppConfig

    @classmethod
    def default(cls) -> "GroundedLLM":
        return cls(get_config())

    def generate(self, *, query: str, retrieved: list[RetrievedChunk]) -> str:
        api_key = get_gemini_api_key()
        model = self.cfg.gemini_model
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in .env")
        client = genai.Client(api_key=api_key)
        context = format_context(retrieved)
        print("[LLM-DEBUG] final context sent to LLM:")
        print(context[:4000])

        system = (
            "You are an NCERT-grounded AI tutor. Answer primarily using the provided NCERT context.\n"
            "If the context discusses the topic but lacks a specific standard formula (like the distance formula), you MAY use your own mathematical and scientific knowledge to solve the student's problem.\n"
            "Do not refuse unless the context is completely unrelated to the question.\n"
            "Provide a step-by-step solution if solving a problem.\n"
            "Write in a clear, student-friendly way, aligned to the NCERT syllabus.\n"
            "Format strictly as:\n"
            "1) Explanation\n"
            "2) Example\n"
            "3) Formula (if applicable; state all formulas used)\n"
            "4) NCERT reference (chapter + page range)\n"
        )

        user = (
            "NCERT context:\n"
            f"{context}\n\n"
            f"Student question: {query}\n\n"
            "Now answer following the required format."
        )

        prompt = f"{system}\n\n{user}"

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.2},
        )

        text = getattr(resp, "text", None) or ""
        return text.strip()

