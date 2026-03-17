from __future__ import annotations

from dataclasses import dataclass
import os

from google import genai

from ..config import AppConfig, get_config
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

    def generate(
        self,
        *,
        query: str,
        retrieved: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        api_key = os.environ.get("GEMINI_API_KEY")
        model = self.cfg.gemini_model
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        if not model:
            raise ValueError("GEMINI_MODEL not set in .env")
        client = genai.Client(api_key=api_key)
        context = format_context(retrieved)

        history = history or []
        history_text = ""
        if history:
            lines: list[str] = []
            for m in history:
                role = str(m.get("role", "")).strip().lower()
                text = str(m.get("text", "")).strip()
                if not role or not text:
                    continue
                if role not in {"user", "assistant"}:
                    continue
                label = "Student" if role == "user" else "Tutor"
                lines.append(f"{label}: {text}")
            if lines:
                history_text = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

        allow_outside_examples = os.environ.get("ALLOW_OUTSIDE_EXAMPLES", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

        system = (
            "You are an NCERT-grounded AI tutor.\n"
            "Your primary job is to teach using ONLY the provided NCERT context for factual claims.\n"
            "If the answer is not explicitly supported by the context, you must refuse.\n"
            "Do not use outside knowledge. Do not guess.\n"
            "Write in a clear, student-friendly way, aligned to the NCERT syllabus.\n"
            "Format strictly as:\n"
            "1) Explanation (NCERT-grounded)\n"
            "2) Textbook example(s) (NCERT-grounded; if none, write 'Not available in the provided NCERT context')\n"
            "3) Formula (if applicable; otherwise write 'Not applicable')\n"
            "4) NCERT reference (chapter + page range)\n"
            "5) Follow-up questions (ask 2–4 short questions the student should answer next; NCERT-grounded)\n"
        )

        if allow_outside_examples:
            system += (
                "Exception: If the student explicitly asks for 'real-life examples' or 'everyday examples', you MAY add a final section:\n"
                "6) Real-life examples (NOT from NCERT)\n"
                "These must be clearly labeled as general examples and must not claim NCERT citations.\n"
                "Keep them simple, plausible, and aligned with the NCERT concept, but do not introduce new facts that contradict NCERT.\n"
            )

        user = (
            f"{history_text}"
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

