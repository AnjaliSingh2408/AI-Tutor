from __future__ import annotations

from dataclasses import dataclass
import os

from google import genai

from ..config import AppConfig, get_config


@dataclass(frozen=True)
class SummaryLLM:
    cfg: AppConfig

    @classmethod
    def default(cls) -> "SummaryLLM":
        return cls(get_config())

    def _client_and_model(self) -> tuple[genai.Client, str]:
        api_key = os.environ.get("GEMINI_API_KEY")
        model = self.cfg.gemini_model
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        if not model:
            raise ValueError("GEMINI_MODEL not set in .env")
        return genai.Client(api_key=api_key), model

    def summarize(
        self,
        *,
        class_: str,
        subject: str,
        chapter: str | None,
        topic_hint: str,
        source_text: str,
        summary_request: str,
    ) -> str:
        """
        Summarize a topic for the student, using either the previous tutor
        explanation or student-provided content, at the appropriate class level.
        """
        client, model = self._client_and_model()

        class_level = str(class_).strip() or "unknown"
        subj = str(subject).strip() or "the subject"
        chap = str(chapter).strip() if chapter else None

        system = (
            f"You are an NCERT-aligned school tutor for Class {class_level} ({subj}).\n"
            + (f"Current chapter context: {chap}.\n" if chap else "")
            + "Your task is to write a clear summary of the topic for this student.\n"
            "Match the student's grade level:\n"
            "- Use simple words and short sentences.\n"
            "- Focus on the most important ideas only.\n"
            "- Avoid advanced details beyond the NCERT syllabus for this class.\n"
            "Adjust the length and style of the summary based on the student's request "
            "(for example: 'in short', 'in 3 points', 'in about 100 words').\n"
            "If the request is very short like 'summarise this', assume a medium-length summary "
            "suitable for quick revision.\n"
        )

        user = (
            "Topic hint (from the conversation or student):\n"
            f"{topic_hint.strip()}\n\n"
            "Student's summary request (including any length/format hints):\n"
            f"{summary_request.strip()}\n\n"
            "Content to summarise (this may be a previous explanation or text the student provided):\n"
            f"{source_text.strip()}\n\n"
            "Now write the summary in bullet points or short paragraphs, according to the student's request.\n"
        )

        prompt = f"{system}\n\n{user}"

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.35},
        )

        text = getattr(resp, "text", None) or ""
        return text.strip()

