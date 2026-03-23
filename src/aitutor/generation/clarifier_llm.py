from __future__ import annotations

from dataclasses import dataclass

from google import genai

from ..config import AppConfig, get_config, get_gemini_api_key


@dataclass(frozen=True)
class ClarifierLLM:
    cfg: AppConfig

    @classmethod
    def default(cls) -> "ClarifierLLM":
        return cls(get_config())

    def _client_and_model(self) -> tuple[genai.Client, str]:
        api_key = get_gemini_api_key()
        model = self.cfg.gemini_model
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in .env")
        return genai.Client(api_key=api_key), model

    def clarify(
        self,
        *,
        class_: str,
        subject: str,
        chapter: str | None,
        question: str,
        previous_answer: str,
        student_followup: str,
        thread: list[dict] | None = None,
    ) -> str:
        """
        Use a cloud LLM to re-explain a concept when the student
        was not satisfied with the earlier reply.
        """
        client, model = self._client_and_model()
        recent_thread = (thread or [])[-10:]
        thread_text = "\n".join(
            f"{m.get('role', 'unknown').strip().upper()}: {str(m.get('text', '')).strip()}"
            for m in recent_thread
            if isinstance(m, dict) and str(m.get("text", "")).strip()
        ).strip()

        class_level = str(class_).strip() or "unknown"
        subj = str(subject).strip() or "the subject"
        chap = str(chapter).strip() if chapter else None

        system = (
            f"You are a helpful NCERT-aligned school tutor for Class {class_level} ({subj}).\n"
            + (f"Current chapter context: {chap}.\n" if chap else "")
            + "You must match the student's grade level:\n"
            "- Use simple words and short sentences.\n"
            "- Explain step-by-step.\n"
            "- Use examples that a Class "
            + f"{class_level}"
            + " student can relate to.\n"
            "- Do not use advanced topics beyond the syllabus level.\n"
            "A student did not fully understand the previous explanation.\n"
            "Your job is to explain the SAME original question again in a clearer, step-by-step way.\n"
            "Treat vague words like 'this' or 'these' or 'it' in the follow-up as referring to the original question\n"
            "and your previous answer, unless the student has clearly asked a completely new question.\n"
            "Use simpler language and more concrete examples.\n"
            "When the student asks for examples, you MUST provide new examples that are\n"
            "similar in style and difficulty to typical NCERT textbook examples, but do NOT\n"
            "repeat or closely copy the examples already used earlier in this conversation.\n"
            "Prefer giving 2–3 short, varied examples that all match the same concept.\n"
            "Stay aligned with standard NCERT content for the specified class, but you may use your general knowledge to\n"
            "give better intuitions, alternative methods, and extra examples.\n"
            "Avoid saying you cannot answer; focus on teaching the idea carefully.\n"
            "Format strictly as:\n"
            "1) Explanation\n"
            "2) Example (include at least one fresh example)\n"
            "3) Formula (if applicable; otherwise write 'Not applicable')\n"
            "4) NCERT-style reference (chapter/topic name and an indicative page range if you can infer it; "
            "otherwise write 'Approximate – align with textbook section on this topic').\n"
        )

        user = (
            "Conversation thread so far (most recent last):\n"
            f"{thread_text or '(no prior thread available)'}\n\n"
            "Original student question:\n"
            f"{question.strip()}\n\n"
            "Previous answer the tutor gave:\n"
            f"{previous_answer.strip()}\n\n"
            "Student's follow-up message (they were not satisfied):\n"
            f"{student_followup.strip()}\n\n"
            "Now re-explain the same concept more clearly following the required format."
        )

        prompt = f"{system}\n\n{user}"

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.4},
        )

        text = getattr(resp, "text", None) or ""
        return text.strip()

