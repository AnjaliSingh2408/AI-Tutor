from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from google import genai

from ..config import AppConfig, get_config, get_gemini_api_key


def _normalize_subject(subject: str) -> str:
    s = (subject or "").strip().lower()
    if s in {"math", "maths", "mathematics"}:
        return "Maths"
    if s == "science":
        return "Science"
    return subject.strip()


def _subject_tokens(subject: str) -> tuple[str, ...]:
    if _normalize_subject(subject) == "Maths":
        return ("math", "maths", "mathematics")
    return ("science",)


def _safe_text(v: Any) -> str:
    return str(v).strip() if v is not None else ""


def _normalize_student_answers(raw: Any) -> tuple[dict[str, str], str]:
    if isinstance(raw, dict):
        return {str(k): _safe_text(v) for k, v in raw.items()}, ""
    if isinstance(raw, str):
        text = raw.strip()
        return {}, text
    return {}, ""


def _extract_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            raw = "\n".join(lines[1:-1]).strip()
    return json.loads(raw)


def _sample_pdf_text(pdf_path: Path, *, max_pages: int = 2, max_chars: int = 1800) -> str:
    try:
        with fitz.open(pdf_path) as doc:
            pages = min(doc.page_count, max_pages)
            out: list[str] = []
            total = 0
            for i in range(pages):
                t = doc.load_page(i).get_text("text").strip()
                if not t:
                    continue
                remaining = max_chars - total
                if remaining <= 0:
                    break
                t = t[:remaining]
                out.append(t)
                total += len(t)
            return "\n".join(out).strip()
    except Exception:
        return ""


@dataclass(frozen=True)
class ExamEngine:
    cfg: AppConfig

    @classmethod
    def default(cls) -> "ExamEngine":
        return cls(get_config())

    def _collect_subject_pdfs(self, *, root: Path, subject: str) -> list[Path]:
        tokens = _subject_tokens(subject)
        pdfs = sorted(root.rglob("*.pdf"))
        if not pdfs:
            return []
        filtered = [p for p in pdfs if any(t in str(p).lower() for t in tokens)]
        return filtered or pdfs

    def _get_sources(self, *, subject: str) -> tuple[list[Path], list[Path], list[str]]:
        normalized = _normalize_subject(subject)
        warnings: list[str] = []

        if normalized == "Maths":
            ms_root = self.cfg.project_root / "marking_scheme" / "MS_maths"
            pyq_root = self.cfg.project_root / "pyqs" / "maths_pyq"
        else:
            ms_root = self.cfg.project_root / "marking_scheme" / "MS_science"
            pyq_root = self.cfg.project_root / "pyqs" / "science_pyq"

        ms_pdfs = self._collect_subject_pdfs(root=ms_root.parent if not ms_root.exists() else ms_root, subject=normalized)
        pyq_pdfs = self._collect_subject_pdfs(root=pyq_root.parent if not pyq_root.exists() else pyq_root, subject=normalized)

        if not ms_pdfs:
            warnings.append(f"No marking scheme PDFs found for {normalized}.")
        if not pyq_pdfs:
            warnings.append(f"No PYQ PDFs found for {normalized}.")
        return ms_pdfs, pyq_pdfs, warnings

    def _build_reference_block(self, *, title: str, docs: list[Path], max_docs: int = 5) -> str:
        if not docs:
            return f"{title}: (no docs found)"
        lines = [f"{title}:"]
        for p in docs[:max_docs]:
            sample = _sample_pdf_text(p)
            if not sample:
                continue
            lines.append(f"\n[Source: {p.name}]\n{sample}")
        return "\n".join(lines).strip()

    def _prompt(self, payload: dict[str, Any], reference_text: str, warnings: list[str]) -> str:
        chapters = payload.get("chapters") or []
        student_answers, student_answers_text = _normalize_student_answers(payload.get("student_answers"))
        return (
            "You are an AI exam system for CBSE Class 9 and 10.\n"
            "Generate an exam response in strict JSON only.\n\n"
            "Rules:\n"
            "- Follow CBSE latest pattern.\n"
            "- Keep content in-syllabus and exam-oriented.\n"
            "- Total marks must exactly match requested total_marks.\n"
            "- Cover all requested chapters with balanced difficulty.\n"
            "- Use PYQ patterns and marking scheme style.\n"
            "- If student answers are provided (structured map or free text), evaluate with partial marking and method-based scoring.\n\n"
            f"Input:\n"
            f"- class: {_safe_text(payload.get('class'))}\n"
            f"- subject: {_safe_text(payload.get('subject'))}\n"
            f"- chapters: {json.dumps(chapters, ensure_ascii=True)}\n"
            f"- difficulty_level: {_safe_text(payload.get('difficulty_level'))}\n"
            f"- total_marks: {int(payload.get('total_marks', 0))}\n"
            f"- paper_pattern: {_safe_text(payload.get('paper_pattern') or 'CBSE latest pattern')}\n"
            f"- student_answers (structured map): {json.dumps(student_answers, ensure_ascii=True)}\n"
            f"- student_answers_text (free text): {json.dumps(student_answers_text, ensure_ascii=True)}\n"
            f"- existing_warnings: {json.dumps(warnings, ensure_ascii=True)}\n\n"
            f"Reference snippets from marking scheme + PYQ PDFs:\n{reference_text}\n\n"
            "Output schema (JSON only):\n"
            "{\n"
            '  "paper": {\n'
            '    "class": "",\n'
            '    "subject": "",\n'
            '    "total_marks": 0,\n'
            '    "sections": [\n'
            "      {\n"
            '        "section_name": "",\n'
            '        "questions": [\n'
            "          {\n"
            '            "question_id": "",\n'
            '            "question": "",\n'
            '            "marks": 0,\n'
            '            "chapter": ""\n'
            "          }\n"
            "        ]\n"
            "      }\n"
            "    ]\n"
            "  },\n"
            '  "answer_key": [\n'
            "    {\n"
            '      "question_id": "",\n'
            '      "question": "",\n'
            '      "answer": "",\n'
            '      "steps": ""\n'
            "    }\n"
            "  ],\n"
            '  "evaluation": [\n'
            "    {\n"
            '      "question_id": "",\n'
            '      "question": "",\n'
            '      "student_answer": "",\n'
            '      "correct_answer": "",\n'
            '      "marks_awarded": 0,\n'
            '      "max_marks": 0,\n'
            '      "feedback": ""\n'
            "    }\n"
            "  ],\n"
            '  "result": {\n'
            '    "total_score": "",\n'
            '    "percentage": "",\n'
            '    "grade": "",\n'
            '    "analysis": {\n'
            '      "strong_areas": [],\n'
            '      "weak_areas": [],\n'
            '      "suggestions": []\n'
            "    }\n"
            "  },\n"
            '  "warnings": []\n'
            "}\n"
        )

    def _ensure_shape(self, data: dict[str, Any], *, fallback_warnings: list[str]) -> dict[str, Any]:
        out = data if isinstance(data, dict) else {}
        out.setdefault("paper", {"class": "", "subject": "", "total_marks": 0, "sections": []})
        out.setdefault("answer_key", [])
        out.setdefault("evaluation", [])
        out.setdefault("result", {"total_score": "", "percentage": "", "grade": "", "analysis": {"strong_areas": [], "weak_areas": [], "suggestions": []}})
        out.setdefault("warnings", [])

        paper = out.get("paper") or {}
        sections = paper.get("sections") or []
        computed_total = 0
        for section in sections:
            for q in (section or {}).get("questions", []):
                try:
                    computed_total += int(q.get("marks", 0))
                except Exception:
                    pass
        try:
            requested = int(paper.get("total_marks", 0))
        except Exception:
            requested = 0
        if requested and computed_total and requested != computed_total:
            out["warnings"].append(
                f"Model output marks mismatch: declared total_marks={requested}, computed={computed_total}."
            )
        if fallback_warnings:
            out["warnings"].extend(fallback_warnings)
        return out

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = get_gemini_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in .env")

        subject = _safe_text(payload.get("subject"))
        ms_pdfs, pyq_pdfs, source_warnings = self._get_sources(subject=subject)
        references = "\n\n".join(
            [
                self._build_reference_block(title="Marking Scheme", docs=ms_pdfs),
                self._build_reference_block(title="PYQ", docs=pyq_pdfs),
            ]
        )
        prompt = self._prompt(payload, references, source_warnings)

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=self.cfg.gemini_model,
            contents=prompt,
            config={"temperature": 0.2},
        )
        text = getattr(resp, "text", None) or ""
        parsed = _extract_json(text)
        return self._ensure_shape(parsed, fallback_warnings=source_warnings)
