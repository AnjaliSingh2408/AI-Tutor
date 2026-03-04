from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .pdf_extract import PageText


@dataclass(frozen=True)
class ConceptChunk:
    concept_title: str
    text: str
    page_start: int
    page_end: int


_HEADER_NUMERIC_RE = re.compile(r"^\d+(?:\.\d+){0,3}\s+[A-Za-z].{2,}$")
_CHAPTER_RE = re.compile(r"^CHAPTER$", re.IGNORECASE)
_ACTIVITY_RE = re.compile(r"^Activity\s+\d+(\.\d+)?", re.IGNORECASE)
_EXERCISES_RE = re.compile(r"^(EXERCISES|QUESTIONS|WHAT YOU HAVE LEARNT|GROUP ACTIVITY)\b", re.IGNORECASE)


def _is_heading(line: str) -> bool:
    if _CHAPTER_RE.match(line):
        return True
    if _HEADER_NUMERIC_RE.match(line):
        return True
    if _ACTIVITY_RE.match(line):
        return True
    if _EXERCISES_RE.match(line):
        return True
    # All-caps-ish headings (short)
    if len(line) <= 70 and sum(1 for c in line if c.isalpha() and c.isupper()) >= max(6, int(0.6 * sum(1 for c in line if c.isalpha()))):
        return True
    return False


def _iter_lines_with_pages(pages: Iterable[PageText]) -> Iterable[tuple[int, str]]:
    for p in pages:
        for line in p.text.splitlines():
            line = line.strip()
            if not line:
                continue
            yield (p.page_num, line)


def concept_chunk_pages(
    pages: list[PageText],
    *,
    default_title: str = "Chapter Introduction",
    max_chars: int = 3500,
) -> list[ConceptChunk]:
    chunks: list[ConceptChunk] = []

    current_title = default_title
    current_lines: list[str] = []
    current_pages: list[int] = []

    def flush() -> None:
        nonlocal current_title, current_lines, current_pages
        if not current_lines:
            current_pages = []
            return
        text = "\n".join(current_lines).strip()
        if not text:
            current_lines = []
            current_pages = []
            return
        ps = min(current_pages) if current_pages else 1
        pe = max(current_pages) if current_pages else ps
        chunks.append(ConceptChunk(concept_title=current_title, text=text, page_start=ps, page_end=pe))
        current_lines = []
        current_pages = []

    for page_num, line in _iter_lines_with_pages(pages):
        if _is_heading(line) and current_lines:
            flush()
            current_title = line
            current_lines = []
            current_pages = []
            continue
        if _is_heading(line) and not current_lines:
            current_title = line
            continue

        current_lines.append(line)
        current_pages.append(page_num)

        # Hard split: keep chunks concept-sized and LLM-friendly
        if sum(len(x) + 1 for x in current_lines) >= max_chars:
            flush()
            current_title = current_title + " (cont.)"

    flush()

    # Post-process: drop tiny fragments (usually artifacts) by merging into previous
    merged: list[ConceptChunk] = []
    for c in chunks:
        if merged and len(c.text) < 250:
            prev = merged.pop()
            merged.append(
                ConceptChunk(
                    concept_title=prev.concept_title,
                    text=(prev.text + "\n" + c.text).strip(),
                    page_start=min(prev.page_start, c.page_start),
                    page_end=max(prev.page_end, c.page_end),
                )
            )
        else:
            merged.append(c)
    return merged

