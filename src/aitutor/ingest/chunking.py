from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def _looks_meaningful(text: str) -> bool:
    # Keep chunks that are likely explanatory paragraph content.
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < 18:
        return False
    # Avoid heading-only / metadata-ish snippets.
    if len(text.strip()) < 120:
        return False
    return True


def _clean_paragraph_block(lines: list[str]) -> str:
    cleaned: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip obvious tiny heading-only lines inside a paragraph block.
        if _is_heading(line) and len(line.split()) <= 6:
            continue
        cleaned.append(line)
    return " ".join(cleaned).strip()


def concept_chunk_pages(
    pages: list[PageText],
    *,
    default_title: str = "Chapter Introduction",
    max_chars: int = 3500,
) -> list[ConceptChunk]:
    # Requirement: recursive chunking with overlap.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # 1) Build section-level buffers first (heading + paragraph lines with page ranges).
    sections: list[tuple[str, str, int, int]] = []
    current_title = default_title
    section_lines: list[str] = []
    section_pages: list[int] = []

    def flush_section() -> None:
        nonlocal section_lines, section_pages
        if not section_lines:
            section_pages = []
            return
        raw = _clean_paragraph_block(section_lines)
        if raw:
            ps = min(section_pages) if section_pages else 1
            pe = max(section_pages) if section_pages else ps
            sections.append((current_title, raw, ps, pe))
        section_lines = []
        section_pages = []

    for page_num, line in _iter_lines_with_pages(pages):
        if _is_heading(line):
            flush_section()
            current_title = line
            continue
        section_lines.append(line)
        section_pages.append(page_num)
        if sum(len(x) + 1 for x in section_lines) >= max_chars:
            flush_section()

    flush_section()

    # 2) Apply recursive splitter to each meaningful section and keep only useful chunks.
    out: list[ConceptChunk] = []
    for title, section_text, ps, pe in sections:
        for piece in splitter.split_text(section_text):
            text = piece.strip()
            if not _looks_meaningful(text):
                continue
            out.append(
                ConceptChunk(
                    concept_title=title or default_title,
                    text=text,
                    page_start=ps,
                    page_end=pe,
                )
            )
    return out

