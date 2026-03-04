from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PageText:
    page_num: int  # 1-indexed
    text: str


_REPRINT_RE = re.compile(r"^Reprint\s+\d{4}-\d{2}\s*$", re.IGNORECASE)
_PAGE_OF_RE = re.compile(r"^--\s*\d+\s+of\s+\d+\s*--\s*$")
_ONLY_PAGE_NUM_RE = re.compile(r"^\s*\d+\s*$")


def _collapse_tab_repetitions(line: str) -> str:
    parts = [p.strip() for p in line.split("\t") if p.strip()]
    if len(parts) >= 2 and all(p == parts[0] for p in parts[1:]):
        return parts[0]
    return line


def _collapse_whitespace_repetitions(line: str) -> str:
    # Handle cases like "Activity 1.1  Activity 1.1  Activity 1.1"
    parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
    if len(parts) >= 2 and all(p == parts[0] for p in parts[1:]):
        return parts[0]
    return line


def clean_extracted_text(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        line = _collapse_tab_repetitions(line)
        line = _collapse_whitespace_repetitions(line)

        if _REPRINT_RE.match(line):
            continue
        if _PAGE_OF_RE.match(line):
            continue
        if _ONLY_PAGE_NUM_RE.match(line):
            # Avoid stray page-number lines that are common in PDFs.
            continue

        # Normalize weird multiple spaces
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    pages: list[PageText] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            # "text" mode is usually best for NCERT PDFs
            raw = page.get_text("text")
            cleaned = clean_extracted_text(raw)
            pages.append(PageText(page_num=i + 1, text=cleaned))
    finally:
        doc.close()
    return pages

