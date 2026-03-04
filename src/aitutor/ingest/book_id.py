from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class BookId:
    class_: str
    subject: str
    chapter: str | None
    source_pdf: str


_NCERT_CODE_RE = re.compile(r"^(?P<prefix>[a-z]{4})(?P<classdigit>\d)(?P<chap>\d{2})$", re.IGNORECASE)


def guess_chapter_from_filename(pdf_path: str | Path) -> str | None:
    stem = Path(pdf_path).stem
    m = _NCERT_CODE_RE.match(stem)
    if not m:
        return None
    return str(int(m.group("chap")))


def guess_class_from_filename(pdf_path: str | Path) -> str | None:
    stem = Path(pdf_path).stem
    m = _NCERT_CODE_RE.match(stem)
    if not m:
        return None
    # In common NCERT naming, class 10 books often use "1" (e.g., jesc101)
    if m.group("classdigit") == "1":
        return "10"
    return None


def should_ingest_pdf(pdf_path: str | Path) -> bool:
    stem = Path(pdf_path).stem.lower()
    # Keep only chapter PDFs like jesc101/jemh114. Skip appendices (ps/an/a1/etc.)
    return bool(re.match(r"^[a-z]{4}\d{3}$", stem))

