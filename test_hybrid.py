from __future__ import annotations

"""
Temporary script to exercise the hybrid retriever pipeline.

Uses the same Retriever constructor pattern as the main project:
    from aitutor.retrieval import Retriever
    retriever = Retriever.default()

Query:
    "why does southwest monsoon reverse direction"

Prints only the final retrieved chunks (after vector + BM25 + rerank/fallback).
"""

import sys
from pathlib import Path

# --- TEMP: ensure local `src/` is importable when running directly ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# --- END TEMP ---

from aitutor.retrieval import Retriever


def main() -> None:
    query = "why does southwest monsoon reverse direction"

    # Adjust these as needed to match indexed NCERT content.
    class_ = "10"
    subject = "Science"
    chapter = None  # or e.g. "Geography" if you use chapter filters

    retriever = Retriever.default()
    results = retriever.retrieve(
        query=query,
        class_=class_,
        subject=subject,
        chapter=chapter,
        top_k=5,
    )

    print("\n=== HYBRID TEST: FINAL RETRIEVED CHUNKS ===")
    if not results:
        print("No chunks retrieved.")
        return

    for i, rc in enumerate(results, start=1):
        m = rc.chunk.metadata or {}
        pages = ""
        if "page_start" in m or "page_end" in m:
            pages = f" [pages {m.get('page_start')}–{m.get('page_end')}]"
        title = f" | concept: {m.get('concept_title')}" if m.get("concept_title") else ""
        header = (
            f"{i}. id={rc.chunk.id}"
            f"{pages}"
            f"{title}"
            f" | similarity={rc.similarity:.4f}"
        )
        print(header)
        print("-" * len(header))
        text = rc.chunk.text.replace("\n", " ")
        print(text[:500])
        print()


if __name__ == "__main__":
    main()

