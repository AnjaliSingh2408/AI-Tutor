from __future__ import annotations

from pathlib import Path
import uuid

from tqdm import tqdm

from ..vectorstore import ChromaStore
from .book_id import guess_chapter_from_filename, should_ingest_pdf
from .chunking import concept_chunk_pages
from .pdf_extract import extract_pages


def ingest_books(
    *,
    books_dir: str | Path,
    class_: str,
    subject: str,
    chapter_override: str | None = None,
) -> None:
    books_dir = Path(books_dir)
    if not books_dir.exists():
        raise FileNotFoundError(f"books_dir not found: {books_dir}")

    # Allow subject-wise subfolders, e.g. books/Science/*.pdf, books/Maths/*.pdf
    pdfs = sorted([p for p in books_dir.rglob("*.pdf") if should_ingest_pdf(p)])
    if not pdfs:
        print(f"[ingest] No chapter PDFs found under: {books_dir} (nothing to ingest)")
        return

    store = ChromaStore.default()

    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict] = []

    for pdf in tqdm(pdfs, desc=f"Ingesting {subject} PDFs"):
        chapter = chapter_override or guess_chapter_from_filename(pdf) or "unknown"
        pages = extract_pages(pdf)
        concept_chunks = concept_chunk_pages(pages)

        for idx, c in enumerate(concept_chunks):
            if not c.text or not c.text.strip():
                continue
            chunk_id = f"{class_}|{subject}|{chapter}|{pdf.stem}|{idx}|{uuid.uuid4().hex[:8]}"
            ids.append(chunk_id)
            texts.append(c.text)
            metas.append(
                {
                    "class": str(class_),
                    "subject": str(subject),
                    "chapter": str(chapter),
                    "concept_title": c.concept_title,
                    "source_pdf": pdf.name,
                    "pdf_stem": pdf.stem,
                    "page_start": int(c.page_start),
                    "page_end": int(c.page_end),
                }
            )

        # batch write per PDF (keeps memory bounded)
        if ids:
            store.add_texts(ids=ids, texts=texts, metadatas=metas)
            # Requirement: print 2-3 stored chunks to verify content quality.
            sample = store.get(
                where={
                    "$and": [
                        {"class": {"$eq": str(class_)}},
                        {"subject": {"$eq": str(subject)}},
                        {"chapter": {"$eq": str(chapter)}},
                    ]
                },
                limit=3,
                include=["documents", "metadatas"],
            )
            docs = sample.get("documents") or []
            smeta = sample.get("metadatas") or []
            print("[INGEST-DEBUG] sample stored chunks")
            for i, (d, m) in enumerate(zip(docs, smeta), start=1):
                title = (m or {}).get("concept_title", "Concept")
                text = (d or "").replace("\n", " ").strip()
                print(f"[INGEST-DEBUG]  C{i}: concept={title!r} text={text[:220]}")
        ids, texts, metas = [], [], []

