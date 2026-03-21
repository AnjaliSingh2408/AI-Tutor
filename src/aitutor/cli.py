from __future__ import annotations

import argparse
from .config import load_project_dotenv


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aitutor", description="NCERT-grounded RAG AI Tutor (v1)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Ingest NCERT PDFs into vector store")
    ingest.add_argument("--books-dir", required=True, help="Path to folder containing NCERT PDFs")
    ingest.add_argument("--class", dest="class_", required=True, help="Class (e.g., 10)")
    ingest.add_argument("--subject", required=True, help="Subject (e.g., Science)")
    ingest.add_argument("--chapter", default=None, help="Optional chapter override")

    ask = sub.add_parser("ask", help="Ask a question grounded in NCERT")
    ask.add_argument("--class", dest="class_", required=True, help="Class (e.g., 10)")
    ask.add_argument("--subject", required=True, help="Subject (e.g., Science)")
    ask.add_argument("--chapter", default=None, help="Optional chapter filter")
    ask.add_argument("--top-k", type=int, default=5, help="Top-K chunks to retrieve")
    ask.add_argument("query", help="Student question (text)")

    return p


def main(argv: list[str] | None = None) -> int:
    load_project_dotenv()
    args = build_parser().parse_args(argv)

    if args.cmd == "ingest":
        from .ingest.pipeline import ingest_books

        ingest_books(
            books_dir=args.books_dir,
            class_=str(args.class_),
            subject=str(args.subject),
            chapter_override=args.chapter,
        )
        return 0

    if args.cmd == "ask":
        from .rag.tutor import Tutor

        tutor = Tutor()
        result = tutor.answer(
            query=args.query,
            class_=str(args.class_),
            subject=str(args.subject),
            chapter=str(args.chapter) if args.chapter else None,
            top_k=args.top_k,
        )
        print(result)
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd}")

