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

    mm_index = sub.add_parser("mm-index", help="Build multimodal index (text + optional diagrams)")
    mm_index.add_argument("--class", dest="class_", required=True, help="Class (e.g., 10)")
    mm_index.add_argument("--subject", required=True, help="Subject (e.g., Science)")
    mm_index.add_argument("--chapter", default=None, help="Optional chapter filter")
    mm_index.add_argument("--diagrams-dir", default=None, help="Optional folder containing diagram images")
    mm_index.add_argument(
        "--extract-pdf-diagrams",
        action="store_true",
        help="Extract embedded diagrams from PDFs in --diagrams-dir before indexing images",
    )
    mm_index.add_argument(
        "--diagrams-cache-dir",
        default=None,
        help="Output folder for extracted PDF diagrams (default: data/diagram_cache)",
    )
    mm_index.add_argument(
        "--max-text-chunks",
        type=int,
        default=5000,
        help="Max text chunks to embed into multimodal index (lower for free-tier quota)",
    )
    mm_index.add_argument(
        "--text-batch-size",
        type=int,
        default=32,
        help="Batch size per embed API request",
    )
    mm_index.add_argument(
        "--resume",
        action="store_true",
        help="Resume indexing without resetting multimodal collection; skip already indexed items",
    )

    diagram_ask = sub.add_parser("diagram-ask", help="Explain uploaded diagram with NCERT grounding")
    diagram_ask.add_argument("--class", dest="class_", required=True, help="Class (e.g., 10)")
    diagram_ask.add_argument("--subject", required=True, help="Subject (e.g., Science)")
    diagram_ask.add_argument("--chapter", default=None, help="Optional chapter filter")
    diagram_ask.add_argument("--image-path", required=True, help="Path to diagram image")
    diagram_ask.add_argument("--top-k", type=int, default=5, help="Top-K multimodal retrieval")

    voice_ask = sub.add_parser("voice-ask", help="Answer spoken student query from audio file")
    voice_ask.add_argument("--class", dest="class_", required=True, help="Class (e.g., 10)")
    voice_ask.add_argument("--subject", required=True, help="Subject (e.g., Science)")
    voice_ask.add_argument("--chapter", default=None, help="Optional chapter filter")
    voice_ask.add_argument("--audio-path", required=True, help="Path to audio file")
    voice_ask.add_argument("--mime-type", default="audio/mpeg", help="Audio MIME type")
    voice_ask.add_argument("--top-k", type=int, default=5, help="Top-K text retrieval")

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

    if args.cmd == "mm-index":
        from .multimodal import MultiModalTutor

        mm = MultiModalTutor.default()
        result = mm.rebuild_multimodal_index(
            class_=str(args.class_),
            subject=str(args.subject),
            chapter=str(args.chapter) if args.chapter else None,
            diagrams_dir=args.diagrams_dir,
            extract_from_pdfs=bool(args.extract_pdf_diagrams),
            diagrams_cache_dir=args.diagrams_cache_dir,
            max_text_chunks=args.max_text_chunks,
            text_batch_size=args.text_batch_size,
            resume=bool(args.resume),
        )
        print(result)
        return 0

    if args.cmd == "diagram-ask":
        from pathlib import Path
        import mimetypes
        from .multimodal import MultiModalTutor

        image_path = Path(args.image_path)
        image_bytes = image_path.read_bytes()
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        mm = MultiModalTutor.default()
        result = mm.diagram_explain(
            image_bytes=image_bytes,
            mime_type=mime,
            class_=str(args.class_),
            subject=str(args.subject),
            chapter=str(args.chapter) if args.chapter else None,
            top_k=args.top_k,
        )
        print(result.get("answer", ""))
        return 0

    if args.cmd == "voice-ask":
        from pathlib import Path
        from .multimodal import MultiModalTutor

        audio_path = Path(args.audio_path)
        audio_bytes = audio_path.read_bytes()
        mm = MultiModalTutor.default()
        result = mm.voice_answer(
            audio_bytes=audio_bytes,
            mime_type=str(args.mime_type),
            class_=str(args.class_),
            subject=str(args.subject),
            chapter=str(args.chapter) if args.chapter else None,
            top_k=args.top_k,
        )
        print(f"Transcript: {result.get('transcript', '')}")
        print(f"Answer: {result.get('answer', '')}")
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd}")

