# NCERT-Grounded RAG AI Tutor (v1)

Retrieval-Augmented Generation (RAG) tutor that answers **strictly from NCERT textbook PDFs** (in `books/`) using:

- Concept-based chunking (heuristic section splits, not arbitrary token windows)
- Metadata-filtered retrieval (class/subject/chapter)
- Top-k vector similarity retrieval (Chroma)
- Grounded LLM generation with refusal on low retrieval confidence

## Folder layout

- `books/`: NCERT chapter PDFs (already present in your workspace)
- `src/aitutor/`: ingestion + retrieval + generation code
- `data/`: local persistent Chroma index (created after ingestion)

## Setup

Create a venv and install dependencies:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
```

Set a Gemini LLM key. Create `.env`:

```bash
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-1.5-pro
```

## Ingest NCERT PDFs

For your Class 10 Science PDFs:

```bash
python -m aitutor ingest --books-dir books\\Science --class 10 --subject Science
```

For Mathematics:

```bash
python -m aitutor ingest --books-dir books\\Mathematics --class 10 --subject Mathematics
```

## Ask questions (CLI)

```bash
python -m aitutor ask --class 10 --subject Science --chapter 1 "What is a balanced chemical equation?"
```

If you omit `--chapter`, it searches across the subject.

## Run API (FastAPI)

```bash
uvicorn aitutor.api.main:app --reload
```

Example request body for `/ask`:

```json
{
  "query": "What is a displacement reaction?",
  "class": "10",
  "subject": "Science",
  "chapter": "1"
}
```

