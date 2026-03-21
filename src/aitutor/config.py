from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

# Default Gemini model for this project (Gemini Developer API id).
DEFAULT_GEMINI_MODEL = "models/gemini-2.5-flash"


def normalize_gemini_model_id(raw: str) -> str:
    """
    Return a model id suitable for `google.genai` `generate_content`.

    Accepts either `gemini-2.5-flash` or `models/gemini-2.5-flash`.
    Empty input falls back to `DEFAULT_GEMINI_MODEL`.
    """
    s = (raw or "").strip()
    if not s:
        return DEFAULT_GEMINI_MODEL
    if s.startswith("models/"):
        return s
    return f"models/{s}"


def repo_root() -> Path:
    """Repository root (contains `books/`, `data/`, `.env`)."""
    # .../src/aitutor/config.py -> parents[2] is the repo root
    return Path(__file__).resolve().parents[2]


def load_project_dotenv(*, override: bool = True) -> None:
    """Load `.env` from the repo root so it works even if the shell cwd differs."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = repo_root() / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=override)


def get_gemini_api_key() -> str:
    """
    Gemini Developer API key.

    Accepts `GEMINI_API_KEY` (this project) or `GOOGLE_API_KEY` (google-genai SDK convention).
    """
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def get_gemini_model() -> str:
    return normalize_gemini_model_id(os.getenv("GEMINI_MODEL") or "")


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    data_dir: Path
    chroma_dir: Path

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_collection: str = "ncert_chunks_v1"
    chroma_space: str = "cosine"

    # Hallucination prevention: refuse if top similarity is below this
    min_similarity: float = 0.35

    # LLM (Gemini) — default `models/gemini-2.5-flash`; override with GEMINI_MODEL in .env
    gemini_model: str = DEFAULT_GEMINI_MODEL


def get_config(project_root: str | Path | None = None) -> AppConfig:
    root = Path(project_root) if project_root else repo_root()
    data_dir = root / "data"
    chroma_dir = data_dir / "chroma"
    return AppConfig(
        project_root=root,
        data_dir=data_dir,
        chroma_dir=chroma_dir,
        gemini_model=get_gemini_model(),
    )

