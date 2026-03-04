from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


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

    # LLM (Gemini, Pro)
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")


def get_config(project_root: str | Path | None = None) -> AppConfig:
    # .../AI Tutor/src/aitutor/config.py -> project root is parents[2] (AI Tutor)
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    data_dir = root / "data"
    chroma_dir = data_dir / "chroma"
    return AppConfig(project_root=root, data_dir=data_dir, chroma_dir=chroma_dir)

