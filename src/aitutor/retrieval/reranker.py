from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    return CrossEncoder(MODEL_NAME)


def rerank(query: str, candidates: List[str], top_k: int = 5) -> list[str]:
    """
    Rerank candidate chunk texts for a given query using a cross-encoder.

    Args:
        query: The user query.
        candidates: List of chunk texts to rerank.
        top_k: Number of top-ranked chunks to return.

    Returns:
        List of top_k chunk texts, ordered by descending relevance score.
    """
    if not candidates:
        return []

    model = _get_model()

    pairs = [(query, cand) for cand in candidates]
    scores = model.predict(pairs)

    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: float(x[1]), reverse=True)

    top_indices = [idx for idx, _ in indexed_scores[:top_k]]

    return [candidates[idx] for idx in top_indices]

