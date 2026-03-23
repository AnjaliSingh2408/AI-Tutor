from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rank_bm25 import BM25Okapi

from ..types import Chunk, RetrievedChunk


def _simple_tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass(frozen=True)
class BM25Retriever:
    chunks: List[Chunk]
    _bm25: BM25Okapi

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> "BM25Retriever":
        tokenized_corpus = [_simple_tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(chunks=chunks, _bm25=bm25)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.chunks:
            return []

        tokenized_query = _simple_tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        top_indices = [idx for idx, _ in indexed_scores[:top_k]]

        out: list[RetrievedChunk] = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = float(scores[idx])
            out.append(RetrievedChunk(chunk=chunk, similarity=score))

        return out

