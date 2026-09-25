"""Hybrid retrieval: dense + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank.

For a question naming several companies, retrieval runs once per company and
the final context gets an equal share of slots per company. Otherwise the
company with the most matching text (usually the one with the biggest filing)
crowds the others out, and a "compare A and B" answer ends up citing only A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app import embeddings
from app.retrieval.query import QueryInfo, analyze
from app.store import ChunkRecord, Index

RRF_K = 60
DOC_TYPE_BOOST = 1.15


@dataclass
class Hit:
    chunk: ChunkRecord
    dense: float  # cosine similarity
    bm25: float
    fused: float
    rerank: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None

    @property
    def score(self) -> float:
        return self.rerank if self.rerank is not None else self.fused


@dataclass
class RetrievalResult:
    hits: list[Hit]
    info: QueryInfo
    best_similarity: float


def _top(scores: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
    """Row ids of the k best scores among `rows`, best first."""
    sub = scores[rows]
    k = min(k, len(rows))
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    part = np.argpartition(-sub, k - 1)[:k]
    return rows[part[np.argsort(-sub[part])]]


def _candidates(
    index: Index,
    rows: np.ndarray,
    dense: np.ndarray,
    bm25: np.ndarray,
    mode: str,
    candidate_k: int,
    doc_type: str | None,
) -> list[Hit]:
    dense_top = _top(dense, rows, candidate_k) if mode in ("dense", "hybrid") else np.zeros(0, dtype=np.int64)
    bm25_top = _top(bm25, rows, candidate_k) if mode in ("bm25", "hybrid") else np.zeros(0, dtype=np.int64)
    if mode == "bm25":
        bm25_top = bm25_top[bm25[bm25_top] > 0]

    fused: dict[int, float] = {}
    dense_rank = {int(r): i + 1 for i, r in enumerate(dense_top)}
    bm25_rank = {int(r): i + 1 for i, r in enumerate(bm25_top)}
    for ranks in (dense_rank, bm25_rank):
        for row, rank in ranks.items():
            fused[row] = fused.get(row, 0.0) + 1.0 / (RRF_K + rank)

    hits = []
    for row, score in fused.items():
        chunk = index.chunks[row]
        if doc_type and chunk.doc_type == doc_type:
            score *= DOC_TYPE_BOOST
        hits.append(
            Hit(chunk, float(dense[row]), float(bm25[row]), score, dense_rank=dense_rank.get(row), bm25_rank=bm25_rank.get(row))
        )
    hits.sort(key=lambda h: h.fused, reverse=True)
    return hits[:candidate_k]


def retrieve(
    index: Index,
    query: str,
    top_k: int = 8,
    candidate_k: int = 30,
    mode: str = "hybrid",
    use_filters: bool = True,
    rerank: bool = True,
    rerank_pool: int = 20,
) -> RetrievalResult:
    info = analyze(query, index.companies())
    if not index.chunks:
        return RetrievalResult([], info, 0.0)

    qvec = embeddings.embed_query(query)
    # Scoring reads chunks, vectors and BM25 together; hold the index lock so an
    # upload finishing mid-query can't hand us arrays of different lengths.
    with index.lock:
        dense = index.vectors @ qvec
        bm25 = index.bm25.scores(query)
        if use_filters and info.companies:
            groups = [index.rows_for([c]) for c in info.companies]
        else:
            groups = [np.arange(len(index.chunks))]
        doc_type = info.doc_type if use_filters else None
        group_cands = [_candidates(index, rows, dense, bm25, mode, candidate_k, doc_type) for rows in groups]

    per_group = max(top_k // len(groups), 3)
    selected: list[Hit] = []
    for cands in group_cands:
        if rerank and cands:
            pool = cands[:rerank_pool]
            for hit, score in zip(pool, embeddings.rerank(query, [h.chunk.search_text() for h in pool]), strict=False):
                hit.rerank = score
            cands = sorted(pool, key=lambda h: h.rerank, reverse=True)
        selected.extend(cands[:per_group])

    best = max((h.dense for h in selected), default=0.0)
    return RetrievalResult(selected, info, best)
