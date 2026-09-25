"""Local embedding + reranking models (ONNX via fastembed, no torch, no API quota).

Models load lazily and once per process: the first request after boot pays the
load (~1s from the image's model cache), everything after reuses them.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np

from app.config import get_settings

_lock = threading.Lock()

# bge models expect this instruction on the query side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(get_settings().embed_model, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))


@lru_cache(maxsize=1)
def _reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(get_settings().rerank_model, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


def embed_documents(texts: list[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    with _lock:
        vecs = list(_embedder().embed(texts, batch_size=batch_size))
    return _normalize(np.asarray(vecs, dtype=np.float32))


def embed_query(text: str) -> np.ndarray:
    with _lock:
        vec = next(iter(_embedder().embed([QUERY_PREFIX + text])))
    return _normalize(np.asarray([vec], dtype=np.float32))[0]


def rerank(query: str, passages: list[str]) -> list[float]:
    """Cross-encoder relevance logits, one per passage (higher is better)."""
    if not passages:
        return []
    with _lock:
        return [float(s) for s in _reranker().rerank(query, passages, batch_size=16)]
