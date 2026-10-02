"""app.embeddings against a fake fastembed model: normalisation, query prefix, empty input, locking."""

import numpy as np
import pytest

from app import embeddings
from tests.conftest import REAL


class FakeEmbedder:
    def __init__(self):
        self.seen: list[str] = []

    def embed(self, texts, batch_size=32):
        self.seen.extend(texts)
        for t in texts:
            yield np.full(384, float(len(t)), dtype=np.float32)


class FakeReranker:
    def rerank(self, query, passages, batch_size=4):
        return (len(p) / 100 for p in passages)


@pytest.fixture
def real(monkeypatch):
    """Undo conftest's stubs of the public functions, but keep the ONNX models faked."""
    for name in ("embed_documents", "embed_query", "rerank"):
        monkeypatch.setattr(embeddings, name, REAL[name])
    fake = FakeEmbedder()
    monkeypatch.setattr(embeddings, "_embedder", lambda: fake)
    monkeypatch.setattr(embeddings, "_reranker", lambda: FakeReranker())
    return fake


def test_documents_are_unit_normalised(real):
    vecs = embeddings.embed_documents(["a", "bb", "ccc"])
    assert vecs.shape == (3, 384) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_empty_input_returns_empty_matrix(real):
    assert embeddings.embed_documents([]).shape == (0, 384)
    assert embeddings.rerank("q", []) == []
    assert real.seen == []


def test_query_gets_the_bge_instruction_prefix_but_documents_do_not(real):
    embeddings.embed_documents(["plain passage"])
    q = embeddings.embed_query("revenue?")
    assert real.seen == ["plain passage", embeddings.QUERY_PREFIX + "revenue?"]
    assert q.shape == (384,) and np.isclose(np.linalg.norm(q), 1.0)


def test_rerank_returns_one_float_per_passage(real):
    assert embeddings.rerank("q", ["x" * 10, "y" * 50]) == [0.1, 0.5]


def test_normalize_survives_zero_vectors():
    out = embeddings._normalize(np.zeros((2, 4), dtype=np.float32))
    assert np.all(out == 0) and not np.isnan(out).any()
