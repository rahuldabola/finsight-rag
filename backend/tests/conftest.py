import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fake_vec(text: str) -> np.ndarray:
    """Deterministic bag-of-words hash embedding: similar word sets -> similar vectors."""
    v = np.zeros(384, dtype=np.float32)
    for word in text.lower().split():
        v[int(hashlib.md5(word.strip(".,:;?").encode()).hexdigest(), 16) % 384] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    """Tests never download ONNX models: embeddings and reranker are stubbed."""
    from app import embeddings

    monkeypatch.setattr(
        embeddings,
        "embed_documents",
        lambda texts, batch_size=32: np.array([_fake_vec(t) for t in texts]).reshape(len(texts), 384),
    )
    monkeypatch.setattr(embeddings, "embed_query", lambda text: _fake_vec(text))
    monkeypatch.setattr(embeddings, "rerank", lambda q, ps: [float(_fake_vec(q) @ _fake_vec(p)) for p in ps])
