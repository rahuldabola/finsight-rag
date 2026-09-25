import numpy as np
import pytest

from app import embeddings
from app.retrieval.hybrid import retrieve
from app.store import ChunkRecord, Index

TEXTS = [
    ("Infosys", "earnings", "Infosys revenue for Q1 was $5,082 million, growth of 2.4% in constant currency."),
    ("Infosys", "annual", "Infosys attrition for the fiscal year was 14.1% among employees."),
    ("Wipro", "earnings", "Wipro revenue for Q1 increased 0.9% YoY in constant currency."),
    ("Wipro", "annual", "Wipro employees attrition voluntary was 15%."),
    ("Accenture", "earnings", "Accenture revenues were $18.7 billion for the third quarter."),
]


@pytest.fixture
def index(tmp_path):
    idx = Index(tmp_path / "index")
    for company, dtype, text in TEXTS:
        rec = ChunkRecord(-1, f"{company.lower()}-{dtype}", company, "FY", dtype, 1, "text", "", text)
        idx.add_document(
            {"id": rec.doc_id, "company": company, "title": ""}, [rec], embeddings.embed_documents([rec.search_text()])
        )
    return idx


def test_company_filter_restricts_results(index):
    res = retrieve(index, "What was Wipro attrition?", top_k=3, candidate_k=5)
    assert res.info.companies == ["Wipro"]
    assert {h.chunk.company for h in res.hits} == {"Wipro"}


def test_comparison_fans_out_per_company(index):
    res = retrieve(index, "Compare Infosys and Accenture revenue", top_k=4, candidate_k=5)
    assert {h.chunk.company for h in res.hits} == {"Infosys", "Accenture"}


def test_no_filter_mode_searches_everything(index):
    res = retrieve(index, "attrition employees", top_k=5, candidate_k=5, use_filters=False, rerank=False)
    top_two = {h.chunk.company for h in sorted(res.hits, key=lambda h: -h.fused)[:2]}
    assert top_two == {"Infosys", "Wipro"}


def test_bm25_only_mode_drops_zero_scores(index):
    res = retrieve(index, "attrition", top_k=5, candidate_k=5, mode="bm25", use_filters=False, rerank=False)
    assert res.hits and all("attrition" in h.chunk.text for h in res.hits)


def test_save_and_load_roundtrip(index):
    index.save()
    loaded = Index.load(index.root)
    assert len(loaded.chunks) == len(TEXTS)
    assert np.allclose(loaded.vectors, index.vectors, atol=1e-3)
    assert loaded.companies() == ["Accenture", "Infosys", "Wipro"]
