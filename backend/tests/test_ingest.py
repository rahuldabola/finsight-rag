"""PDF parsing and ingestion on a small generated PDF (no network, no model downloads)."""

import pymupdf
import pytest

from app.ingest.parse import parse_pdf
from app.ingest.pipeline import ingest_pdf, slugify
from app.store import Index


@pytest.fixture
def pdf(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Liquidity and capital resources", fontsize=18)
    page.insert_text((72, 120), "Cash and cash equivalents were $3,412 million at the end of the quarter.", fontsize=10)
    page.insert_text((72, 140), "1", fontsize=10)  # bare page number: must be dropped
    # A ruled 3x2 table.
    x0, y0, w, h = 72, 200, 120, 24
    for r in range(4):
        page.draw_line((x0, y0 + r * h), (x0 + 2 * w, y0 + r * h))
    for c in range(3):
        page.draw_line((x0 + c * w, y0), (x0 + c * w, y0 + 3 * h))
    for r, row in enumerate([("Metric", "Q1"), ("Revenue", "5,082"), ("Margin", "21.0%")]):
        for c, cell in enumerate(row):
            page.insert_text((x0 + c * w + 6, y0 + r * h + 16), cell, fontsize=10)
    doc.new_page().insert_text((72, 80), "Second page text about attrition of 14.1 percent.", fontsize=10)
    path = tmp_path / "sample.pdf"
    doc.save(path)
    return str(path)


def test_parse_pdf_finds_headings_tables_and_pages(pdf):
    blocks, pages = parse_pdf(pdf)
    assert pages == 2
    assert {"heading", "text", "table"} <= {b.kind for b in blocks}
    heading = next(b for b in blocks if b.kind == "heading")
    assert heading.text == "Liquidity and capital resources" and heading.page == 1
    table = next(b for b in blocks if b.kind == "table")
    assert "| Revenue | 5,082 |" in table.text
    assert all(b.text.strip() != "1" for b in blocks)
    assert {b.page for b in blocks} == {1, 2}


def test_parse_pdf_without_table_detection(pdf):
    blocks, _ = parse_pdf(pdf, detect_tables=False)
    assert all(b.kind != "table" for b in blocks)


def test_ingest_pdf_indexes_chunks_per_page(pdf, tmp_path):
    index = Index(tmp_path / "index")
    doc = ingest_pdf(index, pdf, {"id": "acme-q1", "company": "Acme", "period": "Q1", "doc_type": "earnings", "title": "Acme Q1"})
    assert doc["pages"] == 2 and doc["chunks"] >= 2 and doc["tables"] >= 1
    assert index.get_document("acme-q1")["company"] == "Acme"
    reloaded = Index.load(tmp_path / "index")
    assert len(reloaded.chunks) == doc["chunks"]
    assert {c.page for c in reloaded.chunks} == {1, 2}


def test_slugify():
    assert slugify("Infosys / Q1 FY27 (earnings)") == "infosys-q1-fy27-earnings"
    assert slugify("???") == "document"
