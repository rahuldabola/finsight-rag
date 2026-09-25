"""PDF file + metadata -> chunks + vectors -> appended to an Index."""

from __future__ import annotations

import logging
import re
import time

from app import embeddings
from app.config import get_settings
from app.ingest.chunk import build_chunks
from app.ingest.parse import parse_pdf
from app.store import ChunkRecord, Index

log = logging.getLogger(__name__)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "document"


def ingest_pdf(index: Index, pdf_path: str, doc: dict, save: bool = True) -> dict:
    """Parse, chunk, embed and index one PDF. `doc` needs id/company/period/doc_type/title."""
    settings = get_settings()
    started = time.time()
    blocks, pages = parse_pdf(pdf_path)
    raw = build_chunks(blocks, settings.chunk_chars, settings.chunk_overlap)
    records = [
        ChunkRecord(
            id=-1,
            doc_id=doc["id"],
            company=doc["company"],
            period=doc["period"],
            doc_type=doc["doc_type"],
            page=c.page,
            kind=c.kind,
            section=c.section[:160],
            text=c.text,
        )
        for c in raw
    ]
    vectors = embeddings.embed_documents([r.search_text(doc.get("title", "")) for r in records])
    doc = {**doc, "pages": pages, "chunks": len(records), "tables": sum(r.kind == "table" for r in records)}
    index.add_document(doc, records, vectors)
    if save:
        index.save()
    log.info("indexed %s: %d pages, %d chunks in %.1fs", doc["id"], pages, len(records), time.time() - started)
    return doc
