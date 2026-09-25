"""On-disk index: document metadata, chunk records and a dense matrix.

Layout of an index directory:
    documents.json   list of document records (id, company, period, ...)
    chunks.jsonl     one chunk per line, in the same order as the matrix rows
    vectors.npy      float16 [n_chunks, dim] L2-normalised embeddings
    pdfs/<id>.pdf    the source files, served to the viewer for citations

A few thousand chunks fit comfortably in memory, so search is a single matrix
product; no vector database to run or pay for. Writes go to temp files and are
renamed into place so a crash mid-upload can't leave a half-written index.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.retrieval.bm25 import BM25


@dataclass
class ChunkRecord:
    id: int
    doc_id: str
    company: str
    period: str
    doc_type: str
    page: int
    kind: str
    section: str
    text: str

    def search_text(self, title: str = "") -> str:
        """What gets embedded / keyword-indexed: the chunk plus where it came from."""
        head = f"{self.company} {self.period} {title}".strip()
        section = f"Section: {self.section}\n" if self.section else ""
        return f"{head}\n{section}{self.text}"


class Index:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.documents: list[dict] = []
        self.chunks: list[ChunkRecord] = []
        self.vectors = np.zeros((0, 384), dtype=np.float32)
        self.bm25 = BM25([])
        self._company_rows: dict[str, np.ndarray] = {}
        # Seed PDFs stay in the image; only uploads are written under root/pdfs.
        self.extra_pdf_dirs: list[Path] = []

    # ---- persistence -------------------------------------------------
    @classmethod
    def load(cls, root: Path) -> Index:
        idx = cls(root)
        docs_path = idx.root / "documents.json"
        if docs_path.exists():
            idx.documents = json.loads(docs_path.read_text(encoding="utf-8"))
            with open(idx.root / "chunks.jsonl", encoding="utf-8") as fh:
                idx.chunks = [ChunkRecord(**json.loads(line)) for line in fh if line.strip()]
            idx.vectors = np.load(idx.root / "vectors.npy").astype(np.float32)
            assert len(idx.chunks) == len(idx.vectors), "chunks.jsonl and vectors.npy out of sync"
        idx._rebuild_derived()
        return idx

    def save(self) -> None:
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / ".tmp"
            tmp.mkdir(exist_ok=True)
            (tmp / "documents.json").write_text(json.dumps(self.documents, indent=2), encoding="utf-8")
            with open(tmp / "chunks.jsonl", "w", encoding="utf-8") as fh:
                for c in self.chunks:
                    fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            np.save(tmp / "vectors.npy", self.vectors.astype(np.float16))
            for name in ("chunks.jsonl", "vectors.npy", "documents.json"):
                os.replace(tmp / name, self.root / name)
            shutil.rmtree(tmp, ignore_errors=True)

    def _rebuild_derived(self) -> None:
        titles = {d["id"]: d.get("title", "") for d in self.documents}
        self.bm25 = BM25([c.search_text(titles.get(c.doc_id, "")) for c in self.chunks])
        companies: dict[str, list[int]] = {}
        for c in self.chunks:
            companies.setdefault(c.company.lower(), []).append(c.id)
        self._company_rows = {k: np.array(v, dtype=np.int64) for k, v in companies.items()}

    # ---- mutation ----------------------------------------------------
    def add_document(self, doc: dict, chunks: list[ChunkRecord], vectors: np.ndarray) -> None:
        with self.lock:
            if any(d["id"] == doc["id"] for d in self.documents):
                raise ValueError(f"document {doc['id']!r} already indexed")
            base = len(self.chunks)
            for i, c in enumerate(chunks):
                c.id = base + i
            self.documents.append(doc)
            self.chunks.extend(chunks)
            vectors = vectors.astype(np.float32)
            self.vectors = np.vstack([self.vectors, vectors]) if len(self.vectors) else vectors
            self._rebuild_derived()

    # ---- queries -----------------------------------------------------
    def pdf_path(self, doc_id: str) -> Path | None:
        for folder in [self.root / "pdfs", *self.extra_pdf_dirs]:
            path = folder / f"{doc_id}.pdf"
            if path.exists():
                return path
        return None

    def get_document(self, doc_id: str) -> dict | None:
        return next((d for d in self.documents if d["id"] == doc_id), None)

    def companies(self) -> list[str]:
        return sorted({d["company"] for d in self.documents})

    def rows_for(self, companies: list[str] | None) -> np.ndarray | None:
        """Row ids restricted to these companies (None = no restriction)."""
        if not companies:
            return None
        parts = [self._company_rows.get(c.lower(), np.zeros(0, dtype=np.int64)) for c in companies]
        return np.concatenate(parts) if parts else None


def ensure_live_index(seed_dir: Path, data_dir: Path) -> Path:
    """Copy the baked-in seed index to the writable data dir on first boot.

    The seed ships in the image; uploads are appended to the copy on the volume.
    Changing seed/index/VERSION forces a re-copy (which drops earlier uploads
    from the index; their PDFs stay on disk).
    """
    live = data_dir / "index"
    seed_version = (seed_dir / "index" / "VERSION").read_text().strip() if (seed_dir / "index" / "VERSION").exists() else "0"
    live_version = (live / "VERSION").read_text().strip() if (live / "VERSION").exists() else None
    if live_version != seed_version and (seed_dir / "index").exists():
        pdfs = live / "pdfs"
        if pdfs.exists():
            shutil.move(pdfs, data_dir / "pdfs-keep")
        if live.exists():
            shutil.rmtree(live)
        shutil.copytree(seed_dir / "index", live)
        if (data_dir / "pdfs-keep").exists():
            shutil.move(data_dir / "pdfs-keep", pdfs)
    return live
