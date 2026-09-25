"""PDF -> page-aware blocks (paragraphs, headings, tables).

Chunks never straddle a page boundary, so every citation points at exactly one
page. Tables are pulled out whole and rendered as Markdown: financial statements
are mostly tables, and splitting a table mid-row destroys the row/column context
the model needs to read a number correctly.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import pymupdf

_WS = re.compile(r"[ \t ]+")
_PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d{1,4}\s*$", re.I)


@dataclass
class Block:
    page: int  # 1-based
    kind: str  # "heading" | "text" | "table"
    text: str


def _clean(text: str) -> str:
    lines = [_WS.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _body_font_size(doc: pymupdf.Document, sample_pages: int = 12) -> float:
    sizes: Counter[float] = Counter()
    for page in doc.pages(0, min(sample_pages, doc.page_count)):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes[round(span["size"], 1)] += len(span["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _is_heading(line_spans: list[dict], text: str, body_size: float) -> bool:
    if not text or len(text) > 120 or text.endswith((".", ",", ";", ":")) and len(text) > 60:
        return False
    if not re.search(r"[A-Za-z]{3}", text):
        return False
    size = max(s["size"] for s in line_spans)
    bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in line_spans if s["text"].strip())
    if size >= body_size + 1.5:
        return True
    # Short bold lines ("Liquidity and capital resources") act as sub-headings.
    return bold and len(text) <= 90 and not text[0].isdigit()


def _table_to_markdown(table) -> str | None:
    rows = table.extract()
    rows = [[_WS.sub(" ", (c or "").replace("\n", " ")).strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return None
    # Drop columns that are empty in every row (common in HTML-printed statements).
    keep = [i for i in range(len(rows[0])) if any(i < len(r) and r[i] for r in rows)]
    rows = [[r[i] if i < len(r) else "" for i in keep] for r in rows]
    if len(keep) < 2:
        return None
    header, *body = rows
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def _inside(rect: pymupdf.Rect, boxes: list[pymupdf.Rect]) -> bool:
    return any(b.contains(rect) or (b & rect).get_area() > 0.6 * rect.get_area() for b in boxes)


def parse_pdf(path: str, detect_tables: bool = True) -> tuple[list[Block], int]:
    """Return (blocks in reading order, page count)."""
    doc = pymupdf.open(path)
    body_size = _body_font_size(doc)
    blocks: list[Block] = []
    for page in doc:
        pno = page.number + 1
        table_boxes: list[pymupdf.Rect] = []
        page_items: list[tuple[float, Block]] = []
        if detect_tables:
            try:
                for table in page.find_tables(strategy="lines").tables:
                    md = _table_to_markdown(table)
                    if md:
                        rect = pymupdf.Rect(table.bbox)
                        table_boxes.append(rect)
                        page_items.append((rect.y0, Block(pno, "table", md)))
            except Exception:  # noqa: BLE001 - a bad table must not sink the page
                table_boxes = []
                page_items = []

        for raw in page.get_text("dict", sort=True)["blocks"]:
            if raw.get("type") != 0:
                continue
            rect = pymupdf.Rect(raw["bbox"])
            if table_boxes and _inside(rect, table_boxes):
                continue
            para: list[str] = []
            for line in raw["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip()]
                text = _clean(" ".join(s["text"] for s in spans))
                if not text or _PAGE_NUMBER.match(text):
                    continue
                if _is_heading(spans, text, body_size):
                    if para:
                        page_items.append((rect.y0, Block(pno, "text", "\n".join(para))))
                        para = []
                    page_items.append((rect.y0, Block(pno, "heading", text)))
                else:
                    para.append(text)
            if para:
                page_items.append((rect.y0, Block(pno, "text", "\n".join(para))))

        page_items.sort(key=lambda item: item[0])
        blocks.extend(b for _, b in page_items)
    count = doc.page_count
    doc.close()
    return blocks, count
