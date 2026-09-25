"""Blocks -> retrieval chunks.

Rules:
- A chunk never crosses a page (citations stay exact).
- Tables are their own chunks, prefixed with the nearest heading and the text
  right above them (usually the caption: "Revenue by segment, in $ millions").
  Tables longer than the budget are split by rows, repeating the header row.
- Text is packed paragraph by paragraph up to `max_chars`, with a small tail
  overlap so a sentence cut at the boundary appears in both neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingest.parse import Block


@dataclass
class Chunk:
    page: int
    kind: str  # "text" | "table"
    section: str
    text: str


def _split_table(md: str, max_chars: int) -> list[str]:
    lines = md.splitlines()
    if len(md) <= max_chars or len(lines) <= 3:
        return [md]
    header, rule, rows = lines[0], lines[1], lines[2:]
    parts: list[str] = []
    current: list[str] = []
    size = len(header) + len(rule)
    for row in rows:
        if current and size + len(row) > max_chars:
            parts.append("\n".join([header, rule, *current]))
            current, size = [], len(header) + len(rule)
        current.append(row)
        size += len(row) + 1
    if current:
        parts.append("\n".join([header, rule, *current]))
    return parts


def _split_long_paragraph(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind(". ", start + max_chars // 2, end)
            end = cut + 1 if cut != -1 else end
        out.append(text[start:end].strip())
        start = end
    return [p for p in out if p]


def build_chunks(blocks: list[Block], max_chars: int = 1400, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""
    buf: list[str] = []
    buf_page = None
    last_text = ""  # most recent paragraph, used as a table caption

    def flush() -> None:
        nonlocal buf
        if buf and buf_page is not None:
            text = "\n".join(buf).strip()
            if len(text) >= 40:  # skip stray footer fragments
                chunks.append(Chunk(buf_page, "text", section, text))
        buf = []

    for block in blocks:
        if block.page != buf_page:
            flush()
            buf_page = block.page
        if block.kind == "heading":
            # Short bold labels ("Revenue") also look like headings, so only a
            # heading that arrives after a decent amount of text starts a new chunk;
            # otherwise it stays inline and just updates the section label.
            if sum(len(p) for p in buf) >= max_chars // 3:
                flush()
            section = block.text
            buf.append(block.text)
            continue
        if block.kind == "table":
            caption = last_text[-300:] if last_text else ""
            for part in _split_table(block.text, max_chars):
                body = f"{caption}\n{part}" if caption else part
                chunks.append(Chunk(block.page, "table", section, body))
            continue

        for para in _split_long_paragraph(block.text, max_chars):
            size = sum(len(p) + 1 for p in buf)
            if buf and size + len(para) > max_chars:
                tail = buf[-1][-overlap:] if overlap else ""
                flush()
                if tail:
                    buf = [tail]
            buf.append(para)
            last_text = para
    flush()
    return chunks
