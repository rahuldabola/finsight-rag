from app.ingest.chunk import build_chunks
from app.ingest.parse import Block


def test_chunks_never_cross_pages_and_keep_tables_whole():
    blocks = [
        Block(1, "heading", "Results"),
        Block(1, "text", "Revenue grew 4% in the quarter driven by financial services."),
        Block(1, "table", "| Segment | Revenue |\n|---|---|\n| FS | 10 |\n| Health | 5 |"),
        Block(2, "text", "Operating margin expanded 40 basis points year over year to 16.0%."),
    ]
    chunks = build_chunks(blocks, max_chars=500, overlap=50)
    assert {c.page for c in chunks} == {1, 2}
    table = [c for c in chunks if c.kind == "table"]
    assert len(table) == 1
    assert "| Health | 5 |" in table[0].text
    assert table[0].text.startswith("Revenue grew 4%")  # caption from the text above
    assert all(c.section == "Results" for c in chunks)


def test_long_tables_split_by_rows_with_header_repeated():
    rows = "\n".join(f"| row{i} | {i} |" for i in range(200))
    blocks = [Block(1, "table", f"| Name | Value |\n|---|---|\n{rows}")]
    chunks = build_chunks(blocks, max_chars=400, overlap=0)
    assert len(chunks) > 1
    assert all(c.text.startswith("| Name | Value |") for c in chunks)
    assert sum(c.text.count("| row") for c in chunks) == 200


def test_text_is_packed_up_to_budget():
    blocks = [Block(1, "text", f"Sentence number {i} about revenue growth and margins.") for i in range(40)]
    chunks = build_chunks(blocks, max_chars=300, overlap=0)
    assert len(chunks) > 3
    assert all(len(c.text) <= 360 for c in chunks)
