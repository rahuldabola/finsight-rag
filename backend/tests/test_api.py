import json
import time

import pytest
from fastapi.testclient import TestClient

from app import answer as answer_mod
from app import embeddings
from app.store import ChunkRecord, Index


class FakePart:
    def __init__(self, text=None, function_call=None):
        self.text, self.function_call, self.thought = text, function_call, False


class FakeCall:
    def __init__(self, expr):
        self.name, self.args = "calculate", {"expression": expr}


class FakeChunk:
    def __init__(self, parts):
        content = type("C", (), {"parts": parts})()
        self.candidates = [type("Cand", (), {"content": content})()]


def _script(monkeypatch):
    """Two model rounds: first a calculator call, then the streamed answer."""
    rounds = iter(
        [
            [FakeChunk([FakePart(function_call=FakeCall("(5082-4941)/4941*100"))])],
            [FakeChunk([FakePart(text="Revenue was $5,082 million [1], ")]), FakeChunk([FakePart(text="up 2.85% [1].")])],
        ]
    )
    monkeypatch.setattr(answer_mod.llm, "stream_generate", lambda contents, config: next(rounds))


@pytest.fixture
def client(tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    idx = Index(seed / "index")
    rec = ChunkRecord(
        -1, "infosys-q1", "Infosys", "Q1 FY2027", "earnings", 2, "text", "Results",
        "Infosys revenue for Q1 was $5,082 million, up from $4,941 million a year ago.",
    )
    idx.add_document(
        {"id": "infosys-q1", "company": "Infosys", "title": "Infosys Q1", "period": "Q1 FY2027"},
        [rec],
        embeddings.embed_documents([rec.search_text("Infosys Q1")]),
    )
    idx.save()
    (seed / "index" / "VERSION").write_text("t1")
    (seed / "pdfs").mkdir()
    (seed / "pdfs" / "infosys-q1.pdf").write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setenv("FINSIGHT_SEED_DIR", str(seed))
    monkeypatch.setenv("FINSIGHT_DATA_DIR", str(tmp_path / "live"))
    monkeypatch.setenv("FINSIGHT_MIN_SIMILARITY", "0.1")
    monkeypatch.setenv("FINSIGHT_ADMIN_PASSWORD", "pw")
    monkeypatch.setenv("FINSIGHT_QUERIES_PER_HOUR", "3")
    from app.config import get_settings

    get_settings.cache_clear()
    from app import main

    main._hits.clear()
    monkeypatch.setattr(main, "_warm_models", lambda: None)
    with TestClient(main.app) as c:
        yield c
    get_settings.cache_clear()


def _events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_health_and_documents(client):
    health = client.get("/api/health").json()
    assert health["chunks"] == 1 and health["commit"] == "local"
    docs = client.get("/api/documents").json()
    assert docs["companies"] == ["Infosys"]
    pdf = client.get("/api/documents/infosys-q1/pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert client.get("/api/documents/nope/pdf").status_code == 404


def test_ask_streams_sources_tool_call_and_citations(client, monkeypatch):
    _script(monkeypatch)
    events = _events(client.post("/api/ask", json={"question": "How much did Infosys revenue grow in Q1?"}))
    assert [e["type"] for e in events][:2] == ["analysis", "sources"]
    tool = next(e for e in events if e["type"] == "tool")
    assert tool["result"] == pytest.approx(2.853674, rel=1e-5)
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert text == "Revenue was $5,082 million [1], up 2.85% [1]."
    done = events[-1]
    assert done["type"] == "done" and done["answered"] and done["cited"] == [1]


def test_relevance_gate_skips_llm(client, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("LLM must not be called for off-topic questions")

    monkeypatch.setattr(answer_mod.llm, "stream_generate", boom)
    monkeypatch.setenv("FINSIGHT_MIN_SIMILARITY", "0.99")
    from app.config import get_settings

    get_settings.cache_clear()
    events = _events(client.post("/api/ask", json={"question": "Who won the football world cup?"}))
    assert events[-1]["gated"] is True and events[-1]["answered"] is False


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(answer_mod.llm, "stream_generate", lambda c, cfg: iter([FakeChunk([FakePart(text="ok [1]")])]))
    for _ in range(5):
        assert client.post("/api/search", json={"question": "revenue"}).status_code == 200  # not rate limited
    codes = [client.post("/api/ask", json={"question": "Infosys revenue?"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_upload_requires_password_and_pdf(client):
    files = {"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}
    data = {"company": "TCS", "period": "FY2026"}
    assert client.post("/api/documents", files=files, data=data).status_code == 401
    assert client.post("/api/documents", files=files, data=data, headers={"X-Admin-Password": "bad"}).status_code == 401
    bad = client.post(
        "/api/documents", files={"file": ("x.txt", b"hello", "text/plain")}, data=data, headers={"X-Admin-Password": "pw"}
    )
    assert bad.status_code == 415


def test_follow_up_is_rewritten_before_retrieval(client, monkeypatch):
    seen = []
    rounds = iter(
        [
            [FakeChunk([FakePart(text="What was Infosys revenue in Q1 FY2027?\n")])],
            [FakeChunk([FakePart(text="Revenue was $5,082 million [1].")])],
        ]
    )

    def fake(contents, config):
        seen.append(contents[0].parts[0].text)
        return next(rounds)

    monkeypatch.setattr(answer_mod.llm, "stream_generate", fake)
    history = [{"question": "How did Infosys do in Q1?", "answer": "Infosys grew revenue [1]."}]
    events = _events(client.post("/api/ask", json={"question": "and revenue?", "history": history}))
    rewrite = events[0]
    assert rewrite == {"type": "rewrite", "original": "and revenue?", "question": "What was Infosys revenue in Q1 FY2027?"}
    assert "[1]" not in seen[0] and "Follow-up question: and revenue?" in seen[0]
    assert "Question: What was Infosys revenue in Q1 FY2027?" in seen[1]
    assert events[1]["companies"] == ["Infosys"]
    assert events[-1]["answered"] is True


def test_follow_up_rewrite_falls_back_to_raw_question(client, monkeypatch):
    calls = iter([answer_mod.llm.LLMUnavailable("down"), [FakeChunk([FakePart(text="ok [1]")])]])

    def fake(contents, config):
        nxt = next(calls)
        if isinstance(nxt, Exception):
            raise nxt
        return iter(nxt)

    monkeypatch.setattr(answer_mod.llm, "stream_generate", fake)
    history = [{"question": "Infosys Q1 revenue?", "answer": "$5,082 million [1]."}]
    events = _events(client.post("/api/ask", json={"question": "Infosys revenue growth?", "history": history}))
    assert all(e["type"] != "rewrite" for e in events) and events[-1]["type"] == "done"


def test_upload_is_indexed_in_background_and_job_reports_done(client, monkeypatch):
    import pymupdf

    from app import main

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 80), "TCS revenue for FY2026 was $30 billion.", fontsize=11)
    data = doc.tobytes()

    res = client.post(
        "/api/documents",
        files={"file": ("tcs.pdf", data, "application/pdf")},
        data={"company": "TCS", "period": "FY2026", "doc_type": "annual"},
        headers={"X-Admin-Password": "pw"},
    )
    assert res.status_code == 202
    job = res.json()
    for _ in range(100):
        status = client.get(f"/api/jobs/{job['id']}").json()
        if status["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert status["status"] == "done", status
    assert "TCS" in client.get("/api/documents").json()["companies"]
    assert main.state.index.pdf_path(job["doc_id"]) is not None
    assert client.get("/api/jobs/unknown").status_code == 404


def test_upload_disabled_without_admin_password(client, monkeypatch):
    monkeypatch.setenv("FINSIGHT_ADMIN_PASSWORD", "")
    from app.config import get_settings

    get_settings.cache_clear()
    files = {"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}
    res = client.post("/api/documents", files=files, data={"company": "TCS", "period": "FY2026"}, headers={"X-Admin-Password": "pw"})
    assert res.status_code == 403


def test_corrupt_pdf_upload_marks_job_failed(client):
    res = client.post(
        "/api/documents",
        files={"file": ("bad.pdf", b"%PDF-1.4 not really a pdf", "application/pdf")},
        data={"company": "Bad", "period": "FY2026"},
        headers={"X-Admin-Password": "pw"},
    )
    job_id = res.json()["id"]
    for _ in range(100):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert status["status"] == "failed" and status["error"]
