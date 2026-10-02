"""answer_events edge cases: citation parsing, refusals, calculator errors, LLM failures."""

import pytest
from google.genai import errors

from app import answer as answer_mod
from app import embeddings
from app.store import ChunkRecord, Index
from tests.test_api import FakeCall, FakeChunk, FakePart


@pytest.fixture
def index(tmp_path, monkeypatch):
    monkeypatch.setenv("FINSIGHT_MIN_SIMILARITY", "0.1")
    from app.config import get_settings

    get_settings.cache_clear()
    idx = Index(tmp_path / "index")
    rec = ChunkRecord(-1, "wipro-q1", "Wipro", "Q1 FY2027", "earnings", 3, "text", "Results", "Wipro revenue was $2,600 million.")
    idx.add_document({"id": "wipro-q1", "company": "Wipro", "title": "Wipro Q1"}, [rec], embeddings.embed_documents([rec.search_text()]))
    yield idx
    get_settings.cache_clear()


def _run(index, monkeypatch, rounds, question="Wipro revenue?"):
    it = iter(rounds)

    def fake(contents, config):
        nxt = next(it)
        if isinstance(nxt, Exception):
            raise nxt
        return iter(nxt)

    monkeypatch.setattr(answer_mod.llm, "stream_generate", fake)
    return list(answer_mod.answer_events(index, question))


def test_cited_numbers_dedupes_and_drops_out_of_range():
    assert answer_mod.cited_numbers("A [2], B [1][2], C [9], D [0].", 3) == [1, 2]
    assert answer_mod.cited_numbers("no citations", 3) == []


def test_build_context_labels_each_source():
    ctx = answer_mod.build_context(
        [
            {"n": 1, "title": "T", "page": 4, "section": "Risk", "text": "body"},
            {"n": 2, "title": "U", "page": 1, "section": "", "text": "x"},
        ]
    )
    assert ctx.startswith('[1] (T, page 4, section "Risk")\nbody')
    assert "[2] (U, page 1)\nx" in ctx


def test_condense_without_history_never_calls_llm(monkeypatch):
    monkeypatch.setattr(answer_mod.llm, "stream_generate", lambda *_: pytest.fail("no LLM call expected"))
    assert answer_mod.condense_question("and Wipro?", []) == "and Wipro?"


@pytest.mark.parametrize("reply", ["", "ok", "x" * 700])
def test_condense_rejects_implausible_rewrites(monkeypatch, reply):
    monkeypatch.setattr(answer_mod.llm, "stream_generate", lambda *_: iter([FakeChunk([FakePart(text=reply)])]))
    assert answer_mod.condense_question("and Wipro?", [{"question": "q", "answer": "a"}]) == "and Wipro?"


def test_condense_keeps_first_line_and_strips_quotes(monkeypatch):
    reply = '"What was Wipro revenue in Q1 FY27?"\nExtra commentary'
    monkeypatch.setattr(answer_mod.llm, "stream_generate", lambda *_: iter([FakeChunk([FakePart(text=reply)])]))
    assert answer_mod.condense_question("and Wipro?", [{"question": "q", "answer": "a"}]) == "What was Wipro revenue in Q1 FY27?"


def test_refusal_is_reported_as_not_answered(index, monkeypatch):
    events = _run(index, monkeypatch, [[FakeChunk([FakePart(text=answer_mod.NOT_FOUND + " Nothing on that.")])]])
    assert events[-1]["type"] == "done" and events[-1]["answered"] is False and events[-1]["gated"] is False


def test_calculator_error_is_fed_back_to_the_model(index, monkeypatch):
    rounds = [
        [FakeChunk([FakePart(function_call=FakeCall("__import__('os').getcwd()"))])],
        [FakeChunk([FakePart(text="I could not compute that [1].")])],
    ]
    events = _run(index, monkeypatch, rounds)
    tool = next(e for e in events if e["type"] == "tool")
    assert "error" in tool and "result" not in tool
    assert events[-1]["type"] == "done" and events[-1]["cited"] == [1]


def test_tool_loop_is_bounded(index, monkeypatch):
    looping = [[FakeChunk([FakePart(function_call=FakeCall("1+1"))])]] * (answer_mod.MAX_TOOL_ROUNDS + 1)
    events = _run(index, monkeypatch, looping)
    assert sum(e["type"] == "tool" for e in events) == answer_mod.MAX_TOOL_ROUNDS + 1
    assert events[-1]["type"] == "done"


@pytest.mark.parametrize(
    ("exc", "needle"),
    [
        (errors.APIError(429, {"error": {"message": "quota"}}), "rate-limited"),
        (errors.APIError(500, {"error": {"message": "boom"}}), "failed"),
        (answer_mod.llm.LLMUnavailable("no key"), "failed"),
    ],
)
def test_generation_failures_become_error_events(index, monkeypatch, exc, needle):
    events = _run(index, monkeypatch, [exc])
    assert events[-1]["type"] == "error" and needle in events[-1]["message"]
