"""The SSE events the API really emits must match contracts/ask-events.json (shared with the frontend tests)."""

import json
from pathlib import Path

from app import answer as answer_mod
from tests.test_api import FakeChunk, FakePart, _events, _script, client  # noqa: F401  (client is a fixture)

CONTRACT = json.loads((Path(__file__).resolve().parents[2] / "contracts" / "ask-events.json").read_text(encoding="utf-8"))


def check_event(event: dict) -> None:
    spec = CONTRACT["events"][event["type"]]
    keys = set(event)
    assert set(spec["required"]) <= keys, f"{event['type']} is missing {set(spec['required']) - keys}"
    assert keys <= set(spec["required"]) | set(spec["optional"]), f"{event['type']} has extra {keys - set(spec['required']) - set(spec['optional'])}"


def check_source(source: dict) -> None:
    assert set(source) == set(CONTRACT["source"]["required"])
    assert set(source["scores"]) == set(CONTRACT["source"]["scores"])


def test_answer_stream_matches_contract(client, monkeypatch):  # noqa: F811
    _script(monkeypatch)
    events = _events(client.post("/api/ask", json={"question": "How much did Infosys revenue grow in Q1?"}))
    assert {e["type"] for e in events} == {"analysis", "sources", "tool", "token", "done"}
    for event in events:
        check_event(event)
    for source in next(e for e in events if e["type"] == "sources")["sources"]:
        check_source(source)


def test_rewrite_and_calculator_error_events_match_contract(client, monkeypatch):  # noqa: F811
    rounds = iter(
        [
            [FakeChunk([FakePart(text="What was Infosys revenue in Q1 FY2027?")])],
            [FakeChunk([FakePart(function_call=type("Call", (), {"name": "calculate", "args": {"expression": "1/0"}})())])],
            [FakeChunk([FakePart(text="Cannot compute [1].")])],
        ]
    )
    monkeypatch.setattr(answer_mod.llm, "stream_generate", lambda contents, config: next(rounds))
    history = [{"question": "Infosys Q1?", "answer": "ok [1]"}]
    events = _events(client.post("/api/ask", json={"question": "and revenue?", "history": history}))
    assert {"rewrite", "tool"} <= {e["type"] for e in events}
    for event in events:
        check_event(event)
    assert "error" in next(e for e in events if e["type"] == "tool")


def test_error_event_matches_contract(client, monkeypatch):  # noqa: F811
    def boom(contents, config):
        raise answer_mod.llm.LLMUnavailable("down")

    monkeypatch.setattr(answer_mod.llm, "stream_generate", boom)
    events = _events(client.post("/api/ask", json={"question": "Infosys revenue in Q1?"}))
    assert events[-1]["type"] == "error"
    check_event(events[-1])
