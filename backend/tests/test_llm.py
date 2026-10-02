"""Gemini wrapper: retry only before the first chunk, honour retryDelay, give up on 4xx."""

import pytest
from google.genai import errors

from app import llm


def _api_error(code: int, details: list | None = None) -> errors.APIError:
    return errors.APIError(code, {"error": {"message": "x", "details": details or []}})


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content_stream(self, **_kwargs):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return iter(out)


@pytest.fixture
def models(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    holder = type("Holder", (), {"sleeps": sleeps})()

    def install(outcomes):
        holder.models = FakeModels(outcomes)
        monkeypatch.setattr(llm, "get_client", lambda: type("Client", (), {"models": holder.models})())
        return holder

    return install


def test_streams_all_chunks(models):
    h = models([["a", "b", "c"]])
    assert list(llm.stream_generate([], None)) == ["a", "b", "c"]
    assert h.models.calls == 1


def test_retries_rate_limit_using_retry_delay_hint(models):
    hint = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]
    h = models([_api_error(429, hint), ["ok"]])
    assert list(llm.stream_generate([], None)) == ["ok"]
    assert h.sleeps == [7.0]


def test_backoff_without_hint_and_cap(models):
    h = models([_api_error(503), _api_error(500), ["ok"]])
    assert list(llm.stream_generate([], None)) == ["ok"]
    assert h.sleeps == [2.0, 4.0]
    assert llm._retry_delay(_api_error(429, [{"retryDelay": "300s"}]), 0) == 20.0


def test_client_errors_are_not_retried(models):
    h = models([_api_error(400), ["never"]])
    with pytest.raises(errors.APIError):
        list(llm.stream_generate([], None))
    assert h.models.calls == 1


def test_gives_up_after_last_attempt(models):
    h = models([_api_error(429)] * 3)
    with pytest.raises(errors.APIError):
        list(llm.stream_generate([], None, attempts=3))
    assert h.models.calls == 3


def test_missing_api_key_raises_llm_unavailable(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    llm.get_client.cache_clear()
    with pytest.raises(llm.LLMUnavailable):
        llm.get_client()
    get_settings.cache_clear()
