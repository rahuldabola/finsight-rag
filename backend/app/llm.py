"""Thin Gemini wrapper: one client per process, retry on 429/5xx at stream start."""

from __future__ import annotations

import logging
import time
from functools import lru_cache

from google import genai
from google.genai import errors

from app.config import get_settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    key = get_settings().gemini_api_key
    if not key:
        raise LLMUnavailable("GEMINI_API_KEY is not set")
    return genai.Client(api_key=key)


def _retry_delay(exc: errors.APIError, attempt: int) -> float:
    """Honour Gemini's RetryInfo hint on a 429, capped so a request can't hang."""
    try:
        for detail in (exc.details or {}).get("error", {}).get("details", []):
            if "retryDelay" in detail:
                return min(float(detail["retryDelay"].rstrip("s")), 20.0)
    except (AttributeError, ValueError, TypeError):
        pass
    return min(2.0 * 2**attempt, 20.0)


def stream_generate(contents, config, attempts: int = 3):
    """Yield response chunks. Retries only before the first chunk arrives, so a
    caller never sees duplicated text from a half-finished attempt."""
    client = get_client()
    model = get_settings().chat_model
    for attempt in range(attempts):
        try:
            stream = client.models.generate_content_stream(model=model, contents=contents, config=config)
            first = next(iter(stream), None)
        except errors.APIError as exc:
            retryable = exc.code == 429 or (exc.code or 0) >= 500
            if not retryable or attempt == attempts - 1:
                raise
            delay = _retry_delay(exc, attempt)
            log.warning("gemini %s, retrying in %.1fs", exc.code, delay)
            time.sleep(delay)
            continue
        if first is not None:
            yield first
        yield from stream
        return
