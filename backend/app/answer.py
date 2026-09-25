"""Question -> retrieval -> grounded, cited, streamed answer.

`answer_events` yields plain dicts that the API turns into Server-Sent Events:

    {"type": "analysis", ...}   detected companies / document type
    {"type": "sources", ...}    the numbered passages the model is allowed to use
    {"type": "token", "text"}   answer text, streamed
    {"type": "tool", ...}       a calculator call and its result
    {"type": "done", ...}       which sources were cited, whether it answered
    {"type": "error", ...}
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator

from google.genai import errors, types

from app import llm
from app.calc import CalcError, calculate
from app.config import get_settings
from app.retrieval.hybrid import Hit, retrieve
from app.store import Index

log = logging.getLogger(__name__)

NOT_FOUND = "I couldn't find this in the indexed filings."
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = f"""You are FinSight, an analyst assistant that answers questions about company filings
(annual reports on Form 20-F / 10-K and quarterly earnings releases).

Rules:
1. Use ONLY the numbered sources provided. Never use outside knowledge for facts or figures.
2. Cite every factual sentence with its source number in square brackets, e.g. "Revenue was $5,082 million [2]."
   Cite several like [1][4]. Only cite sources that actually contain the fact.
3. Quote figures exactly as written in the sources, with units and the period they refer to
   (e.g. "Q1 FY27", "fiscal 2025"). Mind the currency (USD vs INR) and scale (million vs billion, crore).
4. For any arithmetic (growth rates, differences, ratios, sums) call the `calculate` tool and use its result.
   Never compute numbers in your head.
5. If the sources do not contain the answer, reply exactly: "{NOT_FOUND}" and, if useful, one sentence on
   what related information the sources do contain. If only part is answered, answer that part and say what is missing.
   Guidance, outlook and forecasts are not actual results: if the question asks for an actual figure and the sources
   only give an expected range, start with "{NOT_FOUND}" and then give the outlook, clearly labelled as an outlook.
   Companies name the same metric differently (e.g. "TCV of large deal wins" and "large deal bookings", "headcount"
   and "employees", "revenues" and "net revenues"); treat such synonyms as the same metric.
6. When comparing companies, use a short Markdown table when it helps, and note if periods differ
   (e.g. fiscal years ending March vs August vs December).
7. Be concise: lead with the direct answer, then supporting detail. No preamble."""

CALC_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="calculate",
            description="Evaluate an arithmetic expression exactly. Supports + - * / ** % and parentheses. "
            "Example: (5082 - 4941) / 4941 * 100",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"expression": types.Schema(type=types.Type.STRING, description="Arithmetic expression")},
                required=["expression"],
            ),
        )
    ]
)


def _source_payload(n: int, hit: Hit, index: Index) -> dict:
    doc = index.get_document(hit.chunk.doc_id) or {}
    return {
        "n": n,
        "doc_id": hit.chunk.doc_id,
        "title": doc.get("title", hit.chunk.doc_id),
        "company": hit.chunk.company,
        "period": hit.chunk.period,
        "doc_type": hit.chunk.doc_type,
        "page": hit.chunk.page,
        "section": hit.chunk.section,
        "kind": hit.chunk.kind,
        "text": hit.chunk.text,
        "scores": {
            "dense": round(hit.dense, 4),
            "bm25": round(hit.bm25, 3),
            "fused": round(hit.fused, 5),
            "rerank": None if hit.rerank is None else round(hit.rerank, 3),
            "dense_rank": hit.dense_rank,
            "bm25_rank": hit.bm25_rank,
        },
    }


def build_context(sources: list[dict]) -> str:
    parts = []
    for s in sources:
        where = f"{s['title']}, page {s['page']}" + (f", section \"{s['section']}\"" if s["section"] else "")
        parts.append(f"[{s['n']}] ({where})\n{s['text']}")
    return "\n\n".join(parts)


def cited_numbers(answer: str, max_n: int) -> list[int]:
    found = {int(m) for m in re.findall(r"\[(\d{1,2})\]", answer)}
    return sorted(n for n in found if 1 <= n <= max_n)


def answer_events(index: Index, question: str, *, rerank: bool | None = None) -> Iterator[dict]:
    settings = get_settings()
    started = time.time()
    result = retrieve(
        index,
        question,
        top_k=settings.top_k,
        candidate_k=settings.candidate_k,
        rerank=settings.rerank if rerank is None else rerank,
        min_similarity=settings.min_similarity,
    )
    yield {
        "type": "analysis",
        "companies": result.info.companies,
        "doc_type": result.info.doc_type,
        "comparison": result.info.comparison,
        "best_similarity": round(result.best_similarity, 4),
        "retrieval_ms": int((time.time() - started) * 1000),
    }
    sources = [_source_payload(i + 1, h, index) for i, h in enumerate(result.hits)]
    yield {"type": "sources", "sources": sources}

    # Relevance gate: when nothing retrieved is even close, don't spend an LLM
    # call inviting it to improvise an answer from unrelated passages.
    if not sources or result.best_similarity < settings.min_similarity:
        yield {"type": "token", "text": NOT_FOUND + " The question doesn't match anything in the indexed documents closely enough."}
        yield {"type": "done", "answered": False, "cited": [], "gated": True, "total_ms": int((time.time() - started) * 1000)}
        return

    prompt = f"Sources:\n\n{build_context(sources)}\n\nQuestion: {question}"
    contents: list[types.Content] = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[CALC_TOOL],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.1,
    )

    answer = ""
    try:
        for _ in range(MAX_TOOL_ROUNDS + 1):
            model_parts: list[types.Part] = []
            calls: list[types.FunctionCall] = []
            for chunk in llm.stream_generate(contents, config):
                cand = (chunk.candidates or [None])[0]
                if cand is None or cand.content is None:
                    continue
                for part in cand.content.parts or []:
                    model_parts.append(part)
                    if part.function_call:
                        calls.append(part.function_call)
                    elif part.text and not part.thought:
                        answer += part.text
                        yield {"type": "token", "text": part.text}
            if not calls:
                break
            contents.append(types.Content(role="model", parts=model_parts))
            responses = []
            for call in calls:
                expr = str((call.args or {}).get("expression", ""))
                try:
                    value: dict = {"result": calculate(expr)}
                except CalcError as exc:
                    value = {"error": str(exc)}
                yield {"type": "tool", "name": "calculate", "expression": expr, **value}
                responses.append(types.Part.from_function_response(name=call.name or "calculate", response=value))
            contents.append(types.Content(role="user", parts=responses))
    except (errors.APIError, llm.LLMUnavailable) as exc:
        log.exception("generation failed")
        code = getattr(exc, "code", None)
        msg = "The language model is rate-limited right now; please retry in a minute." if code == 429 else "Answer generation failed."
        yield {"type": "error", "message": msg}
        return

    yield {
        "type": "done",
        "answered": not answer.strip().startswith(NOT_FOUND),
        "cited": cited_numbers(answer, len(sources)),
        "gated": False,
        "total_ms": int((time.time() - started) * 1000),
    }
