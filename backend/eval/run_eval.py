"""Retrieval ablation + end-to-end answer evaluation.

    python -m eval.run_eval                  # retrieval only (offline, no API calls)
    python -m eval.run_eval --answers        # also generate answers with Gemini

Writes eval/results.json (served at /api/eval and shown in the UI) and, with
--answers, eval/answers.jsonl with every generated answer for inspection.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from app.config import BACKEND_DIR, get_settings  # noqa: E402
from app.retrieval.hybrid import retrieve  # noqa: E402
from app.store import Index  # noqa: E402
from eval.questions import load  # noqa: E402

OUT = Path(__file__).resolve().parent

CONFIGS = [
    ("BM25 only", "keyword search, no query understanding", dict(mode="bm25", use_filters=False, rerank=False)),
    ("Dense only", "bge-small embeddings, cosine similarity", dict(mode="dense", use_filters=False, rerank=False)),
    ("Hybrid (RRF)", "dense + BM25 fused with Reciprocal Rank Fusion", dict(mode="hybrid", use_filters=False, rerank=False)),
    ("Hybrid + query filters", "adds company filters with per-company fan-out", dict(mode="hybrid", use_filters=True, rerank=False)),
    ("Hybrid + filters + rerank", "adds a MiniLM cross-encoder over the top 20 (production)", dict(mode="hybrid", use_filters=True, rerank=True)),
]


def gold_hits(hits, gold: dict[str, list[int]]) -> list[int | None]:
    """For each gold document, the 1-based rank of its first retrieved gold page."""
    ranks = []
    for doc_id, pages in gold.items():
        rank = next((i + 1 for i, h in enumerate(hits) if h.chunk.doc_id == doc_id and h.chunk.page in pages), None)
        ranks.append(rank)
    return ranks


def retrieval_eval(index: Index, answerable: list[dict], unanswerable: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    for name, desc, kwargs in CONFIGS:
        r5, r8, rr, lat = [], [], [], []
        for q in answerable:
            t = time.perf_counter()
            res = retrieve(index, q["question"], top_k=8, candidate_k=30, **kwargs)
            lat.append((time.perf_counter() - t) * 1000)
            ranks = gold_hits(res.hits, q["gold"])
            r5.append(sum(1 for r in ranks if r and r <= 5) / len(ranks))
            r8.append(sum(1 for r in ranks if r and r <= 8) / len(ranks))
            first = min((r for r in ranks if r), default=None)
            rr.append(1 / first if first else 0.0)
        rows.append(
            {
                "config": name,
                "description": desc,
                "recall_at_5": round(statistics.mean(r5), 4),
                "recall_at_8": round(statistics.mean(r8), 4),
                "mrr": round(statistics.mean(rr), 4),
                "latency_ms": int(statistics.median(lat)),
            }
        )
        print(f"{name:28s} R@5 {rows[-1]['recall_at_5']:.3f}  R@8 {rows[-1]['recall_at_8']:.3f}  MRR {rows[-1]['mrr']:.3f}  {rows[-1]['latency_ms']} ms")

    # Relevance-gate calibration: best dense similarity, answerable vs not.
    sims_a = [retrieve(index, q["question"], rerank=False).best_similarity for q in answerable]
    sims_u = [retrieve(index, q["question"], rerank=False).best_similarity for q in unanswerable]
    gate = {
        "answerable_min": round(min(sims_a), 4),
        "answerable_p05": round(sorted(sims_a)[max(0, len(sims_a) // 20 - 1)], 4),
        "unanswerable": {q["id"]: round(s, 4) for q, s in zip(unanswerable, sims_u, strict=False)},
        "threshold": get_settings().min_similarity,
        "gated_unanswerable": sum(s < get_settings().min_similarity for s in sims_u),
        "gated_answerable": sum(s < get_settings().min_similarity for s in sims_a),
    }
    print("gate:", json.dumps(gate))
    return rows, gate


_pages: dict = {}


def page_text(doc_id: str, page: int) -> str:
    import pymupdf

    if doc_id not in _pages:
        _pages[doc_id] = pymupdf.open(BACKEND_DIR / "data" / "seed" / "pdfs" / f"{doc_id}.pdf")
    return re.sub(r"\s+", " ", _pages[doc_id][page - 1].get_text())


def citation_supports(cited: list, expect: list[list[str]]) -> bool:
    """A cited page itself prints one of the expected figures (checkable without gold pages,
    which matters because filings repeat key numbers on several pages)."""
    return any(any(alt in page_text(doc, page) for group in expect for alt in group) for doc, page in cited)


def normalize(text: str) -> str:
    return text.replace("**", "").replace(" ", " ").replace(" ", " ")


def answer_eval(index: Index, answerable: list[dict], unanswerable: list[dict], pause: float) -> dict:
    from app.answer import answer_events

    records = []
    for q in answerable + unanswerable:
        started = time.time()
        answer, sources, done, tools, error = "", [], {}, [], None
        for _attempt in range(3):
            answer, sources, done, tools, error = "", [], {}, [], None
            for e in answer_events(index, q["question"]):
                if e["type"] == "token":
                    answer += e["text"]
                elif e["type"] == "sources":
                    sources = e["sources"]
                elif e["type"] == "tool":
                    tools.append(e)
                elif e["type"] == "done":
                    done = e
                elif e["type"] == "error":
                    error = e["message"]
            if not error:
                break
            print(f"  {q['id']} error: {error}; retrying in 30s")
            time.sleep(30)
        rec = {
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "error": error,
            "answered": bool(done.get("answered")) and not error,
            "gated": bool(done.get("gated")),
            "cited": [(s["doc_id"], s["page"]) for s in sources if s["n"] in done.get("cited", [])],
            "tools": [t["expression"] for t in tools],
            "ms": int((time.time() - started) * 1000),
        }
        if "gold" in q:
            text = normalize(answer)
            rec["figures_ok"] = all(any(alt in text for alt in group) for group in q["expect"])
            rec["citation_ok"] = any(doc in q["gold"] and page in q["gold"][doc] for doc, page in rec["cited"])
            rec["citation_supports"] = citation_supports(rec["cited"], q["expect"])
        records.append(rec)
        flag = "ok " if rec.get("figures_ok", not rec["answered"]) else "BAD"
        print(f"{flag} {q['id']} {rec['ms']:>6} ms  {q['question'][:70]}")
        time.sleep(pause)

    with open(OUT / "answers.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ans = [r for r in records if r["id"].startswith("a") and not r["error"]]
    una = [r for r in records if r["id"].startswith("u") and not r["error"]]
    return {
        "evaluated": len(ans) + len(una),
        "errors": sum(1 for r in records if r["error"]),
        "exact_figure_accuracy": round(sum(r["figures_ok"] for r in ans) / len(ans), 4),
        "citation_page_accuracy": round(sum(r["citation_ok"] for r in ans) / len(ans), 4),
        "citation_support_rate": round(sum(r["citation_supports"] for r in ans) / len(ans), 4),
        "refusal_rate_unanswerable": round(sum(not r["answered"] for r in una) / len(una), 4),
        "false_refusal_rate": round(sum(not r["answered"] for r in ans) / len(ans), 4),
        "calculator_used": sum(1 for r in ans if r["tools"]),
        "avg_latency_ms": int(statistics.mean(r["ms"] for r in ans + una)),
        "model": get_settings().chat_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="also run end-to-end answers (uses the Gemini API)")
    parser.add_argument("--pause", type=float, default=4.0, help="seconds between LLM questions (free-tier RPM)")
    args = parser.parse_args()

    index = Index.load(BACKEND_DIR / "data" / "seed" / "index")
    answerable, unanswerable = load()
    retrieval, gate = retrieval_eval(index, answerable, unanswerable)

    previous = json.loads((OUT / "results.json").read_text(encoding="utf-8")) if (OUT / "results.json").exists() else {}
    results = {
        "generated_at": time.strftime("%Y-%m-%d"),
        "corpus": {
            "documents": len(index.documents),
            "chunks": len(index.chunks),
            "pages": sum(d["pages"] for d in index.documents),
        },
        "questions": {"answerable": len(answerable), "unanswerable": len(unanswerable)},
        "retrieval": retrieval,
        "gate": gate,
        "answers": answer_eval(index, answerable, unanswerable, args.pause) if args.answers else previous.get("answers"),
        "notes": previous.get("notes", []),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results.get("answers"), indent=2))


if __name__ == "__main__":
    main()
