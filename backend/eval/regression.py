"""Retrieval regression gate for CI: fails when recall drops below the published baseline.

    python -m eval.regression

Runs the labelled questions through the real embedding model against the real seed index, without the
reranker (so CI only downloads the small embedding model) and without any LLM call. The floors sit a
little under the numbers in README.md / results.json so harmless noise passes but a real regression
(a chunking, parsing, query-understanding or fusion change that loses evidence) does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BACKEND_DIR, get_settings  # noqa: E402
from app.retrieval.hybrid import retrieve  # noqa: E402
from app.store import Index  # noqa: E402
from eval.questions import load  # noqa: E402
from eval.run_eval import gold_hits  # noqa: E402

# (name, retrieve() kwargs, minimum Recall@8). Published: hybrid 0.873, hybrid + filters 0.922.
FLOORS = [
    ("Hybrid (RRF)", dict(mode="hybrid", use_filters=False, rerank=False), 0.85),
    ("Hybrid + query filters", dict(mode="hybrid", use_filters=True, rerank=False), 0.90),
]


def main() -> int:
    index = Index.load(BACKEND_DIR / "data" / "seed" / "index")
    answerable, unanswerable = load()
    failed = False

    for name, kwargs, floor in FLOORS:
        recalls = []
        for q in answerable:
            hits = retrieve(index, q["question"], top_k=8, candidate_k=30, **kwargs).hits
            ranks = gold_hits(hits, q["gold"])
            recalls.append(sum(1 for r in ranks if r and r <= 8) / len(ranks))
        recall = sum(recalls) / len(recalls)
        ok = recall >= floor
        failed |= not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:24s} Recall@8 {recall:.3f} (floor {floor:.2f}, {len(recalls)} questions)")

    # The relevance gate must never block an answerable question.
    threshold = get_settings().min_similarity
    gated = [q["question"] for q in answerable if retrieve(index, q["question"], rerank=False).best_similarity < threshold]
    blocked = sum(retrieve(index, q["question"], rerank=False).best_similarity < threshold for q in unanswerable)
    print(f"{'ok  ' if not gated else 'FAIL'} relevance gate blocks {len(gated)} answerable, {blocked}/{len(unanswerable)} unanswerable")
    failed |= bool(gated)
    for question in gated:
        print(f"     wrongly gated: {question}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
