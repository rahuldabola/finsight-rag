"""Verify each expected figure appears on at least one of its gold pages.

    python -m eval.check_labels
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.questions import load  # noqa: E402

PDFS = Path(__file__).resolve().parent.parent / "data" / "seed" / "pdfs"


def main() -> int:
    answerable, _ = load()
    docs: dict[str, pymupdf.Document] = {}
    problems = 0
    for q in answerable:
        texts = []
        for doc_id, pages in q["gold"].items():
            doc = docs.setdefault(doc_id, pymupdf.open(PDFS / f"{doc_id}.pdf"))
            texts += [re.sub(r"\s+", " ", doc[p - 1].get_text()) for p in pages]
        blob = " ".join(texts)
        # Calculator questions expect a derived number that isn't printed anywhere.
        derived = "percentage of Infosys employees" in q["question"] or "By how many" in q["question"]
        for group in q["expect"]:
            if not derived and not any(alt in blob for alt in group):
                problems += 1
                print(f"{q['id']}: none of {group} on gold pages {q['gold']}  ({q['question']})")
    print(f"checked {len(answerable)} questions, {problems} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
