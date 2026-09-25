"""Cheap, deterministic query understanding (no LLM call).

Extracts which companies a question is about and whether it points at a
quarterly release or an annual filing. Company mentions become hard filters
(with per-company fan-out for comparisons), the document-type hint only a soft
boost, since "revenue growth" could be answered by either kind of document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Tickers and common short forms, mapped to the company name used in the index.
ALIASES = {
    "infy": "Infosys",
    "wit": "Wipro",
    "ctsh": "Cognizant",
    "acn": "Accenture",
}

_QUARTER = re.compile(r"\b(q[1-4]|quarter(ly)?|earnings (release|call)|press release|guidance|sequential|qoq)\b", re.I)
_ANNUAL = re.compile(r"\b(annual|20-f|10-k|fiscal year|full[- ]year|risk factors?|legal proceedings|subsidiaries|board of directors)\b", re.I)
_COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|between|each of|both|all (four|three|two))\b", re.I)


@dataclass
class QueryInfo:
    companies: list[str] = field(default_factory=list)
    doc_type: str | None = None  # "earnings" | "annual" | None
    comparison: bool = False


def analyze(query: str, known_companies: list[str]) -> QueryInfo:
    found: list[str] = []
    for name in known_companies:
        if re.search(rf"\b{re.escape(name)}(?:'s)?\b", query, re.I):
            found.append(name)
    for alias, name in ALIASES.items():
        if name in known_companies and name not in found and re.search(rf"\b{alias}\b", query, re.I):
            found.append(name)

    doc_type = None
    if _QUARTER.search(query) and not _ANNUAL.search(query):
        doc_type = "earnings"
    elif _ANNUAL.search(query) and not _QUARTER.search(query):
        doc_type = "annual"

    return QueryInfo(companies=found, doc_type=doc_type, comparison=len(found) > 1 or bool(_COMPARE.search(query)))
