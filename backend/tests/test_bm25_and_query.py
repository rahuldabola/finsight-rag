from app.retrieval.bm25 import BM25, tokenize
from app.retrieval.query import analyze

COMPANIES = ["Accenture", "Cognizant", "Infosys", "Wipro"]


def test_tokenize_keeps_financial_tokens():
    assert tokenize("Form 20-F: margin 21.1% in Q1") == ["form", "20-f", "margin", "21.1%", "q1"]


def test_bm25_ranks_exact_term_match_first():
    bm = BM25(["revenue grew strongly", "attrition fell to 12%", "the board met twice"])
    scores = bm.scores("what was attrition")
    assert scores.argmax() == 1
    assert scores[2] == 0


def test_analyze_detects_companies_and_tickers():
    info = analyze("Compare INFY and Wipro operating margin", COMPANIES)
    assert set(info.companies) == {"Infosys", "Wipro"}
    assert info.comparison


def test_analyze_doc_type_hints():
    assert analyze("Q1 revenue guidance for Accenture", COMPANIES).doc_type == "earnings"
    assert analyze("Cognizant risk factors in the 10-K", COMPANIES).doc_type == "annual"
    assert analyze("Cognizant headcount", COMPANIES).doc_type is None


def test_analyze_possessive_and_case():
    assert analyze("what is infosys's attrition?", COMPANIES).companies == ["Infosys"]
