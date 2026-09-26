"""Hand-labelled evaluation set.

Each answerable question lists its gold evidence as {doc_id: [pages]} and the
figures a correct answer must contain (`expect`: every group must match, a group
is a list of acceptable spellings). Pages were found by reading the filings;
`python -m eval.check_labels` verifies every expected figure actually appears on
a gold page, so a typo in the labels can't silently skew the metrics.
"""

INFY_Q = "infosys-q1-fy2027-earnings"
INFY_A = "infosys-20f-fy2026"
WIT_Q = "wipro-q1-fy2027-earnings"
WIT_A = "wipro-20f-fy2026"
CTSH_Q = "cognizant-q2-2026-earnings"
CTSH_A = "cognizant-10k-fy2025"
ACN_Q = "accenture-q3-fy2026-earnings"
ACN_A = "accenture-10k-fy2025"

ANSWERABLE = [
    # Infosys: quarterly release
    ("What were Infosys revenues in Q1 FY27?", {INFY_Q: [1]}, [["5,082", "5.08"]]),
    ("What was Infosys's operating margin in Q1 FY27?", {INFY_Q: [1]}, [["21.1"]]),
    ("What revenue growth guidance did Infosys give for FY27?", {INFY_Q: [1]}, [["1.5"], ["3.0", "3%"]]),
    ("What was the total contract value of Infosys large deal wins in Q1 FY27?", {INFY_Q: [1]}, [["3.6"]]),
    ("How much free cash flow did Infosys generate in the June 2026 quarter?", {INFY_Q: [1]}, [["955", "0.96"]]),
    ("What were Infosys's cash and cash equivalents as of June 30, 2026?", {INFY_Q: [7]}, [["2,287"]]),
    # Infosys: annual report
    ("How many employees did Infosys have as of March 31, 2026?", {INFY_A: [39, 62, 111]}, [["328,594"]]),
    ("What share of Infosys revenue came from financial services clients in fiscal 2026?", {INFY_A: [8]}, [["27.9"]]),
    ("What percentage of Infosys revenues came from North America in fiscal 2026?", {INFY_A: [38]}, [["56.1"]]),
    ("What share of total revenues did Infosys's five largest clients account for in fiscal 2026?", {INFY_A: [9, 61]}, [["12.9"]]),
    ("What final dividend per share did the Infosys board recommend for fiscal 2026?", {INFY_A: [87, 229]}, [["25"]]),
    ("Which firm is Infosys's independent registered public accounting firm?", {INFY_A: [157, 158]}, [["Deloitte"]]),
    ("How many Infosys employees were based in India as of March 31, 2026?", {INFY_A: [111]}, [["258,502"]]),
    # Wipro: annual report
    ("What was Wipro's IT Services segment operating margin for fiscal 2026?", {WIT_A: [93]}, [["17.22", "17.2"]]),
    ("What was Wipro's large deal order booking TCV in fiscal year 2026?", {WIT_A: [93]}, [["7,829", "7.8"]]),
    ("What share of Wipro's IT Services revenue comes from the Americas?", {WIT_A: [15]}, [["62"]]),
    ("What share of total revenues did Wipro's ten largest clients account for in fiscal 2026?", {WIT_A: [15]}, [["23.7"]]),
    ("How did Wipro's IT services revenue change in constant currency in fiscal 2026?", {WIT_A: [93]}, [["1.6"]]),
    ("What was Wipro's total order booking in total contract value terms for fiscal year 2026?", {WIT_A: [93]}, [["16,449", "16.4"]]),
    ("How many employees did Wipro and its subsidiaries have as of March 31, 2026?", {WIT_A: [143]}, [["240,000"]]),
    # Wipro: quarterly release
    ("What was Wipro's IT services segment revenue in the quarter ended June 30, 2026?", {WIT_Q: [1]}, [["2,614.5", "2.61"]]),
    ("What revenue outlook did Wipro give for the quarter ending September 30, 2026?", {WIT_Q: [1]}, [["2,574"], ["2,627"]]),
    ("What was Wipro's voluntary attrition in Q1 FY27?", {WIT_Q: [1]}, [["13.9"]]),
    ("What were Wipro's large deal bookings in the June 2026 quarter?", {WIT_Q: [1]}, [["1,626", "1.6"]]),
    ("Who is the CEO of Wipro?", {WIT_Q: [2], WIT_A: [127]}, [["Pallia"]]),
    # Cognizant: annual report
    ("How many employees did Cognizant have at the end of 2025?", {CTSH_A: [18]}, [["351,600"]]),
    ("What were Cognizant's total revenues for 2025?", {CTSH_A: [54]}, [["21,108", "21.1"]]),
    ("How much did the Belcan acquisition contribute to Cognizant's 2025 revenue growth?", {CTSH_A: [51, 54]}, [["260"]]),
    ("Which firm audits Cognizant's financial statements?", {CTSH_A: [72, 86]}, [["PricewaterhouseCoopers"]]),
    ("How much is authorized under Cognizant's stock repurchase program?", {CTSH_A: [48]}, [["13.5"]]),
    # Cognizant: quarterly release
    ("What was Cognizant's revenue in Q2 2026?", {CTSH_Q: [1]}, [["5,481", "5.5", "5.48"]]),
    ("What is Cognizant's 2026 constant currency revenue growth guidance?", {CTSH_Q: [1]}, [["4.0", "4%"], ["5.5"]]),
    ("What was Cognizant's total headcount as of June 30, 2026?", {CTSH_Q: [2]}, [["356,700"]]),
    ("Which company did Cognizant acquire in Q2 2026 and at what purchase price?", {CTSH_Q: [2]}, [["Astreya"], ["634"]]),
    ("What were Cognizant's trailing 12-month bookings as of Q2 2026?", {CTSH_Q: [1, 2]}, [["29.1"]]),
    # Accenture: annual report
    ("How many people did Accenture employ as of August 31, 2025?", {ACN_A: [5, 9]}, [["779,000"]]),
    ("What were Accenture's revenues for fiscal 2025?", {ACN_A: [34]}, [["69.7"]]),
    ("What were Accenture's new bookings in fiscal 2025?", {ACN_A: [34]}, [["80.6"]]),
    ("Which firm audits Accenture?", {ACN_A: [47, 60]}, [["KPMG"]]),
    ("How much in business optimization costs did Accenture record in the fourth quarter of fiscal 2025?", {ACN_A: [34]}, [["615"]]),
    ("What quarterly cash dividend did Accenture declare in September 2025?", {ACN_A: [32, 97]}, [["1.63"]]),
    # Accenture: quarterly release
    ("What were Accenture's new bookings in Q3 FY26?", {ACN_Q: [1, 3]}, [["19.3"]]),
    ("What was Accenture's operating margin in the third quarter of fiscal 2026?", {ACN_Q: [1, 5]}, [["17.0", "17%"]]),
    ("What revenue range does Accenture expect for Q4 fiscal 2026?", {ACN_Q: [8]}, [["17.75"], ["18.4"]]),
    ("What was Accenture's diluted EPS in Q3 FY26?", {ACN_Q: [1]}, [["3.80", "3.8"]]),
    ("What were Accenture's Q3 FY26 revenues in the Americas?", {ACN_Q: [3]}, [["9.14"]]),
    # Cross-company comparisons
    ("Compare the latest quarterly operating margins of Infosys and Accenture.", {INFY_Q: [1], ACN_Q: [1, 5]}, [["21.1"], ["17.0", "17%"]]),
    ("Which had more employees at their last fiscal year-end, Infosys or Cognizant?", {INFY_A: [39, 62, 111], CTSH_A: [18]}, [["328,594", "328,600", "329"], ["351,600", "352"]]),
    ("Compare the large deal bookings of Wipro and Infosys in the June 2026 quarter.", {INFY_Q: [1], WIT_Q: [1]}, [["3.6"], ["1,626", "1.6"]]),
    # Needs the calculator
    ("What percentage of Infosys employees were based in India as of March 31, 2026?", {INFY_A: [111]}, [["78.7", "78.67", "78.6"]]),
    ("By how many US dollars did Wipro's large deal TCV bookings increase from fiscal 2025 to fiscal 2026?", {WIT_A: [93]}, [["2,461", "2.46"]]),
]

UNANSWERABLE = [
    "What was TCS's revenue in fiscal 2026?",
    "How many employees does HCLTech have?",
    "What was Accenture's revenue in fiscal 2030?",
    "Who won the 2026 FIFA World Cup?",
    "What is a good recipe for butter chicken?",
    "What was Cognizant's revenue in the third quarter of 2026?",
    "What was Wipro's revenue for the quarter ending September 30, 2026?",
    "What is Infosys's share price target for 2027?",
    "What is the capital of France?",
    "What was Accenture's actual revenue for the fourth quarter of fiscal 2026?",
    "How many employees does Tech Mahindra have?",
    "What is the Infosys CEO's favourite movie?",
]


def load():
    answerable = [
        {"id": f"a{i:02d}", "question": q, "gold": gold, "expect": expect}
        for i, (q, gold, expect) in enumerate(ANSWERABLE, 1)
    ]
    unanswerable = [{"id": f"u{i:02d}", "question": q} for i, q in enumerate(UNANSWERABLE, 1)]
    return answerable, unanswerable


# Follow-ups: (earlier question, its answer, follow-up, gold for the follow-up, companies it is about).
# The follow-up alone is ambiguous; retrieval must use the conversation to find the gold page.
FOLLOWUPS = [
    ("What were Infosys revenues in Q1 FY27?", "Infosys revenues in Q1 FY27 were $5,082 million.",
     "What was the operating margin?", {INFY_Q: [1]}, ["Infosys"]),
    ("What was Wipro's voluntary attrition in Q1 FY27?", "Wipro's voluntary attrition in Q1 FY27 was 13.9%.",
     "And its large deal bookings that quarter?", {WIT_Q: [1]}, ["Wipro"]),
    ("How many employees did Infosys have as of March 31, 2026?", "Infosys had 328,594 employees as of March 31, 2026.",
     "How many of them were based in India?", {INFY_A: [111]}, ["Infosys"]),
    ("How many employees did Infosys have as of March 31, 2026?", "Infosys had 328,594 employees as of March 31, 2026.",
     "And Cognizant at the end of 2025?", {CTSH_A: [18]}, ["Cognizant"]),
    ("What was Cognizant's revenue in Q2 2026?", "Cognizant's Q2 2026 revenue was $5,481 million.",
     "Which company did they acquire that quarter, and for how much?", {CTSH_Q: [2]}, ["Cognizant"]),
    ("Which firm is Infosys's independent registered public accounting firm?", "Deloitte Haskins & Sells LLP.",
     "Same question for Cognizant.", {CTSH_A: [72, 86]}, ["Cognizant"]),
    ("What share of Infosys revenue came from financial services clients in fiscal 2026?", "27.9% in fiscal 2026.",
     "What about North America's share of revenues?", {INFY_A: [38]}, ["Infosys"]),
    ("What was Wipro's IT services segment revenue in the quarter ended June 30, 2026?", "$2,614.5 million.",
     "What outlook did they give for the next quarter?", {WIT_Q: [1]}, ["Wipro"]),
    ("What were Wipro's large deal bookings in the June 2026 quarter?", "$1,626 million.",
     "How about for the full fiscal year 2026?", {WIT_A: [93]}, ["Wipro"]),
    ("What were Accenture's revenues for fiscal 2025?", "Accenture's fiscal 2025 revenues were $69.7 billion.",
     "And new bookings?", {ACN_A: [34]}, ["Accenture"]),
    ("What were Accenture's new bookings in Q3 FY26?", "Accenture's Q3 FY26 new bookings were $19.3 billion.",
     "What was diluted EPS in that quarter?", {ACN_Q: [1]}, ["Accenture"]),
    ("What was Cognizant's total headcount as of June 30, 2026?", "356,700 as of June 30, 2026.",
     "Compare that with Wipro's headcount.", {CTSH_Q: [2], WIT_A: [143]}, ["Cognizant", "Wipro"]),
]


def load_followups():
    return [
        {"id": f"f{i:02d}", "history": [{"question": q, "answer": a}], "question": f, "gold": gold, "companies": cos}
        for i, (q, a, f, gold, cos) in enumerate(FOLLOWUPS, 1)
    ]
