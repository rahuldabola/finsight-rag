# Contributing

Everything below is what CI runs, so a green local run means a green PR.

```bash
# backend (from backend/)
pip install -r requirements-dev.txt
ruff check .                  # lint
mypy app                      # types
pytest                        # 54 tests, fails under 85% coverage; models and LLM are stubbed, no network
python -m eval.check_labels   # every expected figure in the eval set is printed on its gold page
python -m eval.regression     # real embeddings on the seed index: Recall@8 must stay above the published floor

# frontend (from frontend/)
npm ci
npm run lint
npm test                      # Vitest unit tests
npx playwright install chromium   # once
npm run test:e2e              # 8 browser tests; builds and serves the app, mocks /api (see e2e/mock-api.ts)
npm run build                 # tsc -b + vite build
```

## Changing retrieval

Retrieval quality is a measured claim in the README, so changes to parsing, chunking, query understanding or fusion
need numbers:

1. `python -m eval.run_eval` regenerates the retrieval ablation in `backend/eval/results.json` (offline).
2. `python -m eval.run_eval --answers --followups` also re-scores generated answers and follow-ups (needs `GEMINI_API_KEY`;
   free-tier rate limits make it slow on purpose).
3. If the published table moves, update the README tables and the floors in `backend/eval/regression.py` in the same commit.
4. Changing the PDFs or the chunker means rebuilding the seed index (`python -m scripts.build_index`) and bumping
   `backend/data/seed/index/VERSION` so deployed volumes pick it up.

## Style

- Python: `ruff` (config in `backend/pyproject.toml`), type hints on public functions, comments explain *why*.
- TypeScript: `eslint` + strict `tsc`; pure logic lives in `src/lib` so it can be unit-tested without a DOM.
- Tests never call the network or download models; use the fakes in `backend/tests/conftest.py`.
