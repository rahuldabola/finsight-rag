"""Deploy the backend to a Hugging Face Docker Space.

    python -m scripts.deploy_space --space rahuldabola/finsight-api \
        --cors https://finsight-ai-rag.vercel.app

Reads HF_TOKEN and GEMINI_API_KEY from the repo-root .env. Creates the Space if
needed, stores the API key / admin password / CORS origins as Space secrets
(never in the repo), and uploads backend/ (Dockerfile at its root). Hugging Face
then builds the image; progress is visible on the Space page.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

BACKEND = Path(__file__).resolve().parent.parent
ENV_FILE = BACKEND.parent / ".env"

SPACE_README = """---
title: FinSight API
emoji: 📈
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: RAG API over SEC filings with hybrid search and citations
---

Backend for [FinSight](https://github.com/rahuldabola/finsight-rag): hybrid retrieval (bge-small + BM25 + RRF),
cross-encoder reranking and cited, streamed Gemini answers over annual reports and earnings releases.

Health check: `/api/health`. The UI lives at https://finsight-ai-rag.vercel.app.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True)
    parser.add_argument("--cors", required=True, help="comma-separated allowed origins")
    parser.add_argument("--admin-password", help="defaults to the existing .env value or a new random one")
    args = parser.parse_args()

    env = dotenv_values(ENV_FILE)
    token = env.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not found: add HF_TOKEN=hf_... to the repo-root .env (write-access token).")
    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)

    password = args.admin_password or env.get("FINSIGHT_ADMIN_PASSWORD") or secrets.token_urlsafe(12)
    if not env.get("FINSIGHT_ADMIN_PASSWORD"):
        with open(ENV_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"\nFINSIGHT_ADMIN_PASSWORD={password}\n")
    for key, value in {
        "GEMINI_API_KEY": env["GEMINI_API_KEY"],
        "FINSIGHT_ADMIN_PASSWORD": password,
        "FINSIGHT_CORS_ORIGINS": args.cors,
    }.items():
        api.add_space_secret(args.space, key, value)

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=BACKEND, capture_output=True, text=True, check=False
    ).stdout.strip() or "local"
    api.add_space_variable(args.space, "FINSIGHT_COMMIT", sha)

    # Code and Space metadata go up in a single commit, so Hugging Face builds the image once.
    # The upload goes through the Hub API, which handles the >10 MB filing PDFs (plain git push would reject them).
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "space"
        shutil.copytree(BACKEND, stage, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".coverage", ".env", "tests", "live",
        ))
        (stage / "eval" / "answers.jsonl").unlink(missing_ok=True)
        (stage / "README.md").write_text(SPACE_README, encoding="utf-8")
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=str(stage),
            commit_message=f"Deploy FinSight backend ({sha[:7]})",
        )
    user, name = args.space.split("/")
    print(f"https://huggingface.co/spaces/{args.space}")
    print(f"https://{user}-{name}.hf.space/api/health")


if __name__ == "__main__":
    main()
