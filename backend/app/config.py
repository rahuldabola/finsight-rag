"""Runtime settings, read once from the environment.

Everything has a working default so `uvicorn app.main:app` runs locally with only
GEMINI_API_KEY set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    chat_model: str
    embed_model: str
    rerank_model: str
    # Where the live index + uploaded PDFs are kept (a Railway volume in prod).
    data_dir: Path
    # Read-only seed corpus baked into the image, copied to data_dir on first boot.
    seed_dir: Path
    chunk_chars: int
    chunk_overlap: int
    top_k: int
    candidate_k: int
    rerank: bool
    # Minimum cosine similarity of the best dense hit before we even ask the LLM.
    min_similarity: float
    admin_password: str
    cors_origins: list[str] = field(default_factory=list)
    queries_per_hour: int = 30
    max_upload_mb: int = 25


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = os.environ.get("FINSIGHT_CORS_ORIGINS", "*")
    return Settings(
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        chat_model=os.environ.get("FINSIGHT_CHAT_MODEL", "gemini-flash-lite-latest"),
        embed_model=os.environ.get("FINSIGHT_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        rerank_model=os.environ.get("FINSIGHT_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
        data_dir=Path(os.environ.get("FINSIGHT_DATA_DIR", BACKEND_DIR / "data" / "live")),
        seed_dir=Path(os.environ.get("FINSIGHT_SEED_DIR", BACKEND_DIR / "data" / "seed")),
        chunk_chars=_env_int("FINSIGHT_CHUNK_CHARS", 1400),
        chunk_overlap=_env_int("FINSIGHT_CHUNK_OVERLAP", 200),
        top_k=_env_int("FINSIGHT_TOP_K", 8),
        candidate_k=_env_int("FINSIGHT_CANDIDATE_K", 30),
        rerank=_env_bool("FINSIGHT_RERANK", True),
        min_similarity=_env_float("FINSIGHT_MIN_SIMILARITY", 0.62),
        admin_password=os.environ.get("FINSIGHT_ADMIN_PASSWORD", ""),
        cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
        queries_per_hour=_env_int("FINSIGHT_QUERIES_PER_HOUR", 30),
        max_upload_mb=_env_int("FINSIGHT_MAX_UPLOAD_MB", 25),
    )
