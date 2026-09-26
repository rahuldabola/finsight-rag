"""FinSight API."""

from __future__ import annotations

import json
import logging
import os
import secrets
import tempfile
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app import embeddings  # noqa: E402
from app.answer import answer_events  # noqa: E402
from app.config import BACKEND_DIR, get_settings  # noqa: E402
from app.ingest.pipeline import ingest_pdf, slugify  # noqa: E402
from app.retrieval.hybrid import retrieve  # noqa: E402
from app.store import Index, ensure_live_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("finsight")

EVAL_RESULTS = BACKEND_DIR / "eval" / "results.json"


class State:
    index: Index
    jobs: dict[str, dict] = {}
    ingest_lock = threading.Lock()


state = State()


def _warm_models() -> None:
    try:
        embeddings.embed_query("warm up")
        embeddings.rerank("warm up", ["warm up"])
        log.info("models warm")
    except Exception:  # noqa: BLE001
        log.exception("model warm-up failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    live = ensure_live_index(settings.seed_dir, settings.data_dir)
    state.index = Index.load(live)
    state.index.extra_pdf_dirs = [settings.seed_dir / "pdfs"]
    log.info(
        "index loaded: %d documents, %d chunks (commit %s)",
        len(state.index.documents),
        len(state.index.chunks),
        os.environ.get("RAILWAY_GIT_COMMIT_SHA", "local")[:7],
    )
    threading.Thread(target=_warm_models, daemon=True).start()
    yield


app = FastAPI(title="FinSight", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Password"],
)


# ---- rate limiting --------------------------------------------------------
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _check_rate(request: Request) -> None:
    """Sliding one-hour window per client IP; protects the shared Gemini quota."""
    limit = get_settings().queries_per_hour
    if limit <= 0:
        return
    now = time.time()
    with _hits_lock:
        window = _hits[_client_ip(request)]
        while window and now - window[0] > 3600:
            window.popleft()
        if len(window) >= limit:
            retry = int(3600 - (now - window[0])) + 1
            raise HTTPException(429, f"Rate limit: {limit} questions per hour. Try again in {retry // 60 + 1} min.")
        window.append(now)


def _require_admin(password: str | None) -> None:
    expected = get_settings().admin_password
    if not expected:
        raise HTTPException(403, "Uploads are disabled on this deployment (no admin password configured).")
    if not password or not secrets.compare_digest(password, expected):
        raise HTTPException(401, "Wrong admin password.")


# ---- routes ---------------------------------------------------------------
class HistoryTurn(BaseModel):
    question: str = Field(max_length=600)
    answer: str = Field(max_length=6000)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=600)
    # Earlier turns of the conversation, oldest first; used only to resolve follow-ups.
    history: list[HistoryTurn] = Field(default_factory=list, max_length=10)


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=600)
    mode: str = Field("hybrid", pattern="^(hybrid|dense|bm25)$")
    rerank: bool = True
    filters: bool = True
    top_k: int = Field(8, ge=1, le=20)


@app.get("/api/health")
def health() -> dict:
    idx = state.index
    return {
        "status": "ok",
        "documents": len(idx.documents),
        "chunks": len(idx.chunks),
        "model": get_settings().chat_model,
        # Railway sets this for git deploys; lets you confirm which commit is live.
        "commit": os.environ.get("RAILWAY_GIT_COMMIT_SHA", "local")[:7],
    }


@app.get("/api/documents")
def list_documents() -> dict:
    return {"documents": state.index.documents, "companies": state.index.companies()}


@app.get("/api/documents/{doc_id}/pdf")
def document_pdf(doc_id: str):
    path = state.index.pdf_path(doc_id)
    if path is None:
        raise HTTPException(404, "No such document")
    return FileResponse(path, media_type="application/pdf", headers={"Content-Disposition": f'inline; filename="{doc_id}.pdf"'})


@app.post("/api/ask")
def ask(body: AskRequest, request: Request):
    _check_rate(request)
    question = body.question.strip()
    history = [h.model_dump() for h in body.history if h.answer.strip()]

    def sse():
        try:
            for event in answer_events(state.index, question, history=history):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception:  # noqa: BLE001 - surface as an event, not a dropped stream
            log.exception("ask failed")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Something went wrong answering that.'})}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/search")
def search(body: SearchRequest) -> dict:
    """Retrieval only (no LLM): powers the retrieval inspector and the eval."""
    started = time.time()
    result = retrieve(state.index, body.question, top_k=body.top_k, mode=body.mode, rerank=body.rerank, use_filters=body.filters)
    return {
        "companies": result.info.companies,
        "doc_type": result.info.doc_type,
        "best_similarity": round(result.best_similarity, 4),
        "ms": int((time.time() - started) * 1000),
        "hits": [
            {
                "doc_id": h.chunk.doc_id,
                "company": h.chunk.company,
                "page": h.chunk.page,
                "section": h.chunk.section,
                "kind": h.chunk.kind,
                "text": h.chunk.text[:600],
                "dense": round(h.dense, 4),
                "bm25": round(h.bm25, 3),
                "fused": round(h.fused, 5),
                "rerank": None if h.rerank is None else round(h.rerank, 3),
            }
            for h in result.hits
        ],
    }


@app.get("/api/eval")
def eval_results():
    if not EVAL_RESULTS.exists():
        raise HTTPException(404, "No evaluation results published")
    return JSONResponse(json.loads(EVAL_RESULTS.read_text(encoding="utf-8")))


def _run_ingest(job_id: str, tmp_path: Path, doc: dict) -> None:
    job = state.jobs[job_id]
    try:
        with state.ingest_lock:
            job["status"] = "indexing"
            final = state.index.root / "pdfs" / f"{doc['id']}.pdf"
            final.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.replace(final)
            job["document"] = ingest_pdf(state.index, str(final), doc)
        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        log.exception("ingest failed")
        job.update(status="failed", error=str(exc)[:300])
        tmp_path.unlink(missing_ok=True)


@app.post("/api/documents", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    company: str = Form(..., min_length=2, max_length=60),
    period: str = Form(..., min_length=2, max_length=30),
    doc_type: str = Form("annual", pattern="^(annual|earnings|other)$"),
    title: str = Form("", max_length=200),
    x_admin_password: str | None = Header(None),
) -> dict:
    _require_admin(x_admin_password)
    settings = get_settings()
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"PDF larger than {settings.max_upload_mb} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Only PDF files are supported")

    company = company.strip()
    known = {c.lower(): c for c in state.index.companies()}
    company = known.get(company.lower(), company)  # reuse existing spelling so filters match
    doc_id = slugify(f"{company}-{period}-{doc_type}")
    if state.index.get_document(doc_id):
        doc_id = f"{doc_id}-{uuid.uuid4().hex[:4]}"
    doc = {"id": doc_id, "company": company, "period": period.strip(), "doc_type": doc_type,
           "title": title.strip() or f"{company} {period} ({file.filename})", "source": "upload"}

    tmp = Path(tempfile.mkstemp(suffix=".pdf", dir=settings.data_dir)[1])
    tmp.write_bytes(data)
    job_id = uuid.uuid4().hex[:12]
    state.jobs[job_id] = {"id": job_id, "status": "queued", "doc_id": doc_id}
    threading.Thread(target=_run_ingest, args=(job_id, tmp, doc), daemon=True).start()
    return state.jobs[job_id]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    return job
