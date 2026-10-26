"""FastAPI service.

uv run uvicorn hybrid_rag.api.main:app --port 8000      # docs at http://localhost:8000/docs

POST /query    question + mode -> cited answer, retrieval debug, per-stage latency
GET  /health   index size, configured LLM and reranker
POST /ingest   rebuild pages/chunks/indexes from data/raw (needs X-Admin-Token; minutes)
GET  /ingest   status of the last ingest

Every /query writes one JSON log line (mode, per-stage latency, citations, refusal), so latency
per stage can be grepped or loaded into a dataframe without extra tooling.
"""

import hmac
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import anthropic
import openai
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from hybrid_rag.api.schemas import Health, IngestStatus, QueryRequest, QueryResponse
from hybrid_rag.config import settings
from hybrid_rag.generation.answer import answer
from hybrid_rag.indexing import bm25_index, vector_store
from hybrid_rag.retrieval.retriever import retrieve

log = logging.getLogger("hybrid_rag.api")
_ingest = {"running": False, "last_result": None}
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.perf_counter()
    try:
        retrieve("warm-up", "hybrid_rerank")  # load embedder, BM25, reranker before first user
    except Exception as e:  # no index yet: start anyway so /health and /ingest work
        log.warning(json.dumps({"event": "warmup_failed", "error": repr(e)}))
    log.info(
        json.dumps({"event": "startup", "warmup_ms": round((time.perf_counter() - t0) * 1000)})
    )
    yield


app = FastAPI(title="Hybrid Search RAG", version="0.1.0", lifespan=lifespan)


def check_rate_limit(client: str) -> None:
    """Sliding one-minute window per client IP.
    ponytail: in-process memory, so limits are per replica; use Redis if this ever scales out."""
    now = time.monotonic()
    with _hits_lock:
        hits = _hits[client]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= settings.rate_limit_per_minute:
            raise HTTPException(429, f"rate limit: {settings.rate_limit_per_minute} queries/minute")
        hits.append(now)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request) -> dict:
    # Plain `def`: FastAPI runs it in a worker thread, so the blocking LLM call doesn't stall
    # the event loop for other requests.
    check_rate_limit(request.client.host if request.client else "unknown")
    if _ingest["running"]:
        raise HTTPException(503, "indexes are being rebuilt; try again in a few minutes")
    request_id = uuid.uuid4().hex[:8]
    try:
        res = answer(req.question, req.mode)
    except (openai.APIConnectionError, anthropic.APIConnectionError) as e:
        log.error(json.dumps({"event": "llm_unreachable", "id": request_id, "error": str(e)}))
        raise HTTPException(503, f"LLM ({settings.llm_provider}) is unreachable") from e
    except (openai.APIStatusError, anthropic.APIStatusError) as e:
        log.error(json.dumps({"event": "llm_error", "id": request_id, "status": e.status_code}))
        raise HTTPException(502, f"LLM ({settings.llm_provider}) returned an error") from e

    log.info(
        json.dumps(
            {
                "event": "query",
                "id": request_id,
                "mode": req.mode,
                "latency_ms": res["latency_ms"],
                "n_citations": len(res["citations"]),
                "invalid_citations": len(res["invalid_citations"]),
                "refused": res["refused"],
            }
        )
    )
    return res


@app.get("/health", response_model=Health)
def health() -> dict:
    try:
        n = vector_store.client().count(settings.qdrant_collection).count
    except Exception:  # missing collection etc.: report it instead of failing the probe
        n = 0
    return {
        "status": "ok" if n else "no_index",
        "chunks_indexed": n,
        "llm": f"{settings.llm_provider}:{settings.llm_model}",
        "reranker": f"{settings.reranker_provider}:{settings.reranker_model}",
        "ingest_running": _ingest["running"],
    }


def _run_ingest() -> None:
    from hybrid_rag.indexing.build import build
    from hybrid_rag.ingestion.pipeline import chunk_all, parse_all

    t0 = time.perf_counter()
    try:
        parse_all()
        chunk_all()
        build()
        bm25_index.load.cache_clear()  # pick up the new BM25 pickle
        _ingest["last_result"] = f"ok in {time.perf_counter() - t0:.0f}s"
    except Exception as e:
        _ingest["last_result"] = f"failed: {e!r}"
        log.exception("ingest failed")
    finally:
        _ingest["running"] = False


@app.post("/ingest", response_model=IngestStatus, status_code=202)
def ingest(tasks: BackgroundTasks, x_admin_token: str | None = Header(default=None)) -> dict:
    if not settings.admin_token:
        raise HTTPException(403, "ingest is disabled (set ADMIN_TOKEN to enable)")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(401, "missing or wrong X-Admin-Token")
    if _ingest["running"]:
        raise HTTPException(409, "an ingest is already running")
    _ingest["running"] = True
    tasks.add_task(_run_ingest)
    return _ingest


@app.get("/ingest", response_model=IngestStatus)
def ingest_status() -> dict:
    return _ingest
