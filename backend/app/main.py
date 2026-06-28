"""
Arabic RAG — FastAPI application entry point.

Endpoints
---------
POST /upload   Ingest a PDF into the hybrid retrieval index.
POST /query    Retrieve and answer a question using GPT-4o.
GET  /health   System status and index statistics.

Run
---
From the project root:
    uvicorn backend.app.main:app --reload --port 8000

All heavy objects (FAISS index, BM25 corpus, LLM client) are loaded once
in the lifespan context and stored in app.state.  Route handlers access
them as FastAPI dependencies — no module-level globals.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Project root on sys.path ──────────────────────────────────────────────────
# Needed so `core/` and `utils/` (at the project root) are importable when
# uvicorn is launched from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.api.routes.documents import router as documents_router
from backend.app.api.routes.query import router as query_router
from backend.app.core.config import settings
from core.embeddings import EmbeddingsStore
from core.llm import LLMClient
from core.retriever import HybridRetriever

import os
if settings.hf_token:
    os.environ["HF_TOKEN"] = settings.hf_token



# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise app.state, attempt to load persisted indexes.
    Shutdown: nothing to clean up (FAISS is file-persisted, not a server).
    """
    # Initialise state slots so route handlers can always read them safely.
    app.state.store: EmbeddingsStore | None = None
    app.state.retriever: HybridRetriever | None = None
    app.state.llm: LLMClient | None = None
    app.state.processed_docs: list[str] = []
    app.state.total_chunks: int = 0
    app.state.jobs: dict = {}          # job_id → status dict; resets on restart

    # ── Try loading persisted vector store ────────────────────────────────────
    vs_dir = settings.vector_store_path
    try:
        store = EmbeddingsStore.load(str(vs_dir))
        retriever = HybridRetriever.load(str(vs_dir), store=store)
        app.state.store = store
        app.state.retriever = retriever
        app.state.total_chunks = store.total_vectors
        app.state.processed_docs = list({
            doc["source"] for doc in retriever._corpus
        })
        logger.info(
            "Loaded persisted index: %d vectors across %d document(s).",
            store.total_vectors,
            len(app.state.processed_docs),
        )
    except FileNotFoundError:
        logger.info(
            "No persisted index found at '%s' — pre-loading embedding model "
            "(first-time download may take a few minutes). "
            "Upload a PDF to begin.",
            vs_dir,
        )
        # Eagerly load the model so the first upload is not blocked by a
        # large HuggingFace download inside the background ingestion job.
        store = EmbeddingsStore(model_name=settings.embedding_model)
        retriever = HybridRetriever(store=store)
        app.state.store = store
        app.state.retriever = retriever
    except Exception as exc:
        logger.warning("Could not load persisted index: %s", exc)

    # ── Initialise LLM client ─────────────────────────────────────────────────
    try:
        app.state.llm = LLMClient(
            model=settings.openai_model,
            api_key=settings.openai_api_key or None,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
        logger.info("LLM client ready (%s).", settings.openai_model)
    except EnvironmentError as exc:
        logger.warning(
            "LLM client not initialised — /query will return 503 until fixed. "
            "Reason: %s",
            exc,
        )

    yield  # ── Application runs here ──────────────────────────────────────────

    logger.info("Arabic RAG backend shutting down.")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Arabic RAG API",
    description=(
        "Hybrid retrieval-augmented generation for Arabic PDF documents.\n\n"
        "Upload PDFs via **POST /upload**, then query them via **POST /query**."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten to your frontend origin in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(query_router)


# ── Health endpoint ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    retriever_ready: bool
    llm_ready: bool
    total_chunks: int
    processed_documents: list[str]
    openai_model: str
    embedding_model: str


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    """
    Return system status and index statistics.

    Always returns HTTP 200.  Inspect retriever_ready and llm_ready to
    determine whether the system is fully operational.
    """
    return HealthResponse(
        status="ok",
        retriever_ready=request.app.state.retriever is not None,
        llm_ready=request.app.state.llm is not None,
        total_chunks=request.app.state.total_chunks,
        processed_documents=list(request.app.state.processed_docs),
        openai_model=settings.openai_model,
        embedding_model=settings.embedding_model,
    )
