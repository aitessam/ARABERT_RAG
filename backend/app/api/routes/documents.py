"""
POST /upload   — accept a PDF, validate it, and start background ingestion.
GET  /upload/status/{job_id} — poll ingestion progress.

Flow
----
1.  Validate file type and size (synchronous — fast).
2.  Persist raw PDF to data/raw/ (synchronous — fast).
3.  Return 202 Accepted with a job_id immediately (client never times out).
4.  Background task (asyncio thread pool):
      a. Extract pages with ArabicPDFReader.
      b. Chunk with ArabicChunker.
      c. Remove any existing chunks for this filename (idempotent re-upload).
      d. Embed chunks and update FAISS index.
      e. Update BM25 index.
      f. Persist both indexes to disk.
5.  Job status is updated in app.state.jobs (in-memory; reset on restart).

Error codes
-----------
400  Wrong file type or file exceeds the size limit.
404  Job ID not found (status endpoint).
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.core.config import settings
from core.chunker import ArabicChunker
from core.embeddings import EmbeddingsStore
from core.retriever import HybridRetriever
from utils.pdf_reader import ArabicPDFReader

logger = logging.getLogger(__name__)
router = APIRouter()

_ALLOWED_SUFFIXES = {".pdf"}


# ── Response models ───────────────────────────────────────────────────────────

class UploadAccepted(BaseModel):
    """Immediate response — returned before ingestion completes."""
    job_id: str
    filename: str
    message: str


class UploadResult(BaseModel):
    """Populated in JobStatus.result once ingestion succeeds."""
    page_count: int
    chunk_count: int
    total_chunks_in_index: int


class JobStatus(BaseModel):
    job_id: str
    status: str  # "processing" | "done" | "error"
    filename: str
    result: UploadResult | None = None
    error: str | None = None


# ── Background ingestion ──────────────────────────────────────────────────────

def _sync_ingest(
    app: Any,
    filename: str,
    dest: Path,
) -> tuple[int, int, int]:
    """
    CPU-heavy ingestion pipeline — runs in a thread pool via asyncio.to_thread.

    Returns (page_count, chunk_count, total_chunks_in_index).
    Raises on any failure so the caller can set job status to 'error'.
    """
    reader = ArabicPDFReader()
    pages = reader.read(dest)
    if not pages:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "The file may be image-only or password-protected."
        )

    chunker = ArabicChunker(
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    chunks = chunker.chunk_pages(pages, source=filename)
    if not chunks:
        raise ValueError("Document produced no usable chunks after processing.")

    store: EmbeddingsStore | None = app.state.store
    retriever: HybridRetriever | None = app.state.retriever

    if store is None:
        store = EmbeddingsStore(model_name=settings.embedding_model)
        retriever = HybridRetriever(store=store)
    else:
        # Remove stale entries so re-uploads are idempotent.
        removed = store.remove_by_source(filename)
        if removed:
            retriever.remove_by_source(filename)
            logger.info("Re-upload '%s': removed %d stale chunks.", filename, removed)

    store.embed_chunks(chunks, show_progress=False)
    retriever.index_chunks(chunks)

    vs_dir = settings.vector_store_path
    store.save(str(vs_dir))
    retriever.save(str(vs_dir))

    # Hand back the (possibly new) store/retriever references so the async
    # wrapper can update app.state in the event-loop thread.
    app._ingest_result = (store, retriever)

    return len(pages), len(chunks), store.total_vectors


async def _ingest_background(
    app: Any,
    job_id: str,
    filename: str,
    dest: Path,
) -> None:
    """Async wrapper: offloads sync work to a thread, then updates app.state."""
    jobs: dict = app.state.jobs
    try:
        page_count, chunk_count, total = await asyncio.to_thread(
            _sync_ingest, app, filename, dest
        )
        # Update shared state in the event-loop thread (safe — no race here
        # because asyncio is single-threaded and we're back on the loop).
        store, retriever = app._ingest_result
        del app._ingest_result
        app.state.store = store
        app.state.retriever = retriever
        app.state.total_chunks = store.total_vectors
        if filename not in app.state.processed_docs:
            app.state.processed_docs.append(filename)

        jobs[job_id].update(
            status="done",
            result={"page_count": page_count, "chunk_count": chunk_count,
                    "total_chunks_in_index": total},
        )
        logger.info(
            "Job %s done — '%s': %d pages, %d chunks, %d total.",
            job_id, filename, page_count, chunk_count, total,
        )

    except Exception as exc:
        logger.exception("Ingestion failed for job '%s' ('%s').", job_id, filename)
        jobs[job_id].update(status="error", error=str(exc))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadAccepted, status_code=202)
async def upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadAccepted:
    """
    Accept a PDF, validate it, save it, and start background ingestion.

    Returns immediately with a job_id.  Poll GET /upload/status/{job_id}
    to check when ingestion completes.
    """
    filename = (file.filename or "upload.pdf").strip()
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are accepted. Received extension: '{suffix or 'none'}'.",
        )

    content = await file.read()
    size_mb = len(content) / (1024 ** 2)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File exceeds the {settings.max_upload_mb} MB limit. "
                f"Received: {size_mb:.1f} MB."
            ),
        )

    logger.info("Upload received: '%s' (%.2f MB).", filename, size_mb)

    raw_dir = settings.raw_data_path
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / filename
    try:
        dest.write_bytes(content)
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not save uploaded file: {exc}"
        ) from exc

    job_id = uuid.uuid4().hex[:12]
    request.app.state.jobs[job_id] = {
        "status": "processing",
        "filename": filename,
        "result": None,
        "error": None,
    }

    background_tasks.add_task(_ingest_background, request.app, job_id, filename, dest)

    return UploadAccepted(
        job_id=job_id,
        filename=filename,
        message=(
            f"'{filename}' accepted for ingestion. "
            f"Poll GET /upload/status/{job_id} to check progress."
        ),
    )


@router.get("/upload/status/{job_id}", response_model=JobStatus)
async def upload_status(job_id: str, request: Request) -> JobStatus:
    """Check the status of a background ingestion job."""
    jobs: dict = request.app.state.jobs
    if job_id not in jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. Job history resets on server restart.",
        )
    job = jobs[job_id]
    result_dict = job.get("result")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        filename=job["filename"],
        result=UploadResult(**result_dict) if result_dict else None,
        error=job.get("error"),
    )
