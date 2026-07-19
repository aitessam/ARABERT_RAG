"""
POST /query — retrieve relevant chunks and optionally generate a grounded answer.

Flow
----
1.  Validate that the retriever is initialised (documents have been uploaded).
2.  Run hybrid retrieval: FAISS dense + BM25 sparse + cross-encoder rerank.
3.  If an LLM client is configured, pass chunks to GPT-4o and return the answer.
4.  If no LLM is configured (no OPENAI_API_KEY), return the retrieved chunks
    directly with retrieval_only=True so the caller can still inspect results.

Error codes
-----------
400  Question is empty, too short, or too long (Pydantic validation).
404  No documents have been uploaded yet.
500  Retrieval or generation error.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.core.config import settings
from core.llm import LLMClient
from core.retriever import HybridRetriever

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="User question in Arabic, English, or mixed.",
    )
    top_k_dense: int = Field(default=10, ge=1, le=50)
    top_k_sparse: int = Field(default=10, ge=1, le=50)
    top_k_final: int = Field(default=5, ge=1, le=20)

    model_config = {"json_schema_extra": {
        "example": {
            "question": "ما هي التوصيات الرئيسية في التقرير؟",
            "top_k_dense": 10,
            "top_k_sparse": 10,
            "top_k_final": 5,
        }
    }}


class SourceChunk(BaseModel):
    chunk_id: str
    source: str
    page_number: int
    chunk_index: int
    text: str
    rerank_score: float
    dense_score: float | None
    sparse_score: float | None


class QueryResponse(BaseModel):
    question: str
    answer: str
    answered: bool
    sources: list[SourceChunk]
    model: str
    input_tokens: int
    output_tokens: int
    retrieval_only: bool = Field(
        default=False,
        description=(
            "True when no LLM is configured. Retrieved chunks are returned "
            "directly without a generated answer."
        ),
    )


# ── Dependency ────────────────────────────────────────────────────────────────

def _get_retriever(request: Request) -> HybridRetriever:
    retriever = request.app.state.retriever
    if retriever is None or not retriever.is_ready:
        raise HTTPException(
            status_code=404,
            detail=(
                "No documents have been indexed yet. "
                "Upload at least one PDF via POST /upload before querying."
            ),
        )
    return retriever


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, status_code=200)
async def query(
    body: QueryRequest,
    request: Request,
    retriever: HybridRetriever = Depends(_get_retriever),
) -> QueryResponse:
    """
    Answer a question using hybrid retrieval and (optionally) GPT-4o generation.

    When OPENAI_API_KEY is not set the endpoint still works: it returns the
    top-ranked retrieved chunks with retrieval_only=True and an empty answer.
    """
    logger.info("Query: %r", body.question[:100])

    # ── Retrieve ──────────────────────────────────────────────────────────────
    try:
        results = retriever.search(
            query=body.question,
            top_k_dense=body.top_k_dense,
            top_k_sparse=body.top_k_sparse,
            top_k_final=body.top_k_final,
        )
    except Exception as exc:
        logger.exception("Retrieval failed.")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}") from exc

    sources = [
        SourceChunk(
            chunk_id=r.chunk_id,
            source=r.source,
            page_number=r.page_number,
            chunk_index=r.chunk_index,
            text=r.text,
            rerank_score=r.rerank_score,
            dense_score=r.dense_score,
            sparse_score=r.sparse_score,
        )
        for r in results
    ]

    # ── Generate (optional) ───────────────────────────────────────────────────
    llm: LLMClient | None = request.app.state.llm
    if llm is None:
        logger.info("LLM unavailable — returning %d chunks (retrieval-only).", len(results))
        return QueryResponse(
            question=body.question,
            answer="",
            answered=False,
            sources=sources,
            model="none",
            input_tokens=0,
            output_tokens=0,
            retrieval_only=True,
        )

    try:
        generated = llm.answer(body.question, results)
    except Exception as exc:
        logger.exception("LLM generation failed.")
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}") from exc

    logger.info(
        "Query complete — answered=%s tokens=%d+%d sources=%d.",
        generated.answered,
        generated.input_tokens,
        generated.output_tokens,
        len(results),
    )

    return QueryResponse(
        question=body.question,
        answer=generated.answer,
        answered=generated.answered,
        sources=sources,
        model=generated.model,
        input_tokens=generated.input_tokens,
        output_tokens=generated.output_tokens,
        retrieval_only=False,
    )
