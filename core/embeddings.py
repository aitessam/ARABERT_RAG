"""
Embeddings module — dense vector store backed by Qdrant.

Model
-----
intfloat/multilingual-e5-large (1 024-dim)

This model was trained with mandatory input prefixes:
  "passage: " — for document chunks added to the index
  "query: "   — for user queries at search time

Omitting the prefixes silently degrades recall by a significant margin;
this module applies them automatically so callers never need to think
about it.

Vector similarity
-----------------
Vectors are stored in a Qdrant collection configured with Distance.COSINE.
Qdrant normalises vectors internally; raw (unnormalised) encoder output is
accepted — no manual L2-normalisation step is required.

Persistence
-----------
Qdrant persists data to its own storage directory automatically.
EmbeddingsStore.save() is a no-op; load() creates a fresh connection and
returns the live collection as-is.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_MODEL = "intfloat/multilingual-e5-large"
_EMBEDDING_DIM = 1024
_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "
_UPSERT_BATCH = 100          # points per Qdrant upsert call


def _chunk_id_to_uint(hex_id: str) -> int:
    """Map a 16-char hex chunk ID to a uint64 Qdrant point ID."""
    return struct.unpack(">Q", bytes.fromhex(hex_id))[0]


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single result returned by EmbeddingsStore.search()."""

    chunk_id: str
    source: str
    page_number: int
    chunk_index: int
    text: str
    score: float          # cosine similarity in [0, 1]; higher is better

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"<SearchResult score={self.score:.4f} "
            f"source='{self.source}' page={self.page_number} "
            f"'{preview}...'>"
        )


# ── Store ─────────────────────────────────────────────────────────────────────

class EmbeddingsStore:
    """
    Dense vector store: embed chunks → upsert to Qdrant → search by query.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Defaults to intfloat/multilingual-e5-large.
    url:
        Qdrant host.  Defaults to "localhost".
    port:
        Qdrant REST port.  Defaults to 6333.
    collection:
        Qdrant collection name.
    device:
        PyTorch device string ('cpu', 'cuda', 'mps').  None = auto-detect.
    batch_size:
        Number of texts encoded per forward pass.  Reduce if OOM on GPU.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        url: str = "localhost",
        port: int = 6333,
        collection: str = "arabic_rag",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._collection = collection

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)

        logger.info("Connecting to Qdrant at %s:%s", url, port)
        self._client = QdrantClient(url=url, port=port)
        self._ensure_collection()

        # Local count cache — avoids a round-trip on every is_empty check.
        self._local_count: int = self._client.count(self._collection).count
        logger.info(
            "Qdrant store ready — collection '%s', %d existing vectors.",
            collection,
            self._local_count,
        )

    # ── Core operations ───────────────────────────────────────────────────────

    def embed_chunks(self, chunks: list, show_progress: bool = True) -> None:
        """
        Encode a list of Chunk objects and upsert them into Qdrant.

        Each item must expose: .text, .chunk_id, .source, .page_number,
        .chunk_index.  Chunk objects from core.chunker satisfy this.

        Parameters
        ----------
        chunks:
            Output of ArabicChunker.chunk_pages() or chunk_document().
        show_progress:
            Display a tqdm bar when encoding more than one batch.
        """
        if not chunks:
            logger.warning("embed_chunks called with an empty list — nothing to do.")
            return

        texts = [_PASSAGE_PREFIX + chunk.text for chunk in chunks]
        vectors = self._encode(texts, show_progress=show_progress)

        points = [
            PointStruct(
                id=_chunk_id_to_uint(chunk.chunk_id),
                vector=vectors[i].tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        for start in range(0, len(points), _UPSERT_BATCH):
            self._client.upsert(
                collection_name=self._collection,
                points=points[start : start + _UPSERT_BATCH],
            )

        self._local_count += len(chunks)
        logger.info(
            "Upserted %d vectors — collection total: %d",
            len(chunks),
            self._local_count,
        )

    def embed_query(self, text: str) -> np.ndarray:
        """
        Encode a single query string.

        Returns a (1, 1024) float32 array.  The "query: " prefix is applied
        automatically.  Qdrant handles cosine normalisation internally.
        """
        return self._encode([_QUERY_PREFIX + text], show_progress=False)

    def remove_by_source(self, source: str) -> int:
        """
        Delete all points whose payload source field matches *source*.

        Uses a single Qdrant filter-delete — no index rebuild required.

        Returns the number of points removed.
        """
        filt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])

        removed = self._client.count(
            collection_name=self._collection,
            count_filter=filt,
            exact=True,
        ).count
        if removed == 0:
            return 0

        self._client.delete(
            collection_name=self._collection,
            points_selector=filt,
        )
        self._local_count -= removed
        logger.info(
            "Deleted %d vectors for source '%s' — %d remaining.",
            removed,
            source,
            self._local_count,
        )
        return removed

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Find the top_k most similar chunks for a query string.

        Parameters
        ----------
        query:
            Natural-language question or keyword string (Arabic or mixed).
        top_k:
            Number of results to return.

        Returns
        -------
        List of SearchResult objects sorted by descending cosine similarity.
        """
        if self.is_empty:
            logger.warning("search() called on an empty collection.")
            return []

        query_vec = self.embed_query(query)
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vec[0].tolist(),
            limit=top_k,
            with_payload=True,
        )

        return [
            SearchResult(
                chunk_id=hit.payload["chunk_id"],
                source=hit.payload["source"],
                page_number=hit.payload["page_number"],
                chunk_index=hit.payload["chunk_index"],
                text=hit.payload["text"],
                score=hit.score,
            )
            for hit in response.points
        ]

    def all_chunks(self) -> list[dict]:
        """
        Scroll all points and return their payloads.

        Used by HybridRetriever.load() to rebuild the BM25 corpus from
        existing Qdrant data on startup.
        """
        chunks: list[dict] = []
        offset = None
        while True:
            batch, offset = self._client.scroll(
                collection_name=self._collection,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                limit=1000,
            )
            chunks.extend(pt.payload for pt in batch)
            if offset is None:
                break
        return chunks

    # ── Persistence (no-ops — Qdrant persists automatically) ─────────────────

    def save(self, directory: str | Path) -> None:
        """No-op. Qdrant persists data automatically."""

    @classmethod
    def load(
        cls,
        directory: str | Path | None = None,
        model_name: str = _DEFAULT_MODEL,
        url: str = "localhost",
        port: int = 6333,
        collection: str = "arabic_rag",
        device: str | None = None,
        batch_size: int = 32,
    ) -> "EmbeddingsStore":
        """
        Connect to Qdrant and return a ready store.

        The *directory* argument is accepted for API compatibility but ignored —
        all data lives in Qdrant, not on disk.
        """
        return cls(
            model_name=model_name,
            url=url,
            port=port,
            collection=collection,
            device=device,
            batch_size=batch_size,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        """Number of vectors currently in the collection (local cache)."""
        return self._local_count

    @property
    def is_empty(self) -> bool:
        return self._local_count == 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=_EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s'.", self._collection)

    def _encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of strings and return a (N, 1024) float32 numpy array.

        Progress bar is shown only when encoding more than one batch.
        """
        multi_batch = len(texts) > self.batch_size
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=(show_progress and multi_batch),
            normalize_embeddings=False,   # Qdrant COSINE handles normalisation
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
