"""
Embeddings module — dense vector store backed by FAISS.

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
Vectors are L2-normalised before being added to a FAISS IndexFlatIP
(inner-product index).  On unit vectors, inner product equals cosine
similarity, so scores returned by search() are in [−1, 1] with 1.0
being a perfect match.

Persistence
-----------
save(directory) writes two files:
  <directory>/index.faiss   — the raw FAISS index
  <directory>/store.json    — model name + per-vector metadata (source,
                              page, chunk text, …)

load(directory) reads both files and reconstructs a ready-to-query store.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_MODEL = "intfloat/multilingual-e5-large"
_EMBEDDING_DIM = 1024
_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "

_INDEX_FILE = "index.faiss"
_META_FILE = "store.json"


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single result returned by EmbeddingsStore.search()."""

    chunk_id: str
    source: str
    page_number: int
    chunk_index: int
    text: str
    score: float          # cosine similarity in [−1, 1]; higher is better

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
    Dense vector store: embed chunks → store in FAISS → search by query.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Defaults to intfloat/multilingual-e5-large.
    device:
        PyTorch device string ('cpu', 'cuda', 'mps').  None = auto-detect.
    batch_size:
        Number of texts encoded per forward pass.  Reduce if OOM on GPU.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)

        # IndexFlatIP: exact inner-product search.  No training required.
        # Correct for cosine similarity when vectors are L2-normalised.
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(_EMBEDDING_DIM)

        # Parallel metadata list — entry i corresponds to FAISS vector id i.
        self._meta: list[dict] = []

    # ── Core operations ───────────────────────────────────────────────────────

    def embed_chunks(self, chunks: list, show_progress: bool = True) -> None:
        """
        Encode a list of Chunk objects and add them to the FAISS index.

        Chunks are accepted as a plain list so this module does not hard-import
        core.chunker (avoids circular dependencies when both are imported from a
        third module).  Each item must expose: .text, .chunk_id, .source,
        .page_number, .chunk_index.

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
        faiss.normalize_L2(vectors)          # in-place; makes IP == cosine sim
        self._index.add(vectors)

        for chunk in chunks:
            self._meta.append({
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            })

        logger.info(
            "Added %d vectors — index total: %d", len(chunks), self._index.ntotal
        )

    def embed_query(self, text: str) -> np.ndarray:
        """
        Encode a single query string.

        Returns a normalised (1, 1024) float32 array ready for FAISS search.
        The "query: " prefix is applied automatically.
        """
        vec = self._encode([_QUERY_PREFIX + text], show_progress=False)
        faiss.normalize_L2(vec)
        return vec

    def remove_by_source(self, source: str) -> int:
        """
        Remove all vectors whose metadata source matches *source*.

        FAISS IndexFlatIP does not support in-place deletion, so surviving
        vectors are reconstructed and the index is rebuilt.  For RAG-scale
        corpora this takes milliseconds.

        Returns the number of vectors removed.
        """
        keep = [i for i, m in enumerate(self._meta) if m["source"] != source]
        removed = len(self._meta) - len(keep)
        if removed == 0:
            return 0

        # Reconstruct surviving vectors BEFORE resetting the index.
        surviving = (
            np.stack([self._index.reconstruct(i) for i in keep]) if keep else None
        )
        self._index = faiss.IndexFlatIP(_EMBEDDING_DIM)
        self._meta = [self._meta[i] for i in keep]
        if surviving is not None:
            self._index.add(surviving)

        logger.info(
            "Removed %d vectors for source '%s' — %d remaining.",
            removed, source, len(self._meta),
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
        if self._index.ntotal == 0:
            logger.warning("search() called on an empty index.")
            return []

        k = min(top_k, self._index.ntotal)
        query_vec = self.embed_query(query)
        scores, indices = self._index.search(query_vec, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:            # FAISS uses −1 for unfilled slots
                continue
            meta = self._meta[idx]
            results.append(
                SearchResult(
                    chunk_id=meta["chunk_id"],
                    source=meta["source"],
                    page_number=meta["page_number"],
                    chunk_index=meta["chunk_index"],
                    text=meta["text"],
                    score=float(score),
                )
            )

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """
        Persist the FAISS index and metadata to disk.

        Creates the directory if it does not exist.  Safe to call after
        incremental embed_chunks() calls to checkpoint progress.

        Files written
        -------------
        <directory>/index.faiss   — raw FAISS binary
        <directory>/store.json    — model config + per-vector metadata
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / _INDEX_FILE))

        store_doc = {
            "model_name": self.model_name,
            "embedding_dim": _EMBEDDING_DIM,
            "total_vectors": self._index.ntotal,
            "chunks": self._meta,
        }
        (path / _META_FILE).write_text(
            json.dumps(store_doc, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "Store saved to '%s'  (%d vectors).", path, self._index.ntotal
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        device: str | None = None,
        batch_size: int = 32,
    ) -> "EmbeddingsStore":
        """
        Load a previously saved store from disk.

        Reconstructs both the FAISS index and the metadata list, then loads
        the same embedding model that was used when the store was created.

        Parameters
        ----------
        directory:
            Path to the directory written by save().
        device:
            Override compute device for the embedding model.
        batch_size:
            Batch size for subsequent embed_chunks() / embed_query() calls.

        Raises
        ------
        FileNotFoundError
            If index.faiss or store.json are missing.
        """
        path = Path(directory)
        index_path = path / _INDEX_FILE
        meta_path = path / _META_FILE

        for p in (index_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(f"Store file not found: {p}")

        store_doc = json.loads(meta_path.read_text(encoding="utf-8"))
        model_name = store_doc.get("model_name", _DEFAULT_MODEL)

        instance = cls(model_name=model_name, device=device, batch_size=batch_size)
        instance._index = faiss.read_index(str(index_path))
        instance._meta = store_doc["chunks"]

        logger.info(
            "Store loaded from '%s'  (%d vectors, model=%s).",
            path,
            instance._index.ntotal,
            model_name,
        )
        return instance

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal

    @property
    def is_empty(self) -> bool:
        return self._index.ntotal == 0

    # ── Internal helpers ──────────────────────────────────────────────────────

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
            normalize_embeddings=False,   # we normalise manually via FAISS
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


# ── Convenience function ──────────────────────────────────────────────────────

def build_store(
    chunks: list,
    save_dir: str | Path,
    *,
    model_name: str = _DEFAULT_MODEL,
    device: str | None = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> EmbeddingsStore:
    """
    Embed a list of Chunk objects, save the store, and return it.

    This is the one-call path for first-time ingestion.

    Example
    -------
    >>> from utils.pdf_reader import read_arabic_pdf
    >>> from core.chunker import chunk_document
    >>> from core.embeddings import build_store
    >>>
    >>> pages  = read_arabic_pdf("تقرير.pdf")
    >>> chunks = chunk_document(pages, source="تقرير.pdf")
    >>> store  = build_store(chunks, save_dir="data/vector_store")
    >>> results = store.search("ما هي التوصيات الرئيسية؟", top_k=5)
    """
    store = EmbeddingsStore(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    store.embed_chunks(chunks, show_progress=show_progress)
    store.save(save_dir)
    return store
