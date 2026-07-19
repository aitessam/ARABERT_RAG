"""
Hybrid retriever: dense (FAISS) + sparse (BM25) + cross-encoder reranking.

Pipeline
--------
                          ┌─────────────────────────┐
  query ─────────────────▶│  Dense retrieval (FAISS) │──▶ top_k_dense hits
                          └─────────────────────────┘
                                                          │
                          ┌─────────────────────────┐    │  merge &
  query ─────────────────▶│  Sparse retrieval (BM25) │──▶│  deduplicate
                          └─────────────────────────┘    │  by chunk_id
                                                          │
                          ┌─────────────────────────┐    │
  (query, candidate) ─────▶│  Cross-encoder reranker  │◀──┘
                          └─────────────────────────┘
                                      │
                              top_k_final results

Dense retrieval
    Uses EmbeddingsStore (multilingual-e5-large, cosine similarity via FAISS).

Sparse retrieval
    BM25Okapi over Arabic-normalised, stopword-filtered token lists.
    Handles Arabic-script and mixed Arabic/Latin documents.

Cross-encoder reranking
    cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 — multilingual model trained
    on mMARCO (26 languages including Arabic).  Scores (query, passage) pairs
    jointly.  Scores are raw logits (not bounded); higher is better.
    The cross-encoder re-scores every candidate from scratch, so the merge
    step only needs to collect unique candidates, not rank them.

Persistence
-----------
save(directory) writes:
  <directory>/retriever.json   — reranker model name

The BM25 corpus is NOT written separately; load() reconstructs it from
<directory>/store.json (written by EmbeddingsStore.save()) so there is
no data duplication.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from core.embeddings import EmbeddingsStore, SearchResult

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_RETRIEVER_FILE = "retriever.json"


# ── Arabic stopwords ──────────────────────────────────────────────────────────
# Covers the most frequent function words across Modern Standard Arabic.
# Removing them reduces BM25 index noise without hurting recall on content words.

_ARABIC_STOPWORDS: frozenset[str] = frozenset({
    # Prepositions & conjunctions
    "في", "من", "إلى", "على", "عن", "مع", "عند", "بعد", "قبل", "خلال",
    "بين", "حول", "حتى", "منذ", "رغم", "دون", "بدون", "لدى", "تجاه",
    "و", "أو", "ثم", "لكن", "بل", "لأن", "إذا", "حين", "كلما", "بينما",
    "كي", "لكي", "كما", "بما", "مما", "عما",
    # Pronouns
    "هو", "هي", "هم", "هن", "هما", "أنا", "نحن", "أنت", "أنتم", "أنتن",
    "هذا", "هذه", "ذلك", "تلك", "هؤلاء", "أولئك",
    "الذي", "التي", "الذين", "اللواتي", "اللائي",
    # Common verbs (auxiliary/copula)
    "كان", "كانت", "كانوا", "يكون", "تكون", "يكونوا",
    "ليس", "ليست",
    # Particles
    "قد", "لا", "لم", "لن", "سوف", "سي", "ما", "أن", "إن", "إنّ",
    "أي", "كل", "بعض", "جميع", "معظم",
    # Article (standalone — after tokenisation the ال prefix stays attached)
    "ال",
    # Discourse markers
    "هناك", "هنا", "الآن", "اليوم", "أيضا", "أيضاً", "فقط",
    "لذلك", "لذا", "بذلك", "وبذلك", "وبالتالي", "بالتالي",
})


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    Final output of HybridRetriever.search().

    Attributes
    ----------
    chunk_id:      Deterministic hex ID from core.chunker.
    source:        Filename or document identifier.
    page_number:   Page where the chunk begins (1-based).
    chunk_index:   0-based position in the source document.
    text:          Chunk body.
    dense_score:   Cosine similarity from FAISS (None if not in dense hits).
    sparse_score:  Raw BM25 score (None if not in sparse hits; not comparable
                   to cosine similarity — use only for diagnostics).
    rerank_score:  Cross-encoder logit.  The final ranking key: higher = better.
    """

    chunk_id: str
    source: str
    page_number: int
    chunk_index: int
    text: str
    dense_score: float | None
    sparse_score: float | None
    rerank_score: float

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"<RetrievalResult rerank={self.rerank_score:.3f} "
            f"dense={self.dense_score} sparse={self.sparse_score:.2f if self.sparse_score else None} "
            f"page={self.page_number} '{preview}...'>"
        )


class _SparseHit(NamedTuple):
    """Internal: one BM25 result before merging."""
    chunk_id: str
    source: str
    page_number: int
    chunk_index: int
    text: str
    score: float


class _Candidate(NamedTuple):
    """Internal: merged candidate ready for cross-encoder scoring."""
    chunk_id: str
    source: str
    page_number: int
    chunk_index: int
    text: str
    dense_score: float | None
    sparse_score: float | None


# ── Retriever ─────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Three-stage retriever: BM25 + FAISS → cross-encoder rerank.

    Parameters
    ----------
    store:
        A loaded (or freshly built) EmbeddingsStore for dense retrieval.
    reranker_model:
        HuggingFace model ID for the CrossEncoder.
    device:
        Torch device for the cross-encoder ('cpu', 'cuda', 'mps').
        None = auto-detect.
    """

    def __init__(
        self,
        store: EmbeddingsStore,
        reranker_model: str = _DEFAULT_RERANKER,
        device: str | None = None,
    ) -> None:
        self._store = store
        self.reranker_model = reranker_model

        logger.info("Loading cross-encoder: %s", reranker_model)
        self._reranker = CrossEncoder(reranker_model, device=device)

        # BM25 state — rebuilt from corpus whenever index_chunks() is called.
        self._corpus: list[dict] = []
        self._bm25: BM25Okapi | None = None

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_chunks(self, chunks: list) -> None:
        """
        Add chunks to the BM25 index.

        Dense indexing must be done separately via EmbeddingsStore.embed_chunks().
        This method only updates the sparse index.

        Parameters
        ----------
        chunks:
            List of Chunk objects (must expose .chunk_id, .source,
            .page_number, .chunk_index, .text).
        """
        for chunk in chunks:
            self._corpus.append({
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            })
        self._rebuild_bm25()
        logger.info("BM25 index updated — %d documents.", len(self._corpus))

    def remove_by_source(self, source: str) -> int:
        """
        Remove all BM25 corpus entries whose source matches *source*.

        Returns the number of entries removed.
        """
        before = len(self._corpus)
        self._corpus = [d for d in self._corpus if d["source"] != source]
        removed = before - len(self._corpus)
        if removed:
            self._rebuild_bm25()
            logger.info(
                "Removed %d BM25 entries for source '%s' — %d remaining.",
                removed, source, len(self._corpus),
            )
        return removed

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k_dense: int = 10,
        top_k_sparse: int = 10,
        top_k_final: int = 5,
    ) -> list[RetrievalResult]:
        """
        Run the full hybrid retrieval pipeline and return reranked results.

        Parameters
        ----------
        query:
            Natural-language question or keyword string (Arabic or mixed).
        top_k_dense:
            Number of candidates to retrieve from FAISS.
        top_k_sparse:
            Number of candidates to retrieve from BM25.
        top_k_final:
            Number of results to return after cross-encoder reranking.

        Returns
        -------
        List of RetrievalResult, sorted by rerank_score descending.
        """
        dense_hits = self._dense_search(query, top_k_dense)
        sparse_hits = self._sparse_search(query, top_k_sparse)
        candidates = self._merge(dense_hits, sparse_hits)

        if not candidates:
            logger.warning("No candidates found for query: %r", query[:80])
            return []

        reranked = self._rerank(query, candidates)
        return reranked[:top_k_final]

    # ── Internal pipeline stages ──────────────────────────────────────────────

    def _dense_search(self, query: str, top_k: int) -> list[SearchResult]:
        """FAISS cosine-similarity search via EmbeddingsStore."""
        if self._store.is_empty:
            logger.warning("Dense index is empty — skipping dense retrieval.")
            return []
        return self._store.search(query, top_k=top_k)

    def _sparse_search(self, query: str, top_k: int) -> list[_SparseHit]:
        """BM25 retrieval over Arabic-normalised tokens."""
        if self._bm25 is None or not self._corpus:
            logger.warning("BM25 index is empty — skipping sparse retrieval.")
            return []

        query_tokens = _tokenize_arabic(query)
        if not query_tokens:
            return []

        scores: np.ndarray = self._bm25.get_scores(query_tokens)

        # Take top_k non-zero scoring documents
        k = min(top_k, len(self._corpus))
        top_indices = np.argsort(scores)[::-1][:k]

        hits: list[_SparseHit] = []
        for idx in top_indices:
            if scores[idx] <= 0.0:
                continue
            doc = self._corpus[idx]
            hits.append(
                _SparseHit(
                    chunk_id=doc["chunk_id"],
                    source=doc["source"],
                    page_number=doc["page_number"],
                    chunk_index=doc["chunk_index"],
                    text=doc["text"],
                    score=float(scores[idx]),
                )
            )
        return hits

    def _merge(
        self,
        dense_hits: list[SearchResult],
        sparse_hits: list[_SparseHit],
    ) -> list[_Candidate]:
        """
        Deduplicate hits from both retrievers by chunk_id.

        When a chunk appears in both result sets, both scores are preserved
        in the candidate so they can be inspected in the final RetrievalResult.
        """
        seen: dict[str, _Candidate] = {}

        for hit in dense_hits:
            seen[hit.chunk_id] = _Candidate(
                chunk_id=hit.chunk_id,
                source=hit.source,
                page_number=hit.page_number,
                chunk_index=hit.chunk_index,
                text=hit.text,
                dense_score=hit.score,
                sparse_score=None,
            )

        for hit in sparse_hits:
            if hit.chunk_id in seen:
                existing = seen[hit.chunk_id]
                seen[hit.chunk_id] = existing._replace(sparse_score=hit.score)
            else:
                seen[hit.chunk_id] = _Candidate(
                    chunk_id=hit.chunk_id,
                    source=hit.source,
                    page_number=hit.page_number,
                    chunk_index=hit.chunk_index,
                    text=hit.text,
                    dense_score=None,
                    sparse_score=hit.score,
                )

        candidates = list(seen.values())
        logger.debug(
            "Merge: %d dense + %d sparse → %d unique candidates.",
            len(dense_hits),
            len(sparse_hits),
            len(candidates),
        )
        return candidates

    def _rerank(
        self, query: str, candidates: list[_Candidate]
    ) -> list[RetrievalResult]:
        """
        Score every candidate jointly with the cross-encoder and sort descending.

        The cross-encoder reads query and passage together (not independently),
        so it captures relevance signals that bi-encoder cosine similarity misses.
        Scores are raw logits — only their relative order matters.
        """
        pairs = [(query, c.text) for c in candidates]
        raw_scores: np.ndarray = self._reranker.predict(
            pairs, show_progress_bar=False
        )

        results: list[RetrievalResult] = []
        for candidate, score in zip(candidates, raw_scores):
            results.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    source=candidate.source,
                    page_number=candidate.page_number,
                    chunk_index=candidate.chunk_index,
                    text=candidate.text,
                    dense_score=candidate.dense_score,
                    sparse_score=candidate.sparse_score,
                    rerank_score=float(score),
                )
            )

        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results

    # ── BM25 internals ────────────────────────────────────────────────────────

    def _rebuild_bm25(self) -> None:
        """
        (Re)build BM25Okapi from the current corpus.

        BM25 construction is O(N * avg_tokens) and typically takes
        milliseconds for RAG-scale corpora (< 100 k chunks).
        """
        if not self._corpus:
            self._bm25 = None
            return
        tokenized = [_tokenize_arabic(doc["text"]) for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """
        Save retriever configuration to disk.

        Only the reranker model name is written — the BM25 corpus is
        reconstructed from store.json (written by EmbeddingsStore.save())
        on load, avoiding data duplication.

        Parameters
        ----------
        directory:
            Same directory used for EmbeddingsStore.save().
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        config = {"reranker_model": self.reranker_model}
        (path / _RETRIEVER_FILE).write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Retriever config saved to '%s'.", path / _RETRIEVER_FILE)

    @classmethod
    def load(
        cls,
        directory: str | Path | None = None,
        store: EmbeddingsStore | None = None,
        device: str | None = None,
    ) -> "HybridRetriever":
        """
        Connect to Qdrant-backed store and rebuild the BM25 corpus.

        Reads retriever.json for the reranker model name if *directory* is
        given and the file exists; otherwise falls back to the default.
        BM25 corpus is populated by scrolling all payloads from Qdrant.

        Parameters
        ----------
        directory:
            Optional directory to read retriever.json from.
        store:
            A connected EmbeddingsStore instance.
        device:
            Torch device for the cross-encoder.
        """
        reranker_model = _DEFAULT_RERANKER
        if directory is not None:
            config_path = Path(directory) / _RETRIEVER_FILE
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                reranker_model = config.get("reranker_model", _DEFAULT_RERANKER)

        instance = cls(store=store, reranker_model=reranker_model, device=device)

        # Rebuild BM25 corpus by scrolling all existing Qdrant payloads.
        instance._corpus = store.all_chunks()
        instance._rebuild_bm25()

        logger.info(
            "Retriever loaded — BM25 corpus: %d docs, reranker: %s.",
            len(instance._corpus),
            reranker_model,
        )
        return instance

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def corpus_size(self) -> int:
        """Number of documents in the BM25 index."""
        return len(self._corpus)

    @property
    def is_ready(self) -> bool:
        """True when both dense and sparse indexes have at least one document."""
        return not self._store.is_empty and self._bm25 is not None


# ── Arabic tokeniser ──────────────────────────────────────────────────────────

def _tokenize_arabic(text: str) -> list[str]:
    """
    Lightweight Arabic tokeniser for BM25 input.

    Steps
    -----
    1. Normalise Alef variants (أ إ آ ٱ) → bare Alef (ا).
    2. Strip Tatweel (ـ) and Tashkeel (harakat U+064B–U+065F, U+0670).
    3. Remove all non-word characters (punctuation, symbols); preserve spaces.
    4. Lowercase Latin characters (mixed-script documents).
    5. Split on whitespace.
    6. Discard tokens in the Arabic stopword list or shorter than 2 characters.
    """
    # Normalise Alef variants
    text = re.sub(r"[أإآٱ]", "ا", text)
    # Remove Tatweel
    text = text.replace("ـ", "")
    # Remove Tashkeel
    text = re.sub(r"[ً-ٰٟ]", "", text)
    # Remove punctuation — keep Unicode letters, digits, spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # Lowercase (normalise any Latin tokens in mixed-language docs)
    text = text.lower()

    return [
        token
        for token in text.split()
        if len(token) >= 2 and token not in _ARABIC_STOPWORDS
    ]


# ── Convenience function ──────────────────────────────────────────────────────

def build_retriever(
    chunks: list,
    save_dir: str | Path,
    *,
    reranker_model: str = _DEFAULT_RERANKER,
    embedding_model: str = "intfloat/multilingual-e5-large",  # noqa: E501
    device: str | None = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> "HybridRetriever":
    """
    Embed chunks, build both indexes, persist everything, return retriever.

    This is the single entry point for first-time ingestion.

    Example
    -------
    >>> from utils.pdf_reader import read_arabic_pdf
    >>> from core.chunker import chunk_document
    >>> from core.retriever import build_retriever
    >>>
    >>> pages    = read_arabic_pdf("تقرير.pdf")
    >>> chunks   = chunk_document(pages, source="تقرير.pdf")
    >>> retriever = build_retriever(chunks, save_dir="data/vector_store")
    >>>
    >>> results = retriever.search("ما هي التوصيات الرئيسية؟", top_k_final=5)
    >>> for r in results:
    ...     print(r.rerank_score, r.source, r.page_number)
    """
    from core.embeddings import EmbeddingsStore

    store = EmbeddingsStore(
        model_name=embedding_model,
        device=device,
        batch_size=batch_size,
    )
    store.embed_chunks(chunks, show_progress=show_progress)
    store.save(save_dir)

    retriever = HybridRetriever(
        store=store,
        reranker_model=reranker_model,
        device=device,
    )
    retriever.index_chunks(chunks)
    retriever.save(save_dir)

    return retriever
