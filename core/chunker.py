"""
Arabic-aware semantic chunker.

Strategy
--------
1. Split text into paragraphs (double newlines -- strongest boundary).
2. Within each paragraph, split further at Arabic and Latin sentence endings.
3. Greedily accumulate sentences into chunks until max_tokens is reached.
4. On chunk boundary, seed the next chunk with overlap_tokens worth of
   trailing sentences from the previous chunk for context continuity.
5. Single sentences that exceed max_tokens are split at word boundaries
   as a last resort.

Each Chunk carries source (filename) and page_number so every retrieved
result can be traced back to its origin.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import NamedTuple


# ── Token estimation ──────────────────────────────────────────────────────────
#
# Arabic BPE tokens average ~3.5 UTF-8 characters due to rich morphology and
# common punctuation.  This constant intentionally errs slightly low so chunks
# stay within most embedding-model context windows even after normalisation.

_CHARS_PER_TOKEN: float = 3.5


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single text chunk with full provenance metadata.

    Attributes
    ----------
    text:           Cleaned chunk body ready for embedding.
    source:         Filename or document identifier.
    page_number:    Page where this chunk BEGINS (1-based).
    chunk_index:    0-based position in the document.
    chunk_id:       Deterministic 16-char hex ID (sha256 of source + index).
    char_count:     Character length of text (computed).
    token_estimate: Estimated BPE token count (computed).
    """

    text: str
    source: str
    page_number: int
    chunk_index: int
    chunk_id: str
    char_count: int = field(init=False)
    token_estimate: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)
        self.token_estimate = max(1, math.ceil(self.char_count / _CHARS_PER_TOKEN))

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate,
        }

    def __repr__(self) -> str:
        preview = self.text[:70].replace("\n", " ")
        return (
            f"<Chunk id={self.chunk_id} page={self.page_number} "
            f"tokens≈{self.token_estimate} '{preview}...'>"
        )


class _Sentence(NamedTuple):
    """Internal: a sentence fragment tagged with the page it came from."""
    text: str
    page_number: int


# ── Chunker ───────────────────────────────────────────────────────────────────

class ArabicChunker:
    """
    Semantic chunker for Arabic (and mixed Arabic/Latin) text.

    Parameters
    ----------
    max_tokens:
        Target maximum tokens per chunk.  Chunks may slightly exceed this
        only when a single sentence cannot be split further.
    overlap_tokens:
        Tokens of trailing context to carry into the next chunk.
        Must be < max_tokens.
    min_chunk_tokens:
        Chunks below this threshold are silently discarded (noise guard).
    """

    # ── Compiled patterns ─────────────────────────────────────────────────────

    # Paragraph separator: two or more consecutive newlines
    _PARA_RE = re.compile(r"\n{2,}")

    # Sentence boundary: immediately AFTER one of these end-of-sentence markers,
    # followed by whitespace.
    #
    # Arabic punctuation covered:
    #   ۔  U+06D4  Arabic Full Stop
    #   ؟  U+061F  Arabic Question Mark
    #   ؛  U+061B  Arabic Semicolon  (weak boundary — included for robustness)
    #   !  U+0021  Exclamation Mark
    #   .  U+002E  Full Stop  (common even in Arabic-script documents)
    #   ?  U+003F  Question Mark
    _SENT_RE = re.compile(
        r"(?<=[.۔?؟!])\s+"
        r"|(?<=؛)\s+",
        re.UNICODE,
    )

    def __init__(
        self,
        max_tokens: int = 400,
        overlap_tokens: int = 60,
        min_chunk_tokens: int = 20,
    ) -> None:
        if overlap_tokens >= max_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be less than "
                f"max_tokens ({max_tokens})."
            )
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_pages(self, pages: list, source: str) -> list[Chunk]:
        """
        Chunk a list of PageContent objects (output of ArabicPDFReader.read()).

        Parameters
        ----------
        pages:
            Each item must expose .text (str) and .page_number (int).
        source:
            Document identifier stored in every Chunk (e.g. the filename).
        """
        sentences: list[_Sentence] = []
        for page in pages:
            for text in self._split_to_sentences(page.text):
                sentences.append(_Sentence(text=text, page_number=page.page_number))
        return self._build_chunks(sentences, source)

    def chunk_text(
        self,
        text: str,
        source: str,
        page_number: int = 1,
    ) -> list[Chunk]:
        """
        Chunk a raw string, assigning all chunks the given page number.
        Useful for plain-text or non-PDF sources.
        """
        sentences = [
            _Sentence(text=s, page_number=page_number)
            for s in self._split_to_sentences(text)
        ]
        return self._build_chunks(sentences, source)

    # ── Sentence splitting ────────────────────────────────────────────────────

    def _split_to_sentences(self, text: str) -> list[str]:
        """
        Two-level split: paragraphs → sentences.

        A paragraph that already fits within max_tokens is kept whole to
        preserve tight semantic cohesion.  Only oversized paragraphs are
        broken further at sentence boundaries.
        """
        output: list[str] = []

        for paragraph in self._PARA_RE.split(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if self._tokens(paragraph) <= self.max_tokens:
                output.append(paragraph)
                continue

            # Oversized paragraph: split at sentence boundaries
            for raw in self._SENT_RE.split(paragraph):
                sent = raw.strip()
                if not sent:
                    continue

                if self._tokens(sent) > self.max_tokens:
                    # Single sentence still too long: word-boundary split
                    output.extend(self._split_by_words(sent))
                else:
                    output.append(sent)

        return output

    def _split_by_words(self, text: str) -> list[str]:
        """
        Last-resort split: walk words and flush whenever max_tokens is reached.
        Arabic words are whitespace-delimited after normalisation.
        """
        parts: list[str] = []
        bucket: list[str] = []
        bucket_tokens = 0

        for word in text.split():
            w_tokens = self._tokens(word)
            if bucket_tokens + w_tokens > self.max_tokens and bucket:
                parts.append(" ".join(bucket))
                bucket = []
                bucket_tokens = 0
            bucket.append(word)
            bucket_tokens += w_tokens

        if bucket:
            parts.append(" ".join(bucket))

        return parts

    # ── Chunk assembly ────────────────────────────────────────────────────────

    def _build_chunks(
        self,
        sentences: list[_Sentence],
        source: str,
    ) -> list[Chunk]:
        """
        Greedy accumulation loop.

        Accumulates sentences into a running buffer.  When the next sentence
        would push the buffer past max_tokens:
          1. Flush the buffer as a chunk.
          2. Rewind to overlap_tokens worth of trailing sentences.
          3. Continue from the current sentence.
        """
        if not sentences:
            return []

        chunks: list[Chunk] = []
        buffer: list[_Sentence] = []
        buffer_tokens = 0
        chunk_index = 0
        i = 0

        while i < len(sentences):
            sent = sentences[i]
            s_tokens = self._tokens(sent.text)

            if buffer_tokens + s_tokens <= self.max_tokens:
                buffer.append(sent)
                buffer_tokens += s_tokens
                i += 1
            else:
                if buffer:
                    chunk = self._make_chunk(buffer, source, chunk_index)
                    if chunk is not None:
                        chunks.append(chunk)
                        chunk_index += 1

                    # Seed next chunk with overlap from the tail of the buffer
                    buffer = self._overlap_tail(buffer)
                    buffer_tokens = sum(self._tokens(s.text) for s in buffer)
                else:
                    # Degenerate: sentence alone exceeds budget; force-accept it
                    buffer.append(sent)
                    i += 1

        # Flush any remaining sentences
        if buffer:
            chunk = self._make_chunk(buffer, source, chunk_index)
            if chunk is not None:
                chunks.append(chunk)

        return chunks

    def _overlap_tail(self, buffer: list[_Sentence]) -> list[_Sentence]:
        """
        Walk backwards through buffer collecting sentences until we have
        accumulated at least overlap_tokens characters-worth of context.
        Returns the collected sentences in original (forward) order.
        """
        tail: list[_Sentence] = []
        accumulated = 0

        for sent in reversed(buffer):
            tail.insert(0, sent)
            accumulated += self._tokens(sent.text)
            if accumulated >= self.overlap_tokens:
                break

        return tail

    # ── Chunk construction ────────────────────────────────────────────────────

    def _make_chunk(
        self,
        buffer: list[_Sentence],
        source: str,
        chunk_index: int,
    ) -> Chunk | None:
        """
        Build a Chunk from a list of _Sentence items.
        Returns None if the resulting text is too short (noise guard).
        """
        text = "\n".join(s.text for s in buffer).strip()
        if not text:
            return None
        if self._tokens(text) < self.min_chunk_tokens:
            return None

        return Chunk(
            text=text,
            source=source,
            page_number=buffer[0].page_number,
            chunk_index=chunk_index,
            chunk_id=_chunk_id(source, chunk_index),
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _tokens(text: str) -> int:
        return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))


# ── Module-level helpers ──────────────────────────────────────────────────────

def _chunk_id(source: str, chunk_index: int) -> str:
    """sha256-based 16-char hex ID — deterministic and collision-resistant."""
    payload = f"{source}::{chunk_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def chunk_document(
    pages: list,
    source: str,
    *,
    max_tokens: int = 400,
    overlap_tokens: int = 60,
    min_chunk_tokens: int = 20,
) -> list[Chunk]:
    """
    One-call convenience wrapper for the common case.

    Example
    -------
    >>> from utils.pdf_reader import read_arabic_pdf
    >>> from core.chunker import chunk_document
    >>> pages = read_arabic_pdf("تقرير.pdf")
    >>> chunks = chunk_document(pages, source="تقرير.pdf")
    >>> print(f"{len(chunks)} chunks produced")
    """
    return ArabicChunker(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_chunk_tokens=min_chunk_tokens,
    ).chunk_pages(pages, source)
