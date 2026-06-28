"""
Arabic PDF reader built on PyMuPDF (fitz).

Responsibilities:
  - Extract text per page in correct logical Unicode order (RTL-aware)
  - Remove repeated headers/footers detected across pages
  - Strip page numbers (Latin and Arabic-Indic digit forms)
  - Normalise Arabic text: Alef variants, Tatweel, Tashkeel
  - Return one PageContent object per non-empty page
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class PageContent:
    page_number: int          # 1-based
    text: str
    word_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.word_count = len(self.text.split())

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"<PageContent page={self.page_number} words={self.word_count} preview='{preview}...'>"


# ── Reader ────────────────────────────────────────────────────────────────────

class ArabicPDFReader:
    """
    Extracts clean Arabic text from PDF files using PyMuPDF.

    Parameters
    ----------
    header_margin:
        Fraction of page height (from top) treated as a header zone.
    footer_margin:
        Fraction of page height (from bottom) treated as a footer zone.
    repeat_threshold:
        A text block appearing on this many pages is classified as a
        repeated header/footer and suppressed.
    min_block_chars:
        Blocks shorter than this (after strip) are discarded as noise.
    normalise_arabic:
        When True, applies Alef normalisation, Tatweel and Tashkeel removal.
    """

    # Arabic-Indic digit range: ٠ (U+0660) – ٩ (U+0669)
    _ARABIC_INDIC = r"٠-٩"

    _PAGE_NUMBER_RE: list[re.Pattern[str]] = [
        re.compile(rf"^\s*[\d{_ARABIC_INDIC}]+\s*$"),
        re.compile(rf"^\s*[-–]\s*[\d{_ARABIC_INDIC}]+\s*[-–]\s*$"),
        re.compile(rf"^\s*صفحة\s*[\d{_ARABIC_INDIC}]+\s*$"),
        re.compile(r"^\s*page\s*\d+\s*$", re.IGNORECASE),
    ]

    # Tashkeel (harakat + shadda + sukun + superscript alef)
    _TASHKEEL_RE = re.compile(r"[ً-ٰٟ]")

    def __init__(
        self,
        header_margin: float = 0.08,
        footer_margin: float = 0.08,
        repeat_threshold: int = 3,
        min_block_chars: int = 3,
        normalise_arabic: bool = True,
    ) -> None:
        self.header_margin = header_margin
        self.footer_margin = footer_margin
        self.repeat_threshold = repeat_threshold
        self.min_block_chars = min_block_chars
        self.normalise_arabic = normalise_arabic

    # ── Public API ────────────────────────────────────────────────────────────

    def read(self, pdf_path: str | Path) -> list[PageContent]:
        """
        Open a PDF and return a list of PageContent objects, one per
        non-empty page.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

        doc = fitz.open(str(path))
        try:
            repeated = self._detect_repeated_blocks(doc)
            return [
                page_content
                for page in doc
                if (page_content := self._process_page(page, repeated)).text.strip()
            ]
        finally:
            doc.close()

    # ── Per-page extraction ───────────────────────────────────────────────────

    def _process_page(
        self, page: fitz.Page, repeated_texts: set[str]
    ) -> PageContent:
        page_h = page.rect.height
        header_limit = page_h * self.header_margin
        footer_limit = page_h * (1.0 - self.footer_margin)

        # Flags: preserve whitespace + clip to page bounds
        flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP

        # get_text("blocks") → list of (x0,y0,x1,y1,text,block_no,block_type)
        raw_blocks = page.get_text("blocks", flags=flags)

        # Keep only text blocks (block_type == 0); discard image blocks
        text_blocks = [b for b in raw_blocks if b[6] == 0]

        # Sort: top-to-bottom first; within same band, right-to-left (RTL)
        # Rounding y0 to 10 px groups lines at the same visual height.
        text_blocks.sort(key=lambda b: (round(b[1] / 10) * 10, -b[0]))

        kept: list[str] = []
        for x0, y0, x1, y1, text, *_ in text_blocks:
            text = text.strip()

            # ── Positional noise filters ──────────────────────────────────
            if y0 < header_limit or y1 > footer_limit:
                continue

            # ── Content noise filters ─────────────────────────────────────
            if len(text) < self.min_block_chars:
                continue
            if self._is_page_number(text):
                continue
            if text in repeated_texts:
                continue

            kept.append(text)

        raw_text = "\n".join(kept)
        clean = self._clean_text(raw_text) if self.normalise_arabic else raw_text
        return PageContent(page_number=page.number + 1, text=clean)

    # ── Repeated-block detection ──────────────────────────────────────────────

    def _detect_repeated_blocks(self, doc: fitz.Document) -> set[str]:
        """
        Scan header/footer zones across all pages.  Any block text appearing
        on >= repeat_threshold pages (or >=1/3 of total pages, whichever is
        smaller) is flagged as a repeated element.
        """
        num_pages = len(doc)
        counter: Counter[str] = Counter()
        flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP

        for page in doc:
            page_h = page.rect.height
            header_limit = page_h * self.header_margin
            footer_limit = page_h * (1.0 - self.footer_margin)

            for block in page.get_text("blocks", flags=flags):
                x0, y0, x1, y1, text, block_no, block_type = block
                if block_type != 0:
                    continue
                # Only examine the margin zones
                if y0 < header_limit or y1 > footer_limit:
                    t = text.strip()
                    if t:
                        counter[t] += 1

        min_pages = max(2, min(self.repeat_threshold, num_pages // 3))
        return {txt for txt, count in counter.items() if count >= min_pages}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_page_number(self, text: str) -> bool:
        return any(pat.match(text) for pat in self._PAGE_NUMBER_RE)

    def _clean_text(self, text: str) -> str:
        # Normalise Alef variants → bare Alef (ا)
        text = re.sub(r"[أإآٱ]", "ا", text)
        # Remove Tatweel / Kashida (ـ U+0640)
        text = text.replace("ـ", "")
        # Remove Tashkeel (harakat, shadda, sukun, superscript alef)
        text = self._TASHKEEL_RE.sub("", text)
        # Normalise Teh Marbuta variants (optional but common in OCR output)
        text = text.replace("ة", "ه")  # ة → ه (soft normalisation)
        # Collapse multiple spaces on the same line (preserve newlines)
        text = re.sub(r"[^\S\n]+", " ", text)
        # Collapse 3+ consecutive newlines to a double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ── Convenience function ──────────────────────────────────────────────────────

def read_arabic_pdf(
    pdf_path: str | Path,
    *,
    header_margin: float = 0.08,
    footer_margin: float = 0.08,
    repeat_threshold: int = 3,
    normalise_arabic: bool = True,
) -> list[PageContent]:
    """
    Shortcut: instantiate ArabicPDFReader and return pages in one call.

    Example
    -------
    >>> pages = read_arabic_pdf("document.pdf")
    >>> for p in pages:
    ...     print(p.page_number, p.word_count)
    """
    reader = ArabicPDFReader(
        header_margin=header_margin,
        footer_margin=footer_margin,
        repeat_threshold=repeat_threshold,
        normalise_arabic=normalise_arabic,
    )
    return reader.read(pdf_path)
