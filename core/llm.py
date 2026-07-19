"""
LLM module — GPT-4o answer generation over retrieved context.

Responsibilities
----------------
1. Format retrieved chunks into a numbered context block.
2. Send a grounded, citation-aware system prompt + user question to GPT-4o.
3. Return a GeneratedAnswer containing the response text, token usage,
   and the source metadata of every passage that was provided.

Language behaviour
------------------
The system prompt instructs GPT-4o to detect the language of the question
and reply entirely in that language — so an Arabic question gets an Arabic
answer, an English question gets an English answer, and a bilingual question
gets a bilingual answer.  No language detection logic is needed in this module.

Citation format
---------------
Every context passage is labelled with its filename and page number.
The model is instructed to cite inline as [filename, p. N] after every
factual claim.  This format is language-neutral so it works identically
in Arabic and English responses.

Hallucination guard
-------------------
The system prompt mandates a specific "cannot answer" phrase when the
context is insufficient.  GeneratedAnswer.answered is set to False when
either of those phrases is detected in the response, so callers (e.g. the
Streamlit frontend) can surface a clear "not found" state without parsing
free text.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()
logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_MODEL = "gpt-4o"
_DEFAULT_MAX_TOKENS = 1500
_DEFAULT_TEMPERATURE = 0.1

# Exact phrases the model is told to output when it cannot answer.
# Both are checked (lowercased) in _detect_no_answer().
_NO_ANSWER_AR = "لا يمكنني الإجابة على هذا السؤال بناءً على المستندات المقدمة"
_NO_ANSWER_EN = "i cannot answer this question based on the provided documents"

_CONTEXT_DIVIDER = "─" * 48


# ── System prompt ─────────────────────────────────────────────────────────────
#
# Written in English so the model understands it reliably, but rule 1
# forces the reply into the question's language.

_SYSTEM_PROMPT = """\
You are a precise document assistant. You answer questions using only the \
numbered context passages supplied by the user. Each passage carries a source \
filename and page number.

Follow every rule below without exception.

1. LANGUAGE
   Detect the language of the user's question. Write your entire answer in \
that same language. Arabic question → Arabic answer. English question → \
English answer. Do not mix languages in your reply.

2. GROUNDING
   Use only information that appears in the context passages. Do not draw on \
outside knowledge, make inferences beyond what the text states, or combine \
context with prior training data.

3. INLINE CITATIONS
   After every sentence or factual claim, add a citation in this exact format:
     [filename, p. PAGE_NUMBER]
   Example: "The budget was approved in Q3 [annual_report.pdf, p. 12]."
   Keep the citation in Latin script and square brackets regardless of the \
reply language. Cite every passage that contributed to the claim; if two \
passages support the same point, cite both.

4. CANNOT ANSWER
   If the context does not contain sufficient information to answer the \
question, output one of the following sentences and nothing else:
   — Arabic:  "لا يمكنني الإجابة على هذا السؤال بناءً على المستندات المقدمة."
   — English: "I cannot answer this question based on the provided documents."
   Choose the sentence that matches the question language. Do not guess, \
speculate, or add information from memory.

5. BREVITY
   Be direct and concise. Do not open with filler phrases such as \
"Based on the provided context…" or "According to the documents…". \
Start your answer with the substantive content immediately.\
"""


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Source:
    """Provenance record for one context passage shown to the model."""
    source: str
    page_number: int
    chunk_id: str


@dataclass
class GeneratedAnswer:
    """
    Full result of one LLM call.

    Attributes
    ----------
    answer:        The model's response text (may be a "cannot answer" phrase).
    question:      The original user question, unchanged.
    model:         Model ID that produced the answer.
    sources:       One Source record per chunk provided as context, in the
                   order they appeared in the prompt.
    chunks_used:   len(sources) — convenience accessor.
    input_tokens:  Prompt token count reported by the API.
    output_tokens: Completion token count reported by the API.
    answered:      False when the model returned a "cannot answer" phrase;
                   True otherwise.  Reliable because the system prompt mandates
                   an exact, detectable string for that case.
    """

    answer: str
    question: str
    model: str
    sources: list[Source]
    input_tokens: int
    output_tokens: int
    answered: bool
    chunks_used: int = field(init=False)

    def __post_init__(self) -> None:
        self.chunks_used = len(self.sources)

    def __repr__(self) -> str:
        status = "answered" if self.answered else "no-answer"
        preview = self.answer[:80].replace("\n", " ")
        return (
            f"<GeneratedAnswer [{status}] tokens={self.input_tokens}+{self.output_tokens} "
            f"sources={self.chunks_used} '{preview}...'>"
        )


# ── LLM client ────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Thin wrapper around the OpenAI Chat Completions API.

    Parameters
    ----------
    model:
        OpenAI model ID.  Defaults to gpt-4o.
    api_key:
        OpenAI API key.  Falls back to the OPENAI_API_KEY environment variable.
    max_tokens:
        Maximum tokens in the completion.
    temperature:
        Sampling temperature.  0.1 keeps answers factual and consistent.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
    ) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "API key not found. Set OPENAI_API_KEY in your environment "
                "or pass api_key= to LLMClient()."
            )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=base_url or None,   # None = default OpenAI endpoint
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        chunks: list,
        extra_context: str | None = None,
    ) -> GeneratedAnswer:
        """
        Generate a grounded answer for a question given retrieved chunks.

        Parameters
        ----------
        question:
            The user's question (Arabic, English, or mixed).
        chunks:
            List of RetrievalResult objects from HybridRetriever.search().
            Each must expose .text, .source, .page_number, .chunk_id.
        extra_context:
            Optional plain-text addendum appended after the numbered passages.
            Use for document-level metadata (e.g. title, author, date) that
            should inform the answer but is not a direct passage.

        Returns
        -------
        GeneratedAnswer with .answer, .sources, token counts, and .answered flag.
        """
        if not chunks:
            logger.warning("answer() called with no chunks — returning no-answer.")
            return self._empty_answer(question)

        context = self._build_context(chunks, extra_context)
        messages = self._build_messages(question, context)

        logger.debug(
            "Sending request: model=%s chunks=%d max_tokens=%d",
            self.model,
            len(chunks),
            self.max_tokens,
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        answer_text = response.choices[0].message.content or ""
        usage = response.usage

        return GeneratedAnswer(
            answer=answer_text,
            question=question,
            model=response.model,
            sources=[
                Source(
                    source=c.source,
                    page_number=c.page_number,
                    chunk_id=c.chunk_id,
                )
                for c in chunks
            ],
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            answered=not _detect_no_answer(answer_text),
        )

    def stream_answer(
        self,
        question: str,
        chunks: list,
        extra_context: str | None = None,
    ) -> Iterator[str]:
        """
        Stream the answer token by token.

        Yields each text delta as it arrives from the API.  Useful for
        Streamlit's st.write_stream() to display a typing effect.

        Example
        -------
        >>> for token in client.stream_answer(question, chunks):
        ...     print(token, end="", flush=True)
        """
        if not chunks:
            yield _NO_ANSWER_EN
            return

        context = self._build_context(chunks, extra_context)
        messages = self._build_messages(question, context)

        with self._client.chat.completions.stream(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta

    # ── Context & message construction ────────────────────────────────────────

    def _build_context(
        self,
        chunks: list,
        extra_context: str | None,
    ) -> str:
        """
        Format retrieved chunks into a numbered passage block.

        Output format for each passage:
            [N] Source: filename.pdf | Page: 7
            ────────────────────────────────────────────────
            <chunk text>
        """
        blocks: list[str] = []

        for i, chunk in enumerate(chunks, start=1):
            header = f"[{i}] Source: {chunk.source} | Page: {chunk.page_number}"
            blocks.append(f"{header}\n{_CONTEXT_DIVIDER}\n{chunk.text.strip()}")

        context = "\n\n".join(blocks)

        if extra_context:
            context += f"\n\n{_CONTEXT_DIVIDER}\nAdditional context:\n{extra_context.strip()}"

        return context

    def _build_messages(
        self,
        question: str,
        context: str,
    ) -> list[ChatCompletionMessageParam]:
        """
        Assemble the messages list for the Chat Completions API.

        Structure:
            system  — grounding rules and citation format
            user    — context passages + the question, clearly separated
        """
        user_content = (
            f"CONTEXT PASSAGES:\n\n{context}"
            f"\n\n{'─' * 48}\n\nQUESTION:\n{question}"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _empty_answer(self, question: str) -> GeneratedAnswer:
        """Return a well-formed GeneratedAnswer when no chunks are available."""
        text = (
            "لا يمكنني الإجابة على هذا السؤال بناءً على المستندات المقدمة."
            if _looks_arabic(question)
            else "I cannot answer this question based on the provided documents."
        )
        return GeneratedAnswer(
            answer=text,
            question=question,
            model=self.model,
            sources=[],
            input_tokens=0,
            output_tokens=0,
            answered=False,
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

def _detect_no_answer(text: str) -> bool:
    """
    Return True when the model's response is a "cannot answer" phrase.

    Checks are case-insensitive and strip punctuation so minor formatting
    variations (trailing full stop, etc.) do not cause false negatives.
    """
    normalised = text.lower().strip().rstrip(".")
    return _NO_ANSWER_AR in normalised or _NO_ANSWER_EN in normalised


def _looks_arabic(text: str) -> bool:
    """Heuristic: True when the majority of word characters are Arabic script."""
    import re
    arabic = len(re.findall(r"[؀-ۿ]", text))
    total = len(re.findall(r"\w", text))
    return total > 0 and (arabic / total) >= 0.5


# ── Convenience function ──────────────────────────────────────────────────────

def answer_question(
    question: str,
    chunks: list,
    *,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> GeneratedAnswer:
    """
    One-call convenience wrapper — instantiate LLMClient and get an answer.

    Example
    -------
    >>> from core.retriever import HybridRetriever
    >>> from core.llm import answer_question
    >>>
    >>> results  = retriever.search("ما هي نتائج التقرير؟", top_k_final=5)
    >>> response = answer_question("ما هي نتائج التقرير؟", results)
    >>>
    >>> print(response.answer)
    >>> print(f"Answered: {response.answered}  |  Tokens: {response.input_tokens}+{response.output_tokens}")
    """
    client = LLMClient(
        model=model or os.getenv("OPENAI_MODEL", _DEFAULT_MODEL),
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return client.answer(question, chunks)
