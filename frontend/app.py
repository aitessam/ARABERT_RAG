from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
UPLOAD_TIMEOUT = 300
QUERY_TIMEOUT = 60

st.set_page_config(
    page_title="Arabic RAG",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── Base ───────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .block-container { padding-top: 1.4rem; max-width: 860px; }

    /* ── Arabic text ────────────────────────────────── */
    .rtl {
        direction: rtl;
        text-align: right;
        font-family: "Segoe UI", "Noto Sans Arabic", Arial, sans-serif;
        line-height: 2.1;
    }

    /* ── Chat answer body ───────────────────────────── */
    .chat-answer {
        line-height: 1.85;
        font-size: 1.01rem;
        white-space: pre-wrap;
        margin: 2px 0;
        color: #1a1a2e;
    }
    .chat-answer.rtl { line-height: 2.1; }

    /* ── Source cards ───────────────────────────────── */
    .src-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-left: 3px solid #4b6cb7;
        border-radius: 0 6px 6px 0;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.92rem;
        line-height: 1.8;
        white-space: pre-wrap;
        color: #374151;
    }
    .src-card.rtl {
        direction: rtl;
        text-align: right;
        border-left: none;
        border-right: 3px solid #4b6cb7;
        border-radius: 6px 0 0 6px;
        line-height: 2.0;
    }
    .src-meta {
        font-size: 0.76rem;
        color: #6b7280;
        margin-bottom: 6px;
        font-weight: 500;
        letter-spacing: 0.01em;
    }

    /* ── Score badges ───────────────────────────────── */
    .badge {
        display: inline-block;
        background: #f0f4ff;
        color: #3b52a3;
        border: 1px solid #dbe4ff;
        border-radius: 3px;
        padding: 1px 6px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 4px;
        letter-spacing: 0.02em;
    }

    /* ── Status indicator ───────────────────────────── */
    .status-dot { font-size: 0.75rem; margin-right: 4px; }
    .dot-green  { color: #16a34a; }
    .dot-red    { color: #dc2626; }
    .dot-amber  { color: #d97706; }

    /* ── Empty state ────────────────────────────────── */
    .empty-state {
        text-align: center;
        padding: 52px 0;
        color: #9ca3af;
        font-size: 0.95rem;
        line-height: 1.9;
    }

    /* ── Page title ─────────────────────────────────── */
    h1 { font-size: 1.5rem !important; font-weight: 600 !important; color: #111827 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_arabic(text: str) -> bool:
    arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
    alpha  = sum(1 for c in text if c.isalpha())
    return alpha > 0 and arabic / alpha >= 0.30


def _safe_html(text: str) -> str:
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def _fetch_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _upload_pdf(file, status_text=None) -> tuple[bool, dict | str]:
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            timeout=30,
        )
        if r.status_code != 202:
            return False, f"[{r.status_code}] {r.json().get('detail', r.text)}"

        job_id = r.json()["job_id"]

        for attempt in range(300):
            time.sleep(2)
            sr = requests.get(f"{API_BASE}/upload/status/{job_id}", timeout=10)
            sr.raise_for_status()
            job = sr.json()
            if job["status"] == "done":
                return True, job
            if job["status"] == "error":
                return False, job.get("error", "Ingestion failed.")
            if status_text is not None:
                status_text.caption(
                    f"Processing `{file.name}`... ({(attempt + 1) * 2}s elapsed)"
                )

        return False, "Timed out waiting for ingestion to complete."

    except requests.exceptions.ConnectionError:
        return False, "Cannot reach the backend. Is the server running on port 8000?"
    except Exception as exc:
        return False, str(exc)


def _run_query(
    question: str, top_k_dense: int, top_k_sparse: int, top_k_final: int
) -> dict | str:
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={
                "question": question,
                "top_k_dense": top_k_dense,
                "top_k_sparse": top_k_sparse,
                "top_k_final": top_k_final,
            },
            timeout=QUERY_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json()
        return f"[{r.status_code}] {r.json().get('detail', r.text)}"
    except requests.exceptions.ConnectionError:
        return "Cannot reach the backend. Is the server running on port 8000?"
    except requests.exceptions.Timeout:
        return "Query timed out. Try reducing the number of candidates."
    except Exception as exc:
        return str(exc)


def _render_assistant_message(msg: dict) -> None:
    """Render one assistant turn inside a st.chat_message context."""
    answer        = msg.get("answer", "")
    answered      = msg.get("answered", False)
    retrieval_only = msg.get("retrieval_only", False)
    sources       = msg.get("sources", [])
    model         = msg.get("model", "")
    in_tok        = msg.get("input_tokens", 0)
    out_tok       = msg.get("output_tokens", 0)

    if retrieval_only:
        st.info(
            "Running in retrieval-only mode. No LLM is configured — "
            "the passages below are the top-ranked chunks for your query. "
            "Set OPENAI_API_KEY (or a compatible provider) to enable generated answers."
        )
    elif answered:
        cls = "chat-answer rtl" if _is_arabic(answer) else "chat-answer"
        st.markdown(
            f'<div class="{cls}">{_safe_html(answer)}</div>',
            unsafe_allow_html=True,
        )
        if model and model != "none":
            st.caption(f"Model: `{model}` — {in_tok} in / {out_tok} out tokens")
    else:
        st.warning(
            "The indexed documents do not contain sufficient information "
            "to answer this question."
        )
        if answer:
            cls = "chat-answer rtl" if _is_arabic(answer) else "chat-answer"
            st.markdown(
                f'<div class="{cls}">{_safe_html(answer)}</div>',
                unsafe_allow_html=True,
            )

    if sources:
        with st.expander(f"Retrieved passages ({len(sources)})", expanded=False):
            for i, src in enumerate(sources, start=1):
                doc    = src.get("source", "unknown")
                page   = src.get("page_number", "?")
                text   = src.get("text", "")
                rerank = src.get("rerank_score", 0.0)
                d_sc   = src.get("dense_score")
                s_sc   = src.get("sparse_score")

                badges = f'<span class="badge">Rerank {rerank:.3f}</span>'
                if d_sc is not None:
                    badges += f'<span class="badge">Dense {d_sc:.3f}</span>'
                if s_sc is not None:
                    badges += f'<span class="badge">BM25 {s_sc:.1f}</span>'

                card_cls = "src-card rtl" if _is_arabic(text) else "src-card"
                st.markdown(
                    f'<div class="src-meta">[{i}]  {doc}  —  Page {page}'
                    f'&emsp;{badges}</div>'
                    f'<div class="{card_cls}">{_safe_html(text)}</div>',
                    unsafe_allow_html=True,
                )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Arabic RAG")
    st.caption("Hybrid retrieval — Arabic & English")
    st.divider()

    health = _fetch_health()

    with st.expander("System Status", expanded=True):
        if health is None:
            st.markdown(
                '<span class="status-dot dot-red">●</span> Backend offline',
                unsafe_allow_html=True,
            )
            st.caption(f"Expected at `{API_BASE}`")
            st.caption("Start with:\n`uvicorn backend.app.main:app --reload`")
        else:
            retriever_ok = health.get("retriever_ready", False)
            llm_ok       = health.get("llm_ready", False)

            if retriever_ok and llm_ok:
                label, dot = "Operational", "dot-green"
            elif retriever_ok or llm_ok:
                label, dot = "Partial", "dot-amber"
            else:
                label, dot = "Not ready", "dot-red"

            st.markdown(
                f'<span class="status-dot {dot}">●</span> {label}',
                unsafe_allow_html=True,
            )

            col_a, col_b = st.columns(2)
            col_a.metric("Chunks", health.get("total_chunks", 0))
            col_b.metric("Docs", len(health.get("processed_documents", [])))

            docs = health.get("processed_documents", [])
            if docs:
                st.caption("Indexed documents")
                for doc in docs:
                    st.markdown(f"- `{doc}`")
            else:
                st.caption("No documents indexed.")

            st.caption(f"LLM: `{health.get('openai_model', 'N/A')}`")
            embed = health.get("embedding_model", "N/A").split("/")[-1]
            st.caption(f"Embeddings: `{embed}`")

            if not llm_ok:
                st.warning("No LLM key configured. Queries return retrieved passages only.")

        if st.button("Refresh status", use_container_width=True):
            st.rerun()

    with st.expander("Retrieval Settings", expanded=False):
        top_k_dense  = st.slider("Dense candidates",  1, 50, 10)
        top_k_sparse = st.slider("Sparse candidates", 1, 50, 10)
        top_k_final  = st.slider("Final results",     1, 20, 5)

    if st.session_state.messages:
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Arabic Document Q&A")
st.caption("Hybrid retrieval-augmented generation — Arabic and English queries supported.")

tab_upload, tab_chat = st.tabs(["Upload Documents", "Chat"])


# ── Upload tab ────────────────────────────────────────────────────────────────

with tab_upload:
    st.subheader("Upload PDF Documents")
    st.caption(
        "Each file is extracted, chunked with Arabic-aware sentence boundaries, "
        "embedded with a multilingual model, and indexed in Qdrant. "
        "Re-uploading a file replaces its existing index entries."
    )

    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Maximum 50 MB per file. Image-only (scanned) PDFs are not supported.",
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.markdown(
            '<div class="empty-state">'
            "No files selected.<br>"
            "Use the file picker above to choose one or more PDF documents."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            size_mb = len(f.getvalue()) / (1024 ** 2)
            st.markdown(f"- `{f.name}` &nbsp;({size_mb:.1f} MB)", unsafe_allow_html=True)

        st.markdown("")
        if st.button(
            f"Process {len(uploaded_files)} file(s)",
            type="primary",
            use_container_width=True,
        ):
            progress    = st.progress(0, text="Starting...")
            status_text = st.empty()
            results: list[tuple[str, bool, str]] = []

            for idx, file in enumerate(uploaded_files):
                progress.progress(
                    idx / len(uploaded_files),
                    text=f"Uploading `{file.name}` ({idx + 1}/{len(uploaded_files)})...",
                )
                ok, payload = _upload_pdf(file, status_text=status_text)
                if ok and isinstance(payload, dict):
                    res = payload.get("result") or {}
                    msg = (
                        f"{res.get('page_count', '?')} pages, "
                        f"{res.get('chunk_count', '?')} chunks indexed, "
                        f"{res.get('total_chunks_in_index', '?')} total in index"
                    )
                else:
                    msg = payload if isinstance(payload, str) else str(payload)
                results.append((file.name, ok, msg))

            status_text.empty()
            progress.progress(1.0, text="Complete.")
            progress.empty()

            successes = [(n, m) for n, ok, m in results if ok]
            failures  = [(n, m) for n, ok, m in results if not ok]

            if successes:
                st.success(f"{len(successes)} file(s) processed successfully.")
                for name, msg in successes:
                    st.markdown(f"- **{name}** — {msg}")
            if failures:
                st.error(f"{len(failures)} file(s) failed.")
                for name, msg in failures:
                    st.markdown(f"- **{name}** — {msg}")
            if successes:
                st.info(
                    "Index updated. Switch to the Chat tab to query your documents."
                )
                st.rerun()


# ── Chat tab ──────────────────────────────────────────────────────────────────

with tab_chat:
    if not st.session_state.messages:
        st.markdown(
            '<div class="empty-state">'
            "No conversation yet.<br>"
            "Upload documents and type a question below.<br>"
            "يدعم النظام الاستعلامات باللغتين العربية والإنجليزية."
            "</div>",
            unsafe_allow_html=True,
        )

    for msg in st.session_state.messages:
        with st.chat_message("user"):
            q = msg["question"]
            if _is_arabic(q):
                st.markdown(
                    f'<div class="rtl">{_safe_html(q)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(q)

        with st.chat_message("assistant"):
            _render_assistant_message(msg)

    question = st.chat_input(
        "Type your question in Arabic or English...",
        disabled=(health is None),
    )

    if question:
        with st.chat_message("user"):
            if _is_arabic(question):
                st.markdown(
                    f'<div class="rtl">{_safe_html(question)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving..."):
                result = _run_query(
                    question,
                    int(top_k_dense),
                    int(top_k_sparse),
                    int(top_k_final),
                )

            if isinstance(result, str):
                st.error(result)
            else:
                entry = {
                    "question":       question,
                    "answer":         result.get("answer", ""),
                    "answered":       result.get("answered", False),
                    "retrieval_only": result.get("retrieval_only", False),
                    "sources":        result.get("sources", []),
                    "model":          result.get("model", ""),
                    "input_tokens":   result.get("input_tokens", 0),
                    "output_tokens":  result.get("output_tokens", 0),
                }
                _render_assistant_message(entry)
                st.session_state.messages.append(entry)
