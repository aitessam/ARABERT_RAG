from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
UPLOAD_TIMEOUT = 300
QUERY_TIMEOUT = 60

st.set_page_config(
    page_title="Arabic RAG | نظام الاسترجاع",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .rtl {
        direction: rtl;
        text-align: right;
        font-family: "Segoe UI", Arial, sans-serif;
        line-height: 2.0;
        font-size: 1.05rem;
    }
    .answer-card {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        border-radius: 6px;
        padding: 18px 22px;
        margin: 10px 0 4px 0;
        line-height: 1.9;
        font-size: 1.05rem;
        white-space: pre-wrap;
    }
    .answer-card.rtl {
        border-left: none;
        border-right: 4px solid #1f77b4;
    }
    .chunk-text {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 5px;
        padding: 12px 16px;
        margin-top: 8px;
        font-size: 0.97rem;
        line-height: 1.85;
        white-space: pre-wrap;
    }
    .badge {
        display: inline-block;
        background: #eef4fb;
        color: #1f77b4;
        border-radius: 4px;
        padding: 2px 9px;
        font-size: 0.80rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .dot-green { color: #22c55e; }
    .dot-red   { color: #ef4444; }
    .dot-amber { color: #f59e0b; }
    .block-container { padding-top: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "query_result" not in st.session_state:
    st.session_state.query_result = None
if "query_error" not in st.session_state:
    st.session_state.query_error = None


def _is_arabic(text: str) -> bool:
    arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
    alpha = sum(1 for c in text if c.isalpha())
    return alpha > 0 and arabic / alpha >= 0.30


def _safe_html(text: str) -> str:
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def _render_text(text: str) -> None:
    escaped = _safe_html(text)
    cls = "chunk-text rtl" if _is_arabic(text) else "chunk-text"
    st.markdown(f'<div class="{cls}">{escaped}</div>', unsafe_allow_html=True)


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
            detail = r.json().get("detail", r.text)
            return False, f"[{r.status_code}] {detail}"

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
                status_text.caption(f"Processing `{file.name}`... ({(attempt + 1) * 2}s)")

        return False, "Timed out waiting for ingestion to complete."

    except requests.exceptions.ConnectionError:
        return False, "Cannot reach the backend. Is the server running on port 8000?"
    except Exception as exc:
        return False, str(exc)


def _run_query(question: str, top_k_dense: int, top_k_sparse: int, top_k_final: int) -> dict | str:
    try:
        response = requests.post(
            f"{API_BASE}/query",
            json={
                "question": question,
                "top_k_dense": top_k_dense,
                "top_k_sparse": top_k_sparse,
                "top_k_final": top_k_final,
            },
            timeout=QUERY_TIMEOUT,
        )
        if response.status_code == 200:
            return response.json()
        detail = response.json().get("detail", response.text)
        return f"[{response.status_code}] {detail}"
    except requests.exceptions.ConnectionError:
        return "Cannot reach the backend. Is the server running on port 8000?"
    except requests.exceptions.Timeout:
        return "Query timed out. Try reducing the number of candidates."
    except Exception as exc:
        return str(exc)


with st.sidebar:
    st.markdown("### System Status")

    health = _fetch_health()

    if health is None:
        st.markdown('<span class="dot-red">●</span> Backend offline', unsafe_allow_html=True)
        st.caption(f"Expected at `{API_BASE}`")
        st.caption("Start the server with:  \n`uvicorn backend.app.main:app --reload`")
    else:
        retriever_ok = health.get("retriever_ready", False)
        llm_ok = health.get("llm_ready", False)

        if retriever_ok and llm_ok:
            st.markdown('<span class="dot-green">●</span> **Ready**', unsafe_allow_html=True)
        elif llm_ok or retriever_ok:
            st.markdown('<span class="dot-amber">●</span> **Partial**', unsafe_allow_html=True)
        else:
            st.markdown('<span class="dot-red">●</span> **Not ready**', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        col_a.metric("Chunks", health.get("total_chunks", 0))
        col_b.metric("Docs", len(health.get("processed_documents", [])))

        docs = health.get("processed_documents", [])
        if docs:
            st.caption("Indexed documents")
            for doc in docs:
                st.markdown(f"- `{doc}`")
        else:
            st.caption("No documents indexed yet.")

        st.divider()
        st.caption(f"LLM: `{health.get('openai_model', 'N/A')}`")
        embed = health.get("embedding_model", "N/A").split("/")[-1]
        st.caption(f"Embeddings: `{embed}`")

        if not llm_ok:
            st.warning(
                "No OPENAI_API_KEY set. Queries will return retrieved chunks only.",
                icon="⚠️",
            )

    if st.button("Refresh status", use_container_width=True):
        st.rerun()


st.title("📄 Arabic Document Q&A")
st.info(
    "**This system supports Arabic and English queries.** "
    "Upload your PDF documents in the first tab, then ask questions about them in the second.\n\n"
    "يدعم هذا النظام الاستعلامات باللغتين العربية والإنجليزية. "
    "قم بتحميل ملفات PDF في التبويب الأول، ثم اطرح أسئلتك في التبويب الثاني.",
    icon="ℹ️",
)

tab_upload, tab_query = st.tabs(["📤  Upload Documents", "🔍  Ask a Question"])

with tab_upload:
    st.subheader("Upload Arabic PDF Documents")
    st.caption(
        "Select one or more PDF files. Each file is extracted, chunked using "
        "Arabic-aware sentence boundaries, embedded, and added to the search index."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Maximum 50 MB per file. Image-only (scanned) PDFs are not supported.",
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.markdown(
            "<div style='text-align:center;padding:40px 0;color:#999;'>"
            "No files selected. Use the uploader above to choose one or more PDFs."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"**{len(uploaded_files)} file(s) ready to process:**")
        for f in uploaded_files:
            size_mb = len(f.getvalue()) / (1024 ** 2)
            st.markdown(f"- `{f.name}` &nbsp; ({size_mb:.1f} MB)", unsafe_allow_html=True)

        st.markdown("")

        if st.button(
            f"Process {len(uploaded_files)} file(s)",
            type="primary",
            use_container_width=True,
        ):
            progress = st.progress(0, text="Starting...")
            results: list[tuple[str, bool, str]] = []

            status_text = st.empty()
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
                        f"{res.get('chunk_count', '?')} chunks added "
                        f"({res.get('total_chunks_in_index', '?')} total in index)."
                    )
                else:
                    msg = payload if isinstance(payload, str) else str(payload)
                results.append((file.name, ok, msg))
            status_text.empty()

            progress.progress(1.0, text="Done.")
            progress.empty()

            successes = [(n, m) for n, ok, m in results if ok]
            failures = [(n, m) for n, ok, m in results if not ok]

            if successes:
                st.success(f"{len(successes)} file(s) processed successfully.")
                for name, msg in successes:
                    st.markdown(f"✅ **{name}** - {msg}")

            if failures:
                st.error(f"{len(failures)} file(s) could not be processed.")
                for name, msg in failures:
                    st.markdown(f"❌ **{name}** - {msg}")

            if successes:
                st.info(
                    "Index updated. Switch to **Ask a Question** to query your documents.",
                    icon="💡",
                )
                st.rerun()

with tab_query:
    st.subheader("Ask a Question")
    st.caption(
        "The model will answer in the same language as your question and will "
        "cite the document and page number for every claim."
    )

    question = st.text_area(
        "question",
        placeholder="اكتب سؤالك هنا...  /  Type your question here...",
        height=110,
        label_visibility="collapsed",
    )

    with st.expander("Retrieval settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        top_k_dense = col1.number_input(
            "Dense candidates",
            min_value=1, max_value=50, value=10,
            help="Candidates retrieved from the FAISS vector index.",
        )
        top_k_sparse = col2.number_input(
            "Sparse candidates",
            min_value=1, max_value=50, value=10,
            help="Candidates retrieved from the BM25 keyword index.",
        )
        top_k_final = col3.number_input(
            "Final results",
            min_value=1, max_value=20, value=5,
            help="Results kept after cross-encoder reranking.",
        )

    can_submit = bool(question.strip()) and health is not None
    if st.button(
        "Get Answer",
        type="primary",
        use_container_width=True,
        disabled=not can_submit,
    ):
        with st.spinner("Searching and generating answer..."):
            result = _run_query(
                question.strip(),
                int(top_k_dense),
                int(top_k_sparse),
                int(top_k_final),
            )

        if isinstance(result, str):
            st.session_state.query_result = None
            st.session_state.query_error = result
        else:
            st.session_state.query_result = result
            st.session_state.query_error = None

    if st.session_state.query_error:
        st.divider()
        st.error(st.session_state.query_error)

    elif st.session_state.query_result:
        result = st.session_state.query_result
        answer = result.get("answer", "")
        answered = result.get("answered", False)
        sources = result.get("sources", [])
        model = result.get("model", "")
        in_tok = result.get("input_tokens", 0)
        out_tok = result.get("output_tokens", 0)
        retrieval_only = result.get("retrieval_only", False)

        st.divider()

        if retrieval_only:
            st.info(
                "Running in **retrieval-only mode** - no LLM is configured. "
                "The passages below are the top-ranked chunks for your query. "
                "Set OPENAI_API_KEY to enable generated answers.",
                icon="ℹ️",
            )
        elif answered:
            st.markdown("#### Answer")
            ans_cls = "answer-card rtl" if _is_arabic(answer) else "answer-card"
            st.markdown(
                f'<div class="{ans_cls}">{_safe_html(answer)}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Model: `{model}` &nbsp;|&nbsp; Tokens: {in_tok} in / {out_tok} out",
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "The system could not find a sufficient answer in the uploaded documents.",
                icon="⚠️",
            )
            st.markdown("#### Response")
            ans_cls = "answer-card rtl" if _is_arabic(answer) else "answer-card"
            st.markdown(
                f'<div class="{ans_cls}">{_safe_html(answer)}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Model: `{model}` &nbsp;|&nbsp; Tokens: {in_tok} in / {out_tok} out",
                unsafe_allow_html=True,
            )

        if sources:
            st.markdown("#### Sources")
            for i, src in enumerate(sources, start=1):
                doc = src.get("source", "unknown")
                page = src.get("page_number", "?")
                score = src.get("rerank_score", 0.0)
                text = src.get("text", "")
                d_score = src.get("dense_score")
                s_score = src.get("sparse_score")

                label = f"[{i}]  {doc} - Page {page}"
                with st.expander(label, expanded=(i == 1)):
                    badges = f'<span class="badge">Rerank {score:.3f}</span>'
                    if d_score is not None:
                        badges += f'<span class="badge">Dense {d_score:.3f}</span>'
                    if s_score is not None:
                        badges += f'<span class="badge">BM25 {s_score:.1f}</span>'
                    st.markdown(badges, unsafe_allow_html=True)

                    st.markdown("")
                    _render_text(text)
