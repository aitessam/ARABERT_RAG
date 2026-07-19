# Arabic Document RAG System

A hybrid retrieval-augmented generation (RAG) system for Arabic PDF documents. Upload Arabic reports, policies, or research papers and ask questions about them in Arabic or English. The system cites the exact document and page number for every claim and explicitly says when it cannot find an answer rather than fabricating one.

---

## Why Arabic RAG is technically interesting

Building a RAG system for Arabic is not a translation of an English pipeline. Arabic presents a set of challenges that require specific engineering decisions at every stage.

**Morphological richness.** Arabic is a highly inflected, root-based language. The single word "وَسَيَكْتُبُونَهَا" encodes subject (they), tense (future), verb (write), and object (it) simultaneously. This compresses meaning into fewer tokens, which means chunking by character count produces uneven semantic units. The chunker here splits at paragraph and sentence boundaries instead.

**Right-to-left text in PDFs.** PDFs store text blocks as positioned rectangles on a page. A naive `get_text()` call returns Arabic blocks in left-to-right visual order (backwards). The PDF reader sorts blocks by their Y coordinate (top to bottom) and then by their X coordinate in reverse (right to left), which matches the actual Arabic reading order.

**Orthographic variation.** The letter alef alone has four Unicode code points (ا أ إ آ) depending on the preceding context and typographer preference. Without normalisation, "الإجراء" and "الاجراء" are treated as completely different strings. The system collapses all alef variants, removes optional vowel marks (tashkeel), and strips the decorative tatweel character before any text enters the index.

**Mixed-script documents.** Arabic technical documents routinely mix Arabic prose with Latin acronyms, English terms, and Western numerals. The BM25 tokeniser preserves both scripts, lowercases Latin tokens, and applies language-appropriate stopword filtering so that neither script drowns the other out.

**Stopword challenge.** Arabic has a rich set of clitic prefixes (ب، ل، و، ك، ف) that attach directly to the following word with no space. Without normalisation and a carefully constructed stopword list, BM25 treats "والتوصية" and "التوصية" as different terms even though they share the same content word.

**Diglossia.** Modern Standard Arabic (used in formal documents) differs substantially from spoken dialects. The embedding model (`intfloat/multilingual-e5-large`) was trained on formal multilingual text and handles MSA well. Dialect support would require a different model.

---

## Architecture

### Ingestion pipeline

```
                        ┌──────────────────────────────────┐
  PDF upload            │   POST /upload  (202 Accepted)    │
  ─────────────────────▶│   validate → save raw → job_id   │
                        └──────────────┬───────────────────┘
                                       │ BackgroundTask
                                       ▼
                        ┌──────────────────────────────────┐
                        │   ArabicPDFReader  (PyMuPDF)     │
                        │   ├─ RTL block sort (y↓, x←)    │
                        │   ├─ Header/footer removal       │
                        │   │   position-based (top 8%)   │
                        │   │   repetition-based (≥3 pg)  │
                        │   └─ Arabic normalisation        │
                        │       Alef · Tatweel · Tashkeel  │
                        └──────────────┬───────────────────┘
                                       │ list[PageContent]
                                       ▼
                        ┌──────────────────────────────────┐
                        │   ArabicChunker                  │
                        │   ├─ Paragraph boundaries (\n\n) │
                        │   ├─ Sentence endings (۔ ؟ ؛ . ?)│
                        │   ├─ 400-token target size       │
                        │   └─ 60-token overlap between   │
                        │       consecutive chunks         │
                        └──────────┬───────────┬───────────┘
                                   │           │
                      list[Chunk]  │           │  list[Chunk]
                                   ▼           ▼
              ┌────────────────────┐   ┌───────────────────────┐
              │  EmbeddingsStore   │   │   HybridRetriever     │
              │  ─────────────     │   │   ──────────────────  │
              │  "passage: "+text  │   │   Arabic tokenisation │
              │  → e5-large embed  │   │   stopword removal    │
              │  → L2 normalise    │   │   → BM25Okapi corpus  │
              │  → Qdrant upsert   │   └──────────┬────────────┘
              └────────┬───────────┘              │
                       │                          │
                       └────────────┬─────────────┘
                                    │ persist
                              ┌─────┴──────────┐
                              │ Qdrant          │
                              │  collection     │
                              │ + BM25 corpus   │
                              │  (JSON on disk) │
                              └────────────────┘
```

### Query pipeline

```
  User question  ─────────────────────────────────────────────────────────┐
                                                                           │
                   ┌─────────────────────┐   ┌─────────────────────────┐  │
                   │  Dense retrieval    │   │  Sparse retrieval       │  │
                   │  ─────────────────  │   │  ───────────────────── │  │
                   │  "query: "+question │   │  Arabic tokenise        │  │
                   │  → e5-large embed  │   │  → BM25 score all docs  │  │
                   │  → Qdrant search   │   │  → top 10 by score      │  │
                   │  → top 10 by       │   │  (zero-score excluded)  │  │
                   │    cosine sim      │   └──────────────┬──────────┘  │
                   └──────────┬──────────┘                 │             │
                              │                            │             │
                              └──────────────┬─────────────┘             │
                                             │ merge + deduplicate        │
                                             │ by chunk_id               │
                                             ▼                           │
                              ┌──────────────────────────────┐           │
                              │  CrossEncoder reranker       │◀──────────┘
                              │  mmarco-mMiniLMv2-L12        │  (query, passage) pairs
                              │  jointly scores every pair   │
                              │  → sort by logit desc        │
                              │  → top 5 chunks              │
                              └──────────────┬───────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │  GPT-4o  (LLMClient)         │
                              │  ─────────────────────────   │
                              │  System prompt rules:        │
                              │  1. Reply in question lang   │
                              │  2. Use only context text    │
                              │  3. Cite [file, p.N] inline  │
                              │  4. "Cannot answer" phrase   │
                              │     when context insufficient│
                              └──────────────┬───────────────┘
                                             │
                                             ▼
                                    GeneratedAnswer
                                    ├─ answer text
                                    ├─ answered: bool
                                    ├─ sources (5 chunks)
                                    └─ token usage
```

---

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| PDF extraction | PyMuPDF (`fitz`) | RTL-aware block extraction |
| Arabic NLP | `pyarabic`, regex | Normalisation, stopwords |
| Chunking | Custom (`ArabicChunker`) | Semantic sentence-boundary splitting |
| Bi-encoder | `intfloat/multilingual-e5-large` | 1024-dim multilingual embeddings |
| Dense index | Qdrant | Vector similarity search |
| Sparse index | `BM25Okapi` (rank-bm25) | Arabic keyword matching |
| Cross-encoder | `mmarco-mMiniLMv2-L12-H384-v1` | Multilingual reranking |
| LLM | OpenAI GPT-4o | Grounded answer generation |
| Backend | FastAPI + uvicorn | Async REST API |
| Frontend | Streamlit | Browser UI with RTL rendering |

---

## What makes this production-grade vs a simple RAG

A minimal RAG system does three things: split text by character count, store vectors in a database, retrieve top-k by cosine similarity, and prompt an LLM. This system adds nine layers on top of that.

**1. Background task ingestion (202 Accepted + polling)**
A simple system blocks the HTTP connection for the entire duration of embedding, which can be minutes on a large document and will time out. Here, `POST /upload` returns a `job_id` in milliseconds, and the CPU-heavy work runs in `asyncio.to_thread`. The client polls `GET /upload/status/{job_id}` to check progress.

**2. Idempotent re-uploads**
Uploading the same filename twice in a simple system duplicates every chunk in the index, degrading retrieval quality. Here, `remove_by_source(filename)` purges all existing vectors for that file before re-indexing, so the index always reflects the current version of each document.

**3. Hybrid retrieval**
Dense-only retrieval misses exact keyword matches. Sparse-only retrieval misses semantic paraphrase. Running both and merging the candidate pools captures what each alone would miss. This is especially important in Arabic where a technical term like "ISO 27001" may not have a good embedding representation but will be matched exactly by BM25.

**4. Cross-encoder reranking**
Bi-encoders embed the query and passage independently and compare them by dot product. They cannot see how a specific phrase in the query relates to a specific phrase in the chunk. The cross-encoder reads both at once in a single forward pass. This is slower (called only on the ~20 merged candidates, not the full corpus) but significantly more precise.

**5. Arabic-specific PDF parsing**
A generic PDF reader returns Arabic text in the wrong order and does not remove repeated headers, footers, or page numbers. The custom `ArabicPDFReader` sorts blocks correctly for RTL, detects repeated elements across pages, and normalises the text before it enters the pipeline.

**6. Semantic chunking with overlap**
Fixed-size character splitting cuts in the middle of sentences. Semantic chunking respects paragraph and sentence boundaries. The 60-token overlap ensures that answers spanning a chunk boundary are not lost.

**7. Structured error taxonomy**
HTTP status codes are used semantically: 400 (bad input), 404 (no index yet), 422 (valid format, bad content), 503 (LLM unavailable). Callers can handle each case differently without parsing error strings.

**8. Index persistence with startup reload**
The Qdrant collection and BM25 corpus are persisted across restarts. Qdrant runs as a local server with its own storage directory; the BM25 corpus is saved to disk as JSON. On server restart, both are reloaded in the lifespan hook. A simple system loses all processed documents on restart and requires re-ingestion.

**9. Hallucination guard**
The system prompt mandates two specific refusal phrases (one Arabic, one English) when the context is insufficient. The `answered` boolean in the response is set by detecting those exact strings, not by heuristic text classification. The frontend surfaces a distinct warning state when `answered=False`.

---

## Project structure

```
ArabicRAG/
├── core/                         shared processing modules
│   ├── chunker.py                Arabic-aware semantic chunker
│   ├── embeddings.py             Qdrant vector store + e5-large embeddings
│   ├── retriever.py              hybrid BM25 + Qdrant + cross-encoder
│   └── llm.py                   GPT-4o answer generation
│
├── utils/
│   └── pdf_reader.py             PyMuPDF RTL extractor + Arabic normaliser
│
├── backend/
│   └── app/
│       ├── main.py               FastAPI app, lifespan, /health
│       ├── core/
│       │   └── config.py         pydantic-settings (.env loader)
│       └── api/routes/
│           ├── documents.py      POST /upload, GET /upload/status/{job_id}
│           └── query.py          POST /query
│
├── frontend/
│   └── app.py                    Streamlit UI (upload + query tabs)
│
├── data/
│   ├── raw/                      uploaded PDFs (git-ignored)
│   ├── processed/                intermediate outputs (git-ignored)
│   └── vector_store/             Qdrant collection + BM25 JSON metadata (git-ignored)
│
├── tests/
│   ├── backend/
│   └── frontend/
│
├── .env.example                  configuration template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.11 or 3.12
- An OpenAI API key with access to `gpt-4o`
- ~4 GB of free disk space (for the `multilingual-e5-large` model download on first run)
- A modern CPU; a GPU is not required but will speed up embedding significantly

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd ArabicRAG

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The first run will download `intfloat/multilingual-e5-large` (~2.2 GB) and `mmarco-mMiniLMv2-L12-H384-v1` (~120 MB) from HuggingFace. They are cached locally after the first download.

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

All other values have working defaults and do not need to be changed to get started.

### 4. Create the data directories

```bash
mkdir -p data/raw data/processed data/vector_store
```

---

## Running

Open two terminals from the project root.

**Terminal 1: Backend**

```bash
uvicorn backend.app.main:app --reload --port 8000
```

The server starts in under a second. The first upload triggers the model download.

**Terminal 2: Frontend**

```bash
streamlit run frontend/app.py
```

Opens at `http://localhost:8501`.

**API documentation** (auto-generated by FastAPI) is available at:
- `http://localhost:8000/docs` (Swagger UI)
- `http://localhost:8000/redoc` (ReDoc)

---

## API reference

### `POST /upload`

Accepts a PDF file. Returns `202 Accepted` immediately with a `job_id`.

```bash
curl -X POST http://localhost:8000/upload \
     -F "file=@report.pdf"
```

```json
{
  "job_id": "a3f9c12b8e04",
  "filename": "report.pdf",
  "message": "'report.pdf' accepted for ingestion. Poll GET /upload/status/a3f9c12b8e04 to check progress."
}
```

### `GET /upload/status/{job_id}`

Poll until `status` is `"done"` or `"error"`.

```bash
curl http://localhost:8000/upload/status/a3f9c12b8e04
```

```json
{
  "job_id": "a3f9c12b8e04",
  "status": "done",
  "filename": "report.pdf",
  "result": {
    "page_count": 42,
    "chunk_count": 187,
    "total_chunks_in_index": 187
  }
}
```

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "ما هي التوصيات الرئيسية في التقرير؟"}'
```

```json
{
  "question": "ما هي التوصيات الرئيسية في التقرير؟",
  "answer": "أوصى التقرير بثلاثة إجراءات رئيسية: أولاً، تعزيز البنية التحتية الرقمية [report.pdf, p. 12]. ثانياً، إنشاء لجنة متخصصة للرقابة [report.pdf, p. 15]. ثالثاً، مراجعة الميزانية التشغيلية سنوياً [report.pdf, p. 23].",
  "answered": true,
  "sources": [
    {
      "source": "report.pdf",
      "page_number": 12,
      "rerank_score": 4.821,
      "text": "..."
    }
  ],
  "model": "gpt-4o",
  "input_tokens": 1843,
  "output_tokens": 124
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "retriever_ready": true,
  "llm_ready": true,
  "total_chunks": 187,
  "processed_documents": ["report.pdf"],
  "openai_model": "gpt-4o",
  "embedding_model": "intfloat/multilingual-e5-large"
}
```

---

## Example queries

These work once at least one Arabic PDF has been uploaded.

**Arabic questions (answers returned in Arabic):**

```
ما هي التوصيات الرئيسية في التقرير؟
What are the main recommendations? (replied in Arabic if document is Arabic)
```

```
ما هو إجمالي الميزانية المقترحة؟
```

```
من هم الأطراف المعنية بتنفيذ الخطة؟
```

```
متى يبدأ تنفيذ المرحلة الأولى؟
```

**English questions (answers returned in English):**

```
What is the total proposed budget?
```

```
Which departments are responsible for implementation?
```

```
Summarise the risk assessment findings.
```

**When the answer is not in the documents:**

The model will respond with one of two exact phrases rather than hallucinating:

- Arabic: `"لا يمكنني الإجابة على هذا السؤال بناءً على المستندات المقدمة."`
- English: `"I cannot answer this question based on the provided documents."`

The `answered` field in the API response will be `false` in this case.

---

## Configuration reference

All values are read from `.env`. Defaults are shown.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key. |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model for generation. |
| `LLM_MAX_TOKENS` | `1500` | Maximum tokens in the generated answer. |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature. Keep low for factual answers. |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | HuggingFace bi-encoder model. |
| `VECTOR_STORE_DIR` | `data/vector_store` | Where Qdrant data and BM25 metadata are saved. |
| `RAW_DATA_DIR` | `data/raw` | Where uploaded PDFs are stored. |
| `CHUNK_MAX_TOKENS` | `400` | Target token count per chunk. |
| `CHUNK_OVERLAP_TOKENS` | `60` | Overlap tokens between consecutive chunks. |
| `TOP_K_DENSE` | `10` | Qdrant candidates per query. |
| `TOP_K_SPARSE` | `10` | BM25 candidates per query. |
| `TOP_K_FINAL` | `5` | Final results after cross-encoder reranking. |
| `PDF_HEADER_MARGIN` | `0.08` | Top fraction of page treated as header zone. |
| `PDF_FOOTER_MARGIN` | `0.08` | Bottom fraction of page treated as footer zone. |
| `PDF_REPEAT_THRESHOLD` | `3` | Pages a block must appear on to be flagged as a repeated element. |

---

## Known limitations

- **Image-only PDFs** (scanned documents with no embedded text layer) are rejected. OCR support is not included. To process scanned PDFs, run them through an OCR tool such as Tesseract with Arabic language support before uploading.
- **The BM25 job pool resets on server restart.** In-progress job status is stored in memory. A restarted server will have no record of jobs submitted before the restart. The underlying index files are persisted and reloaded correctly; only the job status dict is lost.
- **The cross-encoder is English-trained.** `mmarco-mMiniLMv2-L12-H384-v1` is trained on the multilingual MS-MARCO dataset and handles Arabic reasonably, but an Arabic-specific cross-encoder would improve reranking precision on exclusively Arabic corpora.
- **Single-worker deployment.** Qdrant handles concurrent reads safely, but the BM25 corpus is held in memory and rebuilt on each ingestion. Running multiple uvicorn workers will cause each worker to maintain its own BM25 state. Use a single worker (`--workers 1`) unless you move BM25 state to a shared store.
