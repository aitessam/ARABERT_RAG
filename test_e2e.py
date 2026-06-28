#!/usr/bin/env python3
"""
End-to-end test for ArabicRAG.

Creates a bilingual (English + Arabic) sample PDF, uploads it via the API,
runs one Arabic query and one English query, and prints the results.
Also prints a static analysis of known implementation issues.

Usage
-----
    # Server must be running first:
    uvicorn backend.app.main:app --reload --port 8000

    python test_e2e.py
    python test_e2e.py --url http://localhost:8000
    python test_e2e.py --keep-pdf          # don't delete the generated PDF
    python test_e2e.py --skip-upload       # re-use whatever is already indexed
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import requests

try:
    import fitz  # PyMuPDF — already in requirements.txt
except ImportError:
    print("ERROR: PyMuPDF not installed.  Run: pip install pymupdf")
    sys.exit(1)


# ── Sample document content ───────────────────────────────────────────────────

_ENGLISH_BODY = """\
Patient Health Record Management — Technical Overview

Electronic health records (EHR) are digital versions of patients' paper charts.
They make patient information available instantly and securely to authorised users.
EHR systems improve care coordination, reduce medical errors, and streamline
clinical workflows while enhancing data security and regulatory compliance.

Blood Pressure Clinical Guidelines

Normal blood pressure: 120 over 80 mmHg (120/80).
Stage 1 hypertension: 130-139 systolic or 80-89 diastolic mmHg.
Stage 2 hypertension: 140 or higher systolic or 90 or higher diastolic mmHg.
Hypertensive crisis: readings exceeding 180/120 mmHg — requires immediate care.

First-line treatment for Stage 1 hypertension is lifestyle modification:
  - Reduce dietary sodium intake to below 2,300 mg per day.
  - Achieve at least 150 minutes of moderate aerobic exercise per week.
  - Maintain a healthy body weight (BMI 18.5-24.9 kg/m2).
  - Limit alcohol to two standard drinks per day for men, one for women.
  - Cessation of tobacco products.

Pharmacological therapy is added when lifestyle changes are insufficient.
Common drug classes include ACE inhibitors, angiotensin receptor blockers (ARBs),
calcium channel blockers (CCBs), and thiazide diuretics. Combination therapy is
often required for Stage 2 and resistant hypertension cases.

Data Privacy and Consent Framework

All patient health data must be processed under explicit informed consent.
Consent records are stored with a timestamp, document version, and patient
identifier. Patients retain the right to withdraw consent at any time, which
triggers immediate anonymisation of their data within the platform. Consent
audit logs are retained for a minimum of seven years in accordance with
applicable healthcare data regulations.
"""

_ARABIC_BODY = """\
نظرة عامة على نظام إدارة السجلات الصحية

السجلات الصحية الإلكترونية هي نسخ رقمية من ملفات المرضى الورقية.
تُتيح هذه السجلات معلومات المريض فوراً وبأمان للمستخدمين المخوّلين.
تُحسّن أنظمة السجلات الصحية الإلكترونية تنسيق الرعاية، وتُقلّل من الأخطاء الطبية،
وتُبسّط سير العمل السريري مع تعزيز أمن البيانات والامتثال التنظيمي.

الإرشادات السريرية لضغط الدم

ضغط الدم الطبيعي: 120 على 80 ملم زئبق.
ارتفاع ضغط الدم من المرحلة الأولى: 130-139 انقباضي أو 80-89 انبساطي.
ارتفاع ضغط الدم من المرحلة الثانية: 140 أو أعلى انقباضي أو 90 أو أعلى انبساطي.
أزمة ارتفاع ضغط الدم: قراءات تتجاوز 180 على 120 تستوجب رعاية فورية.

العلاج الخطي الأول لضغط الدم من المرحلة الأولى هو تعديل نمط الحياة:
  - تقليل تناول الصوديوم الغذائي إلى أقل من 2300 ملغ يومياً.
  - ممارسة 150 دقيقة على الأقل من التمارين الهوائية المعتدلة أسبوعياً.
  - الحفاظ على وزن صحي (مؤشر كتلة الجسم 18.5-24.9 كغ/م2).
  - الحدّ من الكحول إلى مشروبين يومياً للرجال، ومشروب واحد للنساء.
  - الإقلاع عن منتجات التبغ.

يُضاف العلاج الدوائي عندما تكون تغييرات نمط الحياة غير كافية.
تشمل الفئات الدوائية الشائعة مثبطات الإنزيم المحوّل للأنجيوتنسين،
وحاصرات مستقبلات الأنجيوتنسين، وحاصرات قنوات الكالسيوم، ومدرات البول الثيازيدية.

إطار الخصوصية والموافقة

يجب معالجة جميع بيانات صحة المريض بموجب موافقة مستنيرة صريحة.
تُخزَّن سجلات الموافقة مع طابع زمني وإصدار المستند ومعرّف المريض.
يحتفظ المرضى بحق سحب موافقتهم في أي وقت، مما يُطلق عملية إخفاء هويتهم فوراً.
تُحتفظ سجلات تدقيق الموافقة لمدة سبع سنوات على الأقل.
"""


# ── PDF creation ──────────────────────────────────────────────────────────────

_ARABIC_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\arialuni.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _find_arabic_font() -> str | None:
    for path in _ARABIC_FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def create_sample_pdf(out_path: Path) -> None:
    """Write a two-page bilingual health document PDF to *out_path*."""
    doc = fitz.open()
    arabic_font = _find_arabic_font()

    def _add_page(body: str, title: str, use_arabic_font: bool = False) -> None:
        page = doc.new_page(width=595, height=842)  # A4
        y = 55
        # Title
        page.insert_text((50, y), title, fontsize=13)
        y += 30
        for line in body.strip().splitlines():
            if not line.strip():
                y += 8
                continue
            try:
                if use_arabic_font and arabic_font:
                    page.insert_text(
                        (50, y), line,
                        fontfile=arabic_font,
                        fontsize=11,
                    )
                else:
                    page.insert_text((50, y), line, fontsize=11)
            except Exception:
                page.insert_text((50, y), "[line encoding error]", fontsize=10)
            y += 17
            if y > 810:
                break  # stay within a single page

    _add_page(_ENGLISH_BODY, "BlockMed Pro — Health Records Overview (EN)", use_arabic_font=False)
    _add_page(_ARABIC_BODY, "BlockMed Pro - نظرة عامة على السجلات الصحية", use_arabic_font=True)

    doc.save(str(out_path))
    doc.close()

    if arabic_font:
        print(f"  Arabic font : {arabic_font}")
    else:
        print("  WARNING: No Arabic font found — Arabic page will contain placeholder text.")
    print(f"  PDF written : {out_path}  ({out_path.stat().st_size // 1024} KB)")


# ── API helpers ───────────────────────────────────────────────────────────────

def health_check(base: str) -> dict:
    r = requests.get(f"{base}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def upload_pdf(
    base: str,
    pdf_path: Path,
    poll_interval: float = 2.0,
    max_wait: float = 300.0,
) -> dict:
    """Upload a PDF and poll until ingestion completes. Returns the final JobStatus dict."""
    with open(pdf_path, "rb") as fh:
        r = requests.post(
            f"{base}/upload",
            files={"file": (pdf_path.name, fh, "application/pdf")},
            timeout=30,
        )
    r.raise_for_status()
    accepted = r.json()
    job_id = accepted["job_id"]
    print(f"  Job ID  : {job_id}  (polling every {poll_interval:.0f}s…)")

    elapsed = 0.0
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval
        sr = requests.get(f"{base}/upload/status/{job_id}", timeout=10)
        sr.raise_for_status()
        status = sr.json()
        if status["status"] == "done":
            return status
        if status["status"] == "error":
            raise RuntimeError(f"Ingestion failed: {status.get('error', 'unknown')}")
        print(f"  Processing… ({elapsed:.0f}s elapsed)")

    raise TimeoutError(f"Ingestion did not complete within {max_wait:.0f}s.")


def run_query(base: str, question: str, top_k_final: int = 5) -> dict:
    r = requests.post(
        f"{base}/query",
        json={"question": question, "top_k_final": top_k_final},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


# ── Pretty printers ───────────────────────────────────────────────────────────

_LINE = "─" * 72


def _header(text: str) -> None:
    print(f"\n{_LINE}\n  {text}\n{_LINE}")


def _print_query_result(label: str, question: str, resp: dict) -> None:
    print(f"\n  [{label}] Question : {question}")
    if resp.get("retrieval_only"):
        print("\n  [retrieval-only — no LLM configured]")
    else:
        print(f"\n  Answer (answered={resp['answered']}):\n")
        for line in resp["answer"].splitlines():
            print(f"    {line}")
    sources: list[dict] = resp.get("sources", [])
    if sources:
        print(f"\n  Sources ({len(sources)}):")
        for i, s in enumerate(sources, 1):
            score = s.get("rerank_score", 0.0)
            preview = (s.get("text") or "").replace("\n", " ")[:100]
            print(f"    [{i}] {s['source']}  p.{s['page_number']}  rerank={score:.4f}")
            print(f"         {preview}…")
    print(
        f"\n  Tokens : {resp.get('input_tokens', '?')} in / "
        f"{resp.get('output_tokens', '?')} out   model={resp.get('model', '?')}"
    )


# ── Static issues report ──────────────────────────────────────────────────────

_ISSUES: list[tuple[str, str, str, str]] = [
    (
        "MEDIUM",
        "Synchronous embedding and indexing inside the HTTP request handler",
        "backend/app/api/routes/documents.py",
        "PDF extraction, chunking, SentenceTransformer encoding, and FAISS/BM25 indexing all "
        "run synchronously in the upload handler. For documents larger than ~15-20 pages the "
        "total processing time can exceed typical reverse-proxy or load balancer timeouts "
        "(commonly 30-60 s), silently dropping the connection while work continues server-side.\n"
        "    Fix: return a job ID immediately and run ingestion in a FastAPI BackgroundTask "
        "or Celery worker, with a GET /upload/{job_id}/status polling endpoint.",
    ),
    (
        "LOW",
        "Zero-vector risk in embedding normalisation",
        "core/embeddings.py",
        "Chunk vectors are L2-normalised before FAISS insertion. If a chunk produces an "
        "all-zero embedding (pathological input that slips past the minimum-token filter), "
        "dividing by norm=0 produces a NaN vector. FAISS silently accepts NaN vectors but "
        "returns garbage cosine scores for those entries.\n"
        "    Fix: guard with: if np.linalg.norm(vec) == 0: skip or replace with zeros.",
    ),
]


def print_issues() -> None:
    _header("Implementation Issues Found")
    for severity, title, location, detail in _ISSUES:
        print(f"\n  [{severity}] {title}")
        print(f"  Location : {location}")
        print(f"  Detail   :")
        for line in detail.splitlines():
            print(f"    {line}")


# ── Main ──────────────────────────────────────────────────────────────────────

ARABIC_QUESTION = "ما هي إرشادات ضغط الدم الطبيعي وكيف يتم العلاج؟"
ENGLISH_QUESTION = "What are the blood pressure guidelines and treatment options for hypertension?"


def main() -> None:
    parser = argparse.ArgumentParser(description="ArabicRAG end-to-end test")
    parser.add_argument("--url", default="http://localhost:8000", metavar="URL",
                        help="Backend base URL (default: http://localhost:8000)")
    parser.add_argument("--keep-pdf", action="store_true",
                        help="Do not delete the generated sample PDF after the test")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip PDF creation and upload — query whatever is already indexed")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    passed = failed = 0

    # ── 1. Health check ───────────────────────────────────────────────────────
    _header("1 / 5  Health Check")
    try:
        h = health_check(base)
        print(f"  Status          : {h.get('status', '?')}")
        print(f"  LLM ready       : {h.get('llm_ready', '?')}")
        print(f"  Index ready     : {h.get('index_ready', '?')}")
        print(f"  Documents       : {h.get('document_count', '?')}")
        print(f"  Total chunks    : {h.get('total_chunks', '?')}")
        print(f"  Embedding model : {h.get('embedding_model', '?')}")
        passed += 1
    except Exception as exc:
        print(f"  FAILED — {exc}")
        print(f"\n  Is the server running?")
        print(f"  → uvicorn backend.app.main:app --reload --port 8000")
        sys.exit(1)

    # ── 2. Upload ─────────────────────────────────────────────────────────────
    if not args.skip_upload:
        _header("2 / 5  Create and Upload Sample Document")
        tmp_pdf = Path(tempfile.mktemp(suffix=".pdf", prefix="arabicrag_test_"))
        try:
            create_sample_pdf(tmp_pdf)
            print("  Uploading…")
            up = upload_pdf(base, tmp_pdf)
            res = up.get("result") or {}
            print(f"  Filename        : {up['filename']}")
            print(f"  Pages           : {res.get('page_count', '?')}")
            print(f"  Chunks added    : {res.get('chunk_count', '?')}")
            print(f"  Total in index  : {res.get('total_chunks_in_index', '?')}")
            passed += 1
        except Exception as exc:
            print(f"  FAILED — {exc}")
            failed += 1
        finally:
            if not args.keep_pdf:
                tmp_pdf.unlink(missing_ok=True)
            else:
                print(f"\n  PDF kept at: {tmp_pdf}")
        # Give indexing a moment (it's synchronous, but just in case)
        time.sleep(0.5)
    else:
        _header("2 / 5  Upload  [SKIPPED — --skip-upload flag]")
        print("  Using whatever documents are already indexed.")

    # ── 3. Arabic query ───────────────────────────────────────────────────────
    _header("3 / 5  Arabic Query")
    try:
        result = run_query(base, ARABIC_QUESTION)
        _print_query_result("AR", ARABIC_QUESTION, result)
        passed += 1
    except Exception as exc:
        print(f"  FAILED — {exc}")
        failed += 1

    # ── 4. English query ──────────────────────────────────────────────────────
    _header("4 / 5  English Query")
    try:
        result = run_query(base, ENGLISH_QUESTION)
        _print_query_result("EN", ENGLISH_QUESTION, result)
        passed += 1
    except Exception as exc:
        print(f"  FAILED — {exc}")
        failed += 1

    # ── 5. Issues report ──────────────────────────────────────────────────────
    print_issues()

    # ── Summary ───────────────────────────────────────────────────────────────
    _header(f"Summary  —  {passed} passed  /  {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
