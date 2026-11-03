"""The faculty workflow: upload a syllabus PDF → read courses and skills → compare with the trained job-market
model → recommendations. One background worker processes uploads in order (SQLite allows one writer at a time).
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime, timezone

from sqlalchemy import func, select

from ..common import normalize_name
from ..database import SessionLocal
from ..models import Course, CurriculumVersion, FacultyAnalysis, Programme, SyllabusDocument, University
from .market_model import OTHER_ROLE, get_market, market_ready

log = logging.getLogger("skillpulse.faculty")

WORKSPACE_UNIVERSITY = "Faculty workspace"
STEPS = {
    "queued": "Waiting to start",
    "reading": "Reading the PDF and finding courses and topics",
    "comparing": "Comparing with industry job postings using the trained models",
    "done": "Done",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def job_field_label(value: str) -> str:
    return {"it": "All Software & IT jobs", "all": "All jobs (any industry)"}.get(value, value)


def valid_job_field(value: str) -> bool:
    ok, _ = market_ready()
    if not ok:
        return value in ("it", "all")
    return value in ("it", "all") or (value in get_market().roles and value != OTHER_ROLE)


def market_status() -> dict:
    ok, err = market_ready()
    if not ok:
        return {"state": "missing", "message": err, "postings": 0, "period": None, "job_fields": [
            {"value": "it", "label": job_field_label("it"), "postings": 0},
            {"value": "all", "label": job_field_label("all"), "postings": 0}]}
    m = get_market()
    return {"state": "ready", "message": None, "postings": m.n, "period": m.period, "job_fields": m.role_options()}


# --- Analyses ------------------------------------------------------------------------------------------------
# One background worker does all heavy work in order (SQLite allows a single writer at a time).
_jobs: "queue.Queue[tuple[str, int]]" = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()


def _workspace_version(db, course_name: str, institution: str | None) -> CurriculumVersion:
    name = (institution or "").strip() or WORKSPACE_UNIVERSITY
    uni = db.scalar(select(University).where(University.normalized_name == normalize_name(name)))
    if uni is None:
        uni = University(name=name, normalized_name=normalize_name(name), type="other")
        db.add(uni)
        db.flush()
    prog = db.scalar(select(Programme).where(Programme.university_id == uni.id, Programme.degree == "Uploaded syllabi"))
    if prog is None:
        prog = Programme(university_id=uni.id, name="Uploaded syllabi", degree="Uploaded syllabi")
        db.add(prog)
        db.flush()
    n = (db.scalar(select(func.count()).select_from(Course).where(Course.programme_id == prog.id)) or 0) + 1
    course = Course(programme_id=prog.id, code=f"UPLOAD-{n:04d}", title=course_name[:300])
    db.add(course)
    db.flush()
    ver = CurriculumVersion(course_id=course.id, version_label="v1")
    db.add(ver)
    db.flush()
    return ver


def create_analysis(db, data: bytes, filename: str, course_name: str, institution: str | None,
                    job_field: str, entry_level: bool = False) -> FacultyAnalysis:
    from ..storage import store_bytes
    from .pdf_processing import sha256_bytes, validate_pdf_bytes

    pages = validate_pdf_bytes(data)  # raises PDFValidationError with a clear message
    digest = sha256_bytes(data)
    doc = db.scalar(select(SyllabusDocument).where(SyllabusDocument.file_hash == digest))
    if doc is None:
        ver = _workspace_version(db, course_name, institution)
        doc = SyllabusDocument(curriculum_version_id=ver.id, original_filename=(filename or "syllabus.pdf")[:300],
                               stored_path=store_bytes(digest, data), file_hash=digest, file_size=len(data),
                               status="uploaded", page_count=pages, status_message="Queued")
        db.add(doc)
        db.flush()
    a = FacultyAnalysis(document_id=doc.id, course_name=course_name[:300], institution=(institution or None),
                        job_field=job_field, entry_level=entry_level, status="queued", step="queued",
                        message=STEPS["queued"])
    db.add(a)
    db.commit()
    enqueue(a.id)
    return a


def warm_up() -> None:
    """Load the NLP pipeline and skill matchers in the background at start-up, so the first upload is fast."""
    def _go():
        try:
            from ..config import get_settings
            from .skill_extractor import get_nlp
            from .taxonomy import get_matchers

            get_nlp(get_settings().spacy_model)
            wdb = SessionLocal()
            try:
                get_matchers(wdb)
            finally:
                wdb.close()
            get_market()
            log.info("warm-up done: NLP pipeline, skill matchers and ML models are loaded")
        except Exception:  # warm-up is only an optimisation
            log.exception("warm-up failed")

    threading.Thread(target=_go, daemon=True, name="warm-up").start()


def _ensure_worker() -> None:
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            threading.Thread(target=_worker, daemon=True, name="skillpulse-worker").start()
            _worker_started = True


def enqueue(analysis_id: int) -> None:
    _ensure_worker()
    _jobs.put(("analysis", analysis_id))


def resume_unfinished() -> None:
    """Restart interrupted analyses, and re-run analyses made by older versions (no stored ML result) so they
    open with the current models. Their PDFs are read again with the current extractor."""
    db = SessionLocal()
    try:
        legacy = db.scalars(select(FacultyAnalysis).where(FacultyAnalysis.status == "done",
                                                          FacultyAnalysis.result.is_(None))).all()
        for a in legacy:
            a.status, a.step, a.message, a.finished_at = "queued", "queued", "Updating to the new ML analysis", None
            if a.document and a.document.status == "analysed":
                a.document.status = "uploaded"
        db.commit()
        ids = list(db.scalars(select(FacultyAnalysis.id).where(FacultyAnalysis.status.in_(("queued", "processing")))
                              .order_by(FacultyAnalysis.id)))
    finally:
        db.close()
    for i in ids:
        enqueue(i)


def _worker() -> None:
    while True:
        kind, ident = _jobs.get()
        try:
            run_analysis(ident)
        except Exception:  # never let the worker die
            log.exception("Background job %s %s crashed", kind, ident)
            if kind == "analysis":
                _mark_failed(ident, "An unexpected error occurred. Please try again.")
        finally:
            _jobs.task_done()


def _mark_failed(analysis_id: int, message: str) -> None:
    db = SessionLocal()
    try:
        a = db.get(FacultyAnalysis, analysis_id)
        if a and a.status != "done":
            a.status, a.message, a.finished_at = "failed", message[:500], _now()
            db.commit()
    finally:
        db.close()


def _update(db, a: FacultyAnalysis, **kw) -> None:
    for k, v in kw.items():
        setattr(a, k, v)
    if "step" in kw and "message" not in kw:
        a.message = STEPS.get(kw["step"])
    db.commit()


def run_analysis(analysis_id: int) -> None:
    from .comparison import compare
    from .pipeline import process_document

    db = SessionLocal()
    try:
        a = db.get(FacultyAnalysis, analysis_id)
        if a is None or a.status == "done":
            return
        _update(db, a, status="processing", step="reading")
        doc = db.get(SyllabusDocument, a.document_id)
        if doc.status != "analysed":
            process_document(doc.id, None)
            db.expire_all()
            doc = db.get(SyllabusDocument, a.document_id)
            a = db.get(FacultyAnalysis, analysis_id)
        if doc.status != "analysed":
            reason = doc.status_message or "The PDF could not be read."
            if doc.status == "needs_ocr":
                reason = ("This PDF is a scanned image and no OCR engine is installed, so its text cannot be read. "
                          "Run: pip install -r requirements.txt (installs RapidOCR), then restart the backend.")
            _update(db, a, status="failed", step="reading", message=reason, finished_at=_now())
            return
        _update(db, a, step="comparing")
        ok, err = market_ready()
        if not ok:
            _update(db, a, status="failed", step="comparing", message=err, finished_at=_now())
            return
        try:
            result = compare(db, doc, get_market(), a.job_field, a.entry_level, a.course_name)
        except ValueError as exc:
            _update(db, a, status="failed", step="comparing", message=str(exc), finished_at=_now())
            return
        _update(db, a, status="done", step="done", result=result, finished_at=_now())
    finally:
        db.close()


# --- Result for the faculty view ----------------------------------------------------------------------------
def analysis_brief(a: FacultyAnalysis) -> dict:
    return {"id": a.id, "course_name": a.course_name, "institution": a.institution, "job_field": a.job_field,
            "job_field_label": job_field_label(a.job_field), "entry_level": a.entry_level, "status": a.status,
            "step": a.step, "message": a.message,
            "filename": a.document.original_filename if a.document else None,
            "created_at": a.created_at, "finished_at": a.finished_at}


def analysis_result(db, a: FacultyAnalysis, full: bool = True) -> dict:
    out = analysis_brief(a)
    if a.status == "done" and a.result:
        res = json.loads(a.result) if isinstance(a.result, str) else a.result
        out.update(res if full else {"summary": res.get("summary")})
    return out
