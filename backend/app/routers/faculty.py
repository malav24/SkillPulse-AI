"""Faculty API: upload a syllabus PDF, follow its progress, read the comparison and recommendations."""
from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..models import FacultyAnalysis
from ..services import faculty
from ..services.pdf_processing import PDFValidationError

router = APIRouter(prefix="/api", tags=["faculty"])


@router.get("/health", summary="Is the server and database up?")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        ok = True
    except Exception:
        ok = False
    return {"status": "ok" if ok else "degraded", "database": ok, "time": datetime.now(timezone.utc).isoformat()}


@router.get("/market", summary="The industry reference: postings, period and the job fields you can compare with")
def market():
    return {**faculty.market_status(), "max_upload_mb": get_settings().max_upload_mb}


@router.get("/models", summary="How the machine-learning models were trained and how well they perform")
def models():
    from ..services.market_model import get_market, market_ready

    ok, err = market_ready()
    if not ok:
        raise HTTPException(503, err)
    return get_market().metrics


def _get(db: Session, analysis_id: int) -> FacultyAnalysis:
    a = db.get(FacultyAnalysis, analysis_id)
    if a is None:
        raise HTTPException(404, "Analysis not found")
    return a


@router.post("/analyses", status_code=201, summary="Upload a syllabus PDF and start the analysis")
async def create(file: UploadFile = File(...), course_name: str = Form(...), institution: str | None = Form(None),
                 job_field: str = Form("it"), entry_level: bool = Form(False), db: Session = Depends(get_db)):
    course_name = course_name.strip()
    if not course_name:
        raise HTTPException(422, "Please enter the course or programme name")
    if not faculty.valid_job_field(job_field):
        raise HTTPException(422, "Unknown job field")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(422, "Please upload a PDF file (.pdf)")
    limit = get_settings().max_upload_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"The PDF is larger than {get_settings().max_upload_mb:g} MB")
    try:
        a = faculty.create_analysis(db, data, file.filename or "syllabus.pdf", course_name,
                                    (institution or "").strip() or None, job_field, entry_level)
    except PDFValidationError as exc:
        raise HTTPException(422, str(exc))
    return faculty.analysis_brief(a)


@router.get("/analyses", summary="Previous analyses, newest first")
def list_analyses(db: Session = Depends(get_db)):
    rows = db.scalars(select(FacultyAnalysis).order_by(FacultyAnalysis.id.desc()).limit(200)).all()
    return [faculty.analysis_result(db, a, full=False) for a in rows]


@router.get("/analyses/{analysis_id}", summary="Progress, or the full result with recommendations")
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    return faculty.analysis_result(db, _get(db, analysis_id))


@router.post("/analyses/{analysis_id}/retry", summary="Run a failed analysis again")
def retry(analysis_id: int, db: Session = Depends(get_db)):
    a = _get(db, analysis_id)
    if a.status != "failed":
        raise HTTPException(409, "Only a failed analysis can be retried")
    doc = a.document
    if doc.status in ("failed", "needs_ocr"):
        doc.status = "uploaded"
    a.status, a.step, a.message, a.finished_at, a.result = "queued", "queued", faculty.STEPS["queued"], None, None
    db.commit()
    faculty.enqueue(a.id)
    return faculty.analysis_brief(a)


@router.delete("/analyses/{analysis_id}", status_code=204, summary="Remove an analysis from the history")
def delete(analysis_id: int, db: Session = Depends(get_db)):
    a = _get(db, analysis_id)
    if a.status in ("queued", "processing"):
        raise HTTPException(409, "Wait until the analysis has finished")
    db.delete(a)
    db.commit()
