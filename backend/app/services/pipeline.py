"""Document processing pipeline: extraction -> structure -> skills.

Each stage records an AnalysisRun. A failure sets the document to `failed` (or `needs_ocr`)
with the stage and message; it is never reported as processed.
"""
from __future__ import annotations

import logging
import re
import traceback
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import SessionLocal
from ..models import (
    AnalysisRun,
    CurriculumSkillEvidence,
    DocumentPage,
    ExtractedSection,
    LearningOutcome,
    SyllabusDocument,
)
from .pdf_processing import PDFValidationError, extract_pdf
from .skill_extractor import SectionInput, extract_skills
from .structure_parser import load_rules, parse_syllabus
from .taxonomy import get_matchers

log = logging.getLogger("skillpulse.pipeline")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class StageError(Exception):
    def __init__(self, stage: str, message: str, status: str = "failed"):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.status = status


def _start_run(db: Session, doc: SyllabusDocument, run_type: str, user_id: int | None, params: dict | None = None):
    run = AnalysisRun(document_id=doc.id, run_type=run_type, status="running", parameters=params,
                      triggered_by_id=user_id)
    db.add(run)
    db.commit()
    return run


def _finish_run(db: Session, run: AnalysisRun, stats: dict | None = None, error: str | None = None):
    run.status = "failed" if error else "succeeded"
    run.finished_at = _now()
    run.stats = stats
    run.error = error
    db.commit()


def clear_derived_data(db: Session, doc_id: int, keep_reviewed: bool = False) -> None:
    ev = delete(CurriculumSkillEvidence).where(CurriculumSkillEvidence.document_id == doc_id)
    if keep_reviewed:
        ev = ev.where(CurriculumSkillEvidence.review_status == "pending")
    db.execute(ev)
    if not keep_reviewed:
        db.execute(delete(LearningOutcome).where(LearningOutcome.document_id == doc_id))
        db.execute(delete(ExtractedSection).where(ExtractedSection.document_id == doc_id))
        db.execute(delete(DocumentPage).where(DocumentPage.document_id == doc_id))


# --- Stage 1: text extraction ------------------------------------------------------------
def run_extraction(db: Session, doc: SyllabusDocument, user_id: int | None) -> None:
    s = get_settings()
    run = _start_run(db, doc, "extraction", user_id, {"ocr_enabled": s.ocr_enabled,
                                                      "min_page_chars": s.min_page_chars})
    try:
        result = extract_pdf(doc.stored_path, s)
    except PDFValidationError as exc:
        _finish_run(db, run, error=str(exc))
        raise StageError("extraction", str(exc))
    except FileNotFoundError:
        _finish_run(db, run, error="Stored file is missing")
        raise StageError("extraction", "The stored PDF file is missing from the upload directory.")

    db.execute(delete(DocumentPage).where(DocumentPage.document_id == doc.id))
    for p in result.pages:
        db.add(DocumentPage(document_id=doc.id, page_number=p.page_number, text=p.text,
                            char_count=p.char_count, method=p.method, status=p.status))
    doc.page_count = result.page_count
    doc.text_char_count = result.total_chars
    doc.empty_pages = result.empty_pages
    doc.ocr_pages = result.ocr_pages
    doc.extraction_method = result.method if result.total_chars else None
    stats = {"page_count": result.page_count, "chars": result.total_chars, "empty_pages": result.empty_pages,
             "ocr_pages": result.ocr_pages, "ocr_status": result.ocr_status}
    meaningful = sum(len("".join(p.text.split())) for p in result.pages if p.status == "ok")
    if meaningful < s.min_document_chars and result.unreadable_without_images == result.page_count:
        msg = ("No readable text was found and the PDF contains no scanned images either, so the file is blank "
               "or damaged. Open it in a PDF viewer to check, or export/download it again and re-upload.")
        _finish_run(db, run, stats, error=msg)
        raise StageError("extraction", msg)
    if meaningful < s.min_document_chars:
        if result.ocr_status in ("unavailable", "disabled"):
            msg = ("Insufficient extractable text: the document appears to be scanned. "
                   + (result.ocr_message or "OCR is not available."))
            _finish_run(db, run, stats, error=msg)
            raise StageError("extraction", msg, status="needs_ocr")
        msg = ("Insufficient extractable text: only "
               f"{meaningful} readable characters were found across {result.page_count} page(s)."
               + (f" {result.ocr_message}" if result.ocr_message else ""))
        _finish_run(db, run, stats, error=msg)
        raise StageError("extraction", msg)
    doc.status = "extracted"
    notes = []
    if result.empty_pages:
        notes.append(f"{len(result.empty_pages)} empty/unreadable page(s): {result.empty_pages}")
    if result.ocr_message:
        notes.append(result.ocr_message)
    doc.status_message = "; ".join(notes) or None
    _finish_run(db, run, stats)


# --- Stage 2: structure ------------------------------------------------------------------
def run_structure(db: Session, doc: SyllabusDocument, user_id: int | None) -> None:
    s = get_settings()
    run = _start_run(db, doc, "structure", user_id, {"rules": s.parsing_rules_path})
    try:
        pages = db.execute(
            select(DocumentPage.page_number, DocumentPage.text)
            .where(DocumentPage.document_id == doc.id, DocumentPage.status == "ok")
            .order_by(DocumentPage.page_number)
        ).all()
        result = parse_syllabus(strip_running_lines([(p, t) for p, t in pages]), load_rules(s.parsing_rules_path))
        db.execute(delete(LearningOutcome).where(LearningOutcome.document_id == doc.id))
        db.execute(delete(ExtractedSection).where(ExtractedSection.document_id == doc.id))
        section_ids: list[int] = []
        for i, sec in enumerate(result.sections):
            row = ExtractedSection(
                document_id=doc.id, curriculum_version_id=doc.curriculum_version_id,
                section_type=sec.section_type, title=sec.title[:500], unit_number=sec.unit_number,
                course_label=(sec.course or None) and sec.course[:300],
                content=sec.content, items=sec.items or None, page_start=sec.page_start, page_end=sec.page_end,
                order_index=i, confidence=sec.confidence, source="parser",
            )
            db.add(row)
            db.flush()
            section_ids.append(row.id)
        for o in result.outcomes:
            db.add(LearningOutcome(
                document_id=doc.id,
                section_id=section_ids[o.section_index] if o.section_index is not None else None,
                code=o.code, text=o.text, outcome_kind=o.kind, bloom_level=o.bloom_level,
                page_number=o.page, source="parser",
            ))
        # Preserve any manual metadata corrections across re-runs.
        previous = doc.detected_metadata or {}
        merged = dict(result.metadata)
        for k, v in previous.items():
            if isinstance(v, dict) and v.get("source") == "manual":
                merged[k] = v
        doc.detected_metadata = merged
        missing = [k for k, v in merged.items() if v.get("value") in (None, "")]
        stats = {
            "sections": len(result.sections),
            "by_type": {t: sum(1 for x in result.sections if x.section_type == t)
                        for t in {x.section_type for x in result.sections}},
            "outcomes": len(result.outcomes),
            "missing_fields": missing,
        }
        _finish_run(db, run, stats)
    except Exception as exc:
        db.rollback()
        log.exception("Structure parsing failed for document %s", doc.id)
        _finish_run(db, run, error=f"{exc.__class__.__name__}: {exc}")
        raise StageError("structure", f"Structure detection failed: {exc}")


# --- Stage 3: skills ---------------------------------------------------------------------
def run_skills(db: Session, doc: SyllabusDocument, user_id: int | None) -> dict:
    s = get_settings()
    matcher, fuzzy, semantic = get_matchers(db)
    params = {"semantic": semantic is not None, "fuzzy_threshold": s.fuzzy_threshold,
              "semantic_threshold": s.semantic_threshold}
    run = _start_run(db, doc, "skills", user_id, params)
    try:
        sections = db.scalars(
            select(ExtractedSection).where(ExtractedSection.document_id == doc.id)
            .order_by(ExtractedSection.order_index)
        ).all()
        # In a multi-course document the header block is the cover / programme name, not course content.
        if any(sec.section_type == "course" for sec in sections):
            sections = [sec for sec in sections if sec.section_type != "header"]
        inputs: list[SectionInput] = []
        for sec in sections:
            lines: list[tuple[int, str]] = []
            page = sec.page_start or 1
            for ln in sec.content.split("\n"):
                lines.append((page, ln))
            inputs.append(SectionInput(sec.id, _effective_type(sec), _evidence_title(sec), lines))
        # Re-derive per-line pages from stored page text (sections can span pages).
        _assign_line_pages(db, doc.id, inputs, sections)
        cands = extract_skills(
            inputs, matcher, model_name=s.spacy_model, fuzzy=fuzzy, fuzzy_threshold=s.fuzzy_threshold,
            semantic=semantic, semantic_threshold=s.semantic_threshold,
        )
        # Keep human-reviewed evidence; replace pending evidence.
        db.execute(delete(CurriculumSkillEvidence).where(
            CurriculumSkillEvidence.document_id == doc.id, CurriculumSkillEvidence.review_status == "pending"))
        reviewed = {
            (e.skill_id, e.section_id, e.page_number, e.char_start)
            for e in db.scalars(select(CurriculumSkillEvidence).where(CurriculumSkillEvidence.document_id == doc.id))
        }
        added = 0
        for c in cands:
            if (c.skill_id, c.section_index, c.page_number, c.char_start) in reviewed:
                continue
            db.add(CurriculumSkillEvidence(
                document_id=doc.id, curriculum_version_id=doc.curriculum_version_id, skill_id=c.skill_id,
                section_id=c.section_index, analysis_run_id=run.id, matched_text=c.matched_text[:300],
                evidence_text=c.evidence_text, page_number=c.page_number, section_type=c.section_type,
                section_title=(c.section_title or "")[:500], char_start=c.char_start, char_end=c.char_end,
                extraction_method=c.extraction_method, evidence_type=c.evidence_type, confidence=c.confidence,
                confidence_reason=c.confidence_reason, review_status="pending",
            ))
            added += 1
        stats = {
            "candidates": added,
            "distinct_skills": len({c.skill_id for c in cands}),
            "by_method": {m: sum(1 for c in cands if c.extraction_method == m)
                          for m in {c.extraction_method for c in cands}},
            "ambiguous": sum(1 for c in cands if c.evidence_type == "ambiguous"),
            "semantic_available": semantic is not None,
        }
        _finish_run(db, run, stats)
        return stats
    except Exception as exc:
        db.rollback()
        log.exception("Skill extraction failed for document %s", doc.id)
        _finish_run(db, run, error=f"{exc.__class__.__name__}: {exc}")
        raise StageError("skills", f"Skill extraction failed: {exc}")


_DIGITS = re.compile(r"\d+")


def strip_running_lines(pages: list[tuple[int, str]], edge: int = 2, share: float = 0.6) -> list[tuple[int, str]]:
    """Remove running headers/footers: lines found among the first/last `edge` lines of at least `share` of the
    pages (page numbers ignored). The first page is kept intact because it carries the document's title block."""
    if len(pages) < 3:
        return pages

    def key(line: str) -> str:
        return _DIGITS.sub("#", " ".join(line.split()).lower())

    counts: dict[str, int] = {}
    for _, text in pages:
        lines = [ln for ln in text.split("\n") if ln.strip()]
        for k in {key(ln) for ln in lines[:edge] + lines[-edge:]}:
            counts[k] = counts.get(k, 0) + 1
    running = {k for k, n in counts.items() if n >= max(3, share * len(pages)) and len(k) > 3}
    if not running:
        return pages
    out = [pages[0]]
    for pno, text in pages[1:]:
        out.append((pno, "\n".join(ln for ln in text.split("\n") if key(ln) not in running)))
    return out


_LAB_RE = re.compile(r"\b(lab|laboratory|practicals?|workshop|project)\b", re.I)


def _effective_type(sec: ExtractedSection) -> str:
    """Content inside a lab/project course counts as practical evidence, even without a 'List of experiments' heading."""
    if sec.course_label and _LAB_RE.search(sec.course_label) and sec.section_type in ("course", "unit", "other"):
        return "practical"
    return sec.section_type


def _evidence_title(sec: ExtractedSection) -> str:
    if sec.course_label and sec.section_type != "course":
        return f"{sec.course_label} › {sec.title}"
    return sec.title


def _assign_line_pages(db: Session, doc_id: int, inputs: list[SectionInput], sections) -> None:
    """Map each section line back to the page it came from, walking pages in order."""
    pages = db.execute(
        select(DocumentPage.page_number, DocumentPage.text).where(DocumentPage.document_id == doc_id)
        .order_by(DocumentPage.page_number)
    ).all()
    page_lines: list[tuple[int, str]] = []
    for pno, text in pages:
        for raw in text.split("\n"):
            t = " ".join(raw.split())
            if t:
                page_lines.append((pno, t))
    cursor = 0
    for inp, sec in zip(inputs, sections):
        fixed: list[tuple[int, str]] = []
        fallback = sec.page_start or 1
        for _, line in inp.lines:
            norm = " ".join(line.split())
            page = None
            if norm:
                for j in range(cursor, min(cursor + 400, len(page_lines))):
                    if norm in page_lines[j][1] or page_lines[j][1] in norm:
                        page = page_lines[j][0]
                        cursor = j
                        break
            fixed.append((page or fallback, line))
            if page:
                fallback = page
        inp.lines = fixed


# --- Orchestration --------------------------------------------------------------------------
def process_document(document_id: int, user_id: int | None = None) -> None:
    """Run the full pipeline in its own session (used from background tasks)."""
    db = SessionLocal()
    try:
        doc = db.get(SyllabusDocument, document_id)
        if doc is None:
            log.error("process_document: document %s not found", document_id)
            return
        doc.status = "processing"
        doc.status_message = "Extracting text"
        doc.error_stage = None
        doc.attempts = (doc.attempts or 0) + 1
        db.commit()
        try:
            run_extraction(db, doc, user_id)
            doc.status_message = (doc.status_message or "")
            db.commit()
            run_structure(db, doc, user_id)
            stats = run_skills(db, doc, user_id)
            doc.status = "analysed"
            note = doc.status_message
            doc.status_message = (f"{stats['candidates']} candidate skill mentions found"
                                  + (f"; {note}" if note else ""))
            doc.processed_at = _now()
            db.commit()
        except StageError as exc:
            db.rollback()
            doc = db.get(SyllabusDocument, document_id)
            doc.status = exc.status
            doc.error_stage = exc.stage
            doc.status_message = exc.message
            db.commit()
            log.warning("Document %s %s at %s: %s", document_id, exc.status, exc.stage, exc.message)
    except Exception:
        log.error("Unexpected pipeline failure for document %s\n%s", document_id, traceback.format_exc())
        db.rollback()
        doc = db.get(SyllabusDocument, document_id)
        if doc is not None:
            doc.status = "failed"
            doc.error_stage = doc.error_stage or "pipeline"
            doc.status_message = "Unexpected processing error; see server logs."
            db.commit()
    finally:
        db.close()


def rerun_skills(document_id: int, user_id: int | None = None) -> None:
    db = SessionLocal()
    try:
        doc = db.get(SyllabusDocument, document_id)
        if doc is None:
            return
        try:
            stats = run_skills(db, doc, user_id)
            doc.status = "analysed"
            doc.status_message = f"Skill extraction re-run: {stats['candidates']} new candidate mentions"
            doc.processed_at = _now()
            db.commit()
        except StageError as exc:
            doc = db.get(SyllabusDocument, document_id)
            doc.error_stage = exc.stage
            doc.status_message = exc.message
            db.commit()
    finally:
        db.close()
