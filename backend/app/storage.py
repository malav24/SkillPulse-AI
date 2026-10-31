"""Secure file storage for uploaded PDFs (content-addressed, outside any static web root)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import Course, CurriculumVersion, Programme, SyllabusDocument

log = logging.getLogger("skillpulse.storage")


def store_bytes(file_hash: str, data: bytes) -> str:
    root = Path(get_settings().upload_dir).resolve()
    folder = root / file_hash[:2]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{file_hash}.pdf"
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:  # Windows / restricted FS
        pass
    return str(path)


def safe_path(stored_path: str) -> Path:
    """Resolve a stored path and make sure it is inside the upload directory."""
    root = Path(get_settings().upload_dir).resolve()
    p = Path(stored_path).resolve()
    if root not in p.parents:
        raise PermissionError("Stored path is outside the upload directory")
    return p


def remove_files(paths: list[str]) -> None:
    for sp in paths:
        try:
            p = safe_path(sp)
            if p.exists():
                p.unlink()
        except Exception as exc:  # never let cleanup break the request
            log.warning("Could not remove stored file %s: %s", sp, exc)


def document_paths(db: Session, *, university_id=None, programme_id=None, course_id=None, version_id=None) -> list[str]:
    stmt = select(SyllabusDocument.stored_path).join(CurriculumVersion).join(Course).join(Programme)
    if university_id is not None:
        stmt = stmt.where(Programme.university_id == university_id)
    if programme_id is not None:
        stmt = stmt.where(Course.programme_id == programme_id)
    if course_id is not None:
        stmt = stmt.where(CurriculumVersion.course_id == course_id)
    if version_id is not None:
        stmt = stmt.where(SyllabusDocument.curriculum_version_id == version_id)
    return list(db.scalars(stmt))
