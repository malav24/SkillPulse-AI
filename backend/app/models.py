"""ORM models for SkillPulse AI (Stage 1).

Hierarchy: University -> Programme -> Course -> CurriculumVersion -> SyllabusDocument
-> (DocumentPage, ExtractedSection, LearningOutcome, CurriculumSkillEvidence).
Every piece of skill evidence links back to its source document, page and section.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


# --- Enumerations (stored as strings; validated with CHECK constraints) -------------
USER_ROLES = ("admin", "editor", "viewer")
UNIVERSITY_TYPES = (
    "central", "state", "deemed", "private", "institute_of_national_importance", "autonomous_college", "other",
)
ACADEMIC_LEVELS = ("diploma", "undergraduate", "postgraduate", "doctoral", "certificate", "other")
VERSION_STATUSES = ("draft", "active", "archived")
DOCUMENT_STATUSES = ("uploaded", "processing", "extracted", "analysed", "failed", "needs_ocr")
SECTION_TYPES = (
    "header", "course", "objectives", "outcomes", "unit", "practical", "project", "textbooks", "references",
    "assessment", "other",
)
EVIDENCE_TYPES = ("topic_mention", "learning_outcome", "practical", "ambiguous")
EXTRACTION_METHODS = ("dictionary", "alias", "fuzzy", "semantic", "manual")
REVIEW_STATUSES = ("pending", "confirmed", "rejected", "edited", "merged")
RUN_STATUSES = ("running", "succeeded", "failed")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


class User(TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint(_in("role", USER_ROLES), name="ck_users_role"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="viewer")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class University(TimestampMixin, Base):
    __tablename__ = "universities"
    __table_args__ = (
        CheckConstraint(_in("type", UNIVERSITY_TYPES), name="ck_universities_type"),
        Index("ix_universities_state_type", "state", "type"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(300), nullable=False, unique=True, index=True)
    type: Mapped[str] = mapped_column(String(40), nullable=False, default="other")
    state: Mapped[str | None] = mapped_column(String(100))
    district: Mapped[str | None] = mapped_column(String(100))
    official_website: Mapped[str | None] = mapped_column(String(500))
    source_url: Mapped[str | None] = mapped_column(String(500))
    # Only stored when supplied from a verifiable source (e.g. AISHE / UGC code). Never generated.
    verified_identifier: Mapped[str | None] = mapped_column(String(100), unique=True)
    identifier_source: Mapped[str | None] = mapped_column(String(200))
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    programmes: Mapped[list[Programme]] = relationship(
        back_populates="university", cascade="all, delete-orphan", passive_deletes=True
    )


class Programme(TimestampMixin, Base):
    __tablename__ = "programmes"
    __table_args__ = (
        UniqueConstraint("university_id", "degree", "branch", "specialization", name="uq_programme_identity"),
        CheckConstraint(_in("academic_level", ACADEMIC_LEVELS), name="ck_programmes_level"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    university_id: Mapped[int] = mapped_column(
        ForeignKey("universities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    degree: Mapped[str] = mapped_column(String(100), nullable=False)
    branch: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    specialization: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    academic_level: Mapped[str] = mapped_column(String(30), nullable=False, default="undergraduate")
    institution: Mapped[str | None] = mapped_column(String(300))  # affiliated college / department
    duration_years: Mapped[float | None] = mapped_column(Float)

    university: Mapped[University] = relationship(back_populates="programmes")
    courses: Mapped[list[Course]] = relationship(
        back_populates="programme", cascade="all, delete-orphan", passive_deletes=True
    )


class Course(TimestampMixin, Base):
    __tablename__ = "courses"
    __table_args__ = (
        UniqueConstraint("programme_id", "code", "academic_year", name="uq_course_identity"),
        CheckConstraint("semester IS NULL OR (semester >= 1 AND semester <= 12)", name="ck_courses_semester"),
        CheckConstraint("credits IS NULL OR (credits >= 0 AND credits <= 40)", name="ck_courses_credits"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    programme_id: Mapped[int] = mapped_column(ForeignKey("programmes.id", ondelete="CASCADE"), nullable=False, index=True)
    code: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    semester: Mapped[int | None] = mapped_column(Integer)
    credits: Mapped[float | None] = mapped_column(Float)
    academic_year: Mapped[str] = mapped_column(String(20), nullable=False, default="")
    syllabus_version: Mapped[str | None] = mapped_column(String(50))

    programme: Mapped[Programme] = relationship(back_populates="courses")
    versions: Mapped[list[CurriculumVersion]] = relationship(
        back_populates="course", cascade="all, delete-orphan", passive_deletes=True
    )


class CurriculumVersion(TimestampMixin, Base):
    __tablename__ = "curriculum_versions"
    __table_args__ = (
        UniqueConstraint("course_id", "version_label", name="uq_version_label"),
        CheckConstraint(_in("status", VERSION_STATUSES), name="ck_versions_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True)
    version_label: Mapped[str] = mapped_column(String(50), nullable=False)
    academic_year: Mapped[str | None] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    notes: Mapped[str | None] = mapped_column(Text)

    course: Mapped[Course] = relationship(back_populates="versions")
    documents: Mapped[list[SyllabusDocument]] = relationship(
        back_populates="curriculum_version", cascade="all, delete-orphan", passive_deletes=True
    )


class SyllabusDocument(TimestampMixin, Base):
    __tablename__ = "syllabus_documents"
    __table_args__ = (
        CheckConstraint(_in("status", DOCUMENT_STATUSES), name="ck_documents_status"),
        Index("ix_documents_status", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    curriculum_version_id: Mapped[int] = mapped_column(
        ForeignKey("curriculum_versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    original_filename: Mapped[str] = mapped_column(String(300), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(600), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False, default="application/pdf")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="uploaded")
    status_message: Mapped[str | None] = mapped_column(Text)
    error_stage: Mapped[str | None] = mapped_column(String(50))
    page_count: Mapped[int | None] = mapped_column(Integer)
    text_char_count: Mapped[int | None] = mapped_column(Integer)
    empty_pages: Mapped[list | None] = mapped_column(JSON)
    ocr_pages: Mapped[list | None] = mapped_column(JSON)
    extraction_method: Mapped[str | None] = mapped_column(String(20))  # text | ocr | mixed
    detected_metadata: Mapped[dict | None] = mapped_column(JSON)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    uploaded_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    curriculum_version: Mapped[CurriculumVersion] = relationship(back_populates="documents")
    pages: Mapped[list[DocumentPage]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True,
        order_by="DocumentPage.page_number",
    )
    sections: Mapped[list[ExtractedSection]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True,
        order_by="ExtractedSection.order_index",
    )


class DocumentPage(Base):
    """Per-page extracted text. Preserves original page numbers for evidence."""
    __tablename__ = "document_pages"
    __table_args__ = (UniqueConstraint("document_id", "page_number", name="uq_document_page"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("syllabus_documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-based
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    method: Mapped[str] = mapped_column(String(20), nullable=False, default="text")  # text | ocr | none
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="ok")  # ok | empty | ocr_failed

    document: Mapped[SyllabusDocument] = relationship(back_populates="pages")


class ExtractedSection(TimestampMixin, Base):
    __tablename__ = "extracted_sections"
    __table_args__ = (
        CheckConstraint(_in("section_type", SECTION_TYPES), name="ck_sections_type"),
        Index("ix_sections_doc_order", "document_id", "order_index"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("syllabus_documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    curriculum_version_id: Mapped[int] = mapped_column(
        ForeignKey("curriculum_versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    section_type: Mapped[str] = mapped_column(String(20), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    unit_number: Mapped[str | None] = mapped_column(String(20))
    course_label: Mapped[str | None] = mapped_column(String(300))  # course heading in multi-course PDFs
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    items: Mapped[list | None] = mapped_column(JSON)  # topics / practicals / references
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="parser")  # parser | manual
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    document: Mapped[SyllabusDocument] = relationship(back_populates="sections")


class LearningOutcome(TimestampMixin, Base):
    __tablename__ = "learning_outcomes"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("syllabus_documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    section_id: Mapped[int | None] = mapped_column(ForeignKey("extracted_sections.id", ondelete="SET NULL"))
    code: Mapped[str | None] = mapped_column(String(20))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    outcome_kind: Mapped[str] = mapped_column(String(20), nullable=False, default="course_outcome")  # or objective
    bloom_level: Mapped[str | None] = mapped_column(String(20))
    page_number: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="parser")
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class Skill(TimestampMixin, Base):
    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    slug: Mapped[str] = mapped_column(String(200), nullable=False, unique=True, index=True)
    category: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    is_custom: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    aliases: Mapped[list[SkillAlias]] = relationship(
        back_populates="skill", cascade="all, delete-orphan", passive_deletes=True
    )


class SkillAlias(TimestampMixin, Base):
    __tablename__ = "skill_aliases"

    id: Mapped[int] = mapped_column(primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True)
    alias: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_alias: Mapped[str] = mapped_column(String(200), nullable=False, unique=True, index=True)
    case_sensitive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # If context_terms is set, the alias only maps confidently when one of these terms is nearby.
    context_terms: Mapped[list | None] = mapped_column(JSON)
    ambiguity_policy: Mapped[str] = mapped_column(String(10), nullable=False, default="flag")  # flag | skip

    skill: Mapped[Skill] = relationship(back_populates="aliases")


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (CheckConstraint(_in("status", RUN_STATUSES), name="ck_runs_status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("syllabus_documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_type: Mapped[str] = mapped_column(String(30), nullable=False)  # extraction | structure | skills
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    parameters: Mapped[dict | None] = mapped_column(JSON)
    stats: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    triggered_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


class CurriculumSkillEvidence(TimestampMixin, Base):
    __tablename__ = "curriculum_skill_evidence"
    __table_args__ = (
        CheckConstraint(_in("evidence_type", EVIDENCE_TYPES), name="ck_evidence_type"),
        CheckConstraint(_in("extraction_method", EXTRACTION_METHODS), name="ck_evidence_method"),
        CheckConstraint(_in("review_status", REVIEW_STATUSES), name="ck_evidence_review"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_evidence_confidence"),
        Index("ix_evidence_doc_skill", "document_id", "skill_id"),
        Index("ix_evidence_review", "review_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("syllabus_documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    curriculum_version_id: Mapped[int] = mapped_column(
        ForeignKey("curriculum_versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True)
    section_id: Mapped[int | None] = mapped_column(ForeignKey("extracted_sections.id", ondelete="SET NULL"))
    analysis_run_id: Mapped[int | None] = mapped_column(ForeignKey("analysis_runs.id", ondelete="SET NULL"))
    matched_text: Mapped[str] = mapped_column(String(300), nullable=False)
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    section_type: Mapped[str | None] = mapped_column(String(20))
    section_title: Mapped[str | None] = mapped_column(String(500))
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    extraction_method: Mapped[str] = mapped_column(String(20), nullable=False)
    evidence_type: Mapped[str] = mapped_column(String(20), nullable=False, default="topic_mention")
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_reason: Mapped[str | None] = mapped_column(String(300))
    review_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    reviewer_note: Mapped[str | None] = mapped_column(Text)
    reviewed_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    skill: Mapped[Skill] = relationship()


class HumanReview(Base):
    __tablename__ = "human_reviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(40), nullable=False)  # evidence | section | outcome | metadata
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    evidence_id: Mapped[int | None] = mapped_column(
        ForeignKey("curriculum_skill_evidence.id", ondelete="SET NULL"), index=True
    )
    document_id: Mapped[int | None] = mapped_column(ForeignKey("syllabus_documents.id", ondelete="CASCADE"), index=True)
    action: Mapped[str] = mapped_column(String(30), nullable=False)
    before: Mapped[dict | None] = mapped_column(JSON)
    after: Mapped[dict | None] = mapped_column(JSON)
    note: Mapped[str | None] = mapped_column(Text)
    reviewer_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_entity", "entity_type", "entity_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[int | None] = mapped_column(Integer)
    details: Mapped[dict | None] = mapped_column(JSON)
    ip_address: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)


# =====================================================================================================
# Stage 2 — Indian job-market intelligence and skill-gap engine
# =====================================================================================================
BATCH_STATUSES = ("importing", "extracting", "completed", "failed", "rolled_back")
DEDUP_STATUSES = ("unique", "exact_duplicate", "near_duplicate", "reviewed_distinct")
REQUIREMENT_LEVELS = ("required", "preferred", "unspecified")
JOB_SKILL_SOURCES = ("structured_required", "structured_preferred", "structured_listed", "nlp_description", "nlp_title")
MAPPING_STATUSES = ("proposed", "equivalent", "partial", "rejected")
GAP_CATEGORIES = ("covered", "partially_covered", "potentially_missing", "underemphasized", "uncertain", "not_applicable")
RECOMMENDATION_TYPES = ("add_topic", "expand_practical", "add_project", "review_prerequisite", "investigate_technology")
RECOMMENDATION_STATUSES = ("draft", "approved", "rejected")


class JobDataset(TimestampMixin, Base):
    """A named source of job records (e.g. a Naukri dump) with provenance and licence metadata."""
    __tablename__ = "job_datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    source_name: Mapped[str | None] = mapped_column(String(200))
    source_url: Mapped[str | None] = mapped_column(String(500))
    licence: Mapped[str | None] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text)
    default_country: Mapped[str | None] = mapped_column(String(100))
    collection_notes: Mapped[str | None] = mapped_column(Text)
    adapter: Mapped[str] = mapped_column(String(50), nullable=False, default="generic")
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


class JobImportBatch(Base):
    __tablename__ = "job_import_batches"
    __table_args__ = (CheckConstraint(_in("status", BATCH_STATUSES), name="ck_batches_status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("job_datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(300), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)
    column_mapping: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="importing")
    status_message: Mapped[str | None] = mapped_column(Text)
    total_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    imported_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rejected_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    warning_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    errors: Mapped[list | None] = mapped_column(JSON)       # row-level errors/warnings (capped)
    stats: Mapped[dict | None] = mapped_column(JSON)        # missingness, date coverage, unmapped skills
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rolled_back_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DuplicateCluster(Base):
    __tablename__ = "job_duplicate_clusters"

    id: Mapped[int] = mapped_column(primary_key=True)
    cluster_type: Mapped[str] = mapped_column(String(10), nullable=False)  # exact | near
    block_key: Mapped[str] = mapped_column(String(400), nullable=False, index=True)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    representative_id: Mapped[int | None] = mapped_column(Integer)
    similarity: Mapped[float | None] = mapped_column(Float)
    review_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")  # pending|confirmed|split
    reviewed_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class JobPosting(Base):
    """One job record exactly as imported (raw_record keeps the original row for provenance)."""
    __tablename__ = "job_postings"
    __table_args__ = (
        CheckConstraint(_in("dedup_status", DEDUP_STATUSES), name="ck_postings_dedup"),
        Index("ix_postings_eligible", "is_canonical", "excluded"),
        Index("ix_postings_state_city", "state", "city"),
        Index("ix_postings_observed", "observed_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("job_datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("job_import_batches.id", ondelete="CASCADE"), nullable=False, index=True)
    row_number: Mapped[int] = mapped_column(Integer, nullable=False)
    external_job_id: Mapped[str | None] = mapped_column(String(200), index=True)
    job_title: Mapped[str] = mapped_column(String(500), nullable=False)
    company_name: Mapped[str | None] = mapped_column(String(300))
    job_description: Mapped[str | None] = mapped_column(Text)
    required_skills: Mapped[str | None] = mapped_column(Text)
    preferred_skills: Mapped[str | None] = mapped_column(Text)
    listed_skills: Mapped[str | None] = mapped_column(Text)      # skills tagged without a requirement level
    education_requirement: Mapped[str | None] = mapped_column(String(500))
    experience_level: Mapped[str | None] = mapped_column(String(100))
    experience_min_years: Mapped[float | None] = mapped_column(Float)
    experience_max_years: Mapped[float | None] = mapped_column(Float)
    employment_type: Mapped[str | None] = mapped_column(String(100))
    country: Mapped[str | None] = mapped_column(String(100), index=True)
    state: Mapped[str | None] = mapped_column(String(100))
    city: Mapped[str | None] = mapped_column(String(150))
    location_raw: Mapped[str | None] = mapped_column(String(500))
    location_source: Mapped[str | None] = mapped_column(String(40))  # as_supplied | derived_from_city | dataset_default
    industry: Mapped[str | None] = mapped_column(String(300), index=True)
    job_role: Mapped[str | None] = mapped_column(String(300), index=True)
    role_category: Mapped[str | None] = mapped_column(String(300))
    salary_min: Mapped[float | None] = mapped_column(Float)
    salary_max: Mapped[float | None] = mapped_column(Float)
    salary_currency: Mapped[str | None] = mapped_column(String(10))
    posting_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    collection_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    observed_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # posting_date, else collection_date
    source_url: Mapped[str | None] = mapped_column(String(1000))
    source_dataset: Mapped[str | None] = mapped_column(String(200))
    licence_or_usage_terms: Mapped[str | None] = mapped_column(String(500))
    is_entry_level: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_internship: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    normalized_title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    normalized_company: Mapped[str | None] = mapped_column(String(300))
    quality_flags: Mapped[list | None] = mapped_column(JSON)
    dedup_status: Mapped[str] = mapped_column(String(20), nullable=False, default="unique")
    duplicate_cluster_id: Mapped[int | None] = mapped_column(ForeignKey("job_duplicate_clusters.id", ondelete="SET NULL"),
                                                            index=True)
    is_canonical: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    excluded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    excluded_reason: Mapped[str | None] = mapped_column(String(300))
    raw_record: Mapped[dict | None] = mapped_column(JSON)
    skills_extracted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class JobSkillEvidence(Base):
    __tablename__ = "job_skill_evidence"
    __table_args__ = (
        CheckConstraint(_in("requirement_level", REQUIREMENT_LEVELS), name="ck_jobev_level"),
        CheckConstraint(_in("source", JOB_SKILL_SOURCES), name="ck_jobev_source"),
        CheckConstraint(_in("review_status", REVIEW_STATUSES), name="ck_jobev_review"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_jobev_confidence"),
        Index("ix_jobev_skill_posting", "skill_id", "posting_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    posting_id: Mapped[int] = mapped_column(ForeignKey("job_postings.id", ondelete="CASCADE"), nullable=False, index=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(30), nullable=False)
    requirement_level: Mapped[str] = mapped_column(String(20), nullable=False, default="unspecified")
    matched_text: Mapped[str] = mapped_column(String(300), nullable=False)
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(20), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_ambiguous: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    review_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    reviewed_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    skill: Mapped[Skill] = relationship()


class SkillMapping(TimestampMixin, Base):
    """Proposed relation between an industry skill and a (different) curriculum skill, reviewed by a human.

    equivalent -> curriculum evidence fully counts for the industry skill
    partial    -> related but not the same (e.g. "Python basics" vs "ML engineering"): counts at a reduced weight
    rejected   -> not related; proposed -> not counted until reviewed (result marked uncertain)
    """
    __tablename__ = "skill_mappings"
    __table_args__ = (
        UniqueConstraint("industry_skill_id", "curriculum_skill_id", name="uq_skill_mapping_pair"),
        CheckConstraint(_in("status", MAPPING_STATUSES), name="ck_mapping_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    industry_skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True)
    curriculum_skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id", ondelete="CASCADE"), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(20), nullable=False)  # keyword | semantic | tfidf | manual
    similarity: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="proposed")
    note: Mapped[str | None] = mapped_column(Text)
    proposed_in_run_id: Mapped[int | None] = mapped_column(Integer)
    reviewed_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SavedAnalysisConfig(TimestampMixin, Base):
    __tablename__ = "saved_analysis_configs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    kind: Mapped[str] = mapped_column(String(20), nullable=False, default="gap")  # gap | cohort
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


class GapAnalysisRun(Base):
    __tablename__ = "gap_analysis_runs"
    __table_args__ = (CheckConstraint(_in("status", RUN_STATUSES), name="ck_gapruns_status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    scoring_version: Mapped[str] = mapped_column(String(20), nullable=False)
    university_id: Mapped[int | None] = mapped_column(ForeignKey("universities.id", ondelete="SET NULL"), index=True)
    programme_id: Mapped[int | None] = mapped_column(ForeignKey("programmes.id", ondelete="SET NULL"), index=True)
    document_ids: Mapped[list | None] = mapped_column(JSON)
    cohort_size: Mapped[int | None] = mapped_column(Integer)
    cohort_fingerprint: Mapped[str | None] = mapped_column(String(64))
    curriculum_fingerprint: Mapped[str | None] = mapped_column(String(64))
    summary: Mapped[dict | None] = mapped_column(JSON)
    warnings: Mapped[list | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class GapResult(Base):
    __tablename__ = "gap_results"
    __table_args__ = (
        UniqueConstraint("run_id", "skill_id", name="uq_gap_result_skill"),
        CheckConstraint(_in("category", GAP_CATEGORIES), name="ck_gap_category"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("gap_analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id", ondelete="CASCADE"), nullable=False)
    rank: Mapped[int | None] = mapped_column(Integer)
    demand_score: Mapped[float] = mapped_column(Float, nullable=False)
    demand_share: Mapped[float] = mapped_column(Float, nullable=False)
    jobs_with_skill: Mapped[int] = mapped_column(Integer, nullable=False)
    demand_components: Mapped[dict | None] = mapped_column(JSON)
    coverage_score: Mapped[float] = mapped_column(Float, nullable=False)
    coverage_components: Mapped[dict | None] = mapped_column(JSON)
    coverage_confidence: Mapped[str] = mapped_column(String(10), nullable=False)
    gap_score: Mapped[float] = mapped_column(Float, nullable=False)
    priority: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    category: Mapped[str] = mapped_column(String(30), nullable=False)
    match_method: Mapped[str] = mapped_column(String(20), nullable=False)  # exact | mapping_equivalent | mapping_partial | proposed | none
    matched_curriculum_skill_ids: Mapped[list | None] = mapped_column(JSON)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)

    skill: Mapped[Skill] = relationship()


class Recommendation(TimestampMixin, Base):
    __tablename__ = "recommendations"
    __table_args__ = (
        CheckConstraint(_in("rec_type", RECOMMENDATION_TYPES), name="ck_rec_type"),
        CheckConstraint(_in("status", RECOMMENDATION_STATUSES), name="ck_rec_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("gap_analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id", ondelete="CASCADE"), nullable=False)
    programme_id: Mapped[int | None] = mapped_column(ForeignKey("programmes.id", ondelete="SET NULL"))
    course_id: Mapped[int | None] = mapped_column(ForeignKey("courses.id", ondelete="SET NULL"))
    rec_type: Mapped[str] = mapped_column(String(30), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[dict | None] = mapped_column(JSON)
    confidence: Mapped[str] = mapped_column(String(10), nullable=False)
    limitations: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="draft")
    reviewer_note: Mapped[str | None] = mapped_column(Text)
    reviewed_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# --- Faculty workflow ---------------------------------------------------------------------------------------
FACULTY_STATUSES = ("queued", "processing", "done", "failed")


class FacultyAnalysis(TimestampMixin, Base):
    """One 'upload a syllabus → compare with industry → recommendations' request made by a faculty member."""
    __tablename__ = "faculty_analyses"
    __table_args__ = (
        CheckConstraint(_in("status", FACULTY_STATUSES), name="ck_faculty_status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("syllabus_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    course_name: Mapped[str] = mapped_column(String(300), nullable=False)
    institution: Mapped[str | None] = mapped_column(String(300))
    job_field: Mapped[str] = mapped_column(String(60), nullable=False, default="it")   # it | all | a job role
    entry_level: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")
    step: Mapped[str | None] = mapped_column(String(40))
    message: Mapped[str | None] = mapped_column(String(500))
    gap_run_id: Mapped[int | None] = mapped_column(ForeignKey("gap_analysis_runs.id", ondelete="SET NULL"))
    result: Mapped[dict | None] = mapped_column(JSON)          # the full comparison, computed once
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    document: Mapped[SyllabusDocument] = relationship()
