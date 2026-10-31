"""Application configuration loaded from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(BACKEND_DIR / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SkillPulse AI"
    environment: str = "development"

    # Database: PostgreSQL in production, SQLite fallback for local development.
    database_url: str = f"sqlite:///{(BACKEND_DIR / 'storage' / 'skillpulse.db').as_posix()}"
    auto_migrate: bool = True

    # Uploads
    upload_dir: str = str(BACKEND_DIR / "storage" / "uploads")
    max_upload_mb: float = 20.0

    # Extraction
    min_page_chars: int = 25          # below this a page is treated as empty / scanned
    min_document_chars: int = 150     # below this the document has insufficient text
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = 300
    tesseract_cmd: str | None = None  # path to tesseract binary if not on PATH

    # NLP
    taxonomy_path: str = str(BACKEND_DIR / "data" / "skill_taxonomy.json")
    parsing_rules_path: str = str(BACKEND_DIR / "data" / "parsing_rules.json")
    spacy_model: str = "en_core_web_sm"   # optional; falls back to a blank English pipeline
    semantic_enabled: bool = True
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_threshold: float = 0.62
    fuzzy_threshold: float = 0.78

    # Trained models + pre-computed job-market index (built by `python -m ml.train_all`)
    artifacts_dir: str = str(BACKEND_DIR / "artifacts")

    # Job-market analytics
    stale_after_days: int = 365          # records older than this are flagged "stale" (not removed)
    min_cohort_size: int = 30            # below this, demand results carry a small-sample warning
    min_cohort_size_hard: int = 10       # below this, gap categories become "uncertain"
    near_duplicate_threshold: float = 0.85

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
