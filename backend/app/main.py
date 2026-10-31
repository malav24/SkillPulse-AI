"""SkillPulse AI — FastAPI entry point (simple faculty workflow)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import database
from .config import BACKEND_DIR, get_settings
from .routers import faculty as faculty_router
from .services import faculty
from .services.market_model import market_ready
from .services.taxonomy import seed_taxonomy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("skillpulse")


def run_migrations(url: str | None = None) -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.attributes["configure_logger"] = False
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url or get_settings().database_url)
    command.upgrade(cfg, "head")


def bootstrap() -> None:
    """Seed the skill taxonomy (idempotent)."""
    db = database.SessionLocal()
    try:
        result = seed_taxonomy(db)
        if result["skills_created"] or result["aliases_created"]:
            log.info("Taxonomy seeded: %s", result)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    Path(s.upload_dir).mkdir(parents=True, exist_ok=True)
    if s.auto_migrate:
        run_migrations()
    bootstrap()
    ok, err = market_ready()             # loads the trained models + market index (about a second)
    if not ok:
        log.warning("%s", err)
    faculty.warm_up()
    faculty.resume_unfinished()
    yield


app = FastAPI(
    title="SkillPulse AI API",
    version="3.0.0",
    description="Faculty upload a syllabus PDF; SkillPulse finds its courses and skills, compares them with real "
                "industry job postings and recommends which skills to add or strengthen.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


def _error(status: int, code: str, message: str, details=None, headers=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body, headers=headers)


_CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 409: "conflict",
          410: "gone", 413: "payload_too_large", 422: "validation_error"}


@app.exception_handler(StarletteHTTPException)
async def http_exc_handler(request: Request, exc: StarletteHTTPException):
    detail = exc.detail
    details = None
    if isinstance(detail, dict):
        message = detail.get("message", "Request failed")
        details = {k: v for k, v in detail.items() if k != "message"} or None
    else:
        message = str(detail)
    return _error(exc.status_code, _CODES.get(exc.status_code, "error"), message, details,
                  getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errs = []
    for e in exc.errors():
        loc = [str(p) for p in e.get("loc", []) if p not in ("body", "query", "path")]
        errs.append({"field": ".".join(loc) or None, "message": e.get("msg", "").replace("Value error, ", "")})
    first = errs[0] if errs else {"field": None, "message": "Invalid request"}
    msg = f"{first['field']}: {first['message']}" if first["field"] else first["message"]
    return _error(422, "validation_error", msg, errs)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error(500, "internal_error", "An unexpected error occurred. The issue has been logged.")


app.include_router(faculty_router.router)


@app.get("/", include_in_schema=False)
def root():
    return {"name": "SkillPulse AI API", "docs": "/docs", "health": "/api/health"}


