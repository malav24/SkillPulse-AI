"""Test configuration: isolated SQLite DB and upload dir. Uses the trained models in backend/artifacts/."""
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="skillpulse-test-"))
os.environ.update({
    # Set TEST_DATABASE_URL to run the suite against PostgreSQL (use an empty, disposable database).
    "DATABASE_URL": os.environ.get("TEST_DATABASE_URL", f"sqlite:///{(_TMP / 'test.db').as_posix()}"),
    "UPLOAD_DIR": str(_TMP / "uploads"),
    "SEMANTIC_ENABLED": "false",    # keep tests deterministic and offline
    "AUTO_MIGRATE": "true",
})

FIXTURES = ROOT / "tests" / "fixtures"
if not (FIXTURES / "sample_syllabus.pdf").exists():
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_fixtures import build  # noqa: E402

    build(FIXTURES)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture(scope="session")
def market(client):
    """The real trained models and market index shipped in backend/artifacts/."""
    from app.services.faculty import market_status

    st = market_status()
    if st["state"] != "ready":
        pytest.skip(st["message"])
    return st


def unique_pdf(path: Path) -> bytes:
    """Fixture bytes with a unique trailing comment, so each upload is a new document."""
    return path.read_bytes() + f"\n% test-nonce {uuid.uuid4().hex}\n".encode()
