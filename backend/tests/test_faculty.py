"""The faculty workflow end to end: upload → read PDF → compare using the trained models → recommendations."""
import time

from tests.conftest import FIXTURES, unique_pdf


def _upload(client, data: bytes, name="syllabus.pdf", course="Machine Learning and Data Analytics", field="it",
            entry_level=False):
    return client.post("/api/analyses", data={"course_name": course, "institution": "Example University",
                                              "job_field": field, "entry_level": str(entry_level).lower()},
                       files={"file": (name, data, "application/pdf")})


def _wait(client, aid: int, timeout=90) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        a = client.get(f"/api/analyses/{aid}").json()
        if a["status"] in ("done", "failed"):
            return a
        time.sleep(0.3)
    raise AssertionError(f"analysis {aid} did not finish: {a}")


def test_health_and_market_status(client, market):
    assert client.get("/api/health").json()["status"] == "ok"
    m = client.get("/api/market").json()
    assert m["state"] == "ready" and m["postings"] > 100000 and m["period"]
    values = [f["value"] for f in m["job_fields"]]
    assert values[0] == "it" and values[-1] == "all" and "Data Scientist / ML Engineer" in values
    assert "Other (non-IT)" not in values


def test_model_report_endpoint(client, market):
    r = client.get("/api/models").json()
    for key in ("role_classifier", "requirement_classifier"):
        rep = r[key]
        assert rep["best_model"] in rep["models"] and len(rep["models"]) >= 4
        best = rep["models"][rep["best_model"]]
        assert 0 < best["accuracy"] <= 1 and len(best["confusion_matrix"]) == len(rep["labels"])
    assert r["skill_recommender"]["apriori"]["rule_count"] > 0 and r["skill_recommender"]["kmeans"]["k"] >= 2
    p = r["pipeline"]
    assert p["datasets"]["raw_rows"] > 100000 and len(p["preprocessing"]) >= 8
    assert {t["model"] for t in p["training"]} == {"Model 1", "Model 2", "Model 3"}


def test_upload_validation(client, market):
    r = _upload(client, b"hello", name="notes.txt")
    assert r.status_code == 422 and "PDF" in r.json()["error"]["message"]
    r = _upload(client, (FIXTURES / "not_a_pdf.pdf").read_bytes())
    assert r.status_code == 422 and "PDF signature" in r.json()["error"]["message"]
    r = _upload(client, unique_pdf(FIXTURES / "sample_syllabus.pdf"), course="   ")
    assert r.status_code == 422
    r = _upload(client, unique_pdf(FIXTURES / "sample_syllabus.pdf"), field="medicine")
    assert r.status_code == 422 and "job field" in r.json()["error"]["message"]


def test_upload_size_limit(client, market, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_mb", 0.001)
    r = _upload(client, unique_pdf(FIXTURES / "sample_syllabus.pdf"))
    assert r.status_code == 413


def test_full_analysis_uses_the_ml_models_and_is_fast(client, market):
    t0 = time.time()
    r = _upload(client, unique_pdf(FIXTURES / "sample_syllabus.pdf"))
    assert r.status_code == 201, r.text
    a = _wait(client, r.json()["id"])
    assert a["status"] == "done", a
    assert time.time() - t0 < 30
    s = a["summary"]
    assert s["skills_taught"] >= 10 and s["courses"] >= 1 and s["industry_skills_compared"] == 50
    assert a["market"]["postings_compared"] > 10000
    by = {x["skill"]: x for x in a["skills"]}
    assert by["Python"]["status_key"] in ("covered", "partially_covered") and "practical" in by["Python"]["in_syllabus_as"]
    assert all(0 <= x["required_pct"] <= 100 for x in a["skills"])
    missing = [x for x in a["skills"] if x["status_key"] == "potentially_missing"]
    assert missing and all(x["coverage"] == 0 for x in missing)
    recs = a["recommendations"]
    assert recs and all(r["example_jobs"] for r in recs)
    assert any(r["rule"] for r in recs)                                         # MODEL 3a explains some of them
    assert {r["kind"] for r in recs} <= {"add", "practice", "strengthen"}
    assert a["roles"][0]["role"] == "Data Scientist / ML Engineer"              # MODEL 1 on an ML syllabus
    assert all("role" in c for c in a["courses"])
    assert a["bundles"] and all(0 <= b["readiness"] <= 100 for b in a["bundles"])   # MODEL 3b
    assert a["models"]["role_classifier"]["algorithm"]
    assert s["match_pct"] == round(100 * (s["well_covered"] + 0.5 * s["partly_covered"]) / s["industry_skills_compared"])
    hist = client.get("/api/analyses").json()
    assert hist[0]["id"] == a["id"] and hist[0]["summary"]["match_pct"] == s["match_pct"] and "skills" not in hist[0]


def test_target_role_and_entry_level_narrow_the_comparison(client, market):
    data = unique_pdf(FIXTURES / "sample_syllabus.pdf")
    it = _wait(client, _upload(client, data).json()["id"])
    ds = _wait(client, _upload(client, data, field="Data Scientist / ML Engineer").json()["id"])
    fresh = _wait(client, _upload(client, data, entry_level=True).json()["id"])
    everything = _wait(client, _upload(client, data, field="all").json()["id"])
    n = {k: v["market"]["postings_compared"] for k, v in
         {"it": it, "ds": ds, "fresh": fresh, "all": everything}.items()}
    assert n["ds"] < n["it"] < n["all"] and n["fresh"] < n["it"]
    assert ds["job_field_label"] == "Data Scientist / ML Engineer" and fresh["entry_level"] is True


def test_same_pdf_twice_reuses_the_document(client, market):
    data = unique_pdf(FIXTURES / "sample_syllabus.pdf")
    a1 = _wait(client, _upload(client, data).json()["id"])
    a2 = _wait(client, _upload(client, data, course="Same file, second time").json()["id"])
    assert a1["status"] == a2["status"] == "done"
    assert a1["summary"]["skills_taught"] == a2["summary"]["skills_taught"]
    assert a2["course_name"] == "Same file, second time"


def test_unreadable_pdf_fails_clearly_and_can_be_retried_and_removed(client, market):
    a = _wait(client, _upload(client, unique_pdf(FIXTURES / "blank.pdf")).json()["id"])
    assert a["status"] == "failed" and a["message"]
    r = client.post(f"/api/analyses/{a['id']}/retry")
    assert r.status_code == 200
    again = _wait(client, a["id"])
    assert again["status"] == "failed"
    assert client.post(f"/api/analyses/{a['id']}/retry").status_code == 200
    _wait(client, a["id"])
    assert client.delete(f"/api/analyses/{a['id']}").status_code == 204
    assert client.get(f"/api/analyses/{a['id']}").status_code == 404
    assert a["id"] not in [x["id"] for x in client.get("/api/analyses").json()]


def test_damaged_pdf_gets_a_clear_message(client, market):
    from tests.test_pdf_and_structure import damaged_pdf_bytes

    a = _wait(client, _upload(client, damaged_pdf_bytes(), name="damaged.pdf").json()["id"])
    assert a["status"] == "failed"
    assert "blank or damaged" in a["message"] and "is a scanned image" not in a["message"]


def test_retry_only_for_failed(client, market):
    a = _wait(client, _upload(client, unique_pdf(FIXTURES / "sample_syllabus.pdf")).json()["id"])
    assert client.post(f"/api/analyses/{a['id']}/retry").status_code == 409


def test_migrations_upgrade_and_downgrade(tmp_path):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    from app.config import BACKEND_DIR
    from app.main import run_migrations

    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    run_migrations(url)
    tables = set(inspect(create_engine(url)).get_table_names())
    assert {"faculty_analyses", "syllabus_documents", "curriculum_skill_evidence"} <= tables
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.attributes["configure_logger"] = False
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.downgrade(cfg, "base")
    assert set(inspect(create_engine(url)).get_table_names()) <= {"alembic_version"}
