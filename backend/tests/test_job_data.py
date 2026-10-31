"""Job data import: parsing, mapping, required/preferred skill detection. Fixtures are SYNTHETIC TEST DATA, plus
the first rows of the bundled real job files as a regression check."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.services.job_import import (
    ADAPTERS,
    parse_date,
    parse_experience,
    parse_location,
    parse_salary,
    prepare_rows,
    validate_mapping,
)
from app.services.job_skills import extract_posting_skills

ROOT = Path(__file__).resolve().parent.parent
FUTURE = (datetime.now(timezone.utc) + timedelta(days=60)).strftime("%Y-%m-%d")
RECENT = (datetime.now(timezone.utc) - timedelta(days=20)).strftime("%Y-%m-%d")
OLD = "2019-07-10"

JD_REQ_PREF = """SYNTHETIC TEST DATA. We are hiring a backend developer.
Responsibilities
Build services and review code with the team.
Required Skills:
Strong knowledge of Python and SQL.
Experience with Docker.
Good to have:
Kubernetes and AWS cloud (EC2) experience.
Exposure to React is a plus."""


def synthetic_rows():
    """SYNTHETIC TEST DATA rows covering valid, duplicate, near-duplicate, invalid and edge cases."""
    base = {"job_id": "", "title": "", "company": "", "description": "", "required": "", "preferred": "",
            "location": "", "experience": "", "salary": "", "posted": "", "url": "", "type": "Full-time",
            "education": "B.Tech"}
    rows = [
        dict(base, job_id="T1", title="Python Developer", company="Acme Analytics", description=JD_REQ_PREF,
             required="Python, SQL", preferred="Kubernetes", location="Bengaluru", experience="0 - 2 yrs",
             salary="3,00,000 - 6,00,000 PA.", posted=RECENT, url="https://jobs.example.test/T1"),
        # exact duplicate of T1 content (syndicated copy, different id)
        dict(base, job_id="T1-copy", title="Python Developer", company="Acme Analytics", description=JD_REQ_PREF,
             required="Python, SQL", preferred="Kubernetes", location="Bengaluru", experience="0 - 2 yrs",
             salary="3,00,000 - 6,00,000 PA.", posted=RECENT, url="https://mirror.example.test/T1"),
        # near duplicate: same title+company, tiny wording change
        dict(base, job_id="T2", title="Python Developer", company="Acme Analytics",
             description=JD_REQ_PREF.replace("We are hiring", "Now hiring"), required="Python, SQL",
             preferred="Kubernetes", location="Bengaluru", experience="0 - 2 yrs", posted=RECENT),
        dict(base, job_id="T3", title="Data Analyst Intern", company="Beta Retail",
             description="SYNTHETIC TEST DATA. Must have Excel spreadsheet skills and SQL. Tableau preferred.",
             location="Pune, Maharashtra", experience="Fresher", salary="15000 per month", posted=RECENT),
        dict(base, job_id="T4", title="Java Backend Engineer", company="Gamma Systems",
             description="SYNTHETIC TEST DATA. Java, Spring Boot and Hibernate. Microservices on AWS cloud.",
             location="Hyderabad", experience="5 - 8 yrs", salary="12-18 LPA", posted=OLD),
        dict(base, job_id="T5", title="Frontend Developer", company="Delta Web",
             description="SYNTHETIC TEST DATA. JavaScript, React and CSS. Required: HTML.",
             location="Mumbai", experience="2 - 5 yrs", posted=FUTURE),
        dict(base, job_id="T6", title="Machine Learning Engineer", company="Epsilon AI",
             description="SYNTHETIC TEST DATA. Mandatory: machine learning with TensorFlow and Python.",
             location="Atlantis City", experience="1-3 years", posted="not a date"),
        dict(base, job_id="T7", title="", company="No Title Co", description="SYNTHETIC TEST DATA. Missing title row."),
        dict(base, job_id="T8", title="Lorem ipsum test job", company="Zeta", description="lorem ipsum dolor sit amet",
             location="Chennai", salary="9,00,000 - 4,00,000 PA.", posted=RECENT),
        dict(base, job_id="T9", title="DevOps Engineer", company="Eta Cloud", description="",
             required="Docker | Kubernetes | Linux", location="Noida", experience="3 - 6 yrs", posted=RECENT),
    ]
    return rows


MAPPING = {"job_id": "job_id", "job_title": "title", "company_name": "company", "job_description": "description",
           "required_skills": "required", "preferred_skills": "preferred", "location": "location",
           "experience_level": "experience", "salary": "salary", "posting_date": "posted", "source_url": "url",
           "employment_type": "type", "education_requirement": "education"}






# --- Unit: parsing & mapping ----------------------------------------------------------------------
def test_parsers_handle_indian_formats():
    assert parse_salary("2,00,000 - 3,00,000 PA.") == (200000.0, 300000.0, "INR", None)
    assert parse_salary("12-18 LPA")[:2] == (1200000.0, 1800000.0)
    assert parse_salary("15000 per month")[:2] == (180000.0, 180000.0)
    assert parse_salary(" Not Disclosed by Recruiter ") == (None, None, None, None)
    assert parse_salary("Openings: 1") == (None, None, None, None)
    assert parse_salary("9,00,000 - 4,00,000 PA.")[3] == "salary minimum exceeds maximum"
    assert parse_experience("2 - 5 yrs") == (2.0, 5.0)
    assert parse_experience("Fresher") == (0.0, 1.0)
    assert parse_location("Pune,Pune") == ("Pune", "Maharashtra", "derived_from_city")
    assert parse_location("Bangalore") == ("Bengaluru", "Karnataka", "derived_from_city")
    assert parse_location("Kochi, Kerala") == ("Kochi", "Kerala", "as_supplied")
    assert parse_location("Atlantis City")[1] is None   # unknown city: no state invented
    assert parse_date("2019-07-05 01:46:07 +0000")[0].year == 2019
    assert parse_date("not a date") == (None, True)


def test_mapping_suggestion_and_validation():
    cols = list(synthetic_rows()[0].keys())
    generic = ADAPTERS["generic"].suggest_mapping(cols)
    assert generic["job_title"] == "title" and generic["company_name"] == "company"
    assert validate_mapping({"company_name": "company"}, cols) == ["Required field 'job_title' must be mapped to a column"]
    assert any("not in file" in e for e in validate_mapping({"job_title": "nope"}, cols))
    assert any("more than one" in e for e in validate_mapping({"job_title": "title", "company_name": "title"}, cols))




REAL_INDIA_2025 = ROOT / "data" / "market" / "job_postings_2025.csv.gz"
REAL_GLOBAL_2024 = ROOT / "data" / "market" / "job_postings_2024.csv.gz"


@pytest.mark.skipif(not (REAL_INDIA_2025.exists() and REAL_GLOBAL_2024.exists()), reason="latest samples not present")
def test_latest_real_datasets_map_and_prepare():
    import pandas as pd

    from app.services.job_import import ADAPTERS

    ad = ADAPTERS["generic"]
    india = pd.read_csv(REAL_INDIA_2025, dtype=str, nrows=300, keep_default_na=False)
    m = ad.suggest_mapping(list(india.columns))
    assert m["listed_skills"] == "key_skills" and m["job_description"] == "job_description"
    assert m["posting_date"] == "posting_date" and m["collection_date"] == "collection_date"
    prep = prepare_rows(india.to_dict("records"), m, ad.defaults(), {"name": "in", "licence": None, "default_country": "India"})
    assert prep.rejected == 0
    rec = prep.records[0]
    assert rec["posting_date"].year == 2025 and rec["collection_date"].date().isoformat() == "2025-10-03"
    assert rec["job_description"] and rec["company_name"] and rec["country"] == "India"

    glob = pd.read_csv(REAL_GLOBAL_2024, dtype=str, nrows=300, keep_default_na=False)
    m = ad.suggest_mapping(list(glob.columns))
    assert m["experience_level"] == "seniority_level" and m["country"] == "country"
    prep = prepare_rows(glob.to_dict("records"), m, ad.defaults(), {"name": "gl", "licence": None, "default_country": None})
    rec = prep.records[0]
    assert rec["posting_date"] is None and rec["collection_date"].year == 2024     # snapshot date, not a posting date
    assert rec["experience_min_years"] is None                                     # seniority labels carry no years
    assert rec["company_name"] is None and rec["listed_skills"]


def test_required_preferred_distinction_and_sources(client):
    from app.database import SessionLocal
    from app.services.taxonomy import get_matchers

    db = SessionLocal()
    matcher, _, _ = get_matchers(db)
    db.close()
    hits, _ = extract_posting_skills({"job_title": "Python Developer", "job_description": JD_REQ_PREF,
                                      "required_skills": "Python, SQL", "preferred_skills": "Kubernetes"}, matcher)
    by = {(h.skill_id, h.source): h for h in hits}
    names = {}
    for h in hits:
        names.setdefault(h.source, {})[h.matched_text.lower()] = h
    desc = names["nlp_description"]
    assert desc["python"].requirement_level == "required" and desc["sql"].requirement_level == "required"
    assert desc["docker"].requirement_level == "required"
    assert desc["kubernetes"].requirement_level == "preferred"
    assert desc["react"].requirement_level == "preferred"         # "is a plus"
    assert names["structured_required"]["python"].requirement_level == "required"
    assert names["structured_preferred"]["kubernetes"].requirement_level == "preferred"
    assert "python" in names["nlp_title"] and names["nlp_title"]["python"].requirement_level == "unspecified"
    assert len(by) == len(hits)
    # Responsibility text without cues stays unspecified
    hits2, _ = extract_posting_skills({"job_title": "x", "job_description": "You will work with Git daily."}, matcher)
    assert hits2[0].requirement_level == "unspecified"


# --- API: import flow ------------------------------------------------------------------------------------
























