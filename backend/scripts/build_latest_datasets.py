"""Build the bundled industry job-posting files (data/market/*.csv.gz) from their public sources (reproducible).

1. India — Naukri.com, collected ~3 Oct 2025 (97,929 postings, with full job descriptions)
   Kaggle: shivamshrivastava21/indian-job-market-dataset-2025-2026, mirrored in the public GitHub repo
   Annusharma30/Job-Market-Intelligence-and-Skill-Analysis (data/raw/indian-job-market-dataset-2025.csv).
   The file has no absolute dates, only "N Days Ago". The collection date is estimated as 2025-10-03: for the
   jobIds whose first six digits form a DDMMYY date in 2025, that date + "N days ago" equals 2025-10-03 for
   94% of them. posting_date = 2025-10-03 minus N days ("Today", "Just Now", "Few Hours Ago" = 0).

2. Global — LinkedIn, snapshot scraped January 2024 (US, UK, Canada, Australia)
   Kaggle: asaniczka/1-3m-linkedin-jobs-and-skills-2024 (ODC-By 1.0); derived Parquet files from the GitHub
   release Agnesor/linkedin-skills@data-v1 (postings, skills, posting_skills; SHA-256 verified against
   data.sha256). The derived data has title, country, seniority level and work mode — no company, description
   or posting date. A fixed-seed random sample of 10,000 postings per country is taken (40,000 total) so that
   no single country dominates; the full file has 1,150,307 postings.

Usage:
  python -m scripts.build_latest_datasets --india <indian-job-market-dataset-2025.csv> --linkedin-dir <dir with parquet>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "market"
INDIA_OUT = OUT / "job_postings_2025.csv.gz"
GLOBAL_OUT = OUT / "job_postings_2024.csv.gz"
INDIA_COLLECTED = pd.Timestamp("2025-10-03")
PER_COUNTRY = 10_000
SEED = 42


def _days_ago(s: str) -> int | None:
    s = (s or "").strip().lower()
    if s in {"today", "just now", "few hours ago"}:
        return 0
    m = re.match(r"(\d+)\s+days?\s+ago", s)
    return int(m.group(1)) if m else None


_BLOCK = re.compile(r"<\s*(br|/p|/li|/div|/h\d|/tr|li|p|div|h\d)\b[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")


def html_to_text(s: str) -> str:
    """Naukri descriptions are HTML fragments; keep the text and line breaks, drop the markup."""
    import html

    if "<" not in s and "&" not in s:
        return s
    s = _BLOCK.sub("\n", s)
    s = html.unescape(_TAG.sub(" ", s)).replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\s*\n\s*", "\n", s).strip()


def build_india(src: Path) -> pd.DataFrame:
    d = pd.read_csv(src, dtype=str, keep_default_na=False)
    ago = [_days_ago(x) for x in d["jobUploaded"]]
    posted = [(INDIA_COLLECTED - pd.Timedelta(days=a)).date().isoformat() if a is not None else "" for a in ago]
    out = pd.DataFrame({
        "job_id": d["jobId"],
        "job_title": d["title"],
        "company_name": d["companyName"],
        "job_description": d["jobDescription"].map(html_to_text),
        "key_skills": d["tagsAndSkills"],
        "experience": d["experience"],
        "salary": d["salary"],
        "salary_currency": d["currency"],
        "location": d["location"],
        "posting_date": posted,
        "posted_as_published": d["jobUploaded"],
        "collection_date": INDIA_COLLECTED.date().isoformat(),
        "country": "India",
    })
    return out


def build_global(src_dir: Path) -> pd.DataFrame:
    posts = pd.read_parquet(src_dir / "postings.parquet")
    skills = pd.read_parquet(src_dir / "skills.parquet")
    link = pd.read_parquet(src_dir / "posting_skills.parquet")
    sample = posts.groupby("country").sample(n=PER_COUNTRY, random_state=SEED)
    link = link[link["pid"].isin(sample["pid"])].merge(skills, on="sid")
    joined = link.groupby("pid")["skill"].agg(lambda s: ", ".join(sorted(set(x.replace(";", " ") for x in s))))
    sample = sample.merge(joined.rename("skills"), left_on="pid", right_index=True, how="left")
    return pd.DataFrame({
        "job_id": "li-" + sample["pid"].astype(str),
        "job_title": sample["title_norm"],
        "country": sample["country"],
        "seniority_level": sample["level"],
        "work_mode": sample["job_type"],
        "key_skills": sample["skills"].fillna(""),
        "collection_date": "2024-01-15",   # snapshot scraped in January 2024; exact day not in the derived data
    }).sort_values("job_id", key=lambda s: s.str[3:].astype(int))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--india", type=Path, required=True)
    ap.add_argument("--linkedin-dir", type=Path, required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    india = build_india(a.india)
    india.to_csv(INDIA_OUT, index=False)
    print(f"{INDIA_OUT.name}: {len(india)} rows, {INDIA_OUT.stat().st_size/1e6:.1f} MB")
    glob = build_global(a.linkedin_dir)
    glob.to_csv(GLOBAL_OUT, index=False)
    print(f"{GLOBAL_OUT.name}: {len(glob)} rows, {GLOBAL_OUT.stat().st_size/1e6:.1f} MB")
    print(glob["country"].value_counts().to_string())


if __name__ == "__main__":
    main()
