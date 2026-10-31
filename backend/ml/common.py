"""Shared helpers for training: load the job postings, extract skills, label job roles from titles.

Everything here is deterministic (fixed seeds) so `python -m ml.train_all` reproduces the shipped artifacts.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

BACKEND = Path(__file__).resolve().parent.parent
MARKET_DIR = BACKEND / "data" / "market"
ARTIFACTS = BACKEND / "artifacts"
TAXONOMY = BACKEND / "data" / "skill_taxonomy.json"
SEED = 42

# Requirement-level weights used for demand (required counts most).
LEVEL_CODE = {"preferred": 1, "unspecified": 2, "required": 3}
LEVEL_WEIGHT = {0: 0.0, 1: 0.5, 2: 0.75, 3: 1.0}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- Taxonomy ------------------------------------------------------------------------------------------------
def load_taxonomy():
    """Skill list + matcher built straight from data/skill_taxonomy.json (no database needed)."""
    from app.services.skill_extractor import AliasEntry, TaxonomyMatcher

    data = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    removed = {a.lower() for a in data.get("removed_aliases", [])}
    skills, entries = [], []
    for i, s in enumerate(data["skills"]):
        skills.append({"index": i, "name": s["name"], "category": s["category"]})
        entries.append(AliasEntry(i, s["name"], s["category"], s["name"], True, False, (), "flag"))
        for raw in s.get("aliases", []):
            spec = {"alias": raw} if isinstance(raw, str) else raw
            if spec["alias"].lower() in removed:
                continue
            entries.append(AliasEntry(i, s["name"], s["category"], spec["alias"], False,
                                      spec.get("case_sensitive", False), tuple(spec.get("context_terms") or ()),
                                      spec.get("ambiguity_policy", "flag")))
    return skills, TaxonomyMatcher(entries)


# --- Job roles (weak labels from job titles) ------------------------------------------------------------------
# Ordered: the first matching rule wins. Titles are only used to create training labels; the classifier itself
# learns from the description and skills, so it can label postings with unclear titles (and syllabus text).
ROLE_RULES: list[tuple[str, str]] = [
    ("Data Scientist / ML Engineer", r"data scien|machine learning|\bml\b|\bmlops\b|deep learning|\bnlp\b|computer vision"
                                     r"|\bai\b|artificial intelligence|gen ?ai|\bllm"),
    ("Data Engineer", r"data engineer|\betl\b|big data|data warehouse|databricks|snowflake|hadoop|spark developer"),
    ("Data Analyst / BI", r"data analyst|business intelligence|\bbi\b|power ?bi|tableau|data analytics|mis analyst"),
    ("Cybersecurity", r"cyber|security engineer|security analyst|information security|infosec|\bsoc\b|penetration"
                      r"|vapt|ethical hack"),
    ("DevOps / Cloud", r"devops|\bcloud\b|\bsre\b|site reliability|\baws\b|azure|kubernetes|platform engineer"),
    ("QA / Testing", r"\bqa\b|quality assurance|\btest(er|ing)?\b|\bsdet\b|automation engineer"),
    ("Mobile Developer", r"android|\bios\b|mobile (app|developer|engineer)|flutter|react native"),
    ("Web / Full Stack Developer", r"front ?end|full ?stack|web developer|web designer|\breact|angular|\bui developer"
                                   r"|javascript|\bmern\b|\bmean\b|php|wordpress|node"),
    ("Software Developer", r"software|developer|programmer|\bjava\b|python|\.net|dotnet|back ?end|\bsde\b|golang"
                           r"|c\+\+|application engineer|salesforce"),
    ("Network / IT Support", r"network|system admin|sysadmin|it support|desktop support|infrastructure|linux admin"
                             r"|it engineer|technical support"),
]
OTHER_ROLE = "Other (non-IT)"
ROLES = [r for r, _ in ROLE_RULES] + [OTHER_ROLE]
_ROLE_RE = [(r, re.compile(p, re.I)) for r, p in ROLE_RULES]


def role_from_title(title: str) -> str:
    for role, rx in _ROLE_RE:
        if rx.search(title or ""):
            return role
    return OTHER_ROLE


# --- Loading postings ------------------------------------------------------------------------------------------
def _clean(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()


def _pick_date(df: pd.DataFrame) -> pd.Series:
    posted = df["posting_date"] if "posting_date" in df else pd.Series([""] * len(df))
    collected = df["collection_date"] if "collection_date" in df else pd.Series([""] * len(df))
    return posted.where(posted.str.strip() != "", collected)


def market_files() -> list[Path]:
    """The job-posting files in data/market/ (.csv or compressed .csv.gz; if both exist, the .csv.gz is used)."""
    files = {}
    for path in sorted(MARKET_DIR.glob("*.csv")) + sorted(MARKET_DIR.glob("*.csv.gz")):
        files[path.name.removesuffix(".gz")] = path          # .csv.gz overrides a .csv of the same name
    return [files[k] for k in sorted(files)]


def load_postings() -> pd.DataFrame:
    frames = []
    for path in market_files():
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        frames.append(pd.DataFrame({
            "title": df.get("job_title", ""),
            "company": df.get("company_name", ""),
            "description": df.get("job_description", ""),
            "skills": df.get("key_skills", ""),
            "experience": df.get("experience", df.get("seniority_level", "")),
            "date": _pick_date(df),
            "source_file": path.name,
        }))
        log(f"loaded {path.name}: {len(df):,} rows")
    out = pd.concat(frames, ignore_index=True).fillna("")
    raw_rows = len(out)
    out = out[out["title"].str.strip() != ""].reset_index(drop=True)
    empty_titles = raw_rows - len(out)
    # exact duplicates (same title, company, skills and description) are removed
    key = (out["title"].str.lower().str.strip() + "|" + out["company"].str.lower().str.strip() + "|"
           + out["skills"].str.lower() + "|" + out["description"].str[:500].str.lower())
    h = key.map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
    before = len(out)
    out = out[~h.duplicated()].reset_index(drop=True)
    log(f"{before:,} postings, {before - len(out):,} exact duplicates removed → {len(out):,}")
    out.attrs.update(raw_rows=raw_rows, empty_titles=empty_titles, duplicates=before - len(out),
                     files={path.name: int(n) for path, n in zip(market_files(), [len(f) for f in frames])})
    return out


@dataclass
class ExtractedMarket:
    postings: pd.DataFrame
    skills: list[dict]
    levels: sparse.csr_matrix                  # postings × skills, value = LEVEL_CODE (0 = not asked)
    sentences: list[tuple[str, str]] = field(default_factory=list)   # (heading context, sentence) for model 2


def extract_market(postings: pd.DataFrame, requirement_model=None) -> ExtractedMarket:
    """Find taxonomy skills in every posting. With `requirement_model`, sentences that have no heading context
    get their requirement level from the trained classifier (model 2)."""
    from ml.requirement_classifier import segment_with_headings

    skills, matcher = load_taxonomy()
    from app.services.job_skills import _hits_in
    from app.services.job_import import split_skills

    rows, cols, vals = [], [], []
    n = len(postings)
    t0 = time.time()
    for i, (title, desc, tags) in enumerate(zip(postings["title"], postings["description"], postings["skills"])):
        best: dict[int, int] = {}

        def put(sid: int, level: str):
            code = LEVEL_CODE[level]
            if code > best.get(sid, 0):
                best[sid] = code

        for tag in split_skills(tags):
            for h in _hits_in(tag, matcher, "listed", "unspecified", 0.9):
                if not h.is_ambiguous:
                    put(h.skill_id, "unspecified")
        if desc:
            segs = segment_with_headings(desc)
            pending = [j for j, (ctx, _s) in enumerate(segs) if ctx == "none"]
            predicted = {}
            if requirement_model is not None and pending:
                labels = requirement_model.predict([segs[j][1] for j in pending])
                predicted = dict(zip(pending, labels))
            for j, (ctx, sent) in enumerate(segs):
                level = ctx if ctx != "none" else predicted.get(j, "neutral")
                level = {"neutral": "unspecified"}.get(level, level)
                for h in _hits_in(sent, matcher, "desc", level, 0.8):
                    if not h.is_ambiguous:
                        put(h.skill_id, level)
        for h in _hits_in(title, matcher, "title", "unspecified", 0.7):
            if not h.is_ambiguous:
                put(h.skill_id, "unspecified")
        for sid, code in best.items():
            rows.append(i)
            cols.append(sid)
            vals.append(code)
        if (i + 1) % 20000 == 0:
            log(f"  skills extracted for {i + 1:,}/{n:,} postings ({time.time() - t0:.0f}s)")
    m = sparse.csr_matrix((np.array(vals, dtype=np.int8), (rows, cols)), shape=(n, len(skills)))
    return ExtractedMarket(postings=postings, skills=skills, levels=m)


def skill_text(levels: sparse.csr_matrix, skills: list[dict]) -> list[str]:
    """Each posting's skills as a space-separated token string (used as classifier features)."""
    names = [re.sub(r"\W+", "_", s["name"].lower()).strip("_") for s in skills]
    out = []
    for i in range(levels.shape[0]):
        a, b = levels.indptr[i], levels.indptr[i + 1]
        out.append(" ".join("SK_" + names[j] for j in levels.indices[a:b]))
    return out


def summarise_counts(values) -> dict:
    return dict(Counter(values).most_common())
