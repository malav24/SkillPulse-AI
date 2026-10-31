"""Job-market dataset import: reading, column mapping, validation and normalisation.

Adding a new Indian dataset = writing a small `JobSourceAdapter` subclass (column synonyms and/or a
row transform) and registering it in ADAPTERS. Permitted APIs can implement `JobAPIAdapter`.
Fields absent from a source stay NULL — nothing is invented. Derived values (e.g. state from city)
are labelled with their provenance.
"""
from __future__ import annotations

import hashlib
import io
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

CANONICAL_FIELDS = [
    "job_id", "job_title", "company_name", "job_description", "required_skills", "preferred_skills",
    "listed_skills", "education_requirement", "experience_level", "employment_type", "country", "state",
    "city", "location", "industry", "job_role", "role_category", "salary", "salary_min", "salary_max",
    "salary_currency", "posting_date", "source_url", "collection_date", "source_dataset",
    "licence_or_usage_terms",
]
FIELD_HELP = {
    "listed_skills": "Skills tagged on the posting without saying whether they are required or preferred",
    "location": "Free-text location (city / 'City, State'); split into city and state where possible",
    "salary": "Free-text salary range (e.g. '3,00,000 - 6,00,000 PA.' or '5-8 LPA'); parsed into min/max",
    "job_role": "Specific role label (e.g. 'Software Developer')",
    "role_category": "Broad role family (e.g. 'Programming & Design')",
}
REQUIRED_FIELDS = ["job_title"]
MAX_STORED_ISSUES = 2000

_SYNONYMS: dict[str, list[str]] = {
    "job_id": ["job_id", "jobid", "id", "uniq_id", "uniq id", "posting_id", "job id", "unique id"],
    "job_title": ["job_title", "jobtitle", "title", "job title", "position", "designation", "role title"],
    "company_name": ["company_name", "company", "employer", "organisation", "organization", "company name"],
    "job_description": ["job_description", "jobdescription", "description", "job description", "jd", "details"],
    "required_skills": ["required_skills", "required skills", "required", "must have skills", "mandatory skills"],
    "preferred_skills": ["preferred_skills", "preferred skills", "preferred", "good to have", "nice to have skills", "desired skills"],
    "listed_skills": ["skills", "key skills", "key_skills", "keyskills", "tags", "skill tags"],
    "education_requirement": ["education", "education_requirement", "qualification", "qualifications", "degree"],
    "experience_level": ["experience", "experience_level", "job experience required", "experience required",
                         "exp", "experience_required", "work experience", "seniority_level", "seniority"],
    "employment_type": ["employment_type", "job type", "job_type", "employment type", "work type", "type"],
    "country": ["country"],
    "state": ["state", "region", "province"],
    "city": ["city", "town"],
    "location": ["location", "job_location", "joblocation_address", "job location", "locations"],
    "industry": ["industry", "sector"],
    "job_role": ["role", "job_role", "job role"],
    "role_category": ["role category", "role_category", "functional area", "function", "category", "job_category"],
    "salary": ["salary", "job salary", "payrate", "pay", "ctc", "salary_range", "salary range", "compensation"],
    "salary_min": ["salary_min", "min_salary", "minimum salary"],
    "salary_max": ["salary_max", "max_salary", "maximum salary"],
    "salary_currency": ["salary_currency", "currency"],
    "posting_date": ["posting_date", "postdate", "posted_date", "posted", "date posted", "posted on", "post date", "date"],
    "source_url": ["source_url", "url", "link", "job_url", "apply link"],
    "collection_date": ["collection_date", "crawl timestamp", "crawl_timestamp", "scraped_at", "collected_at"],
    "source_dataset": ["source_dataset", "source", "site_name", "site"],
    "licence_or_usage_terms": ["licence", "license", "licence_or_usage_terms", "usage terms"],
}

# Major Indian cities -> state/UT (used only to derive state when a source gives just a city).
CITY_STATE = {
    "bengaluru": "Karnataka", "bangalore": "Karnataka", "mysuru": "Karnataka", "mysore": "Karnataka",
    "mangaluru": "Karnataka", "mangalore": "Karnataka", "hubli": "Karnataka", "belgaum": "Karnataka",
    "mumbai": "Maharashtra", "navi mumbai": "Maharashtra", "thane": "Maharashtra", "pune": "Maharashtra",
    "nagpur": "Maharashtra", "nashik": "Maharashtra", "aurangabad": "Maharashtra",
    "hyderabad": "Telangana", "secunderabad": "Telangana", "warangal": "Telangana",
    "chennai": "Tamil Nadu", "coimbatore": "Tamil Nadu", "madurai": "Tamil Nadu", "trichy": "Tamil Nadu",
    "tiruchirappalli": "Tamil Nadu", "salem": "Tamil Nadu", "vellore": "Tamil Nadu",
    "kolkata": "West Bengal", "howrah": "West Bengal", "durgapur": "West Bengal", "siliguri": "West Bengal",
    "delhi": "Delhi", "new delhi": "Delhi", "delhi ncr": "Delhi", "gurgaon": "Haryana", "gurugram": "Haryana",
    "faridabad": "Haryana", "panchkula": "Haryana", "noida": "Uttar Pradesh", "greater noida": "Uttar Pradesh",
    "ghaziabad": "Uttar Pradesh", "lucknow": "Uttar Pradesh", "kanpur": "Uttar Pradesh", "agra": "Uttar Pradesh",
    "varanasi": "Uttar Pradesh", "allahabad": "Uttar Pradesh", "prayagraj": "Uttar Pradesh",
    "ahmedabad": "Gujarat", "vadodara": "Gujarat", "baroda": "Gujarat", "surat": "Gujarat", "rajkot": "Gujarat",
    "gandhinagar": "Gujarat", "chandigarh": "Chandigarh", "mohali": "Punjab", "ludhiana": "Punjab",
    "amritsar": "Punjab", "jalandhar": "Punjab", "jaipur": "Rajasthan", "jodhpur": "Rajasthan",
    "udaipur": "Rajasthan", "kota": "Rajasthan", "indore": "Madhya Pradesh", "bhopal": "Madhya Pradesh",
    "gwalior": "Madhya Pradesh", "jabalpur": "Madhya Pradesh", "raipur": "Chhattisgarh",
    "bhubaneswar": "Odisha", "cuttack": "Odisha", "patna": "Bihar", "ranchi": "Jharkhand",
    "jamshedpur": "Jharkhand", "guwahati": "Assam", "kochi": "Kerala", "cochin": "Kerala", "ernakulam": "Kerala",
    "thiruvananthapuram": "Kerala", "trivandrum": "Kerala", "kozhikode": "Kerala", "calicut": "Kerala",
    "thrissur": "Kerala", "visakhapatnam": "Andhra Pradesh", "vizag": "Andhra Pradesh",
    "vijayawada": "Andhra Pradesh", "guntur": "Andhra Pradesh", "tirupati": "Andhra Pradesh",
    "dehradun": "Uttarakhand", "haridwar": "Uttarakhand", "shimla": "Himachal Pradesh", "goa": "Goa",
    "panaji": "Goa", "jammu": "Jammu and Kashmir", "srinagar": "Jammu and Kashmir",
    "puducherry": "Puducherry", "pondicherry": "Puducherry",
}
CITY_ALIASES = {"bangalore": "Bengaluru", "gurugram": "Gurgaon", "baroda": "Vadodara", "cochin": "Kochi",
                "trivandrum": "Thiruvananthapuram", "calicut": "Kozhikode", "vizag": "Visakhapatnam",
                "pondicherry": "Puducherry", "new delhi": "Delhi"}
_NOT_DISCLOSED = re.compile(r"not\s+disclosed|not\s+specified|negotiable|best in (the )?industry|competitive", re.I)
_INTERNSHIP = re.compile(r"\b(intern|internship|trainee|apprentice)\b", re.I)
_FRESHER = re.compile(r"\b(fresher|freshers|graduate trainee|entry[\s-]level)\b", re.I)


# --- Adapters -------------------------------------------------------------------------------------
class JobSourceAdapter:
    """Maps a source's columns onto canonical fields. Subclass to support a new dataset format."""

    name = "generic"
    label = "Generic CSV / Excel (auto-detected column names)"
    description = "Suggests a mapping from common column-name synonyms; review it before importing."
    synonyms: dict[str, list[str]] = _SYNONYMS

    def detect(self, columns: list[str]) -> float:
        return 0.1

    def suggest_mapping(self, columns: list[str]) -> dict[str, str]:
        """Returns {canonical_field: source_column}."""
        norm = {_norm_col(c): c for c in columns}
        mapping: dict[str, str] = {}
        used: set[str] = set()
        for fld, syns in self.synonyms.items():
            for syn in syns:
                col = norm.get(_norm_col(syn))
                if col and col not in used:
                    mapping[fld] = col
                    used.add(col)
                    break
        return mapping

    def defaults(self) -> dict[str, Any]:
        return {}


class NaukriPromptCloudAdapter(JobSourceAdapter):
    """PromptCloud/DataStock 'Jobs on Naukri.com' sample (Uniq Id, Crawl Timestamp, Key Skills, ...)."""

    name = "naukri_promptcloud"
    label = "Naukri.com (PromptCloud / DataStock export)"
    description = ("Key Skills are employer tags without a required/preferred level, so they import as listed "
                   "skills. Crawl Timestamp is the collection date, not the posting date.")
    synonyms = {
        **_SYNONYMS,
        "job_id": ["uniq id"], "collection_date": ["crawl timestamp"], "listed_skills": ["key skills"],
        "experience_level": ["job experience required"], "salary": ["job salary"], "location": ["location"],
        "role_category": ["role category"], "job_role": ["role"], "industry": ["industry"],
        "posting_date": [], "source_dataset": [],
    }

    def detect(self, columns: list[str]) -> float:
        cols = {_norm_col(c) for c in columns}
        return 0.95 if {"uniqid", "crawltimestamp", "keyskills"} <= cols else 0.0

    def defaults(self) -> dict[str, Any]:
        return {"country": "India", "salary_currency": "INR", "source_dataset": "naukri.com"}


ADAPTERS: dict[str, JobSourceAdapter] = {a.name: a for a in (JobSourceAdapter(), NaukriPromptCloudAdapter())}


class JobAPIAdapter(ABC):
    """Interface for permitted job APIs. Implement `fetch` to yield raw dict rows, then reuse the same
    mapping/validation pipeline (rows go through `prepare_rows` exactly like file rows).
    No API adapters are bundled: only use APIs whose terms permit this kind of analysis."""

    name: str = "api"
    licence: str = ""

    @abstractmethod
    def fetch(self, **params: Any) -> Iterator[dict[str, Any]]:
        raise NotImplementedError


def detect_adapter(columns: list[str]) -> JobSourceAdapter:
    return max(ADAPTERS.values(), key=lambda a: a.detect(columns))


# --- Reading --------------------------------------------------------------------------------------
def _norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def read_table(data: bytes, filename: str):
    import pandas as pd

    name = filename.lower()
    if name.endswith((".xlsx", ".xlsm")):
        df = pd.read_excel(io.BytesIO(data), dtype=str, engine="openpyxl")
    elif name.endswith(".csv") or name.endswith(".tsv"):
        sep = "\t" if name.endswith(".tsv") else ","
        last: Exception | None = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(data), dtype=str, sep=sep, encoding=enc, keep_default_na=False)
                break
            except UnicodeDecodeError as exc:
                last = exc
        else:
            raise ValueError(f"Could not decode the file: {last}")
    else:
        raise ValueError("Unsupported file type. Upload .csv, .tsv or .xlsx")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("The file has duplicate column names: " + ", ".join(df.columns[df.columns.duplicated()]))
    return df.fillna("")


# --- Parsing helpers ------------------------------------------------------------------------------
def clean(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s or None


def norm_text(s: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#.]+", " ", (s or "").lower())).strip()


def split_skills(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"\s*[|;,\n•]\s*", raw)
    seen, out = set(), []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip(" .-*")
        if p and p.lower() not in seen and len(p) <= 120:
            seen.add(p.lower())
            out.append(p)
    return out


def parse_date(raw: str | None) -> tuple[datetime | None, bool]:
    """Returns (datetime_utc, invalid_flag)."""
    if not raw:
        return None, False
    import pandas as pd

    try:
        ts = pd.to_datetime(raw, utc=True, dayfirst=bool(re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{4}", raw)))
    except (ValueError, TypeError, OverflowError):
        return None, True
    if pd.isna(ts):
        return None, True
    return ts.to_pydatetime(), False


def _to_number(tok: str) -> float | None:
    tok = tok.replace(",", "").strip()
    try:
        return float(tok)
    except ValueError:
        return None


def parse_salary(raw: str | None) -> tuple[float | None, float | None, str | None, str | None]:
    """Parse '2,00,000 - 3,00,000 PA.', '5-8 LPA', '25000/month', '$90,000'. Returns (min, max, currency, issue)."""
    if not raw:
        return None, None, None, None
    s = raw.strip()
    if _NOT_DISCLOSED.search(s) or not re.search(r"\d", s) or re.match(r"(?i)^openings?\b", s):
        return None, None, None, None
    currency = "USD" if "$" in s else "INR" if re.search(r"₹|inr|rs\.?|lpa|lakh|pa\b", s, re.I) else None
    nums = [n for n in (_to_number(t) for t in re.findall(r"\d[\d,]*(?:\.\d+)?", s)) if n is not None]
    if not nums:
        return None, None, None, "unparseable salary"
    mult = 1.0
    if re.search(r"lpa|lakh|lac", s, re.I):
        mult = 100000.0
    elif re.search(r"\bk\b", s, re.I):
        mult = 1000.0
    if re.search(r"per\s*month|/\s*month|p\.?m\.?\b|monthly", s, re.I):
        mult *= 12
    lo, hi = nums[0] * mult, (nums[1] if len(nums) > 1 else nums[0]) * mult
    if lo > hi:
        return None, None, currency, "salary minimum exceeds maximum"
    return lo, hi, currency, None


def parse_experience(raw: str | None) -> tuple[float | None, float | None]:
    if not raw:
        return None, None
    if _FRESHER.search(raw):
        return 0.0, 1.0
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", raw)]
    if not nums:
        return None, None
    if re.search(r"month", raw, re.I):
        nums = [n / 12 for n in nums]
    return nums[0], (nums[1] if len(nums) > 1 else None)


def parse_location(raw: str | None) -> tuple[str | None, str | None, str | None]:
    """'Pune,Pune' -> ('Pune', 'Maharashtra', 'derived_from_city'); keeps the first city of multi-city lists."""
    if not raw:
        return None, None, None
    parts = [p.strip() for p in re.split(r"[,/;|]", raw) if p.strip()]
    if not parts:
        return None, None, None
    from ..common import normalize_state

    # "City, State" form
    if len(parts) >= 2 and normalize_state(parts[1]):
        city = CITY_ALIASES.get(parts[0].lower(), parts[0])
        return city, normalize_state(parts[1]), "as_supplied"
    first = re.sub(r"\(.*?\)", "", parts[0]).strip()
    key = first.lower()
    if normalize_state(first):
        return None, normalize_state(first), "as_supplied"
    city = CITY_ALIASES.get(key, first.title() if first.islower() or first.isupper() else first)
    state = CITY_STATE.get(key)
    return city, state, "derived_from_city" if state else None


def content_hash(rec: dict) -> str:
    key = "|".join([
        norm_text(rec.get("job_title")), norm_text(rec.get("company_name")),
        norm_text((rec.get("job_description") or "")[:3000]), norm_text(rec.get("location_raw")),
        norm_text(rec.get("required_skills")), norm_text(rec.get("preferred_skills")),
        norm_text(rec.get("listed_skills")), norm_text(rec.get("experience_level")),
    ])
    return hashlib.sha256(key.encode()).hexdigest()


# --- Validation & transformation --------------------------------------------------------------------
@dataclass
class PreparedImport:
    records: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)   # {row, level: error|warning, field, message}
    rejected: int = 0
    warning_rows: int = 0
    stats: dict = field(default_factory=dict)


def validate_mapping(mapping: dict[str, str], columns: list[str]) -> list[str]:
    errs = []
    unknown_fields = [f for f in mapping if f not in CANONICAL_FIELDS]
    if unknown_fields:
        errs.append(f"Unknown canonical field(s): {', '.join(unknown_fields)}")
    missing_cols = [c for c in mapping.values() if c and c not in columns]
    if missing_cols:
        errs.append(f"Mapped column(s) not in file: {', '.join(missing_cols)}")
    for f in REQUIRED_FIELDS:
        if not mapping.get(f):
            errs.append(f"Required field '{f}' must be mapped to a column")
    dup = [c for c, n in Counter(v for v in mapping.values() if v).items() if n > 1]
    if dup:
        errs.append(f"Column(s) mapped to more than one field: {', '.join(dup)}")
    return errs


def prepare_rows(rows: list[dict[str, Any]], mapping: dict[str, str], defaults: dict[str, Any],
                 dataset_meta: dict[str, Any], now: datetime | None = None) -> PreparedImport:
    """Map, validate and normalise raw rows. Invalid rows are reported, never silently dropped."""
    now = now or datetime.now(timezone.utc)
    out = PreparedImport()
    filled = Counter()
    observed: list[datetime] = []
    for i, raw in enumerate(rows, start=2):  # row 1 = header
        def get(f):
            col = mapping.get(f)
            return clean(raw.get(col)) if col else None

        row_issues = []
        title = get("job_title")
        if not title:
            out.issues.append({"row": i, "level": "error", "field": "job_title", "message": "job_title is empty"})
            out.rejected += 1
            continue
        rec: dict[str, Any] = {"row_number": i, "job_title": title[:500]}
        for f in ("company_name", "job_description", "required_skills", "preferred_skills", "listed_skills",
                  "education_requirement", "experience_level", "employment_type", "industry", "job_role",
                  "role_category", "source_url", "source_dataset", "licence_or_usage_terms"):
            rec[f] = get(f)
        rec["external_job_id"] = get("job_id")
        if rec["job_description"]:
            rec["job_description"] = str(raw.get(mapping["job_description"]))[:20000].strip()
        # location
        loc = get("location")
        city, state = get("city"), get("state")
        rec["location_raw"] = loc or ", ".join(x for x in (city, state) if x) or None
        source = "as_supplied" if (city or state) else None
        if loc and not (city and state):
            pc, ps, psrc = parse_location(loc)
            city = city or pc
            if not state and ps:
                state, source = ps, psrc
            elif city and not source:
                source = "as_supplied"
        if state:
            from ..common import normalize_state

            norm = normalize_state(state)
            if norm:
                state = norm
            else:
                row_issues.append(("state", f"'{state}' is not a recognised Indian state/UT; kept as supplied"))
        rec["city"], rec["state"] = (city[:150] if city else None), state
        rec["country"] = get("country") or dataset_meta.get("default_country") or defaults.get("country")
        if rec["country"] and not get("country"):
            source = source or "dataset_default"
        rec["location_source"] = source
        # experience
        rec["experience_min_years"], rec["experience_max_years"] = parse_experience(rec["experience_level"])
        # salary
        smin, smax = get("salary_min"), get("salary_max")
        if smin or smax:
            lo, hi = (_to_number(smin) if smin else None), (_to_number(smax) if smax else None)
            if (smin and lo is None) or (smax and hi is None):
                row_issues.append(("salary", "non-numeric salary_min/salary_max ignored"))
                lo = hi = None
            elif lo is not None and hi is not None and lo > hi:
                row_issues.append(("salary", "salary_min exceeds salary_max; salary ignored"))
                lo = hi = None
            cur = get("salary_currency") or defaults.get("salary_currency")
        else:
            lo, hi, cur, issue = parse_salary(get("salary"))
            if issue:
                row_issues.append(("salary", issue))
            cur = get("salary_currency") or cur or (defaults.get("salary_currency") if lo else None)
        rec["salary_min"], rec["salary_max"], rec["salary_currency"] = lo, hi, (cur if lo is not None else None)
        # dates
        flags: list[str] = []
        pd_, bad = parse_date(get("posting_date"))
        if bad:
            row_issues.append(("posting_date", f"unparseable posting date '{get('posting_date')}' left empty"))
            flags.append("invalid_posting_date")
        cd_, badc = parse_date(get("collection_date"))
        if badc:
            row_issues.append(("collection_date", "unparseable collection date left empty"))
        rec["posting_date"], rec["collection_date"] = pd_, cd_
        rec["observed_date"] = pd_ or cd_
        if pd_ and pd_ > now:
            flags.append("future_posting_date")
        if rec["observed_date"]:
            observed.append(rec["observed_date"])
        rec["source_dataset"] = rec["source_dataset"] or defaults.get("source_dataset") or dataset_meta.get("name")
        rec["licence_or_usage_terms"] = rec["licence_or_usage_terms"] or dataset_meta.get("licence")
        text_for_level = " ".join(x for x in (title, rec["experience_level"] or "", rec["employment_type"] or "") if x)
        rec["is_internship"] = bool(_INTERNSHIP.search(text_for_level))
        rec["is_entry_level"] = rec["is_internship"] or bool(_FRESHER.search(text_for_level)) or (
            rec["experience_min_years"] is not None and rec["experience_min_years"] <= 1)
        rec["normalized_title"] = norm_text(title)[:500]
        rec["normalized_company"] = norm_text(rec["company_name"])[:300] or None
        rec["content_hash"] = content_hash(rec)
        rec["quality_flags"] = flags
        rec["raw_record"] = {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in raw.items()}
        for f in CANONICAL_FIELDS:
            if f in rec and rec[f] not in (None, "", []):
                filled[f] += 1
        filled["job_id"] += bool(rec["external_job_id"])
        filled["location"] += bool(rec["location_raw"])
        filled["salary"] += rec["salary_min"] is not None
        if row_issues:
            out.warning_rows += 1
            for fld, msg in row_issues:
                out.issues.append({"row": i, "level": "warning", "field": fld, "message": msg})
        out.records.append(rec)
    n = len(out.records)
    mapped_fields = set(mapping) | {"city", "state", "salary_min", "posting_date", "collection_date"}
    out.stats = {
        "missingness": {f: round(1 - filled[f] / n, 4) if n else None for f in sorted(mapped_fields) if f in CANONICAL_FIELDS},
        "date_coverage": {
            "min": min(observed).isoformat() if observed else None,
            "max": max(observed).isoformat() if observed else None,
            "rows_with_date": len(observed),
            "basis": "posting_date, else collection_date",
        },
        "rows_with_salary": filled["salary_min"],
        "issues_by_field": dict(Counter(x["field"] for x in out.issues)),
    }
    return out
