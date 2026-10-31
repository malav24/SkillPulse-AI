"""Loads the trained models and the pre-computed job-market index (backend/artifacts/), once, at start-up.

Everything an analysis needs from the 1.3 lakh job postings is here in memory (a sparse postings × skills
matrix of requirement levels, each posting's predicted job role, experience and title), so comparing a
syllabus with the market takes milliseconds instead of database queries.
"""
from __future__ import annotations

import gzip
import json
import logging
import re
import threading
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import get_settings

log = logging.getLogger("skillpulse.market")

LEVEL_WEIGHT = np.array([0.0, 0.5, 0.75, 1.0], dtype=np.float32)   # 0 none, 1 preferred, 2 unspecified, 3 required
OTHER_ROLE = "Other (non-IT)"


class MarketModel:
    def __init__(self, folder: Path):
        import joblib
        from scipy import sparse

        self.folder = folder
        csr = sparse.load_npz(folder / "market_levels.npz").tocsr()           # postings × skills
        self.levels_csr = csr
        self.levels = csr.tocsc()
        with gzip.open(folder / "market_meta.json.gz", "rt", encoding="utf-8") as fh:
            meta = json.load(fh)
        self.skills: list[dict] = meta["skills"]
        self.roles: list[str] = meta["roles"]
        self.role = np.array(meta["role"], dtype=np.int16)
        self.exp_min = np.array(meta["exp_min"], dtype=np.float32)
        self.year = np.array(meta["year"], dtype=np.int16)
        self.title: list[str] = meta["title"]
        self.n = self.levels.shape[0]
        self.skill_index = {s["name"]: s["index"] for s in self.skills}
        self.role_model = joblib.load(folder / "role_model.joblib")
        self.requirement_model = joblib.load(folder / "requirement_model.joblib")
        rec = json.loads((folder / "recommender.json").read_text(encoding="utf-8"))
        self.rules: list[dict] = rec["apriori"]["rules"]
        self.bundles: list[dict] = rec["kmeans"]["clusters"]
        self.metrics: dict = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        years = self.year[self.year > 0]
        self.period = f"{years.min()}–{years.max()}" if len(years) and years.min() != years.max() else (
            str(years.min()) if len(years) else None)
        log.info("market model loaded: %s postings, %s skills, %s rules, %s bundles",
                 f"{self.n:,}", len(self.skills), len(self.rules), len(self.bundles))

    # --- cohorts -----------------------------------------------------------------------------------------------
    def role_options(self) -> list[dict]:
        counts = Counter(self.role.tolist())
        it_total = sum(c for r, c in counts.items() if self.roles[r] != OTHER_ROLE)
        opts = [{"value": "it", "label": "All Software & IT jobs", "postings": it_total}]
        opts += [{"value": r, "label": r, "postings": counts.get(i, 0)} for i, r in enumerate(self.roles)
                 if r != OTHER_ROLE]
        opts.append({"value": "all", "label": "All jobs (any industry)", "postings": self.n})
        return opts

    def mask(self, job_field: str, entry_level: bool = False) -> np.ndarray:
        if job_field == "all":
            m = np.ones(self.n, dtype=bool)
        elif job_field == "it":
            m = self.role != self.roles.index(OTHER_ROLE)
        elif job_field in self.roles:
            m = self.role == self.roles.index(job_field)
        else:
            raise ValueError(f"Unknown job field: {job_field}")
        if entry_level:
            m &= (self.exp_min >= 0) & (self.exp_min <= 2)
        return m

    # --- demand ------------------------------------------------------------------------------------------------
    def demand(self, mask: np.ndarray) -> dict:
        """Per skill: share of cohort postings asking for it, level-weighted share, required/preferred counts."""
        rows = np.where(mask)[0]
        n = len(rows)
        sub = self.levels_csr[rows].tocsc()
        out = {}
        for j, s in enumerate(self.skills):
            a, b = sub.indptr[j], sub.indptr[j + 1]
            if a == b:
                continue
            lv = sub.data[a:b]
            out[s["name"]] = {
                "jobs": int(b - a), "share": (b - a) / n if n else 0.0,
                "weighted_share": float(LEVEL_WEIGHT[lv].sum()) / n if n else 0.0,
                "required": int((lv == 3).sum()), "preferred": int((lv == 1).sum()),
                "unspecified": int((lv == 2).sum()),
            }
        return {"n": n, "skills": out}

    def example_titles(self, mask: np.ndarray, skill: str, k: int = 3) -> list[str]:
        j = self.skill_index[skill]
        col = self.levels[:, j]
        rows = col.indices[mask[col.indices]]
        counts = Counter(self.title[i].lower() for i in rows[:20000])
        out = []
        for t, _ in counts.most_common(40):
            if not t.isascii() or len(t) > 60 or not any(c.isalpha() for c in t):
                continue
            out.append(t.title())
            if len(out) == k:
                break
        return out

    def skill_role(self, mask: np.ndarray, skill: str) -> dict | None:
        """Which job role (as labelled by MODEL 1) asks for this skill most, among the selected postings."""
        col = self.levels[:, self.skill_index[skill]]
        rows = col.indices[mask[col.indices]]
        other = self.roles.index(OTHER_ROLE)
        r = self.role[rows]
        r = r[r != other]
        if len(r) == 0:
            return None
        counts = np.bincount(r, minlength=len(self.roles))
        top = int(np.argmax(counts))
        return {"role": self.roles[top], "share": round(float(counts[top] / len(r)), 3)}

    # --- ML helpers ---------------------------------------------------------------------------------------------
    @staticmethod
    def model_input(text: str, skills: list[str]) -> str:
        """Same input format MODEL 1 was trained on: skill tokens (twice) + free text."""
        tok = " ".join("SK_" + re.sub(r"\W+", "_", s.lower()).strip("_") for s in skills)
        return f"{tok} {tok} {text}".strip()

    def predict_roles(self, text: str, skills: list[str] | None = None, top: int = 5) -> list[dict]:
        """MODEL 1 on syllabus text: which job roles does this content prepare students for?
        Probabilities are re-normalised over the IT roles."""
        x = self.model_input(text, skills or [])
        if not x.strip():
            return []
        proba = self.role_model.predict_proba([x])[0]
        classes = [str(c) for c in self.role_model.classes_]
        pairs = sorted(((classes[i], float(p)) for i, p in enumerate(proba) if classes[i] != OTHER_ROLE),
                       key=lambda x: -x[1])
        total = sum(p for _, p in pairs) or 1.0
        return [{"role": r, "probability": round(p / total, 4)} for r, p in pairs[:top]]

    def course_role(self, text: str, skills: list[str], min_it: float = 0.2) -> dict | None:
        """MODEL 1 on one course. None when the course is not job-specific (IT roles together get < min_it of
        the probability, e.g. Constitution of India, Gender Sensitization)."""
        proba = self.role_model.predict_proba([self.model_input(text, skills)])[0]
        classes = [str(c) for c in self.role_model.classes_]
        other = float(proba[classes.index(OTHER_ROLE)]) if OTHER_ROLE in classes else 0.0
        if 1 - other < min_it:
            return None
        it = [(c, float(p)) for c, p in zip(classes, proba) if c != OTHER_ROLE]
        best, p = max(it, key=lambda x: x[1])
        return {"role": best, "probability": round(p / (1 - other), 4)}

    def rules_for(self, consequent: str, known: set[str], k: int = 1) -> list[dict]:
        """MODEL 3a: association rules X → consequent whose antecedent X is already taught."""
        hits = [r for r in self.rules if r["then"] == consequent and set(r["if"]) <= known]
        hits.sort(key=lambda r: (-r["confidence"], -r["lift"]))
        return hits[:k]


_model: MarketModel | None = None
_error: str | None = None
_lock = threading.Lock()


def artifacts_dir() -> Path:
    return Path(get_settings().artifacts_dir)


def get_market() -> MarketModel:
    global _model, _error
    with _lock:
        if _model is None:
            try:
                _model = MarketModel(artifacts_dir())
                _error = None
            except FileNotFoundError as exc:
                _error = f"Trained models not found ({exc.filename}). Run: python -m ml.train_all"
                raise RuntimeError(_error) from exc
        return _model


def market_ready() -> tuple[bool, str | None]:
    try:
        get_market()
        return True, None
    except Exception as exc:
        return False, str(exc)


def reset() -> None:
    global _model
    with _lock:
        _model = None
