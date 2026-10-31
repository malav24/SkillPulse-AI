"""Train every SkillPulse model and build the market index used by the app.

    cd backend
    python -m ml.train_all          # ~10 minutes on a laptop; writes backend/artifacts/

Steps
  1. Load and de-duplicate the job postings in data/market/*.csv.gz
  2. MODEL 2 — train the requirement-level classifier (required / preferred / neutral sentences)
  3. Extract skills from every posting; sentences without a heading are labelled by MODEL 2
  4. MODEL 1 — train the job-role classifier, then assign a role to every posting
  5. MODEL 3 — Apriori association rules + K-Means skill bundles on the IT postings
  6. Save the models, the compact market index and all evaluation metrics
"""
from __future__ import annotations

import gzip
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from scipy import sparse

from ml import requirement_classifier, role_classifier, skill_recommender
from ml.common import ARTIFACTS, OTHER_ROLE, ROLES, extract_market, load_postings, log, skill_text


def main() -> None:
    t_all = time.time()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    from app.services.job_import import parse_experience

    log("Training all SkillPulse models. This takes about 10 minutes; do not click inside this window "
        "(on Windows that pauses the program) and do not press Ctrl+C.")
    log("STEP 1 — loading job postings")
    postings = load_postings()

    log("STEP 2 — MODEL 2: requirement-level classifier")
    req_model, req_report = requirement_classifier.train(postings["description"].tolist())

    log("STEP 3 — extracting skills from every posting (MODEL 2 labels sentences without headings)")
    market = extract_market(postings, requirement_model=req_model)
    names = [s["name"] for s in market.skills]
    tokens = skill_text(market.levels, market.skills)
    log(f"  {market.levels.nnz:,} posting-skill links; {int((np.diff(market.levels.indptr) > 0).sum()):,} postings "
        f"with at least one skill")

    log("STEP 4 — MODEL 1: job-role classifier")
    texts = [role_classifier.role_text(d, t) for d, t in zip(postings["description"], tokens)]
    role_model, role_report, title_labels = role_classifier.train(texts, postings["title"].tolist())
    roles, role_stats = role_classifier.assign_roles(role_model, texts, title_labels)
    role_report["assignment"] = role_stats
    role_report["final_role_counts"] = {r: roles.count(r) for r in ROLES}
    log(f"  roles assigned: {role_stats}")

    log("STEP 5 — MODEL 3: skill recommender (Apriori + K-Means) on IT postings")
    it_rows = np.array([i for i, r in enumerate(roles) if r != OTHER_ROLE])
    X_it = market.levels[it_rows]
    rules = skill_recommender.apriori(X_it, names)
    bundles = skill_recommender.kmeans_bundles(X_it, names, [roles[i] for i in it_rows])

    log("STEP 6 — saving artifacts")
    exp_min = []
    for e in postings["experience"]:
        lo, _hi = parse_experience(e)
        exp_min.append(-1.0 if lo is None else float(lo))
    years = [int(d[:4]) if d[:4].isdigit() else 0 for d in postings["date"]]
    sparse.save_npz(ARTIFACTS / "market_levels.npz", market.levels.astype(np.int8), compressed=True)
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "skills": market.skills, "roles": ROLES,
        "role": [ROLES.index(r) for r in roles],
        "exp_min": exp_min, "year": years,
        "title": [t.strip()[:80] for t in postings["title"]],
    }
    with gzip.open(ARTIFACTS / "market_meta.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(meta, fh, separators=(",", ":"))
    joblib.dump(role_model, ARTIFACTS / "role_model.joblib", compress=3)
    joblib.dump(req_model, ARTIFACTS / "requirement_model.joblib", compress=3)
    (ARTIFACTS / "recommender.json").write_text(json.dumps({"apriori": rules, "kmeans": bundles}, indent=1),
                                                encoding="utf-8")
    valid_years = [y for y in years if y]
    metrics = {
        "built_at": meta["built_at"],
        "data": {"postings_loaded": int(len(postings)), "it_postings": int(len(it_rows)),
                 "skills_in_taxonomy": len(names), "posting_skill_links": int(market.levels.nnz),
                 "period": f"{min(valid_years)}–{max(valid_years)}" if valid_years else None,
                 "files": postings["source_file"].value_counts().to_dict()},
        "role_classifier": role_report,
        "requirement_classifier": req_report,
        "skill_recommender": {
            "apriori": {k: v for k, v in rules.items() if k != "rules"} | {"top_rules": rules["rules"][:25],
                                                                           "rule_count": len(rules["rules"])},
            "kmeans": {k: v for k, v in bundles.items()},
        },
        "training_minutes": round((time.time() - t_all) / 60, 1),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    log("STEP 7 — writing the data-pipeline report")
    from ml.pipeline_report import pipeline_summary

    metrics["pipeline"] = pipeline_summary(postings)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    log(f"done in {metrics['training_minutes']} min → {ARTIFACTS}")


if __name__ == "__main__":
    main()
