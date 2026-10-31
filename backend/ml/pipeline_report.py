"""Summary of the data pipeline (datasets → preprocessing → features → training → evaluation) with real numbers.

Called at the end of `python -m ml.train_all`. Can also be run alone to refresh the report from the current
artifacts without retraining:   python -m ml.pipeline_report
"""
from __future__ import annotations

import gzip
import json

import joblib
import numpy as np
from scipy import sparse

from ml.common import ARTIFACTS, OTHER_ROLE, log


def _vocab(model) -> int:
    vec = model.named_steps["tfidf"]
    return int(len(vec.vocabulary_))


def pipeline_summary(postings) -> dict:
    from ml.requirement_classifier import segment_with_headings

    levels = sparse.load_npz(ARTIFACTS / "market_levels.npz").tocsr()
    with gzip.open(ARTIFACTS / "market_meta.json.gz", "rt", encoding="utf-8") as fh:
        meta = json.load(fh)
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    role_model = joblib.load(ARTIFACTS / "role_model.joblib")
    req_model = joblib.load(ARTIFACTS / "requirement_model.joblib")
    roles = meta["roles"]
    role_idx = np.array(meta["role"])
    it = role_idx != roles.index(OTHER_ROLE)
    has_desc = int((postings["description"].str.strip() != "").sum())
    has_tags = int((postings["skills"].str.strip() != "").sum())
    headed, unheaded = 0, 0
    for d in postings["description"]:
        if d:
            for ctx, _s in segment_with_headings(d):
                if ctx == "none":
                    unheaded += 1
                else:
                    headed += 1
    lv = levels.data
    rc, qc = metrics["role_classifier"], metrics["requirement_classifier"]
    a = postings.attrs
    return {
        "datasets": {
            "files": a.get("files", {}), "raw_rows": a.get("raw_rows", len(postings)),
            "description": "Real job postings: an Indian job portal scrape (Oct 2025, with full descriptions) and "
                           "a LinkedIn sample (Jan 2024, skill lists); see README → Data sources.",
        },
        "preprocessing": [
            {"step": "Removed rows without a job title", "count": a.get("empty_titles", 0)},
            {"step": "Removed exact duplicate postings (same title, company, skills and description)",
             "count": a.get("duplicates", 0)},
            {"step": "Postings kept for training", "count": int(len(postings))},
            {"step": "Postings with a job description (HTML tags removed, text normalised)", "count": has_desc},
            {"step": "Postings with skill tags", "count": has_tags},
            {"step": "Description sentences split and assigned to their heading (Required / Good to have / "
                     "Responsibilities)", "count": headed},
            {"step": "Description sentences without a heading, labelled by Model 2", "count": unheaded},
            {"step": "Skills recognised with the 188-skill taxonomy (posting–skill links)", "count": int(levels.nnz)},
            {"step": "Postings with at least one recognised skill", "count": int((np.diff(levels.indptr) > 0).sum())},
            {"step": "Required / preferred / unspecified skill links",
             "count": f"{int((lv == 3).sum()):,} / {int((lv == 1).sum()):,} / {int((lv == 2).sum()):,}"},
            {"step": "Postings with a stated experience range (used for the entry-level filter)",
             "count": int(sum(1 for e in meta["exp_min"] if e >= 0))},
            {"step": "Job-role labels from title rules (weak supervision)",
             "count": int(rc.get("assignment", {}).get("from_title", 0))},
            {"step": "Extra IT postings whose role was predicted by Model 1 (vague titles)",
             "count": int(rc.get("assignment", {}).get("from_model", 0))},
        ],
        "features": [
            {"model": "Model 1: job-role classifier",
             "features": f"TF-IDF word 1–2-grams over skill tokens (weighted ×2) + description; job title removed; "
                         f"vocabulary {_vocab(role_model):,} terms, sublinear TF, min document frequency 3"},
            {"model": "Model 2: requirement-level classifier",
             "features": f"TF-IDF word 1–2-grams per sentence, heading words removed; vocabulary "
                         f"{_vocab(req_model):,} terms, sublinear TF, min document frequency 3"},
            {"model": "Model 3a: Apriori",
             "features": f"Binary skill baskets: {int(it.sum()):,} IT postings × {levels.shape[1]} skills"},
            {"model": "Model 3b: K-Means",
             "features": "TF-IDF-weighted skill vectors, L2-normalised, cosine geometry; postings with ≥ 2 skills"},
        ],
        "training": [
            {"model": "Model 1", "split": f"{rc['train_size']:,} train / {rc['test_size']:,} test (80/20, stratified, "
                                          "de-duplicated, random seed 42)",
             "algorithms": [k for k in rc["models"] if "calibrated" not in k], "selected": rc["best_model"],
             "selection": "highest macro-F1 on the test set; probabilities calibrated (sigmoid, 3-fold)"},
            {"model": "Model 2", "split": f"{qc['train_size']:,} train / {qc['test_size']:,} test (80/20, stratified, "
                                          "de-duplicated, random seed 42)",
             "algorithms": list(qc["models"]), "selected": qc["best_model"], "selection": "highest macro-F1 on the test set"},
            {"model": "Model 3", "split": "unsupervised: all IT postings",
             "algorithms": ["Apriori (min support 1%, min confidence 25%, min lift 1.3, up to 3 skills)",
                            "K-Means (k = 6…12, 5 initialisations, random seed 42)"],
             "selected": f"{metrics['skill_recommender']['apriori']['rule_count']:,} rules; "
                         f"k = {metrics['skill_recommender']['kmeans']['k']}",
             "selection": "k with the highest silhouette score"},
        ],
    }


def main() -> None:
    from ml.common import load_postings

    postings = load_postings()
    path = ARTIFACTS / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["pipeline"] = pipeline_summary(postings)
    path.write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    log(f"pipeline report written to {path}")


if __name__ == "__main__":
    main()
