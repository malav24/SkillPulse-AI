"""MODEL 1: Job-role classifier (team module 1).

Task: from the *content* of a text (skills + description, never the title), predict the job role it belongs to:
Data Scientist / ML Engineer, Data Engineer, Data Analyst / BI, Cybersecurity, DevOps / Cloud, QA / Testing,
Mobile Developer, Web / Full Stack Developer, Software Developer, Network / IT Support, or Other (non-IT).

Training labels are created from job titles with transparent rules (ml.common.ROLE_RULES). The title is then
REMOVED from the input, so the classifier has to learn which skills and words characterise each role. That lets
the same model:
  * label job postings whose titles are vague ("Associate", "Consultant"), and
  * read a syllabus (or a single course) and estimate which job roles it prepares students for.

Algorithms compared on the same held-out test set: Complement Naive Bayes, Logistic Regression, Linear SVM and
Random Forest, all on TF-IDF word 1–2-gram features of skills + description. Best macro-F1 is kept; if it has no
probability output it is calibrated so the app can show role percentages.
"""
from __future__ import annotations

import re
import time

import numpy as np

from ml.common import OTHER_ROLE, ROLES, SEED, log, role_from_title


def role_text(description: str, skill_tokens: str) -> str:
    d = re.sub(r"\s+", " ", description or "")[:2500]
    return f"{skill_tokens} {skill_tokens} {d}".strip()     # skills counted twice: they are the strongest signal


def candidate_models():
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import ComplementNB
    from sklearn.svm import LinearSVC

    return {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Complement Naive Bayes": ComplementNB(alpha=0.3),
        "Logistic Regression": LogisticRegression(C=6.0, max_iter=3000, class_weight="balanced"),
        "Linear SVM": LinearSVC(C=0.4, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=1, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=SEED),
    }


def train(texts: list[str], titles: list[str]) -> tuple[object, dict, list[str]]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    from ml.evaluation import evaluate

    labels = [role_from_title(t) for t in titles]
    keep = [i for i, t in enumerate(texts) if len(t.split()) >= 3]
    # de-duplicate identical inputs so near-identical templated ads cannot sit in both train and test
    seen, idx = set(), []
    for i in keep:
        if texts[i] not in seen:
            seen.add(texts[i])
            idx.append(i)
    X = [texts[i] for i in idx]
    y = [labels[i] for i in idx]
    log("role labels from titles: " + ", ".join(f"{r}={y.count(r):,}" for r in ROLES))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    results, fitted = {}, {}
    for name, clf in candidate_models().items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=80000,
                                                   sublinear_tf=True, strip_accents="unicode")),
                         ("clf", clf)])
        log(f"  [model 1] training {name}…" + (" (the slowest, a few minutes)" if name == "Random Forest" else ""))
        t0 = time.time()
        if name == "Random Forest" and len(Xtr) > 40000:
            sub = np.random.default_rng(SEED).choice(len(Xtr), 40000, replace=False)
            pipe.fit([Xtr[i] for i in sub], [ytr[i] for i in sub])
        else:
            pipe.fit(Xtr, ytr)
        secs = time.time() - t0
        results[name] = evaluate(pipe, Xte, yte, ROLES, secs)
        fitted[name] = pipe
        log(f"  [model 1] {name}: accuracy {results[name]['accuracy']:.3f}, macro-F1 {results[name]['macro_f1']:.3f} "
            f"({secs:.0f}s)")
    best = max((k for k in results if not k.startswith("Baseline")), key=lambda k: results[k]["macro_f1"])
    model = fitted[best]
    if not hasattr(model, "predict_proba"):
        log(f"  [model 1] calibrating {best} for probability output")
        vec = model.named_steps["tfidf"]
        cal = CalibratedClassifierCV(model.named_steps["clf"], cv=3, method="sigmoid")
        cal.fit(vec.transform(Xtr), ytr)
        model = Pipeline([("tfidf", vec), ("clf", cal)])
        results[best + " (calibrated)"] = evaluate(model, Xte, yte, ROLES, 0.0)
    report = {
        "task": "Predict the job role from a posting's skills and description (title excluded)",
        "labels": ROLES,
        "train_size": len(Xtr), "test_size": len(Xte),
        "class_counts": {r: y.count(r) for r in ROLES},
        "features": "TF-IDF word uni+bi-grams (max 80,000) of extracted skills + description; job title removed",
        "label_source": "Job-title rules (weak supervision); see ml/common.py ROLE_RULES",
        "models": results, "best_model": best,
    }
    log(f"  [model 1] best: {best}")
    return model, report, labels


def assign_roles(model, texts: list[str], title_labels: list[str], min_prob: float = 0.55) -> tuple[list[str], dict]:
    """Final role per posting: a clear title rule wins; otherwise the classifier decides (if confident)."""
    proba = model.predict_proba(texts)
    classes = list(model.classes_)
    out, stats = [], {"from_title": 0, "from_model": 0, "other": 0}
    for i, lab in enumerate(title_labels):
        if lab != OTHER_ROLE:
            out.append(lab)
            stats["from_title"] += 1
            continue
        j = int(np.argmax(proba[i]))
        if classes[j] != OTHER_ROLE and proba[i][j] >= min_prob:
            out.append(classes[j])
            stats["from_model"] += 1
        else:
            out.append(OTHER_ROLE)
            stats["other"] += 1
    return out, stats
