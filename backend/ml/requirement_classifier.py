"""MODEL 2: Requirement-level classifier (team module 2).

Task: given one sentence from a job description, predict whether it states a REQUIRED skill, a PREFERRED
("good to have") skill, or is NEUTRAL text (responsibilities, company info).

Training labels come for free from job descriptions that use headings ("Required Skills:", "Good to have:",
"Roles and Responsibilities:"): every sentence under such a heading gets that heading's label. The heading words
themselves are removed from the text, so the model must learn from the sentence content ("must", "hands-on",
"exposure to", "is a plus", "you will"...). The trained model then labels the many sentences that have NO
heading, which decides how strongly each skill counts in the demand score.

Algorithms compared on the same held-out test set: Multinomial Naive Bayes, Logistic Regression, Linear SVM and
Random Forest, all on TF-IDF word 1–2-gram features. The best (macro-F1) is saved.
"""
from __future__ import annotations

import re
import time

import numpy as np

from app.services.job_skills import _NEUTRAL_HEAD, _PREF_HEAD, _REQ_HEAD
from app.services.skill_extractor import clean_text
from ml.common import SEED, log

LABELS = ["required", "preferred", "neutral"]
_SPLIT = re.compile(r"\n+|(?<=[.!?])\s+(?=[A-Z•\-*])")


def _strip_heading(line: str, rx: re.Pattern) -> str:
    m = rx.match(line)
    return line[m.end():].lstrip(" :.-–—") if m else line


def segment_with_headings(desc: str) -> list[tuple[str, str]]:
    """Split a description into (context, sentence). context = required | preferred | neutral from the most recent
    heading, or 'none' when no heading has appeared yet. Heading words are stripped from the sentence."""
    text = clean_text(desc or "")
    ctx = "none"
    out: list[tuple[str, str]] = []
    for line in _SPLIT.split(text):
        line = line.strip(" •*-\t")
        if not line:
            continue
        head = line[:80]
        for rx, label in ((_PREF_HEAD, "preferred"), (_REQ_HEAD, "required"), (_NEUTRAL_HEAD, "neutral")):
            if rx.match(head):
                ctx = label
                line = _strip_heading(line, rx)
                break
        if len(line) >= 12:
            out.append((ctx, line[:400]))
    return out


def training_sentences(descriptions, cap_per_class: int = 60000) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    by_label: dict[str, list[str]] = {k: [] for k in LABELS}
    descriptions = list(descriptions)
    log(f"  reading {len(descriptions):,} job descriptions into sentences (about 1 minute)…")
    for n, d in enumerate(descriptions, 1):
        if n % 20000 == 0:
            log(f"    {n:,}/{len(descriptions):,} descriptions read")
        if not d:
            continue
        for ctx, sent in segment_with_headings(d):
            if ctx == "none":
                continue
            key = re.sub(r"\W+", " ", sent.lower()).strip()
            if len(key.split()) < 3 or key in seen:     # drop very short lines and duplicates (templated ads)
                continue
            seen.add(key)
            by_label[ctx].append(sent)
    rng = np.random.default_rng(SEED)
    X, y = [], []
    for label, sents in by_label.items():
        if len(sents) > cap_per_class:
            sents = list(rng.choice(np.array(sents, dtype=object), cap_per_class, replace=False))
        X += sents
        y += [label] * len(sents)
    log("requirement sentences per class: " + ", ".join(f"{k}={min(len(v), cap_per_class):,}" for k, v in by_label.items()))
    return X, y


def candidate_models():
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    return {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.3),
        "Logistic Regression": LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"),
        "Linear SVM": LinearSVC(C=0.5, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced",
                                                n_jobs=-1, random_state=SEED),
    }


def train(descriptions) -> tuple[object, dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    from ml.evaluation import evaluate

    X, y = training_sentences(descriptions)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    results, fitted = {}, {}
    for name, clf in candidate_models().items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000,
                                                   sublinear_tf=True, strip_accents="unicode")),
                         ("clf", clf)])
        log(f"  [model 2] training {name}…" + (" (the slowest, a few minutes)" if name == "Random Forest" else ""))
        t0 = time.time()
        if name == "Random Forest" and len(Xtr) > 40000:   # forests on sparse text are slow: subsample
            idx = np.random.default_rng(SEED).choice(len(Xtr), 40000, replace=False)
            pipe.fit([Xtr[i] for i in idx], [ytr[i] for i in idx])
        else:
            pipe.fit(Xtr, ytr)
        secs = time.time() - t0
        results[name] = evaluate(pipe, Xte, yte, LABELS, secs)
        fitted[name] = pipe
        log(f"  [model 2] {name}: accuracy {results[name]['accuracy']:.3f}, macro-F1 {results[name]['macro_f1']:.3f} "
            f"({secs:.0f}s)")
    best = max((k for k in results if not k.startswith("Baseline")), key=lambda k: results[k]["macro_f1"])
    report = {
        "task": "Classify a job-description sentence as required / preferred / neutral",
        "labels": LABELS,
        "train_size": len(Xtr), "test_size": len(Xte),
        "class_counts": {k: int(sum(1 for v in y if v == k)) for k in LABELS},
        "features": "TF-IDF word uni+bi-grams (max 60,000), heading words removed",
        "models": results, "best_model": best,
    }
    log(f"  [model 2] best: {best}")
    return fitted[best], report
