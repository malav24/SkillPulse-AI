"""Common evaluation for the supervised models: accuracy, precision, recall, F1, confusion matrix."""
from __future__ import annotations

import time


def evaluate(model, X_test, y_test, labels: list[str], train_seconds: float) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    t0 = time.time()
    pred = model.predict(X_test)
    infer = time.time() - t0
    p, r, f, s = precision_recall_fscore_support(y_test, pred, labels=labels, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_test, pred, labels=labels, average="weighted", zero_division=0)), 4),
        "per_class": {lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                            "f1": round(float(f[i]), 4), "support": int(s[i])} for i, lab in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "train_seconds": round(train_seconds, 1),
        "predict_ms_per_1000": round(1000 * infer / max(1, len(X_test)) * 1000, 2),
    }
