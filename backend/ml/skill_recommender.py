"""MODEL 3: Skill recommender (team module 3) — unsupervised learning on the skill sets of IT job postings.

(a) Apriori association-rule mining. Each IT posting is a "basket" of skills. Frequent itemsets (support ≥ 1%,
    up to 3 skills) give rules X → Y with support, confidence and lift, e.g. {Java} → {Spring Framework}.
    The app uses them to explain recommendations: "students already learn Java; 41% of Java jobs also ask for
    Spring (lift 3.2)".

(b) K-Means clustering of postings (TF-IDF-weighted skill vectors, L2-normalised). k is chosen by the
    silhouette score (searched over 6–12 so the bundles stay readable). Each cluster is a "skill bundle" that employers ask for together (e.g. Java · Spring ·
    Microservices). The app shows how well a syllabus covers each bundle.
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np
from scipy import sparse

from ml.common import SEED, log


# --- (a) Apriori -----------------------------------------------------------------------------------------------
def apriori(X: sparse.csr_matrix, names: list[str], min_support: float = 0.01, min_conf: float = 0.25,
            min_lift: float = 1.3, max_len: int = 3) -> dict:
    t0 = time.time()
    B = (X > 0).astype(np.int32).tocsc()
    n = B.shape[0]
    single = np.asarray(B.sum(axis=0)).ravel() / n
    L1 = [i for i in range(B.shape[1]) if single[i] >= min_support]
    support: dict[tuple, float] = {(i,): float(single[i]) for i in L1}
    # L2 from the co-occurrence matrix
    sub = B[:, L1]
    co = (sub.T @ sub).toarray() / n
    L2 = []
    for a in range(len(L1)):
        for b in range(a + 1, len(L1)):
            if co[a, b] >= min_support:
                key = (L1[a], L1[b])
                support[key] = float(co[a, b])
                L2.append(key)
    # L3 candidates: join L2 itemsets sharing the first item, prune by Apriori property
    L3 = []
    if max_len >= 3:
        l2set = set(L2)
        cols = {i: B[:, i] for i in L1}
        by_first: dict[int, list[int]] = {}
        for a, b in L2:
            by_first.setdefault(a, []).append(b)
        for a, bs in by_first.items():
            for b, c in combinations(sorted(bs), 2):
                if (b, c) not in l2set:
                    continue
                s = cols[a].multiply(cols[b]).multiply(cols[c]).sum() / n
                if s >= min_support:
                    support[(a, b, c)] = float(s)
                    L3.append((a, b, c))
    rules = []
    for itemset in L2 + L3:
        for y in itemset:
            x = tuple(i for i in itemset if i != y)
            conf = support[itemset] / support[x]
            lift = conf / support[(y,)]
            if conf >= min_conf and lift >= min_lift:
                rules.append({"if": [names[i] for i in x], "then": names[y], "support": round(support[itemset], 4),
                              "confidence": round(conf, 4), "lift": round(lift, 3)})
    rules.sort(key=lambda r: (-r["lift"] * r["confidence"], -r["support"]))
    log(f"  [model 3a] Apriori: {len(L1)} frequent skills, {len(L2)} pairs, {len(L3)} triples, "
        f"{len(rules)} rules ({time.time() - t0:.1f}s)")
    return {"min_support": min_support, "min_confidence": min_conf, "min_lift": min_lift, "transactions": int(n),
            "frequent_itemsets": {"1": len(L1), "2": len(L2), "3": len(L3)}, "rules": rules}


# --- (b) K-Means ----------------------------------------------------------------------------------------------
def kmeans_bundles(X: sparse.csr_matrix, names: list[str], roles: list[str], k_range=range(6, 13)) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    t0 = time.time()
    B = (X > 0).astype(np.float32)
    rows = np.where(np.asarray(B.sum(axis=1)).ravel() >= 2)[0]          # postings with at least two skills
    B = B[rows]
    V = normalize(TfidfTransformer(sublinear_tf=True).fit_transform(B))
    rng = np.random.default_rng(SEED)
    sample = rng.choice(V.shape[0], min(8000, V.shape[0]), replace=False)
    scores, inertias, models = {}, {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(V)
        scores[k] = float(silhouette_score(V[sample], km.labels_[sample], metric="cosine", random_state=SEED))
        inertias[k] = float(km.inertia_)
        models[k] = km
        log(f"  [model 3b] k={k}: silhouette {scores[k]:.3f}")
    best_k = max(scores, key=scores.get)
    km = models[best_k]
    overall = np.asarray(B.mean(axis=0)).ravel()
    clusters = []
    for c in range(best_k):
        members = np.where(km.labels_ == c)[0]
        freq = np.asarray(B[members].mean(axis=0)).ravel()
        lift = freq / np.maximum(overall, 1e-9)
        score = freq * np.log1p(lift)
        top = [int(i) for i in np.argsort(-score)[:8] if freq[i] >= 0.05]
        role_counts: dict[str, int] = {}
        for r in (roles[rows[m]] for m in members):
            role_counts[r] = role_counts.get(r, 0) + 1
        main_role = max(role_counts, key=role_counts.get)
        clusters.append({
            "id": c, "size": int(len(members)), "share": round(len(members) / len(rows), 4),
            "name": " · ".join(names[i] for i in top[:3]),
            "top_skills": [{"skill": names[i], "frequency": round(float(freq[i]), 4)} for i in top],
            "main_role": main_role, "main_role_share": round(role_counts[main_role] / len(members), 3),
        })
    clusters.sort(key=lambda c: -c["size"])
    log(f"  [model 3b] K-Means: best k={best_k} (silhouette {scores[best_k]:.3f}) ({time.time() - t0:.0f}s)")
    return {"k": best_k, "silhouette": {str(k): round(v, 4) for k, v in scores.items()},
            "inertia": {str(k): round(v, 1) for k, v in inertias.items()}, "postings_clustered": int(len(rows)),
            "clusters": clusters}
