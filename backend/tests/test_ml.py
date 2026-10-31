"""Machine-learning modules: unit tests on SYNTHETIC data + sanity checks on the shipped trained models."""
import numpy as np
import pytest
from scipy import sparse

from ml.common import OTHER_ROLE, ROLES, role_from_title
from ml.requirement_classifier import segment_with_headings
from ml.skill_recommender import apriori, kmeans_bundles


def test_role_labels_from_titles():
    assert role_from_title("Senior Data Scientist") == "Data Scientist / ML Engineer"
    assert role_from_title("Machine Learning Engineer") == "Data Scientist / ML Engineer"
    assert role_from_title("Azure DevOps Engineer") == "DevOps / Cloud"
    assert role_from_title("React Frontend Developer") == "Web / Full Stack Developer"
    assert role_from_title("Java Developer") == "Software Developer"
    assert role_from_title("SDET - Automation Testing") == "QA / Testing"
    assert role_from_title("Telesales Executive") == OTHER_ROLE
    assert set(ROLES) >= {"Data Engineer", "Cybersecurity", OTHER_ROLE}


def test_sentence_segmentation_tracks_headings_and_strips_them():
    jd = ("We are a product company.\nRoles and Responsibilities:\nBuild REST services.\n"
          "Required Skills:\nStrong Java and SQL.\nGood to have: Docker and Kubernetes experience.")
    segs = segment_with_headings(jd)
    assert segs[0] == ("none", "We are a product company.")
    assert ("neutral", "Build REST services.") in segs
    assert ("required", "Strong Java and SQL.") in segs
    assert ("preferred", "Docker and Kubernetes experience.") in segs     # heading words removed from the text


def test_requirement_classifier_learns_cues_on_synthetic_sentences():
    """SYNTHETIC TEST DATA: distinct cue words per class; the pipeline should separate them."""
    from ml import requirement_classifier as rc

    rng = np.random.default_rng(0)
    tools = ["java", "python", "sql", "docker", "react", "aws", "linux", "spark", "kafka", "git"]
    templates = {"required": "Required:\nMust have strong hands-on {} and {} skills {}.",
                 "preferred": "Good to have:\nExposure to {} or {} would be a plus {}.",
                 "neutral": "Roles and Responsibilities:\nYou will coordinate with the {} and {} teams daily {}."}
    descs = []
    for i in range(600):
        label = ["required", "preferred", "neutral"][i % 3]
        a, b = rng.choice(tools, 2, replace=False)
        descs.append(templates[label].format(a, b, i))
    model, report = rc.train(descs)
    assert report["best_model"] in report["models"]
    assert report["models"][report["best_model"]]["accuracy"] > 0.9
    assert list(model.predict(["Must have strong hands-on golang skills"])) == ["required"]
    assert list(model.predict(["Exposure to rust would be a plus"])) == ["preferred"]


def test_apriori_finds_the_planted_rule():
    """SYNTHETIC TEST DATA: skill 1 almost always appears with skill 0."""
    rng = np.random.default_rng(1)
    n = 1000
    X = np.zeros((n, 4), dtype=np.int8)
    X[:300, 0] = 3
    X[:270, 1] = 3                          # 90% of postings with skill 0 also have skill 1
    X[rng.choice(n, 200, replace=False), 2] = 2
    X[rng.choice(n, 200, replace=False), 3] = 2
    res = apriori(sparse.csr_matrix(X), ["A", "B", "C", "D"], min_support=0.05, min_conf=0.5, min_lift=1.5)
    rule = next(r for r in res["rules"] if r["if"] == ["A"] and r["then"] == "B")
    assert rule["confidence"] == pytest.approx(0.9)
    assert rule["lift"] == pytest.approx(0.9 / 0.27, rel=1e-3)
    assert not any(r["then"] in ("C", "D") for r in res["rules"])   # independent skills give no rules


def test_kmeans_recovers_separate_skill_bundles():
    """SYNTHETIC TEST DATA: three disjoint groups of skills (each posting asks for its whole group)."""
    rows = []
    for g in range(3):
        for _ in range(150):
            r = np.zeros(9, dtype=np.int8)
            r[3 * g:3 * g + 3] = 3
            rows.append(r)
    names = [f"S{i}" for i in range(9)]
    res = kmeans_bundles(sparse.csr_matrix(np.array(rows)), names, ["Software Developer"] * 450, k_range=range(2, 4))
    assert res["k"] == 3
    groups = sorted(sorted(s["skill"] for s in c["top_skills"]) for c in res["clusters"])
    assert groups == [["S0", "S1", "S2"], ["S3", "S4", "S5"], ["S6", "S7", "S8"]]


def test_shipped_market_model_is_consistent(market):
    from app.services.market_model import get_market

    m = get_market()
    assert m.levels.shape == (m.n, len(m.skills)) and m.n > 100000
    assert len(m.role) == len(m.title) == len(m.exp_min) == m.n
    opts = {o["value"]: o["postings"] for o in m.role_options()}
    assert opts["all"] == m.n and 0 < opts["it"] < m.n
    assert sum(v for k, v in opts.items() if k not in ("it", "all")) == opts["it"]
    d = m.demand(m.mask("it"))
    assert d["n"] == opts["it"] and 0 < d["skills"]["Python"]["share"] < 1
    assert d["skills"]["Python"]["jobs"] == d["skills"]["Python"]["required"] + d["skills"]["Python"]["preferred"] \
        + d["skills"]["Python"]["unspecified"]
    entry = m.mask("it", entry_level=True)
    assert 0 < entry.sum() < m.mask("it").sum()
    assert m.metrics["role_classifier"]["best_model"] in m.metrics["role_classifier"]["models"]


def test_role_model_reads_syllabus_like_text(market):
    from app.services.market_model import get_market

    m = get_market()
    ml = m.predict_roles("supervised learning, neural networks, deep learning with TensorFlow, Python, pandas, "
                         "regression and classification models, model evaluation", ["Machine Learning", "Python"])
    web = m.predict_roles("HTML, CSS, JavaScript, React components, responsive web pages, Node.js, REST APIs",
                          ["HTML", "CSS", "JavaScript", "React"])
    assert ml[0]["role"] == "Data Scientist / ML Engineer"
    assert web[0]["role"] == "Web / Full Stack Developer"
    assert abs(sum(r["probability"] for r in m.predict_roles("java spring sql", top=20)) - 1) < 1e-3


def test_course_role_is_none_for_non_technical_courses(market):
    from app.services.market_model import get_market

    m = get_market()
    assert m.course_role("Constitution of India: preamble, fundamental rights, parliament, judiciary, elections", []) is None
    web = m.course_role("Node JS, React JS and Django web programming", ["Node.js", "React", "Django", "JavaScript", "HTML"])
    assert web["role"] == "Web / Full Stack Developer" and web["probability"] > 0.5
