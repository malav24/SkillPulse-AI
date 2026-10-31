"""Syllabus vs industry comparison (team module 3: recommendation engine).

Inputs : the skills found in the syllabus (with where they appear) + the trained market model.
Output : one JSON result with coverage, demand, gap, ML role alignment, skill-bundle readiness and
         recommendations explained by association rules. Stored on the analysis, so reopening it is instant.

Scores (0–100)
  Demand  D(s) = level-weighted share of the selected job postings asking for s (required 1.0,
                 unspecified 0.75, preferred 0.5 — levels predicted by MODEL 2), scaled to the top skill.
  Coverage C(s) = topic 35 + outcome 25 + practical 20 + project 10 + depth 10 (pending evidence × 0.8,
                 book-list-only mentions × 0.3).
  Gap     G(s) = max(0, D − C).
"""
from __future__ import annotations

from collections import Counter, defaultdict

from sqlalchemy import select

from ..models import CurriculumSkillEvidence, ExtractedSection, Skill, SyllabusDocument
from .market_model import MarketModel

WEIGHTS = {"topic": 35, "outcome": 25, "practical": 20, "project": 10, "depth": 10}
PENDING_FACTOR, REFERENCE_FACTOR = 0.8, 0.3
TOP_N, MIN_SHARE, MIN_JOBS = 50, 0.01, 3
COVERED_GAP, UNDEREMPHASIS_DEMAND = 15, 60

STATUS_LABELS = {"covered": "Well covered", "partially_covered": "Partly covered",
                 "underemphasized": "Needs practical work", "potentially_missing": "Missing", "uncertain": "Unclear"}
GROUP_LABELS = {
    "data_ai": "Data & AI", "framework_library": "Framework / library", "programming_language": "Programming language",
    "tool_platform": "Tool / platform", "web_mobile": "Web & mobile", "cs_fundamentals": "CS fundamentals",
    "engineering_domain": "Engineering", "professional_competency": "Professional skill",
    "method_technique": "Method / practice", "database": "Database", "cloud_devops": "Cloud & DevOps",
    "mathematics_statistics": "Mathematics & statistics", "security": "Security", "networking": "Networking",
}
GENERAL_SECTION = "General (outside any single course)"
_CONTENT_TYPES = ("course", "objectives", "outcomes", "unit", "practical", "project", "other")


# --- coverage ------------------------------------------------------------------------------------------------
def coverage_by_skill(db, doc: SyllabusDocument) -> dict[str, dict]:
    rows = db.execute(select(CurriculumSkillEvidence, Skill.canonical_name)
                      .join(Skill, Skill.id == CurriculumSkillEvidence.skill_id)
                      .where(CurriculumSkillEvidence.document_id == doc.id,
                             CurriculumSkillEvidence.review_status != "rejected")).all()
    by: dict[str, list] = defaultdict(list)
    for e, name in rows:
        by[name].append(e)
    out = {}
    for name, evs in by.items():
        comp = dict.fromkeys(WEIGHTS, 0.0)
        sections: set = set()
        for e in evs:
            if e.evidence_type == "ambiguous":
                continue
            fac = 1.0 if e.review_status in ("confirmed", "edited", "merged") else PENDING_FACTOR
            if e.section_type in ("textbooks", "references"):
                comp["topic"] = max(comp["topic"], fac * REFERENCE_FACTOR)
                continue
            sections.add(e.section_id or (e.section_title, e.page_number))
            comp["topic"] = max(comp["topic"], fac)
            if e.evidence_type == "learning_outcome":
                comp["outcome"] = max(comp["outcome"], fac)
            elif e.evidence_type == "practical":
                key = "project" if e.section_type == "project" else "practical"
                comp[key] = max(comp[key], fac)
        comp["depth"] = round(min(1.0, max(0, len(sections) - 1) / 3), 3)
        score = round(sum(WEIGHTS[k] * v for k, v in comp.items()), 1)
        if score > 0:
            out[name] = {"score": score, "components": comp}
    return out


# --- syllabus structure ---------------------------------------------------------------------------------------
def course_breakdown(db, doc: SyllabusDocument, fallback: str) -> tuple[list[dict], dict[str, str]]:
    """Skills per course (in PDF order) and the text of each course (for MODEL 1)."""
    rows = db.execute(
        select(ExtractedSection.course_label, Skill.canonical_name, CurriculumSkillEvidence.evidence_type)
        .join(ExtractedSection, ExtractedSection.id == CurriculumSkillEvidence.section_id, isouter=True)
        .join(Skill, Skill.id == CurriculumSkillEvidence.skill_id)
        .where(CurriculumSkillEvidence.document_id == doc.id,
               CurriculumSkillEvidence.review_status != "rejected",
               CurriculumSkillEvidence.section_type.notin_(("textbooks", "references")))).all()
    sections = db.scalars(select(ExtractedSection).where(ExtractedSection.document_id == doc.id)
                          .order_by(ExtractedSection.order_index)).all()
    has_courses = any(s.section_type == "course" for s in sections)
    general = GENERAL_SECTION if has_courses else fallback
    texts: dict[str, list[str]] = defaultdict(list)
    order: list[str] = []
    for s in sections:
        name = (s.course_label or "").strip() or general
        if s.section_type == "course" and name not in order:
            order.append(name)
        if s.section_type in _CONTENT_TYPES:
            texts[name].append(f"{s.title}. {s.content}")
    by_course: dict[str, Counter] = defaultdict(Counter)
    practical: dict[str, set] = defaultdict(set)
    for label, skill, etype in rows:
        name = (label or "").strip() or general
        by_course[name][skill] += 1
        if etype == "practical":
            practical[name].add(skill)
    for name in list(by_course) + list(texts):
        if name not in order:
            order.append(name)
    courses = [{"name": n, "skills": [{"skill": s, "mentions": c, "practical": s in practical[n]}
                                      for s, c in by_course[n].most_common()]}
               for n in order if by_course[n] or texts.get(n)]
    return courses, {n: " ".join(t)[:20000] for n, t in texts.items()}


# --- main computation -----------------------------------------------------------------------------------------
def _category(D: float, C: float, comps: dict | None, low_support: bool) -> str:
    if low_support:
        return "uncertain"
    if C == 0:
        return "potentially_missing"
    if max(0.0, D - C) <= COVERED_GAP:
        return "covered"
    if comps and not (comps["outcome"] or comps["practical"] or comps["project"]) and D >= UNDEREMPHASIS_DEMAND:
        return "underemphasized"
    return "partially_covered"


def _recommendation(r: dict, rule: dict | None) -> dict | None:
    s, pct, C, st = r["skill"], r["demand_pct"], r["coverage"], r["status_key"]
    req = f" Required (not just preferred) in {r['required_pct']:.0f}% of the postings that mention it." \
        if r["required_pct"] >= 25 else ""
    because = ""
    if rule:
        because = (f" Students already learn {' and '.join(rule['if'])}; {100 * rule['confidence']:.0f}% of jobs asking "
                   f"for {' + '.join(rule['if'])} also ask for {s} (lift {rule['lift']:.1f}).")
    if st == "potentially_missing":
        how = ("Build it in through presentations, team projects or written reports and state it as a course outcome."
               if r["group_key"] == "professional_competency" else "Consider adding it as a unit topic with a lab exercise.")
        return {"action": "Add to syllabus", "kind": "add",
                "text": f"{s} is asked for in {pct:.1f}% of the relevant job postings but was not found in this syllabus."
                        f"{because}{req} {how}"}
    if st == "underemphasized":
        return {"action": "Add practical work", "kind": "practice",
                "text": f"{s} is mentioned only as a topic (no lab, project or learning outcome) while {pct:.1f}% of the "
                        f"relevant jobs ask for it.{req} Add hands-on work and an outcome for it."}
    if st == "partially_covered":
        return {"action": "Strengthen", "kind": "strengthen",
                "text": f"{s} is covered only partly (coverage {C:.0f}/100) compared with industry demand "
                        f"({pct:.1f}% of jobs).{because} Add depth, a learning outcome or a practical."}
    return None


def compare(db, doc: SyllabusDocument, market: MarketModel, job_field: str, entry_level: bool,
            course_name: str) -> dict:
    coverage = coverage_by_skill(db, doc)
    mask = market.mask(job_field, entry_level)
    dem = market.demand(mask)
    n = dem["n"]
    if n == 0:
        raise ValueError("No job postings match this selection. Choose a broader job field.")
    cand = [(name, d) for name, d in dem["skills"].items() if d["share"] >= MIN_SHARE and d["jobs"] >= MIN_JOBS]
    cand.sort(key=lambda x: -x[1]["weighted_share"])
    cand = cand[:TOP_N]
    top_ws = cand[0][1]["weighted_share"] if cand else 1.0
    category_of = {s["name"]: s["category"] for s in market.skills}
    taught = set(coverage)
    low_support = n < 30
    skills = []
    for name, d in cand:
        D = round(100 * d["weighted_share"] / top_ws, 1)
        cov = coverage.get(name)
        C = cov["score"] if cov else 0.0
        comps = cov["components"] if cov else None
        cat = _category(D, C, comps, low_support)
        G = round(max(0.0, D - C), 1)
        reliability = min(1.0, d["jobs"] / 30)
        skills.append({
            "skill": name, "group": GROUP_LABELS.get(category_of.get(name, ""), category_of.get(name, "")),
            "group_key": category_of.get(name, ""), "status_key": cat, "status": STATUS_LABELS[cat],
            "demand_pct": round(100 * d["share"], 1), "jobs": d["jobs"], "demand_score": D,
            "required_pct": round(100 * d["required"] / d["jobs"], 1), "preferred_pct": round(100 * d["preferred"] / d["jobs"], 1),
            "coverage": C, "gap": G, "priority": round(G * reliability, 2),
            "in_syllabus_as": [k for k in ("topic", "outcome", "practical", "project") if comps and comps[k]],
        })
    # recommendations (MODEL 3a explains them)
    recs = []
    for r in sorted(skills, key=lambda x: (-x["priority"], -x["demand_pct"])):
        rules = market.rules_for(r["skill"], taught)
        rec = _recommendation(r, rules[0] if rules else None)
        if rec:
            recs.append({"skill": r["skill"], "demand_pct": r["demand_pct"], "required_pct": r["required_pct"],
                         "priority": "High" if r["demand_pct"] >= 10 else "Medium" if r["demand_pct"] >= 4 else "Low",
                         "_p": r["priority"], **rec, "rule": rules[0] if rules else None})
        if len(recs) == 12:
            break
    rank = {"High": 0, "Medium": 1, "Low": 2}
    recs.sort(key=lambda x: (rank[x["priority"]], -x.pop("_p")))
    m = market.metrics
    svm1 = m["role_classifier"]["best_model"]
    svm2 = m["requirement_classifier"]["best_model"]
    for rec in recs:
        rec["example_jobs"] = market.example_titles(mask, rec["skill"])
        rec["top_role"] = market.skill_role(mask, rec["skill"])
        rec["ml"] = [f"Demand weighted by {svm2} (required vs preferred: {rec['required_pct']:.0f}% required)"]
        if rec["top_role"]:
            rec["ml"].append(f"Mostly needed for {rec['top_role']['role']} jobs ({100 * rec['top_role']['share']:.0f}%), "
                             f"roles labelled by {svm1}")
        if rec["rule"]:
            rec["ml"].append(f"Apriori rule: {' + '.join(rec['rule']['if'])} → {rec['skill']} "
                             f"(confidence {100 * rec['rule']['confidence']:.0f}%, lift {rec['rule']['lift']:.1f})")
    # MODEL 1: role alignment of the whole syllabus and of each course
    courses, texts = course_breakdown(db, doc, course_name)
    full_text = " ".join(texts.values())
    roles = market.predict_roles(full_text, list(coverage), top=6)
    for c in courses:
        t = texts.get(c["name"], "")
        c["role"] = market.course_role(t, [k["skill"] for k in c["skills"]]) if c["skills"] else None
    # MODEL 3b: readiness for each skill bundle (K-Means cluster)
    cov_score = {k: v["score"] for k, v in coverage.items()}
    bundles = []
    for b in market.bundles:
        tot = sum(s["frequency"] for s in b["top_skills"]) or 1
        ready = sum(s["frequency"] * cov_score.get(s["skill"], 0) / 100 for s in b["top_skills"]) / tot
        bundles.append({"name": b["name"], "share": b["share"], "main_role": b["main_role"],
                        "readiness": round(100 * ready, 1),
                        "covered": [s["skill"] for s in b["top_skills"] if cov_score.get(s["skill"], 0) > 0],
                        "missing": [s["skill"] for s in b["top_skills"] if cov_score.get(s["skill"], 0) == 0]})
    counts = Counter(s["status_key"] for s in skills)
    partly = counts.get("partially_covered", 0) + counts.get("underemphasized", 0)
    return {
        "summary": {
            "pages": doc.page_count, "courses": len([c for c in courses if c["name"] != GENERAL_SECTION]),
            "skills_taught": len(coverage), "industry_skills_compared": len(skills),
            "well_covered": counts.get("covered", 0), "partly_covered": partly,
            "missing": counts.get("potentially_missing", 0), "unclear": counts.get("uncertain", 0),
            "match_pct": round(100 * (counts.get("covered", 0) + 0.5 * partly) / max(1, len(skills))),
            "top_role": roles[0] if roles else None,
        },
        "market": {"postings_compared": n, "postings": market.n, "period": market.period},
        "skills": sorted(skills, key=lambda x: -x["demand_pct"]),
        "recommendations": recs, "courses": courses, "roles": roles, "bundles": bundles,
        "models": {
            "role_classifier": {"algorithm": m["role_classifier"]["best_model"],
                                "accuracy": m["role_classifier"]["models"][m["role_classifier"]["best_model"]]["accuracy"]},
            "requirement_classifier": {"algorithm": m["requirement_classifier"]["best_model"],
                                       "accuracy": m["requirement_classifier"]["models"][m["requirement_classifier"]["best_model"]]["accuracy"]},
            "recommender": {"rules": m["skill_recommender"]["apriori"]["rule_count"],
                            "clusters": m["skill_recommender"]["kmeans"]["k"]},
        },
        "warnings": ([f"Only {n} job postings match this selection, so results are less reliable."] if n < 100 else []),
        "note": "\"Missing\" means the skill was not found in the uploaded PDF. It may still be taught in practice "
                "or in another semester's syllabus.",
    }
