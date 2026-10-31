"""Skill extraction for job postings (reuses the Stage 1 taxonomy matcher).

Sources are kept separate:
  structured_required / structured_preferred / structured_listed — skills supplied in dataset columns
  nlp_description — matched inside the job description (required/preferred only when the text says so)
  nlp_title       — matched in the job title
A skill mentioned in an advert is not assumed to be mandatory: without an explicit cue it is 'unspecified'.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .skill_extractor import TaxonomyMatcher, _context_supported, clean_text

MATCHER_VERSION = "matcher-2"

_REQ_HEAD = re.compile(
    r"^\s*(required|requirements?|must[\s-]have|mandatory|essential|key\s+requirements?|"
    r"required\s+skills?|technical\s+skills?\s+required|minimum\s+qualifications?|what\s+you\s+(need|bring))\b", re.I)
_PREF_HEAD = re.compile(
    r"^\s*(preferred|good[\s-]to[\s-]have|nice[\s-]to[\s-]have|desirable|desired|bonus|added\s+advantage|"
    r"preferred\s+qualifications?|plus\s+points?)\b", re.I)
_NEUTRAL_HEAD = re.compile(
    r"^\s*(responsibilities|roles?\s+and\s+responsibilities|job\s+description|about\s+(us|the\s+company|the\s+role)|"
    r"what\s+you('|\s+wi)ll\s+do|benefits|perks|company\s+profile)\b", re.I)
_REQ_CUE = re.compile(r"\b(must\s+have|must\s+be|mandatory|required|is\s+a\s+must|essential|strong\s+(knowledge|experience)\s+(of|in))\b", re.I)
_PREF_CUE = re.compile(r"\b(preferred|good\s+to\s+have|nice\s+to\s+have|is\s+a\s+plus|would\s+be\s+a\s+plus|"
                       r"added\s+advantage|desirable|optional|bonus|an\s+advantage)\b", re.I)
_LEVEL_RANK = {"required": 3, "preferred": 2, "unspecified": 1}


@dataclass
class JobSkillHit:
    skill_id: int
    source: str
    requirement_level: str
    matched_text: str
    evidence_text: str
    extraction_method: str
    confidence: float
    is_ambiguous: bool


def _segments(desc: str) -> list[tuple[str, str]]:
    """Split a JD into (requirement_context, sentence) pairs, tracking requirement/preferred headings."""
    text = clean_text(desc)
    context = "unspecified"
    out: list[tuple[str, str]] = []
    for line in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z•\-*])", text):
        line = line.strip(" •*-\t")
        if not line:
            continue
        head = line if len(line) <= 80 else ""
        if head and _PREF_HEAD.match(head):
            context = "preferred"
        elif head and _REQ_HEAD.match(head):
            context = "required"
        elif head and _NEUTRAL_HEAD.match(head):
            context = "unspecified"
        level = context
        if _PREF_CUE.search(line):
            level = "preferred"
        elif _REQ_CUE.search(line):
            level = "required"
        out.append((level, line))
    return out


def _hits_in(text: str, matcher: TaxonomyMatcher, source: str, level: str, base_conf: float,
             evidence: str | None = None) -> list[JobSkillHit]:
    hits = []
    for s, e, entry in matcher.find(text):
        supported = _context_supported(entry, text, s, e)
        if not supported and entry.ambiguity_policy == "skip":
            continue
        ambiguous = not supported
        method = "dictionary" if entry.is_canonical else "alias"
        conf = 0.35 if ambiguous else base_conf - (0.0 if entry.is_canonical else 0.05)
        hits.append(JobSkillHit(entry.skill_id, source, level, text[s:e][:300], (evidence or text)[:600],
                                method, round(conf, 3), ambiguous))
    return hits


def extract_posting_skills(posting: dict, matcher: TaxonomyMatcher) -> tuple[list[JobSkillHit], list[str]]:
    """Returns (hits, unmapped structured tags). One hit per (skill, source), keeping the strongest level."""
    hits: list[JobSkillHit] = []
    unmapped: list[str] = []
    from .job_import import split_skills

    for field, source, level, conf in (("required_skills", "structured_required", "required", 0.92),
                                       ("preferred_skills", "structured_preferred", "preferred", 0.9),
                                       ("listed_skills", "structured_listed", "unspecified", 0.9)):
        raw = posting.get(field)
        for tag in split_skills(raw):
            found = _hits_in(tag, matcher, source, level, conf, evidence=f"{field}: {raw}")
            if found:
                hits.extend(found)
            else:
                unmapped.append(tag.lower())
    desc = posting.get("job_description")
    if desc:
        for level, sent in _segments(desc):
            hits.extend(_hits_in(sent, matcher, "nlp_description", level, 0.8))
    title = posting.get("job_title")
    if title:
        hits.extend(_hits_in(title, matcher, "nlp_title", "unspecified", 0.7, evidence=f"Job title: {title}"))
    best: dict[tuple[int, str], JobSkillHit] = {}
    for h in hits:
        k = (h.skill_id, h.source)
        cur = best.get(k)
        if cur is None or (h.is_ambiguous, -_LEVEL_RANK[h.requirement_level], -h.confidence) < (
                cur.is_ambiguous, -_LEVEL_RANK[cur.requirement_level], -cur.confidence):
            best[k] = h
    return list(best.values()), unmapped
