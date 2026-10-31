"""Rule-based syllabus structure detection.

Detects header metadata (university, programme, course code/title, semester, year, credits),
sections (objectives, outcomes, units, practicals, projects, textbooks, references) and
learning outcomes. Every value keeps its source text and page number; anything not found is
reported as unknown rather than guessed. Rules live in data/parsing_rules.json.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
         "xi": 11, "xii": 12}
METADATA_FIELDS = ("university_name", "programme_name", "course_code", "course_title", "semester",
                   "academic_year", "admission_batch", "credits")
_HOURS_RE = re.compile(r"[\(\[]?\s*\d{1,3}\s*(hours?|hrs?\.?|lectures?|periods?|sessions?)\s*[\)\]]?\s*$", re.I)
_HEADING_MAX_LEN = 160


@dataclass
class Line:
    page: int
    text: str


@dataclass
class ParsedSection:
    section_type: str
    title: str
    unit_number: str | None = None
    lines: list[Line] = field(default_factory=list)
    confidence: float = 0.5
    items: list[dict] = field(default_factory=list)
    course: str | None = None   # course heading this section belongs to (multi-course PDFs)

    @property
    def content(self) -> str:
        return "\n".join(ln.text for ln in self.lines)

    @property
    def page_start(self) -> int | None:
        return self.lines[0].page if self.lines else None

    @property
    def page_end(self) -> int | None:
        return self.lines[-1].page if self.lines else None


@dataclass
class ParsedOutcome:
    code: str | None
    text: str
    page: int | None
    kind: str            # course_outcome | objective
    bloom_level: str | None
    section_index: int | None


@dataclass
class ParseResult:
    metadata: dict
    sections: list[ParsedSection]
    outcomes: list[ParsedOutcome]


class Rules:
    def __init__(self, data: dict):
        self.raw = data
        self.headings: list[tuple[str, re.Pattern]] = []
        for stype, patterns in data["section_headings"].items():
            for p in patterns:
                self.headings.append((stype, re.compile(p, re.I)))
        cs = set(data.get("case_sensitive_fields", []))
        self.metadata = {
            f: [re.compile(p, 0 if f in cs else re.I) for p in pats]
            for f, pats in data["metadata_fields"].items()
        }
        self.scan_lines = int(data.get("metadata_scan_lines", 60))
        self.outcome_items = [re.compile(p, re.I) for p in data["outcome_item_patterns"]]
        self.list_items = [re.compile(p, re.I) for p in data["list_item_patterns"]]
        self.bloom = {lvl: set(v) for lvl, v in data.get("bloom_verbs", {}).items()}
        ch = data.get("course_heading") or {}
        self.course_followers = [re.compile(p, re.I) for p in ch.get("follower_patterns", [])]
        self.course_lookahead = int(ch.get("max_lookahead", 2))
        self.course_min_letters = int(ch.get("min_letters", 6))
        self.course_max_length = int(ch.get("max_length", 90))
        self.outcome_intros = [re.compile(p, re.I) for p in data.get("outcome_intro_patterns", [])]


@lru_cache
def load_rules(path: str) -> Rules:
    return Rules(json.loads(Path(path).read_text(encoding="utf-8")))


def _to_lines(pages: list[tuple[int, str]]) -> list[Line]:
    out: list[Line] = []
    for page_no, text in pages:
        for raw in text.split("\n"):
            t = re.sub(r"[ \t\u00a0]+", " ", raw).strip()
            if t:
                out.append(Line(page_no, t))
    return out


def _match_heading(rules: Rules, text: str):
    if len(text) > _HEADING_MAX_LEN:
        return None
    for stype, pat in rules.headings:
        m = pat.match(text)
        if m:
            return stype, m
    return None


_ACRONYMS = {"AI", "ML", "IOT", "OOP", "DBMS", "UI", "UX", "AR", "VR", "NLP", "OS", "JS", "IT", "II", "III", "IV"}
_SMALL_WORDS = {"and", "of", "the", "in", "for", "to", "with", "on", "a", "an", "through", "or"}


def _course_title(text: str) -> str:
    words = re.sub(r"\s+", " ", text.strip(" :-–.|")).split(" ")
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i and lw in _SMALL_WORDS:
            out.append(lw)
        elif w.strip("()&/,") in _ACRONYMS:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:].lower())
    return " ".join(out)


def _is_course_heading(rules: Rules, lines: list[Line], i: int) -> bool:
    """ALL-CAPS line followed (within a couple of lines) by a programme/semester line."""
    if not rules.course_followers:
        return False
    text = lines[i].text
    letters = [c for c in text if c.isalpha()]
    if len(letters) < rules.course_min_letters or len(text) > rules.course_max_length or ":" in text:
        return False
    if sum(c.isupper() for c in letters) / len(letters) < 0.9:
        return False
    if _match_heading(rules, text) or any(p.match(text) for p in rules.course_followers):
        return False
    for j in range(i + 1, min(len(lines), i + 1 + rules.course_lookahead)):
        if any(p.match(lines[j].text) for p in rules.course_followers):
            return True
    return False


def _clean_title(text: str) -> str:
    t = _HOURS_RE.sub("", text).strip(" :-–.|")
    return re.sub(r"\s+", " ", t)


def _merge_wrapped(lines: list[Line]) -> list[Line]:
    """Join lines that were wrapped mid-sentence by the PDF layout."""
    merged: list[Line] = []
    for ln in lines:
        if (
            merged
            and not re.search(r"[.;:!?)]$", merged[-1].text)
            and ln.text[:1].islower()
        ):
            merged[-1] = Line(merged[-1].page, f"{merged[-1].text} {ln.text}")
        else:
            merged.append(Line(ln.page, ln.text))
    return merged


def _split_topics(lines: list[Line]) -> list[dict]:
    items: list[dict] = []
    for ln in _merge_wrapped(lines):
        text = _HOURS_RE.sub("", ln.text).strip()
        if not text:
            continue
        for part in re.split(r";|\.\s+(?=[A-Z])", text):
            part = part.strip(" .,-–")
            if len(part) < 2:
                continue
            head, sep, rest = part.partition(":")
            if sep and 1 < len(head) <= 70 and rest.strip():
                subs = [s.strip(" .") for s in re.split(r",|\band\b(?=\s+[A-Z])", rest) if s.strip(" .")]
                items.append({"topic": head.strip(), "subtopics": subs, "page": ln.page})
            elif part.count(",") >= 2:
                for piece in part.split(","):
                    piece = piece.strip(" .")
                    if piece:
                        items.append({"topic": piece, "subtopics": [], "page": ln.page})
            else:
                items.append({"topic": part, "subtopics": [], "page": ln.page})
    return items


def _split_list(rules: Rules, lines: list[Line]) -> list[dict]:
    items: list[dict] = []
    for ln in lines:
        matched = None
        for pat in rules.list_items:
            matched = pat.match(ln.text)
            if matched:
                break
        if matched:
            gd = matched.groupdict()
            items.append({"code": gd.get("code"), "text": gd["text"].strip(), "page": ln.page})
        elif items and not re.search(r"[.;]$", items[-1]["text"]):
            items[-1]["text"] += " " + ln.text  # continuation of the previous item
        else:
            items.append({"code": None, "text": ln.text, "page": ln.page})
    return items


def bloom_level(rules: Rules, text: str) -> str | None:
    t = re.sub(r"^(students?|learners?)\s+(will|should)\s+be\s+able\s+to\s*", "", text.strip(), flags=re.I)
    t = re.sub(r"^(to|be\s+able\s+to)\s+", "", t, flags=re.I)
    for word in re.findall(r"[A-Za-z]+", t)[:4]:
        w = word.lower()
        for level, verbs in rules.bloom.items():
            if w in verbs:
                return level
    return None


def _extract_outcomes(rules: Rules, sec: ParsedSection, index: int, kind: str) -> list[ParsedOutcome]:
    out: list[ParsedOutcome] = []
    for ln in sec.lines:
        text = ln.text
        if any(p.search(text) for p in rules.outcome_intros) and not re.match(r"(?i)^C\.?O\.?\s?\d", text):
            continue  # introductory line ("After learning the contents of this course, students must be able to")
        matched = None
        for pat in rules.outcome_items:
            matched = pat.match(text)
            if matched:
                break
        if matched:
            gd = matched.groupdict()
            code = gd.get("code")
            code = re.sub(r"[\s.]", "", code).upper() if code and re.match(r"(?i)c\.?o|clo|lo", code) else code
            out.append(ParsedOutcome(code, gd["text"].strip(), ln.page, kind, None, index))
        elif out and not re.search(r"[.;]$", out[-1].text):
            out[-1].text += " " + text
        elif text.endswith(":") or re.match(r"(?i)^(students?|learners?) will be able to", text):
            continue  # introductory line ("Students will be able to:")
        elif len(text) > 15:
            out.append(ParsedOutcome(None, text, ln.page, kind, None, index))
    for o in out:
        o.text = re.sub(r"\s+", " ", o.text).strip()
        o.bloom_level = bloom_level(rules, o.text)
    return [o for o in out if len(o.text) >= 5]


def _detect_metadata(rules: Rules, lines: list[Line]) -> dict:
    scan = lines[: rules.scan_lines]
    meta: dict = {}
    for fname in METADATA_FIELDS:
        meta[fname] = {"value": None, "status": "unknown", "page": None, "source_text": None, "source": "parser"}
        for pat in rules.metadata.get(fname, []):
            hit = None
            for ln in scan:
                m = pat.search(ln.text)
                if m and m.groupdict().get("v"):
                    hit = (ln, m.group("v"))
                    break
            if hit:
                ln, value = hit
                value = re.sub(r"\s+", " ", value).strip(" :-–|,")
                if fname == "semester":
                    v = value.lower()
                    value = str(ROMAN.get(v, v))
                    if not value.isdigit() or not 1 <= int(value) <= 12:
                        continue
                if fname == "course_code":
                    value = value.upper().replace(" ", "")
                if fname == "course_title":
                    value = re.split(r"\s{2,}|(?i:\s+(course|subject)\s*code)|\s+credits?\b", value)[0].strip(" :-–|")
                if not value:
                    continue
                meta[fname] = {"value": value, "status": "detected", "page": ln.page,
                               "source_text": ln.text[:300], "source": "parser"}
                break
    return meta


def parse_syllabus(pages: list[tuple[int, str]], rules: Rules) -> ParseResult:
    lines = _to_lines(pages)
    sections: list[ParsedSection] = [ParsedSection("header", "Document header", confidence=0.6)]
    pending_unit_title = False
    course: str | None = None

    for idx, ln in enumerate(lines):
        if _is_course_heading(rules, lines, idx):
            course = _course_title(ln.text)
            sections.append(ParsedSection("course", course, confidence=0.8, course=course))
            pending_unit_title = False
            continue
        hit = _match_heading(rules, ln.text)
        if hit:
            stype, m = hit
            gd = m.groupdict()
            if stype == "unit":
                title = _clean_title(gd.get("title") or "")
                num = (gd.get("num") or "").upper()
                sec = ParsedSection("unit", title or f"Unit {num}", unit_number=num, confidence=0.85, course=course)
                pending_unit_title = not title
            else:
                remainder = ln.text[m.end():].lstrip(" ,:-–").rstrip()
                prev = sections[-1]
                if prev.section_type == stype and not prev.lines:
                    # e.g. "Course Outcomes" followed by "After completion of the course, students ...:"
                    if remainder and not remainder.endswith(":"):
                        prev.lines.append(Line(ln.page, remainder))
                    pending_unit_title = False
                    continue
                title = _clean_title(ln.text[: m.end()]) or stype.title()
                sec = ParsedSection(stype, title, confidence=0.85, course=course)
                if remainder and not (stype in ("outcomes", "objectives") and remainder.endswith(":")):
                    sec.lines.append(Line(ln.page, remainder))
                pending_unit_title = False
            sections.append(sec)
            continue
        current = sections[-1]
        # A bare "UNIT - IV" line may be followed by a short title line (not by unit content).
        if pending_unit_title and len(ln.text) <= 60 and "," not in ln.text and not ln.text.endswith("."):
            current.title = _clean_title(ln.text) or current.title
            pending_unit_title = False
            continue
        pending_unit_title = False
        current.lines.append(ln)

    # Drop empty, non-header sections that were probably false-positive headings.
    sections = [s for i, s in enumerate(sections) if i == 0 or s.lines or s.section_type in ("unit", "course")]

    outcomes: list[ParsedOutcome] = []
    for idx, sec in enumerate(sections):
        if sec.section_type == "unit":
            sec.items = _split_topics(sec.lines)
        elif sec.section_type in ("practical", "project", "textbooks", "references"):
            sec.items = _split_list(rules, sec.lines)
        elif sec.section_type in ("outcomes", "objectives"):
            kind = "course_outcome" if sec.section_type == "outcomes" else "objective"
            found = _extract_outcomes(rules, sec, idx, kind)
            sec.items = [{"code": o.code, "text": o.text, "page": o.page} for o in found]
            outcomes.extend(found)

    # Course outcomes stated outside an outcomes section (e.g. CO tables): "CO3: Apply ..."
    known = {o.code for o in outcomes if o.code}
    co_re = re.compile(r"^(?P<code>C\.?O\.?\s?\d{1,2})\s*[:.)\-–]\s*(?P<text>.{10,})$", re.I)
    for idx, sec in enumerate(sections):
        if sec.section_type in ("outcomes", "objectives"):
            continue
        for ln in sec.lines:
            m = co_re.match(ln.text)
            if m:
                code = re.sub(r"[\s.]", "", m.group("code")).upper()
                if code not in known:
                    known.add(code)
                    text = m.group("text").strip()
                    outcomes.append(ParsedOutcome(code, text, ln.page, "course_outcome", bloom_level(rules, text), idx))

    return ParseResult(metadata=_detect_metadata(rules, lines), sections=sections, outcomes=outcomes)
