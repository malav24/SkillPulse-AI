"""Hybrid skill extraction.

1. Clean text without destroying technical tokens (C++, C#, Node.js, .NET, CI/CD, PL/SQL).
2. Segment sections into sentences / list items (spaCy sentencizer; a full spaCy model is used
   for noun chunks when installed).
3. Match a configurable taxonomy (canonical names + aliases) with boundary-aware patterns.
4. Normalise aliases to canonical skills; acronyms with context_terms only map confidently when
   supporting context is nearby (e.g. AWS -> Amazon Web Services needs a cloud term).
5. Fuzzy (TF-IDF char n-gram, scikit-learn) and optional semantic (Sentence Transformers) matching
   of unmatched candidate phrases.
6. Every candidate carries evidence text, page, section, method, confidence and evidence type.

A mention in a syllabus is evidence that a topic is covered — never proof of mastery.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

log = logging.getLogger("skillpulse.nlp")

_BOUNDARY_L = r"(?<![A-Za-z0-9_+#])"
_BOUNDARY_R = r"(?![A-Za-z0-9_+#]|\.[A-Za-z0-9])"
CONTEXT_WINDOW = 160


# --- Text cleaning -------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalise unicode/whitespace and repair hyphenated line breaks, preserving technical tokens."""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-").replace("­", "")
    # "pro-\ngramming" -> "programming" (only lowercase continuation, so "C-\nDAC" style stays)
    t = re.sub(r"([a-z])-\n([a-z])", r"\1\2", t)
    t = re.sub(r"[ \t ]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return t.strip()


# --- Sentence segmentation ------------------------------------------------------------
@lru_cache
def get_nlp(model_name: str = "en_core_web_sm"):
    """Return (nlp, has_parser). Falls back to a blank English pipeline with a sentencizer."""
    import spacy

    try:
        nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
        if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp, "parser" in nlp.pipe_names
    except Exception:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp, False


def segment(text: str, model_name: str) -> list[tuple[int, int, str]]:
    """Split into sentence-like units (list items are kept separate). Returns (start, end, text)."""
    nlp, _ = get_nlp(model_name)
    segments: list[tuple[int, int, str]] = []
    offset = 0
    for block in text.split("\n"):
        start = text.find(block, offset) if block else offset
        offset = start + len(block)
        if not block.strip():
            continue
        if len(block) < 200:
            segments.append((start, start + len(block), block))
            continue
        doc = nlp(block)
        for sent in doc.sents:
            s = sent.text.strip()
            if s:
                segments.append((start + sent.start_char, start + sent.end_char, s))
    return segments


# --- Taxonomy matcher ------------------------------------------------------------------
@dataclass(frozen=True)
class AliasEntry:
    skill_id: int
    skill_name: str
    category: str
    alias: str
    is_canonical: bool
    case_sensitive: bool
    context_terms: tuple[str, ...]
    ambiguity_policy: str


@dataclass
class Candidate:
    skill_id: int
    skill_name: str
    category: str
    matched_text: str
    evidence_text: str
    page_number: int | None
    char_start: int | None
    char_end: int | None
    section_index: int | None
    section_type: str | None
    section_title: str | None
    extraction_method: str
    evidence_type: str
    confidence: float
    confidence_reason: str


def _alias_pattern(alias: str, case_sensitive: bool) -> str:
    parts = re.split(r"(\s+|-)", alias.strip())
    out = []
    for p in parts:
        if not p:
            continue
        if p.isspace():
            out.append(r"[\s\-]+")
        elif p == "-":
            out.append(r"[\s\-]?")
        else:
            out.append(re.escape(p))
    body = "".join(out)
    # Allow simple plurals for case-insensitive word aliases ("neural network" -> "neural networks").
    if not case_sensitive and re.search(r"[a-z]$", alias, re.I) and len(alias) > 3:
        body += r"(?:s|es)?"
    return _BOUNDARY_L + body + _BOUNDARY_R


def _norm_key(text: str) -> str:
    return re.sub(r"[\s\-]+", "", text.lower())


class TaxonomyMatcher:
    """Boundary-aware, longest-match-wins alias matcher.

    All aliases are compiled into one alternation per case mode (longest alias first), so a text is
    scanned twice regardless of taxonomy size — fast enough for tens of thousands of job records.
    Matches are mapped back to taxonomy entries by a normalised key.
    """

    def __init__(self, entries: Iterable[AliasEntry]):
        self.entries = list(entries)
        self.by_skill: dict[int, AliasEntry] = {e.skill_id: e for e in self.entries if e.is_canonical}
        ordered = sorted(self.entries, key=lambda x: (len(x.alias), x.is_canonical), reverse=True)
        self._ci_lookup: dict[str, AliasEntry] = {}
        self._cs_lookup: dict[str, AliasEntry] = {}
        self._ci_patterns: dict[str, re.Pattern] = {}
        ci_parts, cs_parts = [], []
        for e in ordered:
            body = _alias_pattern(e.alias, e.case_sensitive)[len(_BOUNDARY_L): -len(_BOUNDARY_R)]
            if e.case_sensitive:
                if e.alias not in self._cs_lookup:
                    self._cs_lookup[e.alias] = e
                    cs_parts.append(body)
            else:
                key = _norm_key(e.alias)
                if key not in self._ci_lookup or (e.is_canonical and not self._ci_lookup[key].is_canonical):
                    self._ci_lookup[key] = e
                self._ci_patterns.setdefault(key, re.compile(r"^(?:" + body + r")$", re.I))
                ci_parts.append(body)
        self._ci = re.compile(_BOUNDARY_L + "(?:" + "|".join(ci_parts) + ")" + _BOUNDARY_R, re.I) if ci_parts else None
        self._cs = re.compile(_BOUNDARY_L + "(?:" + "|".join(cs_parts) + ")" + _BOUNDARY_R) if cs_parts else None

    def _resolve_ci(self, matched: str) -> AliasEntry | None:
        key = _norm_key(matched)
        for k in (key, key[:-2] if key.endswith("es") else None, key[:-1] if key.endswith("s") else None):
            if k and k in self._ci_lookup:
                return self._ci_lookup[k]
        for k, pat in self._ci_patterns.items():  # rare fallback for unusual spacing/plural forms
            if pat.match(matched):
                return self._ci_lookup[k]
        return None

    def find(self, text: str) -> list[tuple[int, int, AliasEntry]]:
        hits: list[tuple[int, int, AliasEntry]] = []
        if self._ci is not None:
            for m in self._ci.finditer(text):
                entry = self._resolve_ci(m.group(0))
                if entry:
                    hits.append((m.start(), m.end(), entry))
        if self._cs is not None:
            for m in self._cs.finditer(text):
                entry = self._cs_lookup.get(m.group(0))
                if entry:
                    hits.append((m.start(), m.end(), entry))
        # Longest-match-wins, non-overlapping.
        hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
        taken: list[tuple[int, int]] = []
        result = []
        for s, e, entry in hits:
            if any(s < te and e > ts for ts, te in taken):
                continue
            taken.append((s, e))
            result.append((s, e, entry))
        result.sort(key=lambda h: h[0])
        return result


def _context_supported(entry: AliasEntry, text: str, start: int, end: int) -> bool:
    if not entry.context_terms:
        return True
    # Look either side of the match, excluding the matched token itself.
    window = (text[max(0, start - CONTEXT_WINDOW): start] + " " + text[end: end + CONTEXT_WINDOW]).lower()
    return any(re.search(r"(?<![a-z])" + re.escape(t.lower()) + r"(?:s|es)?(?![a-z])", window)
               for t in entry.context_terms)


# --- Evidence typing & confidence ---------------------------------------------------------
PRACTICAL_SECTIONS = {"practical", "project"}
OUTCOME_SECTIONS = {"outcomes", "objectives"}


def evidence_type_for(section_type: str | None, ambiguous: bool) -> str:
    if ambiguous:
        return "ambiguous"
    if section_type in OUTCOME_SECTIONS:
        return "learning_outcome"
    if section_type in PRACTICAL_SECTIONS:
        return "practical"
    return "topic_mention"


def score(method: str, section_type: str | None, ambiguous: bool, similarity: float | None = None) -> tuple[float, str]:
    if ambiguous:
        return 0.35, "Ambiguous term without supporting context; needs review"
    base = {"dictionary": 0.9, "alias": 0.8, "fuzzy": 0.55, "semantic": 0.6}[method]
    reason = {
        "dictionary": "Exact canonical skill name",
        "alias": "Matched a known alias",
        "fuzzy": "Approximate lexical match",
        "semantic": "Semantic similarity to taxonomy",
    }[method]
    if similarity is not None and method in ("fuzzy", "semantic"):
        base = round(base * 0.5 + similarity * 0.4, 3)
        reason += f" (similarity {similarity:.2f})"
    if section_type in OUTCOME_SECTIONS | PRACTICAL_SECTIONS:
        base += 0.05
        reason += "; stated in " + ("an outcome/objective" if section_type in OUTCOME_SECTIONS else "a lab/project")
    elif section_type in ("textbooks", "references"):
        base -= 0.25
        reason += "; only in a book/reference title"
    return round(max(0.05, min(base, 0.97)), 3), reason


# --- Candidate phrase generation for fuzzy/semantic matching ------------------------------
_STOP_PHRASES = re.compile(
    r"^(introduction|overview|basics|fundamentals|concepts?|applications?|types|examples?|case study|"
    r"unit|module|hours?|marks|total|text ?books?|references?)$",
    re.I,
)


def candidate_phrases(text: str, model_name: str) -> list[tuple[int, int, str]]:
    nlp, has_parser = get_nlp(model_name)
    phrases: list[tuple[int, int, str]] = []
    if has_parser:
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            p = chunk.text.strip()
            if 4 <= len(p) <= 60 and 1 <= len(p.split()) <= 5:
                phrases.append((chunk.start_char, chunk.end_char, p))
        return phrases
    for m in re.finditer(r"[^,;:.()\n\[\]]+", text):
        p = m.group(0).strip()
        p = re.sub(r"^(introduction to|basics of|fundamentals of|concepts? of|overview of|and|the|of)\s+", "", p, flags=re.I)
        words = p.split()
        if 1 <= len(words) <= 5 and 4 <= len(p) <= 60 and not _STOP_PHRASES.match(p) and not p.isdigit():
            start = text.find(p, m.start())
            phrases.append((start, start + len(p), p))
    return phrases


class FuzzyMatcher:
    """TF-IDF character n-gram similarity — a local, model-free approximate matcher."""

    def __init__(self, entries: list[AliasEntry]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.labels = [e for e in entries if len(e.alias) >= 5 and not e.case_sensitive and not e.context_terms]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), lowercase=True)
        self.matrix = self.vectorizer.fit_transform([e.alias for e in self.labels]) if self.labels else None

    def best(self, phrases: list[str], threshold: float) -> list[tuple[int, AliasEntry, float]]:
        if self.matrix is None or not phrases:
            return []
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(self.vectorizer.transform(phrases), self.matrix)
        out = []
        for i, row in enumerate(sims):
            j = int(row.argmax())
            if row[j] >= threshold:
                out.append((i, self.labels[j], float(row[j])))
        return out


class SemanticMatcher:
    """Sentence-Transformers matcher; only active when the library and model are available locally."""

    _instance: "SemanticMatcher | None" = None
    status: dict = {"available": False, "reason": "not initialised"}

    def __init__(self, model, entries: list[AliasEntry]):
        self.model = model
        self.labels = [e for e in entries if e.is_canonical]
        self.embeddings = model.encode([e.skill_name for e in self.labels], normalize_embeddings=True)

    @classmethod
    def load(cls, model_name: str, entries: list[AliasEntry], enabled: bool) -> "SemanticMatcher | None":
        if not enabled:
            cls.status = {"available": False, "reason": "disabled by SEMANTIC_ENABLED=false"}
            return None
        try:
            from sentence_transformers import SentenceTransformer  # optional dependency
        except ImportError:
            cls.status = {"available": False, "reason": "sentence-transformers not installed (optional)"}
            return None
        try:
            model = SentenceTransformer(model_name)
            inst = cls(model, entries)
            cls.status = {"available": True, "model": model_name}
            return inst
        except Exception as exc:
            log.warning("Semantic model unavailable: %s", exc)
            cls.status = {"available": False, "reason": f"model could not be loaded: {exc}"}
            return None

    def best(self, phrases: list[str], threshold: float) -> list[tuple[int, AliasEntry, float]]:
        if not phrases:
            return []
        import numpy as np

        emb = self.model.encode(phrases, normalize_embeddings=True)
        sims = np.asarray(emb) @ np.asarray(self.embeddings).T
        out = []
        for i, row in enumerate(sims):
            j = int(row.argmax())
            if row[j] >= threshold:
                out.append((i, self.labels[j], float(row[j])))
        return out


# --- Main extraction ---------------------------------------------------------------------
@dataclass
class SectionInput:
    index: int
    section_type: str
    title: str
    lines: list[tuple[int, str]]   # (page_number, line text)


def extract_skills(
    sections: list[SectionInput],
    matcher: TaxonomyMatcher,
    *,
    model_name: str = "en_core_web_sm",
    fuzzy: FuzzyMatcher | None = None,
    fuzzy_threshold: float = 0.78,
    semantic: SemanticMatcher | None = None,
    semantic_threshold: float = 0.62,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple] = set()

    for sec in sections:
        # Rebuild section text with a char->page map so evidence keeps its page number.
        text_parts: list[str] = []
        page_spans: list[tuple[int, int, int]] = []
        pos = 0
        for page, line in sec.lines:
            cleaned = clean_text(line)
            if not cleaned:
                continue
            text_parts.append(cleaned)
            page_spans.append((pos, pos + len(cleaned), page))
            pos += len(cleaned) + 1
        text = "\n".join(text_parts)
        if not text:
            continue

        def page_at(i: int) -> int | None:
            for s, e, p in page_spans:
                if s <= i <= e:
                    return p
            return page_spans[-1][2] if page_spans else None

        segments = segment(text, model_name)

        def evidence_at(s: int, e: int) -> str:
            for ss, se, st in segments:
                if ss <= s and e <= se + 1:
                    return st[:600]
            return text[max(0, s - 120): e + 120].replace("\n", " ")

        covered: list[tuple[int, int]] = []
        for start, end, entry in matcher.find(text):
            if end - start == 1 and (sec.section_type in ("references", "textbooks")
                                     or text[start - 1:start] == "." or text[end:end + 1] == "."):
                continue  # one-letter names (C, R) next to a dot, or in book lists, are author initials
            supported = _context_supported(entry, text, start, end)
            if not supported and entry.ambiguity_policy == "skip":
                continue
            ambiguous = not supported
            method = "dictionary" if entry.is_canonical else "alias"
            conf, reason = score(method, sec.section_type, ambiguous)
            page = page_at(start)
            key = (entry.skill_id, sec.index, page, start)
            if key in seen:
                continue
            seen.add(key)
            covered.append((start, end))
            candidates.append(Candidate(
                skill_id=entry.skill_id, skill_name=entry.skill_name, category=entry.category,
                matched_text=text[start:end], evidence_text=evidence_at(start, end), page_number=page,
                char_start=start, char_end=end, section_index=sec.index, section_type=sec.section_type,
                section_title=sec.title, extraction_method=method,
                evidence_type=evidence_type_for(sec.section_type, ambiguous), confidence=conf,
                confidence_reason=reason,
            ))

        # Fuzzy / semantic matching for phrases that the dictionary did not cover.
        if sec.section_type in ("header", "assessment", "textbooks", "references") or not (fuzzy or semantic):
            continue
        # Approximate matching only for multi-word phrases: single words ("design", "framework") are too generic.
        phrases = [p for p in candidate_phrases(text, model_name)
                   if len(p[2].split()) >= 2 and not any(p[0] < ce and p[1] > cs for cs, ce in covered)]
        if not phrases:
            continue
        found_skills_here = {c.skill_id for c in candidates if c.section_index == sec.index}
        matches: list[tuple[int, AliasEntry, float, str]] = []
        if semantic:
            matches += [(i, e, s, "semantic") for i, e, s in semantic.best([p[2] for p in phrases], semantic_threshold)]
        if fuzzy:
            already = {i for i, *_ in matches}
            matches += [(i, e, s, "fuzzy") for i, e, s in fuzzy.best([p[2] for p in phrases], fuzzy_threshold)
                        if i not in already]
        for i, entry, sim, method in matches:
            if entry.skill_id in found_skills_here:
                continue
            start, end, phrase = phrases[i]
            found_skills_here.add(entry.skill_id)
            conf, reason = score(method, sec.section_type, False, sim)
            candidates.append(Candidate(
                skill_id=entry.skill_id, skill_name=entry.skill_name, category=entry.category,
                matched_text=phrase, evidence_text=evidence_at(start, end), page_number=page_at(start),
                char_start=start, char_end=end, section_index=sec.index, section_type=sec.section_type,
                section_title=sec.title, extraction_method=method,
                evidence_type=evidence_type_for(sec.section_type, False), confidence=conf,
                confidence_reason=reason,
            ))
    return candidates
