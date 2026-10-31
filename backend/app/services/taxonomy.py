"""Load the configurable skill taxonomy into the database and build matchers from it."""
from __future__ import annotations

import json
import re
import logging
import threading
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..common import slugify
from ..config import get_settings
from ..models import Skill, SkillAlias
from .skill_extractor import AliasEntry, FuzzyMatcher, SemanticMatcher, TaxonomyMatcher

log = logging.getLogger("skillpulse.taxonomy")

_lock = threading.Lock()
_cache: dict = {"version": -1}
_version = 0


def invalidate() -> None:
    """Call after any change to skills/aliases so matchers are rebuilt."""
    global _version
    with _lock:
        _version += 1


def alias_key(alias: str, case_sensitive: bool) -> str:
    # Case-sensitive acronyms ("AWS") and lowercase words ("aws") are distinct keys.
    return ("cs:" + alias.strip()) if case_sensitive else re.sub(r"\s+", " ", alias.strip().lower())


def seed_taxonomy(db: Session, path: str | None = None) -> dict:
    """Idempotently insert taxonomy skills/aliases. Never overwrites user-edited (custom) skills."""
    path = path or get_settings().taxonomy_path
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    existing = {s.slug: s for s in db.scalars(select(Skill).options(selectinload(Skill.aliases)))}
    taken_aliases = set(db.scalars(select(SkillAlias.normalized_alias)))
    created_skills = created_aliases = 0
    for item in data["skills"]:
        slug = slugify(item["name"])
        skill = existing.get(slug)
        if skill is None:
            skill = Skill(canonical_name=item["name"], slug=slug, category=item["category"],
                          description=item.get("description"), is_custom=False)
            db.add(skill)
            db.flush()
            existing[slug] = skill
            created_skills += 1
        for raw in item.get("aliases", []):
            spec = {"alias": raw, "case_sensitive": False} if isinstance(raw, str) else raw
            key = alias_key(spec["alias"], spec.get("case_sensitive", False))
            if key in taken_aliases or key == alias_key(item["name"], False):
                continue
            db.add(SkillAlias(
                skill_id=skill.id, alias=spec["alias"], normalized_alias=key,
                case_sensitive=spec.get("case_sensitive", False),
                context_terms=spec.get("context_terms"),
                ambiguity_policy=spec.get("ambiguity_policy", "flag"),
            ))
            taken_aliases.add(key)
            created_aliases += 1
    removed = 0
    for raw in data.get("removed_aliases", []):  # aliases withdrawn because they caused false matches
        for a in db.scalars(select(SkillAlias).where(SkillAlias.normalized_alias == alias_key(raw, False))):
            db.delete(a)
            removed += 1
    db.commit()
    if created_skills or created_aliases or removed:
        invalidate()
    return {"skills_created": created_skills, "aliases_created": created_aliases,
            "taxonomy_version": data.get("version")}


def _entries(db: Session) -> list[AliasEntry]:
    entries: list[AliasEntry] = []
    skills = db.scalars(select(Skill).where(Skill.is_active.is_(True)).options(selectinload(Skill.aliases))).all()
    for s in skills:
        entries.append(AliasEntry(s.id, s.canonical_name, s.category, s.canonical_name, True,
                                  False, (), "flag"))
        for a in s.aliases:
            entries.append(AliasEntry(s.id, s.canonical_name, s.category, a.alias, False, a.case_sensitive,
                                      tuple(a.context_terms or ()), a.ambiguity_policy))
    return entries


def get_matchers(db: Session) -> tuple[TaxonomyMatcher, FuzzyMatcher, SemanticMatcher | None]:
    s = get_settings()
    with _lock:
        if _cache.get("version") == _version:
            return _cache["matcher"], _cache["fuzzy"], _cache["semantic"]
        version = _version
    entries = _entries(db)
    matcher = TaxonomyMatcher(entries)
    fuzzy = FuzzyMatcher(entries)
    semantic = SemanticMatcher.load(s.semantic_model, entries, s.semantic_enabled)
    with _lock:
        _cache.update(version=version, matcher=matcher, fuzzy=fuzzy, semantic=semantic)
    return matcher, fuzzy, semantic
