from app.services.skill_extractor import (
    AliasEntry,
    FuzzyMatcher,
    SectionInput,
    TaxonomyMatcher,
    clean_text,
    extract_skills,
)


def _matcher():
    E = AliasEntry
    entries = [
        E(1, "Machine Learning", "data_ai", "Machine Learning", True, False, (), "flag"),
        E(1, "Machine Learning", "data_ai", "ML", False, True, ("learning", "model", "data", "algorithm"), "flag"),
        E(2, "Natural Language Processing", "data_ai", "Natural Language Processing", True, False, (), "flag"),
        E(2, "Natural Language Processing", "data_ai", "NLP", False, True, (), "flag"),
        E(3, "SQL", "database", "SQL", True, False, (), "flag"),
        E(4, "Amazon Web Services", "cloud_devops", "Amazon Web Services", True, False, (), "flag"),
        E(4, "Amazon Web Services", "cloud_devops", "AWS", False, True, ("cloud", "ec2", "s3"), "flag"),
        E(5, "C++", "programming_language", "C++", True, False, (), "flag"),
        E(6, "C Programming", "programming_language", "C Programming", True, False, (), "flag"),
        E(6, "C Programming", "programming_language", "C", False, True, ("programming", "pointer"), "skip"),
        E(7, "Node.js", "framework_library", "Node.js", True, False, (), "flag"),
        E(8, "Microsoft Excel", "tool_platform", "Microsoft Excel", True, False, (), "flag"),
        E(8, "Microsoft Excel", "tool_platform", "Excel", False, True, ("spreadsheet",), "flag"),
        E(9, "Neural Networks", "data_ai", "neural network", False, False, (), "flag"),
        E(10, "Convolutional Neural Networks", "data_ai", "convolutional neural network", False, False, (), "flag"),
    ]
    return TaxonomyMatcher(entries), entries


def _run(lines, section_type="unit", **kw):
    matcher, entries = _matcher()
    sec = SectionInput(1, section_type, "Test section", [(p, t) for p, t in lines])
    return extract_skills([sec], matcher, **kw), entries


def test_clean_text_preserves_technical_tokens():
    t = clean_text("Program-\nming in C++ and C#, Node.js, .NET, CI/CD–pipelines")
    assert "Programming" in t and "C++" in t and "C#" in t and "Node.js" in t and ".NET" in t and "CI/CD" in t


def test_alias_normalisation_and_context():
    cands, _ = _run([(4, "Intro to ML models and NLP with SQL; deploy on AWS cloud (EC2).")])
    by = {c.skill_name: c for c in cands}
    assert by["Machine Learning"].matched_text == "ML" and by["Machine Learning"].extraction_method == "alias"
    assert by["Natural Language Processing"].matched_text == "NLP"
    assert by["SQL"].extraction_method == "dictionary"
    aws = by["Amazon Web Services"]
    assert aws.evidence_type == "topic_mention" and aws.confidence >= 0.8
    assert all(c.page_number == 4 for c in cands)


def test_ambiguous_acronym_is_flagged_not_trusted():
    cands, _ = _run([(1, "AWS = Assignment Writing Schedule for the term.")])
    aws = [c for c in cands if c.skill_name == "Amazon Web Services"]
    assert len(aws) == 1 and aws[0].evidence_type == "ambiguous" and aws[0].confidence < 0.5


def test_skip_policy_and_case_sensitivity():
    cands, _ = _run([(1, "Students may excel in C and in teamwork.")])
    names = {c.skill_name for c in cands}
    assert "C Programming" not in names     # 'C' without programming context is skipped
    assert "Microsoft Excel" not in names   # lowercase 'excel' (verb) never matches the acronym form
    cands, _ = _run([(1, "Pointers and arrays in C; object models in C++ and Node.js servers.")])
    names = {c.skill_name: c.matched_text for c in cands}
    assert names == {"C Programming": "C", "C++": "C++", "Node.js": "Node.js"}


def test_longest_match_and_plurals():
    cands, _ = _run([(2, "Convolutional neural networks extend neural networks.")])
    assert [c.skill_name for c in cands] == ["Convolutional Neural Networks", "Neural Networks"]


def test_evidence_types_follow_sections():
    cands, _ = _run([(3, "Implement SQL queries.")], section_type="practical")
    assert cands[0].evidence_type == "practical"
    cands, _ = _run([(1, "CO2: Apply SQL to design schemas.")], section_type="outcomes")
    assert cands[0].evidence_type == "learning_outcome"
    cands, _ = _run([(5, "SQL Fundamentals, Example Press.")], section_type="references")
    assert cands[0].confidence < 0.8  # book titles are weaker evidence


def test_fuzzy_matcher_finds_near_variants():
    matcher, entries = _matcher()
    sec = SectionInput(1, "unit", "U", [(1, "Basics of convolutional neural netwrks")])
    cands = extract_skills([sec], TaxonomyMatcher([e for e in entries if e.skill_id != 10]),
                           fuzzy=FuzzyMatcher(entries), fuzzy_threshold=0.7)
    fz = [c for c in cands if c.extraction_method == "fuzzy"]
    assert fz and fz[0].skill_name == "Convolutional Neural Networks" and fz[0].confidence < 0.8


def test_author_initials_are_not_the_c_or_r_language():
    cands, _ = _run([(1, "Advanced programming with pointers, W.C. Stevens, Pearson programming series.")])
    assert "C Programming" not in {c.skill_name for c in cands}
    cands, _ = _run([(1, "The C programming language: pointers and functions.")])
    assert "C Programming" in {c.skill_name for c in cands}
    cands, _ = _run([(1, "Programming pointers, Rama C, Example Press.")], section_type="references")
    assert "C Programming" not in {c.skill_name for c in cands}


def test_fuzzy_matching_ignores_single_generic_words():
    matcher, entries = _matcher()
    sec = SectionInput(1, "unit", "U", [(1, "Design; framework; networks")])
    cands = extract_skills([sec], TaxonomyMatcher([e for e in entries if e.skill_id != 9]),
                           fuzzy=FuzzyMatcher(entries), fuzzy_threshold=0.5)
    assert not [c for c in cands if c.extraction_method == "fuzzy"]


def test_semantic_matcher_logic_with_stub_model():
    """Exercises the Sentence-Transformers code path with a deterministic stub (no model download)."""
    import numpy as np

    from app.services.skill_extractor import SemanticMatcher

    class StubModel:
        vocab = ["learning", "language", "cloud", "database"]

        def encode(self, texts, normalize_embeddings=True):
            out = []
            for t in texts:
                v = np.array([1.0 if w in t.lower() else 0.0 for w in self.vocab]) + 1e-6
                out.append(v / np.linalg.norm(v))
            return np.array(out)

    _, entries = _matcher()
    sem = SemanticMatcher(StubModel(), entries)
    matcher = TaxonomyMatcher([e for e in entries if e.skill_id not in (1, 2)])
    sec = SectionInput(1, "unit", "U", [(2, "Statistical learning from examples; human language understanding")])
    cands = extract_skills([sec], matcher, semantic=sem, semantic_threshold=0.9)
    sem_hits = {c.skill_name: c for c in cands if c.extraction_method == "semantic"}
    assert "Machine Learning" in sem_hits and "Natural Language Processing" in sem_hits
    assert all(c.page_number == 2 and c.confidence < 0.9 for c in sem_hits.values())
