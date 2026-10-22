"""The consistency check stays bounded whatever the KG author writes.

* ``sh:pattern`` is evaluated with :mod:`re`, which cannot be interrupted and
  holds the GIL, so an exponential pattern used to pin a server thread for as
  long as it liked. Recognisably nested patterns are now skipped with a warning,
  and pyshacl runs in a child process that is killed when its budget runs out.
* pyshacl's ``ReportableRuntimeError`` (shapes it refuses to load) used to escape
  as a bare exception; it is now a :class:`ConsistencyCheckError`.
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from besser.BUML.notations.kg_to_buml.consistency import (
    MAX_SHACL_PATTERN_LENGTH,
    ConsistencyCheckError,
    ConsistencyCheckTimeout,
    _has_nested_quantifier,
    check_kg_consistency,
)
from besser.BUML.notations.owl_to_kg import owl_file_to_knowledge_graph

_PREFIX = """
@prefix : <http://example.org/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
:Person a owl:Class .
:name a owl:DatatypeProperty ; rdfs:domain :Person ; rdfs:range xsd:string .
"""


def _kg(ttl: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ttl", delete=False, encoding="utf-8") as f:
        f.write(_PREFIX + ttl)
        path = f.name
    try:
        return owl_file_to_knowledge_graph(path)
    finally:
        os.unlink(path)


def _pattern_kg(pattern: str, value: str, extra: str = ""):
    escaped = pattern.replace("\\", "\\\\")
    return _kg(f"""
:PersonShape a sh:NodeShape ; sh:targetClass :Person ;
    sh:property [ sh:path :name ; sh:pattern "{escaped}" ; sh:maxLength 5 {extra} ] .
:alice a :Person ; :name "{value}" .
""")


def test_an_exponential_pattern_on_a_40_char_value_returns_promptly():
    kg = _pattern_kg("^(a+)+$", "a" * 40 + "!")

    started = time.monotonic()
    report = check_kg_consistency(kg)
    elapsed = time.monotonic() - started

    assert elapsed < 15, f"took {elapsed:.1f}s"
    skipped = [i for i in report.issues if i.code == "UnsafePatternSkipped"]
    assert len(skipped) == 1
    assert skipped[0].severity == "warning"
    assert skipped[0].spec_kind == "pattern"
    assert "^(a+)+$" in skipped[0].message
    # The shape's other constraints are still checked.
    assert any(i.code == "MaxLengthConstraintComponent" for i in report.issues)
    assert not any(i.code == "PatternConstraintComponent" for i in report.issues)


def test_the_skipped_pattern_issue_id_is_stable():
    kg = _pattern_kg("^(a+)+$", "aaa")
    first = [i.id for i in check_kg_consistency(kg).issues if i.code == "UnsafePatternSkipped"]
    second = [i.id for i in check_kg_consistency(kg).issues if i.code == "UnsafePatternSkipped"]
    assert first and first == second


def test_a_safe_pattern_is_still_enforced():
    report = check_kg_consistency(_pattern_kg("^[A-Z]", "lowercase"))
    assert any(i.code == "PatternConstraintComponent" for i in report.issues)
    assert not any(i.code == "UnsafePatternSkipped" for i in report.issues)


def test_an_overlong_pattern_is_skipped():
    report = check_kg_consistency(_pattern_kg("a" * (MAX_SHACL_PATTERN_LENGTH + 1), "b"))
    assert [i.code for i in report.issues if i.code == "UnsafePatternSkipped"] == ["UnsafePatternSkipped"]


def test_a_runaway_pattern_the_heuristic_misses_is_stopped_by_the_timeout():
    """``(a|a)+`` has no nested quantifier but is still exponential: the child is killed."""
    kg = _pattern_kg("^(a|a)+$", "a" * 40 + "!")

    started = time.monotonic()
    with pytest.raises(ConsistencyCheckTimeout, match="did not finish within 3 seconds"):
        check_kg_consistency(kg, timeout_seconds=3)
    assert time.monotonic() - started < 15


def _path_less_kg():
    # ``:nickname`` is never declared as a property, so the importer keeps no
    # sh:path for the property shape and pyshacl refuses to load it.
    return _kg("""
:alice a :Person ; :nickname "aaa" .
:S a sh:NodeShape ; sh:targetClass :Person ; sh:property :PS .
:PS a sh:PropertyShape ; sh:path :nickname ; sh:minCount 2 .
""")


@pytest.mark.parametrize("timeout_seconds", [None, 30])
def test_shapes_pyshacl_refuses_raise_a_consistency_check_error(timeout_seconds):
    with pytest.raises(ConsistencyCheckError, match="sh:path"):
        check_kg_consistency(_path_less_kg(), timeout_seconds=timeout_seconds)


def test_the_child_process_reports_the_same_issues_as_in_process():
    kg = _pattern_kg("^[A-Z]", "lowercase")
    isolated = check_kg_consistency(kg)
    in_process = check_kg_consistency(kg, timeout_seconds=None)
    assert [(i.id, i.code, i.affected_node_ids, i.constraint_node_id) for i in isolated.issues] == [
        (i.id, i.code, i.affected_node_ids, i.constraint_node_id) for i in in_process.issues
    ]


@pytest.mark.parametrize("pattern", [
    "^(a+)+$", "(a*)*", "(?:a+)+", "((a)+)+", "(.*a){20}", "(a{2,5})*", r"(\w+\s?)+$",
])
def test_nested_quantifiers_are_recognised(pattern):
    assert _has_nested_quantifier(pattern)


@pytest.mark.parametrize("pattern", [
    "^[A-Z]", "^(https?://)?[a-z]+$", r"(\d{3})+", "(ab)+", "[(]a+[)]+", r"\(a+\)+",
    "^[^]a+]+$", "(a{2})*", "^[0-9]{4}-[0-9]{2}$",
])
def test_plain_patterns_are_not_flagged(pattern):
    assert not _has_nested_quantifier(pattern)
