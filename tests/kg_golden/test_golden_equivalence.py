"""Replay the recorded corpus and compare against recorded goldens.

Skipped unless ``KG_GOLDEN_DIR`` is set; see ``README.md``.
``KG_GOLDEN_MODE=record`` writes goldens, anything else compares.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import pytest

from tests.kg_golden import dump

_DIR = os.environ.get("KG_GOLDEN_DIR")
_RECORD = os.environ.get("KG_GOLDEN_MODE") == "record"

if not _DIR:
    # Module level, not a skipif mark: with no corpus the parametrize lists below
    # are empty, and pytest then hands the `ids=` callable a NOTSET sentinel.
    pytest.skip("KG_GOLDEN_DIR not set (see tests/kg_golden/README.md)", allow_module_level=True)


def _corpus():
    if not _DIR:
        return []
    return sorted((Path(_DIR) / "corpus").glob("*.pkl"))


def _id(path) -> str:
    return getattr(path, "stem", "none")[:10]


def _check(name: str, key: str, actual) -> None:
    path = Path(_DIR) / "golden" / name / f"{key}.json"
    text = json.dumps(actual, indent=1, sort_keys=True, default=str)
    if _RECORD:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    assert path.is_file(), f"no golden for {name}/{key}; record first"
    expected = path.read_text(encoding="utf-8")
    if expected != text:
        import difflib

        diff = "\n".join(list(difflib.unified_diff(
            expected.splitlines(), text.splitlines(), "golden", "now", lineterm="", n=2,
        ))[:60])
        pytest.fail(f"{name}/{key} differs from golden:\n{diff}")


def _load(path: Path):
    return pickle.loads(path.read_bytes())          # fresh copy every time


@pytest.mark.parametrize("path", _corpus(), ids=_id)
def test_class_diagram(path: Path):
    from besser.BUML.notations.kg_to_buml import kg_to_class_diagram

    for emit_ocl in (True, False):
        try:
            result = dump.dump_class_result(kg_to_class_diagram(_load(path), emit_ocl=emit_ocl))
        except Exception as exc:   # a failure is output too: it must not change silently
            result = {"raised": type(exc).__name__, "message": str(exc)}
        _check(path.stem, f"class_ocl{int(emit_ocl)}", result)


@pytest.mark.parametrize("path", _corpus(), ids=_id)
def test_conversion_projection(path: Path):
    from besser.BUML.notations.kg_to_buml._common import KGConversionWarning  # noqa: F401
    from besser.BUML.notations.kg_to_buml.kg_to_rdf import kg_to_rdf
    from dataclasses import asdict

    warnings = []
    try:
        graph, base = kg_to_rdf(_load(path), warnings=warnings)
        result = {"base": base, "triples": dump.dump_graph(graph),
                  "warnings": [asdict(w) for w in warnings]}
    except Exception as exc:
        result = {"raised": type(exc).__name__, "message": str(exc)}
    _check(path.stem, "projection_conversion", result)


@pytest.mark.parametrize("vocab", ["owl", "shacl", "both"])
@pytest.mark.parametrize("path", _corpus(), ids=_id)
def test_export_projection(path: Path, vocab: str):
    from besser.utilities.kg_to_owl import knowledge_graph_to_rdf

    try:
        graph = knowledge_graph_to_rdf(_load(path), vocab=vocab)
        result = {"triples": dump.dump_graph(graph)}
    except Exception as exc:
        result = {"raised": type(exc).__name__, "message": str(exc)}
    _check(path.stem, f"projection_export_{vocab}", result)


@pytest.mark.parametrize("inference", ["rdfs", pytest.param("owlrl", marks=pytest.mark.slow)])
@pytest.mark.parametrize("path", _corpus(), ids=_id)
def test_consistency(path: Path, inference: str):
    from besser.BUML.notations.kg_to_buml.consistency import check_kg_consistency

    try:
        result = dump.dump_consistency(check_kg_consistency(_load(path), inference=inference))
    except Exception as exc:
        result = {"raised": type(exc).__name__, "message": str(exc)}
    _check(path.stem, f"consistency_{inference}", result)
