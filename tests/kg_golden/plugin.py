"""Pytest plugin that records every KnowledgeGraph the KG pipeline is run on.

Load with ``PYTHONPATH=. pytest -p tests.kg_golden.plugin ...`` and set
``KG_GOLDEN_DIR``. It wraps the pipeline entry points and pickles each KG into
``$KG_GOLDEN_DIR/corpus/<sha256>.pkl``.

Wrapping replaces the function object wherever it has already been bound by
name, because tests and routers do ``from x import f`` and would otherwise keep
the original.
"""

from __future__ import annotations

import functools
import hashlib
import os
import pickle
import sys
from pathlib import Path

_DIR = os.environ.get("KG_GOLDEN_DIR")
_seen: set[str] = set()


def _record(kg) -> None:
    from besser.BUML.metamodel.kg import KnowledgeGraph

    if not _DIR or not isinstance(kg, KnowledgeGraph):
        return
    try:
        blob = pickle.dumps(kg)
    except Exception:  # an unpicklable KG is a harness gap worth knowing about
        sys.stderr.write("kg_golden: could not pickle a KG; skipped\n")
        return
    digest = hashlib.sha256(blob).hexdigest()
    if digest in _seen:
        return
    _seen.add(digest)
    out = Path(_DIR) / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{digest}.pkl").write_bytes(blob)


def _wrap(module_name: str, attr: str) -> None:
    import importlib

    module = importlib.import_module(module_name)
    original = getattr(module, attr)

    @functools.wraps(original)
    def recorder(kg, *args, **kwargs):
        _record(kg)
        return original(kg, *args, **kwargs)

    # Rebind everywhere the original was already imported by name.
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, attr, None) is original:
                setattr(mod, attr, recorder)
        except Exception:
            continue


def pytest_configure(config):
    if not _DIR:
        return
    # Import the backend first so its by-name bindings exist to be rebound.
    try:
        import besser.utilities.web_modeling_editor.backend.backend  # noqa: F401
    except Exception:
        pass
    _wrap("besser.BUML.notations.kg_to_buml.kg_to_rdf", "kg_to_rdf")
    _wrap("besser.utilities.kg_to_owl", "knowledge_graph_to_rdf")
    _wrap("besser.BUML.notations.kg_to_buml.consistency", "check_kg_consistency")
