"""``safe_load_buml`` never hands a module object to an upload.

A module hands over everything it imported: with ``knowledge_graph`` in the
allowlist, ``knowledge_graph.copy.dispatch_table.clear()`` emptied the
interpreter-wide ``copyreg.dispatch_table``. Converters build their allowlists
from ``dir(package)``, which lists submodules too, so the loader drops module
values itself rather than relying on each converter to filter them.
"""
import copy
import copyreg

import pytest

import besser.BUML.metamodel.kg.knowledge_graph as knowledge_graph_module
import besser.BUML.metamodel.quantum.quantum as quantum_module
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json import (
    _safe_buml_loader as loader,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.quantum_diagram_converter import (
    quantum_buml_to_json,
)


@pytest.fixture
def dispatch_table_unchanged():
    """Assert the payload did not reach ``copyreg``; restore it if it did."""
    before = dict(copyreg.dispatch_table)
    assert before, "copyreg.dispatch_table is never empty in a running interpreter"
    try:
        yield
        assert copyreg.dispatch_table == before
    finally:
        copyreg.dispatch_table.update(before)


def test_a_module_in_the_allowlist_is_refused_like_an_unknown_name(dispatch_table_unchanged):
    with pytest.raises(loader.SafeBumlLoaderError, match="unknown name: 'knowledge_graph'"):
        loader.safe_load_buml(
            "knowledge_graph.copy.dispatch_table.clear()\n",
            {"knowledge_graph": knowledge_graph_module},
        )


def test_the_other_allowed_names_still_work():
    ns = loader.safe_load_buml(
        "graph = KnowledgeGraph(name='g')\n",
        {"knowledge_graph": knowledge_graph_module, "KnowledgeGraph": knowledge_graph_module.KnowledgeGraph},
    )
    assert ns["graph"].name == "g"


def test_the_quantum_converter_does_not_expose_a_module_either(monkeypatch, dispatch_table_unchanged):
    """The quantum converter passes every public name of its metamodel module.

    It has no public module attribute today, so give it one: the first plain
    ``import`` added to quantum.py must not become reachable from an upload.
    """
    monkeypatch.setattr(quantum_module, "copy", copy, raising=False)

    with pytest.raises(ValueError, match="unknown name: 'copy'"):
        quantum_buml_to_json(
            "qc = QuantumCircuit(name='c', qubits=1)\n"
            "copy.dispatch_table.clear()\n"
        )
