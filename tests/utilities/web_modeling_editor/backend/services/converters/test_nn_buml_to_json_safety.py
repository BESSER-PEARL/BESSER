"""Safety tests for the ast-based ``nn_buml_to_json`` parser.

The previous implementation used ``exec()`` on user-supplied content with a
hand-rolled builtins whitelist — that sandbox is escapable via the standard
``().__class__.__bases__[0].__subclasses__()`` trick (no builtins required,
just attribute traversal). The new AST visitor only evaluates a narrow set
of node types, so these probes must be rejected.
"""

import pytest

from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.nn_diagram_converter import (
    nn_buml_to_json,
)


LEGITIMATE_BUML = """
from besser.BUML.metamodel.nn import (
    NN, Conv2D, LinearLayer, FlattenLayer, Configuration,
)

layer_0 = Conv2D(name='c1', kernel_dim=[3, 3], out_channels=16)
layer_1 = FlattenLayer(name='flat')
layer_2 = LinearLayer(name='lin', out_features=10)

config = Configuration(
    batch_size=32, epochs=5, learning_rate=0.001,
    optimizer='adam', loss_function='crossentropy', metrics=['accuracy'],
)

main_nn = NN(name='main')
main_nn.add_layer(layer_0)
main_nn.add_layer(layer_1)
main_nn.add_layer(layer_2)
main_nn.add_configuration(config)
"""


def test_legitimate_nn_buml_parses():
    out = nn_buml_to_json(LEGITIMATE_BUML)
    assert out['type'] == 'NNDiagram'
    # Each layer plus the main container emits one node in v4.
    node_types = {n['type'] for n in out['nodes']}
    assert 'Conv2DLayer' in node_types
    assert 'LinearLayer' in node_types
    assert 'FlattenLayer' in node_types


def test_import_system_call_rejected():
    """An upload that reimports ``os`` and tries to run a shell command via
    an attribute chain must be rejected at parse time, never executed."""
    malicious = """
from besser.BUML.metamodel.nn import NN
main_nn = NN(name='x')
import os
os.system('touch /tmp/pwned')
"""
    with pytest.raises(ValueError, match=r"(Disallowed|non-whitelisted|Unknown)"):
        nn_buml_to_json(malicious)


def test_subclass_escape_rejected():
    """The classic subclass-traversal sandbox escape must fail.

    ``().__class__.__bases__[0].__subclasses__()`` compiles to a Call whose
    func is an Attribute chain. The AST visitor only evaluates Name→Call,
    so any Attribute-based call is refused with a "non-whitelisted" /
    "whitelisted NN metamodel classes" / "Disallowed" message.
    """
    malicious = """
from besser.BUML.metamodel.nn import NN
main_nn = NN(name='x')
pwned = ().__class__.__bases__[0].__subclasses__()
"""
    with pytest.raises(ValueError, match=r"(Disallowed|whitelisted)"):
        nn_buml_to_json(malicious)


def test_method_call_outside_whitelist_rejected():
    """A method call that isn't one of the add_* builder helpers must raise."""
    malicious = """
from besser.BUML.metamodel.nn import NN
main_nn = NN(name='x')
main_nn.some_evil_method('arg')
"""
    with pytest.raises(ValueError, match=r"Disallowed method call"):
        nn_buml_to_json(malicious)


def test_arbitrary_function_call_rejected():
    """A plain call to something that isn't in the NN metamodel allowlist
    (e.g. ``exec``, ``eval``, ``getattr``) must fail — even when the name
    isn't bound in the env, the AST visitor rejects Name→Call chains that
    don't resolve to a whitelisted class."""
    malicious = """
from besser.BUML.metamodel.nn import NN
pwned = getattr(NN, '__class__')
"""
    with pytest.raises(ValueError, match=r"(non-whitelisted|Unknown)"):
        nn_buml_to_json(malicious)


def test_attribute_assignment_rejected():
    """Attribute-target assignment (``obj.attr = value``) is not a simple
    name target and must be refused."""
    malicious = """
from besser.BUML.metamodel.nn import NN
main_nn = NN(name='x')
main_nn.__class__ = object
"""
    with pytest.raises(ValueError, match=r"simple-name assignment targets"):
        nn_buml_to_json(malicious)


def test_empty_content_raises_no_nn():
    """Totally empty content (just whitespace or imports) raises a
    descriptive 'No NN instance found' error, not a silent empty result."""
    with pytest.raises(ValueError, match=r"No NN instance found"):
        nn_buml_to_json("from besser.BUML.metamodel.nn import NN\n")


def test_multi_top_level_nn_rejected():
    """Two unrelated top-level NNs (neither referenced by the other via
    add_sub_nn) must raise — mirrors the processor's multi-container rule."""
    malicious = """
from besser.BUML.metamodel.nn import NN
first = NN(name='first')
second = NN(name='second')
"""
    with pytest.raises(ValueError, match=r"top-level NN instances"):
        nn_buml_to_json(malicious)


def test_dict_unpacking_rejected():
    """``**kwargs`` inside a Dict literal must raise ValueError, not a
    bare TypeError that would leak as a 500. The encoded form is a Dict
    with a None key."""
    malicious = """
from besser.BUML.metamodel.nn import NN
main = NN(name='x')
stuff = {**{'a': 1}}
"""
    with pytest.raises(ValueError, match=r"(Dict unpacking|Disallowed)"):
        nn_buml_to_json(malicious)


def test_iterable_unpacking_in_list_rejected():
    """``*expr`` inside a list literal must raise ValueError explicitly."""
    malicious = """
from besser.BUML.metamodel.nn import NN
main = NN(name='x')
dims = [*[1, 2, 3]]
"""
    with pytest.raises(ValueError, match=r"(Iterable unpacking|Disallowed)"):
        nn_buml_to_json(malicious)


def test_transitively_referenced_sub_nn_not_counted_as_top_level():
    """A grandchild NN reachable only via another sub-NN's chain must
    still count as referenced — the shallow check pre-iter-18 failed
    this and raised multi-top-level incorrectly."""
    buml = """
from besser.BUML.metamodel.nn import NN
a = NN(name='root')
b = NN(name='middle')
c = NN(name='leaf')
b.add_sub_nn(c)
a.add_sub_nn(b)
"""
    out = nn_buml_to_json(buml)
    assert out['type'] == 'NNDiagram'
    # Three containers emitted (root + 2 sub-NNs)
    containers = [n for n in out['nodes'] if n.get('type') == 'NNContainer']
    assert len(containers) == 3


def test_builder_output_roundtrips_through_ast_parser(tmp_path):
    """An NN generated by ``nn_model_to_code`` (the builder) must parse
    back through the new ast-based ``nn_buml_to_json`` without loss of
    layer types. Exercises the real round-trip that ``/export-buml`` →
    ``/get-json-model`` would see."""
    from besser.BUML.metamodel.nn import (
        NN, Conv2D, LinearLayer, FlattenLayer, Configuration,
    )
    from besser.utilities.buml_code_builder.nn_model_builder import nn_model_to_code

    nn = NN(name="EndToEnd")
    nn.add_layer(Conv2D(name="c", kernel_dim=[3, 3], out_channels=16))
    nn.add_layer(FlattenLayer(name="flat"))
    nn.add_layer(LinearLayer(name="lin", out_features=10))
    nn.add_configuration(Configuration(
        batch_size=32, epochs=5, learning_rate=0.001,
        optimizer="adam", loss_function="crossentropy", metrics=["accuracy"],
    ))

    code_path = tmp_path / "nn_end_to_end.py"
    nn_model_to_code(nn, str(code_path))
    code = code_path.read_text(encoding="utf-8")

    out = nn_buml_to_json(code)
    assert out["type"] == "NNDiagram"
    node_types = {n["type"] for n in out["nodes"]}
    assert "Conv2DLayer" in node_types
    assert "LinearLayer" in node_types
    assert "FlattenLayer" in node_types
    assert "Configuration" in node_types
