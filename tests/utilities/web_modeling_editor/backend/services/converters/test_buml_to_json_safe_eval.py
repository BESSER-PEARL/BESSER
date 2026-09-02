"""Safety + round-trip tests for the shared ``safe_exec`` AST evaluator that
replaced ``exec()`` in the class / gui / bpmn / quantum BUML -> JSON converters.

Two concerns:

1. **Security** — ``safe_exec`` must *refuse* (raise, never run) the standard
   sandbox-escape gadgets: dunder attribute traversal
   (``().__class__.__bases__[0].__subclasses__()``), ``__import__(...)`` /
   ``getattr(...)`` -to-globals, imports, lambdas, comprehensions, and
   ``str.format`` attribute-leak gadgets. The four converters that call it must
   propagate that refusal rather than execute the payload.

2. **Round-trip fidelity** — the builder-emitted ``.py`` for each of the four
   de-exec'd model types must still reconstruct through the AST evaluator with
   no loss. OCL ``Constraint`` objects (the explicitly-called-out case) must
   survive with their names *and* opaque expression strings byte-for-byte.
"""

import pytest

from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._safe_eval import (
    safe_exec,
    UnsafeConstruct,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError


# A minimal whitelist mimicking what the converters seed: benign builtins under
# ``__builtins__``, an import sentinel ``__name__``, and one stand-in
# "constructor" (``dict`` is a convenient callable that echoes its kwargs).
ALLOWED = {
    "__name__": "besser_buml_import",
    "__builtins__": {"set": set, "list": list, "dict": dict},
    "Ctor": dict,
}


# ---------------------------------------------------------------------------
# 1. Security — adversarial payloads must be refused, never executed
# ---------------------------------------------------------------------------

# Each of these must raise ``UnsafeConstruct`` (a security refusal), proving the
# construct was rejected at evaluation time and never ran.
UNSAFE_PAYLOADS = {
    "subclass_traversal": "x = ().__class__.__bases__[0].__subclasses__()",
    "dunder_on_str": "x = ''.__class__",
    "dunder_on_list": "x = [].__class__",
    "dunder_import_call": "x = __import__('os').system('echo pwned')",
    "getattr_to_globals": "x = getattr(Ctor, '__globals__')",
    "dunder_kwarg": "x = Ctor(name=().__class__)",
    "dunder_attr_assign": "obj = Ctor(a=1)\nobj.__class__ = list",
    "lambda_expr": "f = lambda: 1",
    "list_comprehension": "g = [i for i in range(3)]",
    "generator_expr": "g = list(i for i in range(3))",
    "dict_unpacking": "z = {**{'a': 1}}",
    "iterable_unpacking": "w = [*[1, 2]]",
    "kwargs_unpacking": "x = Ctor(**{'a': 1})",
    "format_attr_leak": "s = '{0.__class__}'.format(())",
    "fstring": "name = 'a'\ns = f'{name.__class__}'",
    "subscript_read": "d = {'a': 1}\nx = d['a'].__class__",
    "binop": "x = 1 + ().__class__.__name__",
    "walrus": "x = (y := 5)",
    "function_def": "def evil():\n    import os\n    os.system('x')",
    "class_def": "class Evil:\n    pass",
    "for_loop": "for i in range(3):\n    x = i",
    "while_loop": "while False:\n    pass",
    "with_stmt": "with open('x') as f:\n    pass",
    "raise_stmt": "raise RuntimeError('boom')",
    "delete_stmt": "x = 1\ndel x",
    "global_stmt": "global x",
    "await_expr": "async def f():\n    await g()",
}


@pytest.mark.parametrize("name,src", sorted(UNSAFE_PAYLOADS.items()))
def test_safe_exec_refuses_unsafe_construct(name, src):
    """Every escape gadget raises UnsafeConstruct — parse aborts, nothing runs."""
    with pytest.raises(UnsafeConstruct):
        safe_exec(src, ALLOWED)


def test_safe_exec_import_then_use_never_executes():
    """A stripped/kept ``import os`` is a no-op, so a later ``os.system(...)``
    fails with NameError — ``os`` was never actually imported."""
    with pytest.raises((UnsafeConstruct, NameError)):
        safe_exec("import os\nos.system('echo pwned')", ALLOWED)


def test_unsafe_construct_is_valueerror():
    """UnsafeConstruct subclasses ValueError so the converters' existing
    ``except ValueError`` wrappers catch it (mapping to a 400, not a 500)."""
    assert issubclass(UnsafeConstruct, ValueError)


def test_try_except_cannot_swallow_security_refusal():
    """Wrapping a payload in ``try/except Exception: pass`` must NOT let it
    slip through — a security refusal always aborts the whole parse."""
    src = "try:\n    x = ().__class__\nexcept Exception:\n    pass\n"
    with pytest.raises(UnsafeConstruct):
        safe_exec(src, ALLOWED)


# ---------------------------------------------------------------------------
# 1b. Security — the refusal propagates through each real converter entry point
# ---------------------------------------------------------------------------

_ESCAPE = "pwned = ().__class__.__bases__[0].__subclasses__()\n"


def test_class_converter_rejects_escape():
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
        parse_buml_content,
    )
    with pytest.raises(ValueError):
        parse_buml_content(_ESCAPE)


def test_gui_converter_rejects_escape():
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
        gui_buml_to_json,
    )
    with pytest.raises(ValueError):
        gui_buml_to_json(_ESCAPE)


def test_quantum_converter_rejects_escape():
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.quantum_diagram_converter import (
        quantum_buml_to_json,
    )
    with pytest.raises(ValueError):
        quantum_buml_to_json(_ESCAPE)


def test_bpmn_converter_rejects_escape():
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.bpmn_diagram_converter import (
        bpmn_buml_to_json,
    )
    # BPMN wraps parse failures (incl. the ValueError-derived UnsafeConstruct)
    # into ConversionError.
    with pytest.raises(ConversionError):
        bpmn_buml_to_json(_ESCAPE)


# ---------------------------------------------------------------------------
# 2. Import semantics — ``if __name__ == "__main__":`` guard is skipped
# ---------------------------------------------------------------------------

def test_main_guard_body_is_not_executed():
    """The guarded block must be skipped (condition False), never run — mirrors
    Python import semantics for a file with a ``__main__`` guard."""
    src = (
        "value = 1\n"
        "if __name__ == '__main__':\n"
        "    raise RuntimeError('main block must not execute during import')\n"
    )
    env = safe_exec(src, ALLOWED)
    assert env["value"] == 1


# ---------------------------------------------------------------------------
# 3. OCL round-trip — Constraints survive the de-exec'd class converter
# ---------------------------------------------------------------------------

def test_ocl_constraints_survive_class_converter_roundtrip(tmp_path):
    """Build a class diagram with OCL invariants via ``buml_code_builder``,
    export to ``.py``, and re-import it through the (now exec-free)
    ``parse_buml_content``. Constraint names *and* opaque expression strings
    must survive unchanged, with their class context re-wired."""
    from besser.BUML.metamodel.structural import (
        DomainModel, Class, Property, IntegerType, Constraint,
    )
    from besser.utilities.buml_code_builder.domain_model_builder import (
        domain_model_to_code,
    )
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
        parse_buml_content, class_buml_to_json,
    )

    pages = Property(name="pages", type=IntegerType)
    stock = Property(name="stock", type=IntegerType)
    book = Class(name="Book", attributes={pages, stock})

    expressions = {
        "pages_positive": "context Book inv pages_positive: self.pages > 0",
        "stock_non_negative": "context Book inv stock_non_negative: self.stock >= 0",
    }
    constraints = {
        Constraint(name=name, context=book, expression=expr, language="OCL")
        for name, expr in expressions.items()
    }
    model = DomainModel(name="Library", types={book}, constraints=constraints)

    code_path = tmp_path / "library_ocl.py"
    domain_model_to_code(model, str(code_path))
    code = code_path.read_text(encoding="utf-8")

    parsed = parse_buml_content(code)

    # Names + expressions round-trip verbatim (the opaque OCL text is never
    # parsed or re-serialised by the evaluator).
    assert {c.name: c.expression for c in parsed.constraints} == expressions
    # Every constraint keeps its class context wiring.
    assert all(c.context.name == "Book" for c in parsed.constraints)
    # And the JSON walk still emits an OCL box per constraint.
    out = class_buml_to_json(parsed)
    boxes = sorted(
        e["constraint"]
        for e in out["elements"].values()
        if e.get("type") == "ClassOCLConstraint"
    )
    assert boxes == sorted(expressions.values())


def test_project_to_json_preserves_ocl_constraints(tmp_path):
    """The same OCL survives the full project import path (``project_to_json``),
    which routes the class section through ``parse_buml_content``."""
    from besser.BUML.metamodel.structural import (
        DomainModel, Class, Property, IntegerType, Constraint,
    )
    from besser.utilities.buml_code_builder.domain_model_builder import (
        domain_model_to_code,
    )
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
        project_to_json,
    )

    balance = Property(name="balance", type=IntegerType)
    account = Class(name="Account", attributes={balance})
    expr = "context Account inv positive: self.balance >= 0"
    inv = Constraint(name="positive", context=account, expression=expr, language="OCL")
    model = DomainModel(name="Bank", types={account}, constraints={inv})

    code_path = tmp_path / "bank.py"
    domain_model_to_code(model, str(code_path))
    code = code_path.read_text(encoding="utf-8")

    project = project_to_json(code)
    class_model = project["diagrams"]["ClassDiagram"][0]["model"]
    boxes = [
        e for e in class_model["elements"].values()
        if e.get("type") == "ClassOCLConstraint"
    ]
    assert [b["constraint"] for b in boxes] == [expr]


# ---------------------------------------------------------------------------
# 4. Quantum round-trip — control-qubit chains + enum-literal reads
# ---------------------------------------------------------------------------

def test_quantum_builder_roundtrips_through_safe_eval(tmp_path):
    """A circuit whose builder output uses the tricky patterns —
    ``gate.control_qubits.append(...)`` (method call on an attribute of a
    namespace object) and ``ControlState.CONTROL`` (enum-literal attribute
    read) — must reconstruct through the exec-free converter."""
    from besser.BUML.metamodel.quantum.quantum import (
        QuantumCircuit, HadamardGate, PauliXGate, Measurement,
        ControlState, ClassicalRegister,
    )
    from besser.utilities.buml_code_builder.quantum_model_builder import (
        quantum_model_to_code,
    )
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.quantum_diagram_converter import (
        quantum_buml_to_json,
    )

    qc = QuantumCircuit(name="Bell", qubits=2)
    qc.add_operation(HadamardGate(target_qubit=0))
    cx = PauliXGate(target_qubit=1)
    cx.control_qubits = [0]
    cx.control_states = [ControlState.CONTROL]
    qc.add_operation(cx)
    qc.add_creg(ClassicalRegister("c", 2))
    qc.add_operation(Measurement(target_qubit=0, output_bit=0))

    code_path = tmp_path / "bell.py"
    quantum_model_to_code(qc, str(code_path))
    code = code_path.read_text(encoding="utf-8")

    out = quantum_buml_to_json(code)
    assert out["title"] == "Bell"
    # Three operations -> three columns (H, controlled-X, Measurement).
    assert len(out["cols"]) == 3


# ---------------------------------------------------------------------------
# 5. GUI round-trip — fluent builder chains through the exec-free converter
# ---------------------------------------------------------------------------

def test_gui_builder_roundtrips_through_safe_eval(tmp_path):
    """A GUI model exported by ``gui_model_to_code`` must reconstruct through the
    exec-free ``gui_buml_to_json`` and yield the expected page."""
    from besser.BUML.metamodel.gui import GUIModel
    from besser.BUML.metamodel.gui.graphical_ui import (
        Screen, Module, Button, Text, ButtonType, ButtonActionType,
    )
    from besser.utilities.buml_code_builder.gui_model_builder import (
        gui_model_to_code,
    )
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
        gui_buml_to_json,
    )

    btn = Button(
        name="SubmitBtn", label="Submit", description="A submit button",
        buttonType=ButtonType.RaisedButton, actionType=ButtonActionType.Navigate,
    )
    txt = Text(name="WelcomeText", content="Hello World", description="Welcome text")
    screen = Screen(
        name="MainScreen", description="The main screen",
        view_elements={btn, txt}, is_main_page=True,
    )
    module = Module(name="MainModule", screens={screen})
    gui = GUIModel(
        name="TestGUI", package="com.test", versionCode="1", versionName="1.0",
        modules={module}, description="Test GUI model",
    )

    code_path = tmp_path / "gui.py"
    gui_model_to_code(gui, str(code_path))
    code = code_path.read_text(encoding="utf-8")

    out = gui_buml_to_json(code)
    assert out["pages"], "expected at least one GrapesJS page"
    assert any(p.get("name") == "The main screen" for p in out["pages"])
