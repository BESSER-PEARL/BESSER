"""Importing a B-UML ``.py`` written as a runnable script (#836).

A model an LLM or a person writes is a script: helper functions, a validation
check that prints, a generator run, an ``if __name__ == "__main__":`` block.
The besser-user skill tells LLMs to write exactly that and promises the file
also imports into the editor. The importer refused it ("Top-level statement not
allowed: FunctionDef") or dropped generator lines textually, which broke a call
split over several lines.

The loader stays an allowlist: the model-building code is validated and run,
script-runner statements (they use names that are not B-UML, such as ``print``
or a generator) are skipped with a warning and never run.
"""
import pytest

from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json import (
    _safe_buml_loader as loader,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    parse_buml_content,
)

IMPORTS = """\
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Multiplicity, BinaryAssociation, StringType, IntegerType,
)
"""


def _names(model):
    return sorted(c.name for c in model.get_classes())


def _import(src):
    warnings = []
    return parse_buml_content(src, warnings), warnings


# --------------------------------------------------------------------------- #
# Shapes a runnable script has
# --------------------------------------------------------------------------- #
def test_a_model_built_in_a_function_imports():
    """The issue's error: a ``def`` that builds and returns the model."""
    model, _ = _import(IMPORTS + """
def build_model() -> DomainModel:
    title = Property(name="title", type=StringType)
    book = Class(name="Book", attributes={title})
    author = Class(name="Author", attributes={Property(name="name", type=StringType)})
    writes = BinaryAssociation(name="writes", ends={
        Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
        Property(name="author", type=author, multiplicity=Multiplicity(1, 1)),
    })
    return DomainModel(name="Library", types={book, author}, associations={writes})


model = build_model()
""")
    assert _names(model) == ["Author", "Book"]
    assert [a.name for a in model.associations] == ["writes"]
    assert model.name == "Library"


def test_a_model_built_under_the_main_guard_imports():
    model, warnings = _import(IMPORTS + """
def build_model():
    return DomainModel(name="Shop", types={Class(name="Order")})


if __name__ == "__main__":
    model = build_model()
    print(model.validate())
""")
    assert _names(model) == ["Order"]
    assert any("print" in w for w in warnings)


def test_the_skills_scaffold_imports():
    """besser-user/scripts/scaffold_model.py ends with a validation check that prints."""
    model, warnings = _import(IMPORTS + """
book_name = Property(name="name", type=StringType)
book = Class(name="Book", attributes={book_name})
model = DomainModel(name="Library_model", types={book})

result = model.validate()
if not result["success"]:
    print("Validation errors:", result["errors"])
""")
    assert _names(model) == ["Book"]
    assert len(warnings) == 1 and "line 10" in warnings[0] and "print" in warnings[0]


def test_a_generator_call_over_several_lines_is_skipped():
    """The old filter dropped only the lines containing ``Generator(``: a SyntaxError."""
    model, warnings = _import(IMPORTS + """
from besser.generators.django import DjangoGenerator

book = Class(name="Book")
model = DomainModel(name="Library", types={book})

generator = DjangoGenerator(
    model=model,
    project_name="library",
    app_name="books",
)
generator.generate()
""")
    assert _names(model) == ["Book"]
    assert len(warnings) == 2
    assert all("DjangoGenerator" in w or "generator" in w for w in warnings)


# --------------------------------------------------------------------------- #
# What stays refused
# --------------------------------------------------------------------------- #
def test_a_typo_in_a_model_line_is_an_error_not_a_missing_class():
    with pytest.raises(ValueError, match="title"):
        _import(IMPORTS + """
title = Proprety(name="title", type=StringType)
book = Class(name="Book", attributes={title})
""")


def test_an_unsupported_statement_names_its_line():
    with pytest.raises(ValueError, match="line 6"):
        _import(IMPORTS + """
book = Class(name="Book")
for attribute in [Property(name="title", type=StringType)]:
    book.attributes.add(attribute)
""")


def _load(src):
    """A hand-written script, the only content whose helper functions load."""
    return loader.safe_load_buml(src, allowed_names={"Class": object}, warnings=[])


@pytest.mark.parametrize("src", [
    "@Class\ndef f():\n    return 1\n",                       # decorator
    "def f():\n    def g():\n        return 1\n    return g\n",  # nested def
    "def f(*args):\n    return args\n",                       # varargs
    "def f():\n    return f.__globals__\n",                   # dunder escape
    "def f():\n    x = 1\n    return x.format\n",            # forbidden attribute
    "async def f():\n    return 1\n",                         # async
])
def test_unsafe_function_shapes_are_refused(src):
    with pytest.raises(loader.SafeBumlLoaderError):
        _load(src)


def test_runaway_recursion_stops():
    with pytest.raises((RecursionError, loader.SafeBumlLoaderError)):
        _load("def f():\n    return f()\n\nx = f()\n")


def test_helper_functions_cannot_fan_out_exponentially():
    """Each level calls the one below twice: 2**40 calls without the budget."""
    src = "def f0():\n    return 1\n"
    for level in range(1, 41):
        src += f"def f{level}():\n    return [f{level - 1}(), f{level - 1}()]\n"
    src += "x = f40()\n"
    with pytest.raises(loader.SafeBumlLoaderError, match="calls"):
        _load(src)


def test_a_skipped_statement_never_runs(tmp_path):
    """``open`` is not B-UML: the statement is skipped, not executed."""
    target = tmp_path / "should-not-exist"
    warnings = []
    ns = loader.safe_load_buml(
        f"x = Class\nopen({str(target)!r}, 'w')\n", allowed_names={"Class": object}, warnings=warnings,
    )
    assert "x" in ns
    assert not target.exists()
    assert len(warnings) == 1 and "'open'" in warnings[0]


def test_without_warnings_a_script_statement_is_refused():
    """Machine-written content (a trace section, a KG export) has no script to skip."""
    with pytest.raises(loader.SafeBumlLoaderError, match="line 2"):
        loader.safe_load_buml("x = Class\nprint(x)\n", allowed_names={"Class": object})


def test_functions_load_only_from_a_hand_written_script():
    """BESSER's own exports (BPMN, component, trace, ...) never contain one."""
    with pytest.raises(loader.SafeBumlLoaderError, match="a function definition"):
        loader.safe_load_buml("def f():\n    return 1\n", allowed_names={})


# --------------------------------------------------------------------------- #
# Security review of PR #847
# --------------------------------------------------------------------------- #
def _fan_out(prefix):
    src = prefix + "def f0():\n    return 1\n"
    for level in range(1, 41):
        src += f"def f{level}():\n    return [f{level - 1}(), f{level - 1}()]\n"
    return src + "x = f40()\n"


@pytest.mark.parametrize("prefix", [
    # Rebinding the counter in a try body, the one place assignments were not checked.
    "try:\n    _besser_count_call = Class\nexcept NameError:\n    pass\n",
    # Binding NameError so that try block counts as B-UML.
    "def unused(NameError):\n    return 1\n\ntry:\n    _besser_count_call = Class\nexcept NameError:\n    pass\n",
])
def test_the_call_budget_cannot_be_rebound(prefix):
    with pytest.raises(loader.SafeBumlLoaderError):
        _load(_fan_out(prefix))


@pytest.mark.parametrize("src", [
    "try:\n    _besser_guarded_setattr = Class\nexcept NameError:\n    pass\n",
    "try:\n    NameError = Class\nexcept NameError:\n    pass\n",
])
def test_a_try_body_cannot_bind_a_guard_name(src):
    """Also without warnings: this was open in the strict loader too."""
    with pytest.raises(loader.SafeBumlLoaderError):
        loader.safe_load_buml(src, allowed_names={"Class": object})


def test_function_type_parameters_are_refused():
    with pytest.raises(loader.SafeBumlLoaderError, match="type parameters"):
        _load("def f[T: Class]():\n    return 1\n")


def test_skipping_is_linear_in_the_file_size():
    import time
    src = "x0 = print\n" + "".join(f"x{i} = x{i - 1}\n" for i in range(1, 3000))
    warnings = []
    start = time.monotonic()
    loader.safe_load_buml(src, allowed_names={}, warnings=warnings)
    assert time.monotonic() - start < 5
    assert len(warnings) == 21 and "2980 more" in warnings[-1]


def test_the_builders_name_error_guard_is_not_skipped():
    """``domain_model_to_code`` writes it around a method's state machine."""
    model, warnings = _import(IMPORTS + """
book = Class(name="Book")
try:
    book.state_machine = sm
except NameError:
    pass
""")
    assert _names(model) == ["Book"]
    assert warnings == []


def test_a_model_built_twice_imports_once():
    model, warnings = _import(IMPORTS + """
def build():
    return DomainModel(name="Shop", types={Class(name="Order")})


first = build()
if __name__ == "__main__":
    model = build()
""")
    assert _names(model) == ["Order"]
    assert any("2 DomainModels" in w for w in warnings)
