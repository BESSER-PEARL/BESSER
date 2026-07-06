"""Tests for the test-case generator (pytest + Hypothesis suite emitter).

The key guard is that the generated ``test_hypothesis.py`` always compiles:
we ``compile()`` the output (never ``exec`` it, so neither ``hypothesis`` nor a
real ``classes`` module is required) so CI catches any template syntax break.
"""

import os

import pytest

from besser.generators.testgen import TestCaseGenerator


@pytest.fixture
def domain_model(library_book_author_model):
    """Alias the shared fixture from tests/conftest.py."""
    return library_book_author_model


def _generate(model, output_dir, **kwargs):
    TestCaseGenerator(model=model, output_dir=str(output_dir), **kwargs).generate()
    generated = os.path.join(str(output_dir), "test_hypothesis.py")
    assert os.path.exists(generated), "test_hypothesis.py was not generated"
    with open(generated, encoding="utf-8") as f:
        return f.read()


def test_generated_suite_compiles(domain_model, tmp_path):
    """The generated suite must be syntactically valid Python."""
    source = _generate(domain_model, tmp_path)
    # Raises SyntaxError if the template emits invalid Python.
    compile(source, "test_hypothesis.py", "exec")


def test_generated_suite_has_expected_structure(domain_model, tmp_path):
    source = _generate(domain_model, tmp_path)

    # Default module name is "classes".
    assert "from classes import (" in source
    # Domain classes are imported and get structural tests.
    for class_name in ("Library", "Book", "Author"):
        assert class_name in source
    assert "def test_book_is_not_abstract():" in source
    assert "def test_library_constructor_args():" in source
    # Hypothesis strategies are emitted per class.
    assert "Book_strategy = st.builds(" in source
    # Section 3 header is always present (this model has no OCL post-conditions).
    assert "SECTION 3 — OCL TESTS" in source


def test_module_name_is_configurable(domain_model, tmp_path):
    source = _generate(domain_model, tmp_path, module_name="my_domain")
    assert "from my_domain import (" in source
    assert "from classes import (" not in source
