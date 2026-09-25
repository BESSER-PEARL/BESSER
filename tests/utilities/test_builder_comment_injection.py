"""A model name must not be able to escape a ``#`` comment into executable code.

The code builders emit Python that is later ``exec()``'d. ``NamedElement.name``
rejects empty and whitespace-only names, but it does **not** reject an embedded
newline::

    NamedElement(name='Idle\\n__import__("os").system("id")')   # accepted

So every name written into a ``#`` comment line had to go through
``common._comment_safe``, which folds newlines to spaces. Four builders wrote
theirs raw, and ``project_builder`` sanitised two of its eight labels — the
familiar shape where one call site is fixed and its siblings are not.
"""
import inspect

import pytest

from besser.utilities.buml_code_builder.common import _comment_safe

BUILDERS = [
    "agent_model_builder",
    "domain_model_builder",
    "gui_model_builder",
    "quantum_model_builder",
    "project_builder",
    "bpmn_model_builder",
    "nn_model_builder",
]

PAYLOAD = 'Idle\n__import__("os").system("id")'


def test_the_helper_neutralises_every_newline_form():
    for raw in (PAYLOAD, "a\rb", "a\r\nb", "a\nb\nc"):
        cleaned = _comment_safe(raw)
        assert "\n" not in cleaned and "\r" not in cleaned
    assert _comment_safe("") == ""
    assert _comment_safe(None) == ""


def test_a_name_cannot_escape_the_comment_line():
    """One line in, one line out — the payload stays commented."""
    line = f"# {_comment_safe(PAYLOAD)} state\n"
    assert line.count("\n") == 1
    assert line.startswith("#")
    assert "__import__" in line  # still visible, but inert


@pytest.mark.parametrize("module_name", BUILDERS)
def test_no_builder_writes_an_unsanitised_name_into_a_comment(module_name):
    """Guards against a new raw site being added later.

    Scans the builder source for f-string comment writes and requires each
    interpolation to be either a sanitising call or a generated identifier —
    never a bare ``*.name``.
    """
    import importlib

    mod = importlib.import_module(f"besser.utilities.buml_code_builder.{module_name}")
    source = inspect.getsource(mod)

    offenders = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if 'f"#' not in stripped and "f'#" not in stripped:
            continue
        if "_comment_safe" in stripped or "_safe_comment" in stripped:
            continue
        # A bare `.name` interpolated into a comment is the bug.
        if ".name}" in stripped or ".name!r}" in stripped:
            offenders.append(f"{module_name}.py:{lineno}: {stripped[:90]}")

    assert not offenders, (
        "unsanitised name(s) written into a comment line:\n  "
        + "\n  ".join(offenders)
        + "\nRoute them through common._comment_safe."
    )
