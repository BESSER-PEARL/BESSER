"""Unit tests for the custom-code lint in services/validators/python_code_validator.py."""

import pytest

from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    CodeValidationError,
    ValidationError,
)
from besser.utilities.web_modeling_editor.backend.services.validators.python_code_validator import (
    lint_simulation_tool_code,
    validate_custom_code_action,
)

VALID_ACTION = (
    "def greet(session):\n"
    "    import json\n"
    "    session.reply(json.dumps({'hello': 'world'}))\n"
)


def test_code_validation_error_is_a_validation_error():
    # Needed so @handle_endpoint_errors and the app handlers map it to HTTP 400.
    assert issubclass(CodeValidationError, ValidationError)


@pytest.mark.parametrize("simulation", [False, True])
def test_accepts_valid_action(simulation):
    validate_custom_code_action(VALID_ACTION, simulation=simulation)


def test_accepts_typed_session_parameter_and_extra_parameters():
    validate_custom_code_action("def act(session: 'Session', retries=3):\n    return retries\n")


@pytest.mark.parametrize(
    "source, message",
    [
        ("", "cannot be empty"),
        ("   \n  ", "cannot be empty"),
        ("def broken(session:\n    pass", "Syntax error"),
        ("import os\ndef act(session):\n    pass\n", "exactly one top-level function"),
        ("x = 1\n", "must start with a function definition"),
        ("async def act(session):\n    pass\n", "Async function definitions are not allowed"),
        ("def act(ctx):\n    pass\n", "first parameter must be named 'session'"),
        ("def act():\n    pass\n", "first parameter must be named 'session'"),
    ],
)
def test_rejects_structural_violations(source, message):
    with pytest.raises(CodeValidationError, match=message):
        validate_custom_code_action(source)


@pytest.mark.parametrize(
    "body",
    [
        "    import os\n",
        "    import os.path\n",
        "    from subprocess import run\n",
        "    from urllib.request import urlopen\n",
        "    eval('1 + 1')\n",
        "    exec('x = 1')\n",
        "    open('/etc/passwd')\n",
        "    __import__('socket')\n",
    ],
)
def test_denylist_applies_only_in_simulation(body):
    source = f"def act(session):\n{body}"
    # Standard code generation does not restrict imports or builtins...
    validate_custom_code_action(source)
    # ...but simulation lints them.
    with pytest.raises(CodeValidationError, match="not allowed during agent simulation"):
        validate_custom_code_action(source, simulation=True)


def test_denylist_catches_nested_functions():
    source = "def act(session):\n    def inner():\n        import subprocess\n    inner()\n"
    with pytest.raises(CodeValidationError, match="subprocess"):
        validate_custom_code_action(source, simulation=True)


def test_tool_code_does_not_need_a_session_parameter():
    lint_simulation_tool_code("def add(a: int, b: int) -> int:\n    return a + b\n")


def test_tool_code_may_have_helpers_and_imports():
    lint_simulation_tool_code("import math\n\ndef area(r):\n    return math.pi * r * r\n")


@pytest.mark.parametrize(
    "source, message",
    [
        ("", "cannot be empty"),
        ("def add(a, b:\n", "Syntax error"),
        ("import os\ndef ls():\n    return os.listdir('.')\n", "Import of 'os'"),
        ("def run(cmd):\n    return eval(cmd)\n", r"Call to 'eval\(\)'"),
    ],
)
def test_tool_code_rejections(source, message):
    with pytest.raises(CodeValidationError, match=message):
        lint_simulation_tool_code(source)
