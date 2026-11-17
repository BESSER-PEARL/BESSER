"""Every property of every tool, sent each badly typed value a weak model
produces: ``execute_typed`` never raises, never leaks the workspace path, and a
non-string value never surfaces as a raw ``<tool> failed: ...`` exception."""
import json

import pytest

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.agent.tools import get_all_tools_including_generators

# A valid value for each required string argument, so the fuzzed one is the
# only thing wrong with a call.
_VALID = {"path": "a.py", "command": "true", "pattern": "x", "old_text": "x = 1",
          "new_text": "x = 2", "content": "y = 1\n", "name": "A", "predicate": "is_abstract",
          "class_name": "A", "read_id": "r"}
_NON_STRINGS = [None, 12, [], [1], {}, {"a": 1}, 10**30, float("inf"), float("nan"), True, 1.5]
_STRINGS = ["abc", ""]


def _valid(name, spec):
    if "enum" in spec:
        return spec["enum"][0]
    return {"string": _VALID.get(name, "x"), "integer": 1, "boolean": False, "array": []}[spec["type"]]


_CASES = [
    (tool["name"], {n: _valid(n, tool["input_schema"]["properties"][n])
                    for n in tool["input_schema"].get("required", ())}, prop)
    for tool in get_all_tools_including_generators()
    for prop in tool["input_schema"].get("properties", {})
]


@pytest.fixture(scope="module")
def executor(tmp_path_factory):
    ex = ToolExecutor(str(tmp_path_factory.mktemp("fuzz")), per_write_diagnostics=False, allow_shell=True)
    yield ex
    ex.close()


@pytest.mark.parametrize("tool, base, prop", _CASES, ids=[f"{t}.{p}" for t, _, p in _CASES])
def test_bad_argument_types_are_answered_by_name(executor, tool, base, prop):
    for value in _NON_STRINGS + _STRINGS:
        with open(f"{executor.workspace}/a.py", "w", encoding="utf-8") as fh:
            fh.write("x = 1\n")
        payload = json.loads(executor.execute(tool, {**base, prop: value}))
        error = payload.get("error")
        if not error:
            continue
        assert executor.workspace not in error, (prop, value, error)
        if " was not run: " in error:
            assert prop in error, (prop, value, error)
        if value in _NON_STRINGS:
            assert not error.startswith(f"{tool} failed:"), (prop, value, error)
