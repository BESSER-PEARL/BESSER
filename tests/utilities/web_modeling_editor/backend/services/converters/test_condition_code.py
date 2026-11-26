"""Naming the function of a custom transition condition."""
import ast

from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.condition_code import (
    rename_condition_function,
)


def _function_names(code):
    return [node.name for node in ast.parse(code).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def test_a_single_function_is_renamed():
    code = "def condition(session: 'Session', params: dict) -> bool:\n    return True\n"
    renamed = rename_condition_function(code, "condition_3_1")
    assert _function_names(renamed) == ["condition_3_1"]
    assert "params: dict" in renamed


def test_a_helper_above_the_condition_keeps_its_name():
    code = (
        "def _is_q1(session):\n"
        "    return session.get('came_from') == 'q1'\n"
        "\n"
        "def condition(session):\n"
        "    return _is_q1(session)\n"
    )
    renamed = rename_condition_function(code, "condition_3_1")
    assert _function_names(renamed) == ["_is_q1", "condition_3_1"]
    assert "return _is_q1(session)" in renamed


def test_a_helper_below_the_condition_keeps_its_name_too():
    code = (
        "def condition(session):\n"
        "    return helper(session)\n"
        "\n"
        "def helper(session):\n"
        "    return True\n"
    )
    assert _function_names(rename_condition_function(code, "condition_3_1")) == ["condition_3_1", "helper"]


def test_a_recursive_condition_calls_itself_by_its_new_name():
    code = "def condition(session, n=0):\n    return n > 2 or condition(session, n + 1)\n"
    renamed = rename_condition_function(code, "condition_3_1")
    assert "condition(session, n + 1)" not in renamed
    assert "condition_3_1(session, n + 1)" in renamed


def test_names_that_merely_contain_the_old_name_are_left_alone():
    code = "def condition(session):\n    return condition_ok(session)\n\ndef condition_ok(session):\n    return True\n"
    renamed = rename_condition_function(code, "condition_3_1")
    assert _function_names(renamed) == ["condition_3_1", "condition_ok"]
    assert "return condition_ok(session)" in renamed


def test_code_that_does_not_parse_still_gets_its_first_def_renamed():
    code = "def condition(session):\n    return (\n"
    assert rename_condition_function(code, "condition_3_1").startswith("def condition_3_1(session):")


def test_code_without_a_function_is_left_alone():
    assert rename_condition_function("x = 1\n", "condition_3_1") == "x = 1\n"
