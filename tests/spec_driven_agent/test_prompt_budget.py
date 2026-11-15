"""The Phase 2 request prefix (system prompt + tool schemas) stays within its budget.

Every Phase 2 turn re-sends that prefix; without prompt caching it is paid in
full on each turn, so it must not grow back silently. Provider caching also
needs it byte-identical for the same inputs. ``prompt_budget.py`` builds the
prefix through the orchestrator's own code paths.
"""
import json

import pytest

from besser.spec_driven_agent.agent.tools import get_all_tools_including_generators
from tests.spec_driven_agent.prompt_budget import FIXTURES, breakdown, phase2_request

# Room for small wording fixes; anything larger should be a deliberate raise.
HEADROOM = 0.06

# Characters measured when the budget was pinned. Raise one only on purpose,
# saying in the commit what the extra prefix buys. Re-measure with
# `python -m tests.spec_driven_agent.prompt_budget` from the repo root: it
# prints chars per fixture, section and tool.
BASELINES = {
    "shelf": {"system_chars": 23_312, "tools_chars": 14_861},
    "shelf_no_requirements": {"system_chars": 20_291, "tools_chars": 14_861},
    "conference": {"system_chars": 44_175, "tools_chars": 14_861},
    "state_machine": {"system_chars": 13_614, "tools_chars": 12_603},
}

# Per-section and per-tool sizes at the same point, only to name what grew.
BASELINE_SECTIONS = {
    "shelf": {"role_plan": 1589, "rules": 9635, "model_tools": 375, "models": 803, "data_contract": 1618,
              "tail_header": 60, "inventory": 2071, "endpoint_manifest": 4747, "requirements": 351,
              "request": 326, "checklist": 1124, "phase2_appendix": 613},
    "shelf_no_requirements": {"role_plan": 1589, "rules": 9635, "model_tools": 375, "models": 803,
                              "data_contract": 1618, "tail_header": 60, "inventory": 2071,
                              "endpoint_manifest": 2077, "request": 326, "checklist": 1124,
                              "phase2_appendix": 613},
    "conference": {"role_plan": 1589, "rules": 9635, "model_tools": 375, "models": 5205, "data_contract": 1725,
                   "tail_header": 60, "inventory": 5758, "endpoint_manifest": 17310, "requirements": 351,
                   "request": 326, "checklist": 1228, "phase2_appendix": 613},
    "state_machine": {"role_plan": 993, "rules": 9823, "models": 464, "tail_header": 60, "requirements": 351,
                      "request": 275, "checklist": 1035, "phase2_appendix": 613},
}
BASELINE_TOOLS = {
    "generate_python_classes": 195, "generate_java_classes": 170, "generate_rdf": 156, "generate_supabase": 358,
    "list_files": 245, "read_file": 1238, "write_file": 673, "modify_file": 1473, "replace_file_lines": 1175,
    "search_in_files": 701, "delete_file": 558, "query_class": 416, "list_classes_with": 414,
    "get_constraints_for": 375, "test_api": 2745, "validate_app": 468, "validate_model": 166, "check_syntax": 384,
    "task_list": 2931,
}


@pytest.fixture(scope="module")
def prefixes(tmp_path_factory):
    """Each fixture's first Phase 2 request, built twice in two workspaces."""
    built = {}
    for name, fixture in FIXTURES.items():
        first, second = tmp_path_factory.mktemp(f"{name}_a"), tmp_path_factory.mktemp(f"{name}_b")
        built[name] = (first, phase2_request(fixture(), first), phase2_request(fixture(), second))
    return built


def _growth(measured: dict, baseline: dict) -> str:
    grown = [f"{key} +{size - baseline.get(key, 0):,}" for key, size in measured.items()
             if size > baseline.get(key, 0)]
    return ", ".join(grown) or "no single part grew; the parts' total did"


def _over_budget(what: str, fixture: str, size: int, key: str, growth: str) -> str:
    baseline = BASELINES[fixture][key]
    return (f"{fixture}: {what} {size:,} chars, {size - baseline:+,} on the pinned {baseline:,} "
            f"(budget {int(baseline * (1 + HEADROOM)):,}). Grew: {growth}. Every Phase 2 turn re-sends "
            f"this. If the growth is wanted, raise BASELINES[{fixture!r}][{key!r}] deliberately and say why.")


@pytest.mark.parametrize("fixture", FIXTURES)
def test_system_prompt_within_budget(prefixes, fixture):
    _, (system, tools), _ = prefixes[fixture]
    measured = breakdown(system, tools)
    assert measured["system_chars"] <= BASELINES[fixture]["system_chars"] * (1 + HEADROOM), _over_budget(
        "system prompt", fixture, measured["system_chars"], "system_chars",
        _growth(measured["sections"], BASELINE_SECTIONS[fixture]))


@pytest.mark.parametrize("fixture", FIXTURES)
def test_tool_schemas_within_budget(prefixes, fixture):
    _, (system, tools), _ = prefixes[fixture]
    measured = breakdown(system, tools)
    assert measured["tools_chars"] <= BASELINES[fixture]["tools_chars"] * (1 + HEADROOM), _over_budget(
        "tool schemas", fixture, measured["tools_chars"], "tools_chars",
        _growth(measured["tools"], BASELINE_TOOLS))


@pytest.mark.parametrize("fixture", FIXTURES)
def test_prefix_is_byte_identical_for_the_same_inputs(prefixes, fixture):
    """A prompt cache hits only on an identical prefix: no set order, time or path may leak in."""
    workspace, (system, tools), (system_again, tools_again) = prefixes[fixture]
    assert system == system_again
    assert json.dumps(tools) == json.dumps(tools_again)
    assert str(workspace) not in system + json.dumps(tools)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_tool_order_is_the_canonical_order(prefixes, fixture):
    _, (_, tools), _ = prefixes[fixture]
    canonical = [tool["name"] for tool in get_all_tools_including_generators()]
    names = [tool["name"] for tool in tools]
    assert names == [name for name in canonical if name in names]


def test_both_mutation_coverage_paths_are_measured(prefixes):
    """The fixtures with requirements carry the coverage block; the known-empty ledger omits it."""
    def system(name):
        return prefixes[name][1][0]
    block = "Relationship mutation coverage"
    assert block in system("shelf") and block in system("conference")
    assert block not in system("shelf_no_requirements")
    assert "Request shapes (inherited fields included):" in system("shelf_no_requirements")
