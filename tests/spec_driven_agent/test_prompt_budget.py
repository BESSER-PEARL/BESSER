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
# saying in the commit what the extra prefix buys.
BASELINES = {
    "shelf": {"system_chars": 23_986, "tools_chars": 16_683},
    "conference": {"system_chars": 44_454, "tools_chars": 16_683},
    "state_machine": {"system_chars": 14_923, "tools_chars": 14_297},
}

# Per-section and per-tool sizes at the same point, only to name what grew.
BASELINE_SECTIONS = {
    "shelf": {"role_plan": 1728, "rules": 10301, "model_tools": 375, "models": 803, "data_contract": 1515,
              "tail_header": 60, "inventory": 2071, "endpoint_manifest": 4270, "requirements": 351,
              "request": 333, "checklist": 1327, "request_footer": 194, "phase2_appendix": 658},
    "conference": {"role_plan": 1728, "rules": 10301, "model_tools": 375, "models": 5205, "data_contract": 1622,
                   "tail_header": 60, "inventory": 5758, "endpoint_manifest": 16542, "requirements": 351,
                   "request": 333, "checklist": 1327, "request_footer": 194, "phase2_appendix": 658},
    "state_machine": {"role_plan": 977, "rules": 10489, "models": 592, "tail_header": 60, "requirements": 351,
                      "request": 275, "checklist": 1327, "request_footer": 194, "phase2_appendix": 658},
}
BASELINE_TOOLS = {
    "generate_python_classes": 195, "generate_java_classes": 170, "generate_rdf": 156, "generate_supabase": 358,
    "list_files": 146, "read_file": 1338, "write_file": 720, "modify_file": 1490, "replace_file_lines": 1300,
    "search_in_files": 746, "delete_file": 610, "query_class": 416, "list_classes_with": 550,
    "get_constraints_for": 375, "test_api": 2988, "validate_app": 468, "validate_model": 166, "check_syntax": 384,
    "task_list": 4107,
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
