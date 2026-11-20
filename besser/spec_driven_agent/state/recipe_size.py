"""The size bound of ``.besser_recipe.json``, applied when the orchestrator writes it.

Readers refuse a recipe over ``MAX_RECIPE_BYTES``: the workspace is writable by
the model's tools, so a planted or runaway file must not be read into memory or
sent in the done event. The writer therefore keeps a legitimate recipe well
under that limit by cutting the detail no reader of the recipe uses, and says
what it cut.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from besser.spec_driven_agent.run_report import RECIPE_FILENAME

logger = logging.getLogger(__name__)

# What the backend reads back from a run's workspace and embeds in the done event.
MAX_RECIPE_BYTES = 256 * 1024
# Half: packaging adds the run and verification sections, and a later run rewrites the file.
RECIPE_WRITE_BUDGET = MAX_RECIPE_BYTES // 2
# Present only on a recipe that was cut, with the count of each cut (0 for a cut not needed).
SHORTENED_KEY = "recipe_shortened"

# A tool-call input value kept in the summary besides the path: a scalar this short (ids, actions, flags).
_SUMMARY_VALUE_CHARS = 80
# The calls ``run_changes`` and the history read back.
_FILE_CHANGE_TOOLS = frozenset({"write_file", "modify_file", "replace_file_lines", "delete_file"})
# Request and response bodies of an API workflow; method, path, status and failures stay.
_BODY_KEYS = ("json", "text")


def _dicts(value: Any) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _short(value: Any) -> bool:
    return not isinstance(value, (dict, list)) and not (isinstance(value, str) and len(value) > _SUMMARY_VALUE_CHARS)


def _cut_tool_call_inputs(recipe: dict) -> int:
    """Each tool call's input down to its path and short values; the trace keeps the rest."""
    cut = 0
    for call in _dicts(recipe.get("tool_calls")):
        given = call.get("input")
        if isinstance(given, dict):
            kept = {key: value for key, value in given.items() if key == "path" or _short(value)}
            if len(kept) < len(given):
                call["input"] = kept
                cut += 1
    return cut


def _cut_api_bodies(recipe: dict) -> int:
    """The request and response bodies of the API workflows."""
    cut = 0
    for scenario in _dicts(recipe.get("api_scenarios")):
        definition = scenario.get("scenario") if isinstance(scenario.get("scenario"), dict) else {}
        report = scenario.get("report") if isinstance(scenario.get("report"), dict) else {}
        for item in (*_dicts(definition.get("requests")), *_dicts(scenario.get("submitted_requests")),
                     *_dicts(report.get("responses"))):
            for key in _BODY_KEYS:
                if key in item:
                    del item[key]
                    cut += 1
    return cut


def _cut_tool_calls(recipe: dict) -> int:
    """All but the last call, and the last successful one, of each file-changing tool on each file;
    ``.besser_trace.jsonl`` has every call."""
    calls = recipe.get("tool_calls")
    if not isinstance(calls, list):
        return 0
    last = {}
    for index, call in enumerate(calls):
        if isinstance(call, dict) and call.get("tool") in _FILE_CHANGE_TOOLS:
            given = call.get("input") if isinstance(call.get("input"), dict) else {}
            key = (call["tool"], str(given.get("path")))
            last[key] = index
            if call.get("success") is not False:
                last[(*key, "succeeded")] = index
    kept = set(last.values())
    recipe["tool_calls"] = [call for index, call in enumerate(calls) if index in kept]
    return len(calls) - len(kept)


def _cut_api_details(recipe: dict) -> int:
    """Each API workflow down to what the verification report reads: its id, each request's
    method and path, and its status."""
    scenarios = _dicts(recipe.get("api_scenarios"))
    if not scenarios:
        return 0
    kept = []
    for scenario in scenarios:
        definition = scenario.get("scenario") if isinstance(scenario.get("scenario"), dict) else {}
        report = scenario.get("report") if isinstance(scenario.get("report"), dict) else {}
        kept.append({
            "scenario_id": scenario.get("scenario_id"),
            "scenario": {"requests": [{key: request[key] for key in ("method", "path") if key in request}
                                      for request in _dicts(definition.get("requests"))]},
            "report": {key: report[key] for key in ("status",) if key in report},
        })
    recipe["api_scenarios"] = kept
    return len(kept)


# Least loss first: the trace has every call with its input (.besser_tool_inputs.jsonl the
# whole write inputs), no reader uses the bodies, and the files each tool changed stay listed;
# the per-request results of a workflow are in the recipe alone, so they go last.
_CUTS = (
    ("tool_call_inputs_cut", _cut_tool_call_inputs),
    ("api_bodies_cut", _cut_api_bodies),
    ("tool_calls_dropped", _cut_tool_calls),
    ("api_workflows_cut", _cut_api_details),
)


def _dump(recipe: dict) -> str:
    return json.dumps(recipe, indent=2, default=str)


def bounded_recipe(recipe: dict) -> tuple[dict, str]:
    """``recipe`` as plain JSON data and its text, cut to ``RECIPE_WRITE_BUDGET`` when over it.

    A recipe within the budget is written exactly as before. Over it, the cuts
    in ``_CUTS`` apply in order until it fits, and ``SHORTENED_KEY`` records
    each count; true totals (``tool_calls_count``) are not touched.
    """
    text = _dump(recipe)
    size = len(text.encode("utf-8"))
    if size <= RECIPE_WRITE_BUDGET:
        return json.loads(text), text
    data = json.loads(text)  # a copy: the cuts never touch the caller's objects
    shortened = {"bytes_before": size, "budget_bytes": RECIPE_WRITE_BUDGET, **{name: 0 for name, _ in _CUTS}}
    for name, cut in _CUTS:
        shortened[name] = cut(data)
        text = _dump({**data, SHORTENED_KEY: shortened})
        if len(text.encode("utf-8")) <= RECIPE_WRITE_BUDGET:
            break
    size_after = len(text.encode("utf-8"))
    logger.warning("%s was %d bytes, over the %d-byte budget; cut to %d bytes: %s", RECIPE_FILENAME,
                   size, RECIPE_WRITE_BUDGET, size_after,
                   ", ".join(f"{name}={shortened[name]}" for name, _ in _CUTS))
    if size_after > RECIPE_WRITE_BUDGET:
        logger.warning("%s is still %d bytes after every cut: what is left is what its readers need",
                       RECIPE_FILENAME, size_after)
    return {**data, SHORTENED_KEY: shortened}, text
