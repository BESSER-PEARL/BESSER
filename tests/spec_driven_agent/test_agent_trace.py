"""Line-level attribution and the Agent Trace record (issue #620)."""

import json
import os
import shutil
import time
import tracemalloc

from besser.spec_driven_agent.state import agent_trace as at
from besser.utilities.web_modeling_editor.backend.services.spec_driven.secret_redaction import (
    scrub_secret_files,
)


def _write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _ranges(record, path):
    """``[(start, end, contributor type, model_id), ...]`` for one file, by start line."""
    entry = next(f for f in record["files"] if f["path"] == path)
    out = []
    for conversation in entry["conversations"]:
        contributor = conversation["contributor"]
        for r in conversation["ranges"]:
            out.append((r["start_line"], r["end_line"], contributor["type"], contributor.get("model_id")))
    return sorted(out)


def _record(root, run_id="run1", model_id="openai/gpt-4o"):
    state = at.reconcile_workspace(root, at.ai_contributor(model_id, run_id))
    return state, at.build_trace_record(
        root, state, run_id=run_id, besser_version="9.9.9", base_generator="django", model_id=model_id,
    )


def test_untouched_scaffold_lines_stay_with_the_generator_and_edits_go_to_the_ai(tmp_path):
    root = str(tmp_path)
    _write(root, "app/models.py", "a\nb\nc\nd\n")
    at.record_generated_files(root, ["app/models.py"], "django", "9.9.9")
    # The LLM edits one line, inserts two, and writes a new file.
    _write(root, "app/models.py", "a\nB\nc\nx\ny\nd\n")
    _write(root, "app/extra.py", "1\n2\n")

    state, record = _record(root)

    assert _ranges(record, "app/models.py") == [
        (1, 1, "unknown", None),
        (2, 2, "ai", "openai/gpt-4o"),
        (3, 3, "unknown", None),
        (4, 5, "ai", "openai/gpt-4o"),
        (6, 6, "unknown", None),
    ]
    assert _ranges(record, "app/extra.py") == [(1, 2, "ai", "openai/gpt-4o")]
    meta = record["metadata"][at.METADATA_KEY]
    assert meta["lines"] == {"generator": 3, "ai": 5, "human": 0, "unknown": 0}
    assert meta["run_id"] == "run1" and meta["base_generator"] == "django"
    # Generator ranges name the generator; AI ranges name the run.
    models = next(f for f in record["files"] if f["path"] == "app/models.py")
    related = {c["contributor"]["type"]: c["related"] for c in models["conversations"]}
    assert related["unknown"] == [{"type": "generator", "url": "urn:besser:generator:django@9.9.9"}]
    assert related["ai"] == [{"type": "session", "url": "urn:besser:run:run1"}]

    share = at.ai_share(state, "run1")
    assert share["aiLines"] == 5 and share["aiLinesThisRun"] == 5 and share["totalLines"] == 8
    assert share["files"] == [
        {"path": "app/models.py", "aiLines": 3, "totalLines": 6, "aiPct": 50.0},
        {"path": "app/extra.py", "aiLines": 2, "totalLines": 2, "aiPct": 100.0},
    ]


def test_record_has_the_required_agent_trace_fields(tmp_path):
    root = str(tmp_path)
    _write(root, "main.py", "print('hi')\n")
    _, record = _record(root)
    assert record["version"] == at.AGENT_TRACE_VERSION
    assert record["id"] and record["timestamp"]
    assert record["tool"] == {"name": at.TOOL_NAME, "version": "9.9.9"}
    (rng,) = record["files"][0]["conversations"][0]["ranges"]
    assert rng["start_line"] == 1 and rng["end_line"] == 1
    assert rng["content_hash"].startswith("sha256:")
    path = at.write_trace_record(root, record, "run1")
    assert path == os.path.join(root, ".agent-trace", "run1.json")
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle) == record


def test_a_regenerated_file_goes_back_to_the_generator(tmp_path):
    root = str(tmp_path)
    _write(root, "a.py", "x\n")
    at.reconcile_workspace(root, at.ai_contributor("m", "r"))
    at.record_generated_files(root, ["a.py"], "backend")
    _, record = _record(root)
    assert _ranges(record, "a.py") == [(1, 1, "unknown", None)]


def test_human_edits_between_runs_are_attributed_to_a_human(tmp_path):
    root = str(tmp_path)
    _write(root, "main.py", "g1\ng2\n")
    at.record_generated_files(root, ["main.py"], "backend")
    _write(root, "main.py", "g1\nai\ng2\n")
    at.reconcile_workspace(root, at.ai_contributor("openai/gpt-4o", "run1"))

    # Pushed, edited on GitHub, imported back as the seed of a new run.
    _write(root, "main.py", "g1\nai\nhuman\ng2\n")
    _write(root, "notes.md", "added by hand\n")
    at.seed_attribution(root)
    # The new run's LLM appends a line with another model.
    _write(root, "main.py", "g1\nai\nhuman\ng2\nai2\n")
    _, record = _record(root, run_id="run2", model_id="anthropic/claude")

    assert _ranges(record, "main.py") == [
        (1, 1, "unknown", None),
        (2, 2, "ai", "openai/gpt-4o"),
        (3, 3, "human", None),
        (4, 4, "unknown", None),
        (5, 5, "ai", "anthropic/claude"),
    ]
    assert _ranges(record, "notes.md") == [(1, 1, "human", None)]


def test_files_a_github_push_adds_are_not_credited_to_a_human(tmp_path):
    # The push copies the run's state, then writes buml/ and scrubs secrets;
    # continuing from that repo must not credit any of it to a person.
    run, repo = str(tmp_path / "run"), str(tmp_path / "repo")
    _write(run, "main.py", "g1\n")
    _write(run, "app/buml/kept.py", "x\n")
    _write(run, ".env.example", "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz\n")
    at.record_generated_files(run, ["main.py"], "backend")
    at.reconcile_workspace(run, at.ai_contributor("m", "r"))
    scrub_secret_files(run)                          # the run's own packaging scrub
    at.reconcile_workspace(run, at.generator_contributor("secret_redaction"))

    shutil.copytree(run, repo)
    _write(repo, "buml/domain_model.py", "domain = DomainModel('Library')\n")
    _write(repo, "buml/diagrams.json", '{"project": {}}\n')
    scrub_secret_files(repo)                         # the push's scrub
    at.seed_attribution(repo)

    state = at.AttributionState.load(repo)
    assert not any(p.startswith("buml/") for p in state.files)
    assert "app/buml/kept.py" in state.files         # only the top-level model dir
    kinds = {state.contributors[i]["kind"] for _, owners in state.files.values() for i in owners}
    assert at.KIND_HUMAN not in kinds


def test_a_seed_without_attribution_is_unknown(tmp_path):
    root = str(tmp_path)
    _write(root, "main.py", "old\n")
    at.seed_attribution(root)
    _write(root, "main.py", "old\nnew\n")
    _, record = _record(root)
    assert _ranges(record, "main.py") == [(1, 1, "unknown", None), (2, 2, "ai", "openai/gpt-4o")]


def test_internals_binaries_and_excluded_dirs_are_not_attributed(tmp_path):
    root = str(tmp_path)
    _write(root, "main.py", "x\n")
    _write(root, ".besser_recipe.json", "{}\n")
    _write(root, ".agent-trace/old.json", "{}\n")
    _write(root, "BESSER_GENERATION.md", "provenance\n")
    _write(root, "node_modules/pkg/index.js", "x\n")
    _write(root, "app.log", "x\n")
    with open(os.path.join(root, "logo.png"), "wb") as handle:
        handle.write(b"\x89PNG\x00\x01")
    state = at.reconcile_workspace(root, at.ai_contributor("m", "r"), {"node_modules", "*.log"})
    assert set(state.files) == {"main.py"}


def test_deleted_files_drop_out_and_state_round_trips(tmp_path):
    root = str(tmp_path)
    _write(root, "a.py", "1\n2\n")
    _write(root, "b.py", "3\n")
    at.record_generated_files(root, ["a.py", "b.py"], "backend")
    os.remove(os.path.join(root, "b.py"))
    _write(root, "a.py", "1\n2\n3\n")
    state = at.reconcile_workspace(root, at.ai_contributor("m", "r"))
    loaded = at.AttributionState.load(root)
    assert set(loaded.files) == {"a.py"}
    kinds = [loaded.contributors[i]["kind"] for i in loaded.files["a.py"][1]]
    assert kinds == ["generator", "generator", "ai"]
    assert [state.contributors[i] for i in state.files["a.py"][1]] == [
        loaded.contributors[i] for i in loaded.files["a.py"][1]
    ]


def test_a_corrupt_state_file_is_ignored(tmp_path):
    root = str(tmp_path)
    _write(root, at.ATTRIBUTION_FILENAME, '{"contributors": [{"kind": "ai"}], "files": {"a": {"hashes": "x", "runs": [[5, 1]]}}}')
    assert at.AttributionState.load(root) is None


def test_a_crafted_run_count_is_rejected_before_it_is_allocated(tmp_path):
    # An imported repo's state file is untrusted: a one-line file claiming
    # millions of lines used to allocate them before checking the hash count.
    root = str(tmp_path)
    state = {"contributors": [{"kind": "ai"}], "files": {"a": {"hashes": "x" * 12, "runs": [[0, 20_000_000]]}}}
    _write(root, at.ATTRIBUTION_FILENAME, json.dumps(state))
    tracemalloc.start()
    try:
        assert at.AttributionState.load(root) is None
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 10_000_000
    for runs in ([[0, 2], [0, -1]], [[0, 1.0]], [[0, "1"]], [[0, True]], [[0]]):
        state["files"]["a"]["runs"] = runs
        _write(root, at.ATTRIBUTION_FILENAME, json.dumps(state))
        assert at.AttributionState.load(root) is None, runs
    state["files"]["a"]["runs"] = [[0, 1]]
    _write(root, at.ATTRIBUTION_FILENAME, json.dumps(state))
    assert at.AttributionState.load(root).files["a"] == (["x" * 12], [0])


def test_a_long_repetitive_file_reconciles_quickly(tmp_path):
    # The line diff is quadratic here: 60k lines took minutes in _package_result.
    root = str(tmp_path)
    lines = ["{", "  x", "}"] * 20_000
    _write(root, "big.json", "\n".join(lines) + "\n")
    at.record_generated_files(root, ["big.json"], "backend")
    lines[30_000] = "  changed"
    _write(root, "big.json", "\n".join(lines) + "\n")
    started = time.monotonic()
    state = at.reconcile_workspace(root, at.ai_contributor("m", "r"))
    assert time.monotonic() - started < 3
    kinds = [state.contributors[i]["kind"] for i in state.files["big.json"][1]]
    assert kinds.count("ai") == 1 and kinds[30_000] == "ai"


def test_model_ids_use_the_provider_prefix():
    assert at.normalize_model_id("openai", "gpt-4o") == "openai/gpt-4o"
    assert at.normalize_model_id("anthropic", "anthropic/claude") == "anthropic/claude"
    assert at.normalize_model_id("free", "qwen-coder") == "qwen-coder"
    assert at.normalize_model_id("openai", "") is None
