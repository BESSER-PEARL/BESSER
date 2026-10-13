"""Harness writes to a fixed workspace name must not go through a planted link.

``ln -s /app/besser/... .besser_trace.jsonl`` from the sandboxed shell made the
unsandboxed worker append its trace to a file outside the run (another run's
workspace, or the worker's own code). An "only if absent" writer is caught by a
DANGLING link: ``exists()`` follows it, reports False, and ``open("w")`` then
creates the link's outside target.
"""
import os

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.agent.runbook import PROBE_FILENAME, install_probe
from besser.spec_driven_agent.execution.workspace_fs import open_plain_write, write_atomic_plain
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning.stack_metadata import pre_generate_metadata
from besser.spec_driven_agent.providers.llm_client import UsageTracker
from besser.spec_driven_agent.repair import scaffold_repair
from besser.spec_driven_agent.state import checkpoint
from besser.spec_driven_agent.state.tracing import TraceWriter
from besser.spec_driven_agent.validation import toolchain
from tests.spec_driven_agent.test_checkpoint import _make_checkpoint

ORIGINAL = "ORIGINAL - must not change\n"


class _Client:
    model = "mock-model"

    def __init__(self):
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover
        raise AssertionError("no LLM call expected")


def _orch(ws):
    return LLMOrchestrator(
        llm_client=_Client(), domain_model=DomainModel(name="M", types={Class(name="A")}),
        output_dir=str(ws), enable_tracing=False, enable_checkpointing=False,
        enable_toolchain_validation=False,
    )


# name -> (target inside the workspace, writer, dangling?)
WRITERS = {
    "trace_append": (".besser_trace.jsonl", lambda ws: TraceWriter(str(ws)).write("x"), False),
    "tool_input_sidecar": (".besser_tool_inputs.jsonl",
                           lambda ws: _orch(ws)._record_full_tool_input(1, "write_file", {}, True, "ok"), False),
    "recipe": (".besser_recipe.json", lambda ws: _orch(ws)._save_recipe("x", 1.0), False),
    # The final name is replaced by a rename; the temp file is the one opened.
    "checkpoint": (".besser_checkpoint.json.tmp",
                   lambda ws: checkpoint.save_checkpoint(str(ws), _make_checkpoint()), False),
    "probe_install": (PROBE_FILENAME, lambda ws: install_probe(str(ws)), False),
    "scaffold_write": ("Dockerfile", lambda ws: scaffold_repair._write_text(str(ws / "Dockerfile"), "FROM x\n"), False),
    "tsc_probe": (toolchain._TSC_PROBE_NAME, lambda ws: toolchain._tsc_project_arg(str(ws), False), False),
    "gitignore_if_absent": (".gitignore", lambda ws: _orch(ws)._ensure_gitignore(), True),
    "stack_metadata_if_absent": ("Cargo.toml", lambda ws: pre_generate_metadata("rust", str(ws)), True),
    "requirements_if_absent": ("requirements.txt",
                               lambda ws: scaffold_repair._ensure_requirements_txt(str(ws)), True),
}


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_a_planted_link_is_not_written_through(tmp_path, writer):
    name, write, dangling = WRITERS[writer]
    outside = tmp_path / "outside"
    outside.mkdir()
    canary = outside / "canary"
    if not dangling:
        canary.write_text(ORIGINAL, encoding="utf-8")
    ws = tmp_path / "ws"
    ws.mkdir()
    try:
        os.symlink(canary, ws / name)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable here: {exc}")

    try:
        write(ws)
    except Exception:  # a refusal may surface; a write-through may not
        pass

    if dangling:
        assert not canary.exists(), f"{writer} created the link's outside target"
    else:
        assert canary.read_text(encoding="utf-8") == ORIGINAL, f"{writer} wrote through the link"


def test_the_write_helpers(tmp_path):
    (tmp_path / "outside.txt").write_text(ORIGINAL, encoding="utf-8")
    ws = tmp_path / "ws"
    (ws / "real").mkdir(parents=True)
    try:
        os.symlink(tmp_path / "outside.txt", ws / "link.txt")
        os.symlink(tmp_path, ws / "dirlink", target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable here: {exc}")

    with open_plain_write(ws / "link.txt", root=ws, encoding="utf-8") as fh:
        fh.write("new\n")
    write_atomic_plain(ws / "real" / "a.json", "{}", root=ws)
    with open_plain_write(ws / "real" / "log", "ab", root=ws) as fh:
        fh.write(b"one\n")
    with open_plain_write(ws / "real" / "log", "ab", root=ws) as fh:
        fh.write(b"two\n")

    assert (tmp_path / "outside.txt").read_text(encoding="utf-8") == ORIGINAL
    assert not os.path.islink(ws / "link.txt") and (ws / "link.txt").read_text() == "new\n"
    assert (ws / "real" / "a.json").read_text() == "{}"
    assert (ws / "real" / "log").read_bytes() == b"one\ntwo\n"
    with pytest.raises(OSError):
        open_plain_write(ws / "dirlink" / "x.txt", root=ws)
    with pytest.raises(OSError):
        open_plain_write(tmp_path / "escape.txt", root=ws)
    with pytest.raises(OSError):
        write_atomic_plain(ws / "dirlink" / "x.json", "{}", root=ws)
    assert not (tmp_path / "x.txt").exists() and not (tmp_path / "x.json").exists()
