"""No harness reader may surface a file a planted link points at.

The model's shell writes the workspace; the harness reads it unsandboxed and
pastes what it finds into the prompt (scaffold snapshot, inventory symbols,
endpoint manifest, design CSS, compaction summary), copies it (Phase 3
snapshot, probe scratch copies) or cites it (requirements ledger). One
``ln -s /proc/self/environ app/leak.py`` then puts the worker's keys in front
of the model. Each reader below is pointed at a workspace holding such a link
to an outside canary.
"""
import os

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.agent import compaction, design_system, prompt_builder
from besser.spec_driven_agent.execution.workspace_fs import copytree_plain, walk_plain
from besser.spec_driven_agent.planning import requirements_ledger
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker

CANARY = "canary7f3a"
CANARY_TEXT = (
    f"def {CANARY}_fn():\n    pass\n"
    f'@app.get("/{CANARY}-route")\n'
    f":root{{--ds-{CANARY}:#000}}\n.ds-card{{color:red}}\n.ds-page-{CANARY}{{color:red}}\n"
    f"OPENAI_API_KEY=sk-{CANARY}\n"
)


class _Client:
    def __init__(self):
        self.usage = UsageTracker("mock-model")

    def chat(self, **kwargs):  # pragma: no cover
        raise AssertionError("no LLM call expected")


@pytest.fixture
def workspace(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text(CANARY_TEXT, encoding="utf-8")
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "secrets.py").write_text(CANARY_TEXT, encoding="utf-8")
    ws = tmp_path / "ws"
    (ws / "backend").mkdir(parents=True)
    (ws / "frontend" / "src").mkdir(parents=True)
    (ws / "backend" / "main_api.py").write_text(
        'from fastapi import FastAPI\napp = FastAPI()\n@app.get("/health")\ndef h():\n    return 1\n',
        encoding="utf-8")
    links = {
        ws / "backend" / "leak.py": outside,
        ws / "frontend" / "src" / "design.css": outside,
        ws / "notes.txt": outside,
        ws / "linked_dir": outside_dir,
    }
    try:
        for link, target in links.items():
            os.symlink(target, link, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable here: {exc}")
    return ws


def _snapshot_tree(ws):
    orch = LLMOrchestrator(
        llm_client=_Client(), domain_model=DomainModel(name="M", types={Class(name="A")}),
        output_dir=str(ws), enable_tracing=False, enable_checkpointing=False,
        enable_toolchain_validation=False,
    )
    orch._create_snapshot()
    return _read_tree(ws / ".besser_snapshot")


def _read_tree(root):
    out = []
    for folder, _dirs, files in os.walk(root):
        for name in files:
            with open(os.path.join(folder, name), encoding="utf-8", errors="replace") as fh:
                out.append(name + "\n" + fh.read())
    return "\n".join(out)


def _copied_tree(ws):
    dst = ws.parent / "copy"
    copytree_plain(str(ws), str(dst))
    return _read_tree(dst)


READERS = {
    "scaffold_snapshot": lambda ws: prompt_builder.build_scaffold_snapshot(str(ws)),
    "inventory": lambda ws: prompt_builder.build_inventory(str(ws), None, "generate_fastapi_backend"),
    "endpoint_manifest": lambda ws: prompt_builder.build_endpoint_manifest(str(ws)),
    "design_css": lambda ws: str(design_system.find_design_css(str(ws)))
    + design_system.design_system_section(str(ws), None),
    "compaction_summary": lambda ws: str(compaction._summarize_messages([], [], str(ws))),
    "requirements_sources": lambda ws: repr(requirements_ledger._source_files(str(ws))),
    "phase3_snapshot": _snapshot_tree,
    "probe_scratch_copy": _copied_tree,
    "walk_plain": lambda ws: repr(list(walk_plain(str(ws)))),
}


@pytest.mark.parametrize("reader", sorted(READERS))
def test_no_reader_surfaces_a_linked_outside_file(workspace, reader):
    surfaced = READERS[reader](workspace)

    assert CANARY not in surfaced, f"{reader} read through a planted link"
    assert "leak.py" not in surfaced and "linked_dir" not in surfaced
