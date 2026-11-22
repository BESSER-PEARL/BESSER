"""The workspace revision must move on an edit to any authored file.

It used to hash only an extension allow-list, so a Phase 3 repair that edited
an Android manifest, a Gradle Kotlin script, a XAML window, a ``.csproj``, a
``.cc`` file or a QML screen read as "no progress" and kept a stale judge
verdict (probe_revision.py, 2026-10-30: 6 of 7 kinds unchanged).
"""

from __future__ import annotations

import pytest

from besser.BUML.metamodel.structural import Class, DomainModel
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.providers.llm_client import UsageTracker


class _Client:
    model = "mock-model"
    max_tokens = 4096

    def __init__(self) -> None:
        self.usage = UsageTracker("mock-model")


def _orchestrator(tmp_path) -> LLMOrchestrator:
    return LLMOrchestrator(
        llm_client=_Client(),
        domain_model=DomainModel(name="M", types={Class(name="Item")}),
        output_dir=str(tmp_path),
    )


_SOURCES = {
    "app/src/main/AndroidManifest.xml": "<manifest/>",
    "app/build.gradle.kts": "plugins {}",
    "app/build.gradle": "plugins {}",
    "MainWindow.xaml": "<Window/>",
    "Inventory.csproj": "<Project/>",
    "native/engine.cc": "int main(){}",
    "lib/screen.qml": "Item {}",
    "src/main.py": "x = 1",
}


@pytest.mark.parametrize("rel", sorted(_SOURCES))
def test_revision_changes_for_every_source_kind(tmp_path, rel):
    for path, text in _SOURCES.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8")
    orch = _orchestrator(tmp_path)
    before = orch._workspace_revision()
    (tmp_path / rel).write_text(_SOURCES[rel] + " // edited by repair", encoding="utf-8")
    assert orch._workspace_revision() != before


def test_runtime_outputs_do_not_change_the_revision(tmp_path):
    (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")
    orch = _orchestrator(tmp_path)
    before = orch._workspace_revision()
    (tmp_path / "app.db").write_bytes(b"sqlite")
    (tmp_path / "server.log").write_text("booted", encoding="utf-8")
    (tmp_path / ".pytest_cache" / "v").mkdir(parents=True)
    (tmp_path / ".pytest_cache" / "v" / "lastfailed").write_text("{}", encoding="utf-8")
    assert orch._workspace_revision() == before


def test_large_file_edit_past_the_cap_still_counts_by_size(tmp_path, monkeypatch):
    import besser.spec_driven_agent.pipeline.orchestrator as orchestrator_module
    monkeypatch.setattr(orchestrator_module, "_REVISION_HASH_CAP", 8)
    (tmp_path / "asset.bin").write_bytes(b"0123456789")
    orch = _orchestrator(tmp_path)
    before = orch._workspace_revision()
    (tmp_path / "asset.bin").write_bytes(b"0123456789-appended")
    assert orch._workspace_revision() != before


# Review of #705: hashing every file made the app's own runtime writes read as
# source edits (development hashed none of these by content).
@pytest.mark.parametrize("rel", [
    ".coverage", "app.db-journal", "app.db-wal", "app.db-shm", "data/items.csv",
    ".dart_tool/package_config.json", "obj/Debug/net8.0/App.assets.cache",
    "uploads/photo.png", "htmlcov/index.html", "coverage/lcov.info",
])
def test_a_runtime_rewrite_is_not_an_edit(tmp_path, rel):
    (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_bytes(b"first")
    orch = _orchestrator(tmp_path)
    before = orch._workspace_revision()
    (tmp_path / rel).write_bytes(b"second, longer")
    assert orch._workspace_revision() == before


def test_a_new_data_file_still_counts_by_its_path(tmp_path):
    (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")
    orch = _orchestrator(tmp_path)
    before = orch._workspace_revision()
    (tmp_path / "seed.csv").write_text("id\n1\n", encoding="utf-8")
    assert orch._workspace_revision() != before
