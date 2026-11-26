"""Known-incompatible dependency pairs are pinned at every stage.

Production run 438889bc: Phase 2's dependency install installed bcrypt 5.0.0
beside passlib 1.7.4. passlib's startup self-test hashes a >72-byte secret,
bcrypt 5 rejects it, and every password hash raised "ValueError: password
cannot be longer than 72 bytes". The harness pinned bcrypt==4.0.1 only inside
Phase 3, which the cost cap skipped, so the delivered requirements.txt stayed
unpinned and the app could not register users.
"""
from besser.spec_driven_agent.agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.repair.dependency_pins import pin_known_incompatible
from besser.spec_driven_agent.repair.scaffold_repair import _ensure_requirements_txt

RUN_438889BC = "fastapi>=0.103.0\nuvicorn>=0.15.0\npasslib[bcrypt]>=1.7.4\npython-jose[cryptography]>=3.3.0\n"


def _bcrypt_lines(content):
    return [line for line in content.splitlines() if line.lower().startswith("bcrypt")]


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def test_passlib_without_a_bcrypt_pin_gets_one():
    content, notes = pin_known_incompatible(RUN_438889BC)
    assert _bcrypt_lines(content) == ["bcrypt==4.0.1"]
    assert notes and "passlib" in notes[0]


def test_an_incompatible_bcrypt_range_is_rewritten():
    content, _ = pin_known_incompatible("passlib==1.7.4\nbcrypt>=4.0.0\n")
    assert _bcrypt_lines(content) == ["bcrypt==4.0.1"]
    content, _ = pin_known_incompatible("passlib==1.7.4\nbcrypt==5.0.0\n")
    assert _bcrypt_lines(content) == ["bcrypt==4.0.1"]


def test_an_explicit_compatible_pin_is_respected():
    for pin in ("bcrypt==4.0.1", "bcrypt<4.1", "bcrypt~=4.0.0", "bcrypt==3.2.2"):
        content = f"passlib[bcrypt]==1.7.4\n{pin}\n"
        assert pin_known_incompatible(content) == (content, []), pin


def test_unrelated_requirements_are_untouched():
    content = "fastapi==0.110.0\nbcrypt==5.0.0\n# passlib is not used\n-r base.txt\n"
    assert pin_known_incompatible(content) == (content, [])


def test_other_lines_keep_their_text_when_a_pin_is_added():
    content, _ = pin_known_incompatible(RUN_438889BC)
    assert content.splitlines()[:4] == RUN_438889BC.splitlines()


# --------------------------------------------------------------------------- #
# Stage (a): scaffold repair
# --------------------------------------------------------------------------- #
def test_restored_scaffold_requirements_carry_the_pin(tmp_path):
    (tmp_path / "auth.py").write_text("from passlib.context import CryptContext\n")
    assert _ensure_requirements_txt(str(tmp_path)) is True
    content = (tmp_path / "requirements.txt").read_text()
    assert "passlib" in content
    assert _bcrypt_lines(content) == ["bcrypt==4.0.1"]


# --------------------------------------------------------------------------- #
# Stage (b): a write tool writing a requirements.txt
# --------------------------------------------------------------------------- #
def _write(tmp_path, requirements):
    ex = ToolExecutor(workspace=str(tmp_path))
    result = ex.execute_typed("write_file", {"path": "backend/requirements.txt", "content": requirements})
    return result.payload, (tmp_path / "backend" / "requirements.txt").read_text(encoding="utf-8")


def test_a_written_requirements_file_is_pinned_and_says_so(tmp_path):
    result, written = _write(tmp_path, RUN_438889BC)
    assert result["status"] == "written", result
    assert _bcrypt_lines(written) == ["bcrypt==4.0.1"]
    assert any("bcrypt==4.0.1" in note for note in result["pinned"])
    assert "install" in result["pinned_note"]


def test_an_edit_that_adds_the_trigger_is_pinned(tmp_path):
    ex = ToolExecutor(workspace=str(tmp_path))
    (tmp_path / "requirements.txt").write_text("fastapi>=0.103.0\n", encoding="utf-8")
    result = ex.execute_typed("modify_file", {"path": "requirements.txt", "old_text": "fastapi>=0.103.0",
                                              "new_text": "fastapi>=0.103.0\npasslib[bcrypt]>=1.7.4"}).payload
    assert result["status"] == "modified", result
    assert _bcrypt_lines((tmp_path / "requirements.txt").read_text()) == ["bcrypt==4.0.1"]


def test_a_write_with_a_compatible_pin_is_left_as_written(tmp_path):
    requirements = "passlib[bcrypt]==1.7.4\nbcrypt<4.1\n"
    result, written = _write(tmp_path, requirements)
    assert written == requirements
    assert "pinned" not in result


def test_a_write_of_unrelated_requirements_is_left_alone(tmp_path):
    requirements = "fastapi==0.110.0\nbcrypt==5.0.0\n"
    result, written = _write(tmp_path, requirements)
    assert written == requirements
    assert "pinned" not in result


# --------------------------------------------------------------------------- #
# Stage (c): the Phase 3 check uses the same table
# --------------------------------------------------------------------------- #
def _phase3(model, tmp_path, monkeypatch, requirements):
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "requirements.txt").write_text(requirements, encoding="utf-8")

    class _Client:
        model = "mock-model"

        def chat(self, **kwargs):
            raise AssertionError("no LLM call expected")

    from besser.spec_driven_agent.providers.llm_client import UsageTracker
    _Client.usage = UsageTracker("mock-model")
    orch = LLMOrchestrator(
        llm_client=_Client(), domain_model=model, output_dir=str(tmp_path),
        allow_shell_tools=False, enable_toolchain_validation=False,
        enable_checkpointing=False,
    )
    for name in ("_collect_frontend_contract_issues", "_collect_ruff_issues",
                 "_collect_execution_issues", "_collect_tsc_issues",
                 "_collect_requirement_issues", "_collect_task_issues",
                 "_collect_data_contract_issues", "_collect_missing_frontend_issue",
                 "_collect_framework_switch_issues"):
        monkeypatch.setattr(orch, name, lambda: [])
    orch._collect_validation_issues()
    return (backend / "requirements.txt").read_text(encoding="utf-8")


def test_phase3_pins_passlib_bcrypt(simple_library_book_model, tmp_path, monkeypatch):
    content = _phase3(simple_library_book_model, tmp_path, monkeypatch, RUN_438889BC)
    assert _bcrypt_lines(content) == ["bcrypt==4.0.1"]


def test_phase3_respects_a_compatible_pin(simple_library_book_model, tmp_path, monkeypatch):
    requirements = "passlib[bcrypt]==1.7.4\nbcrypt<4.1\n"
    assert _phase3(simple_library_book_model, tmp_path, monkeypatch, requirements) == requirements


def test_phase3_leaves_a_passlib_mention_in_a_comment_alone(
        simple_library_book_model, tmp_path, monkeypatch):
    requirements = "fastapi==0.110.0\n# passlib: not needed yet\n"
    assert _phase3(simple_library_book_model, tmp_path, monkeypatch, requirements) == requirements
