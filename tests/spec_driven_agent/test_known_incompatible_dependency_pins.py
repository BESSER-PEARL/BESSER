"""Known-incompatible dependency pairs are pinned at every stage.

Production run 438889bc: Phase 2's install_dependencies installed bcrypt 5.0.0
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
_OK = {"success": True, "exit_code": 0, "stdout": "", "stderr": ""}


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
# Stage (b): install_dependencies, before pip runs
# --------------------------------------------------------------------------- #
def _install(tmp_path, requirements):
    (tmp_path / "requirements.txt").write_text(requirements, encoding="utf-8")
    ex = ToolExecutor(workspace=str(tmp_path), allow_shell=True)
    seen = []

    def run(args, **_):
        seen.append((tmp_path / "requirements.txt").read_text(encoding="utf-8"))
        return _OK

    ex._run_command = run
    return ex._install_dependencies({}), seen


def test_install_dependencies_pins_before_installing(tmp_path):
    result, seen = _install(tmp_path, RUN_438889BC)
    assert _bcrypt_lines(seen[0]) == ["bcrypt==4.0.1"], "pip ran against the unpinned file"
    assert any("bcrypt==4.0.1" in note for note in result["pinned"])
    assert "success" not in result or result["success"] is not False


def test_install_dependencies_pins_before_a_custom_command(tmp_path):
    (tmp_path / "requirements.txt").write_text(RUN_438889BC, encoding="utf-8")
    ex = ToolExecutor(workspace=str(tmp_path), allow_shell=True)
    ex._run_command = lambda args, **_: _OK
    result = ex._install_dependencies({"command": "pip install -r requirements.txt"})
    assert result["pinned"]
    assert "bcrypt==4.0.1" in (tmp_path / "requirements.txt").read_text()


def test_install_dependencies_respects_a_compatible_pin(tmp_path):
    requirements = "passlib[bcrypt]==1.7.4\nbcrypt<4.1\n"
    result, seen = _install(tmp_path, requirements)
    assert seen[0] == requirements
    assert "pinned" not in result


def test_install_dependencies_leaves_unrelated_requirements_alone(tmp_path):
    requirements = "fastapi==0.110.0\nbcrypt==5.0.0\n"
    result, seen = _install(tmp_path, requirements)
    assert seen[0] == requirements
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
