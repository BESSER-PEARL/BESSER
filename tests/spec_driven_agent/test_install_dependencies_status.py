"""install_dependencies must fail when one of its installs fails.

It returned ``{"installs": [...]}`` and the status classifier only reads the
top level, so a failed ``npm install`` (exit 1: esbuild's postinstall could not
find its binary) was recorded as "ok" in three scorecard runs. The model then
built against a missing ``vite`` and spent repair turns on a "frontend build
failed" blocker whose cause it had been told succeeded.
"""

from besser.spec_driven_agent.agent.tool_executor import ToolExecutor

_NPM_FAILED = {
    "success": False,
    "exit_code": 1,
    "stdout": "",
    "stderr": "npm error code 1\nnpm error command C:\\WINDOWS\\system32\\cmd.exe /d /s /c node install.js",
}
_OK = {"success": True, "exit_code": 0, "stdout": "added 212 packages", "stderr": ""}


def _executor(tmp_path, results):
    (tmp_path / "package.json").write_text('{"name": "app"}', encoding="utf-8")
    ex = ToolExecutor(workspace=str(tmp_path), allow_shell=True)
    calls = iter(results)
    ex._run_command = lambda args: next(calls)
    return ex


def test_a_failed_npm_install_is_an_error(tmp_path):
    ex = _executor(tmp_path, [_NPM_FAILED])
    assert ex.execute_typed("install_dependencies", {}).status == "error"


def test_the_error_names_the_failed_install_and_its_output(tmp_path):
    ex = _executor(tmp_path, [_NPM_FAILED])
    result = ex._install_dependencies({})
    assert result["success"] is False
    assert "npm" in result["error"] and "exit code 1" in result["error"]
    assert "node install.js" in result["error"]
    # The per-install detail is still there for the model to read.
    assert result["installs"][0]["result"]["exit_code"] == 1


def test_a_successful_install_stays_ok(tmp_path):
    ex = _executor(tmp_path, [_OK])
    assert ex.execute_typed("install_dependencies", {}).status == "ok"
