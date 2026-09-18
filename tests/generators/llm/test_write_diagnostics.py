"""Same-turn parser and undefined-name feedback for file writes."""

import json

import pytest

from besser.generators.llm.tool_executor import ToolExecutor
from besser.generators.llm.write_diagnostics import diagnose_written_content


def _call(executor: ToolExecutor, tool: str, arguments: dict) -> dict:
    return json.loads(executor.execute(tool, arguments))


def test_python_syntax_error_is_reported_without_rejecting_write(tmp_path):
    executor = ToolExecutor(workspace=str(tmp_path))
    result = _call(executor, "write_file", {
        "path": "app.py",
        "content": "def broken(:\n    pass\n",
    })
    assert result["status"] == "written"
    assert result["diagnostics"][0]["source"] == "python"
    assert result["diagnostics"][0]["code"] == "syntax"
    assert (tmp_path / "app.py").is_file()


def test_pilot_undefined_annotation_is_reported_same_turn(tmp_path):
    pytest.importorskip("pyflakes")
    executor = ToolExecutor(workspace=str(tmp_path))
    result = _call(executor, "write_file", {
        "path": "services/forecast.py",
        "content": (
            "def forecast():\n"
            "    method: linear_trend = 'linear_trend'\n"
            "    return method\n"
        ),
    })
    finding = next(item for item in result["diagnostics"] if item["source"] == "pyflakes")
    assert finding["code"] == "UndefinedName"
    assert "linear_trend" in finding["message"]


@pytest.mark.parametrize(
    ("path", "content", "source"),
    [
        ("package.json", '{"name": }', "json"),
        ("config.yaml", "items: [one, two", "yaml"),
        ("pyproject.toml", 'name = "unterminated', "toml"),
    ],
)
def test_structured_data_parse_errors_are_reported(tmp_path, path, content, source):
    executor = ToolExecutor(workspace=str(tmp_path))
    result = _call(executor, "write_file", {"path": path, "content": content})
    assert result["diagnostics"][0]["source"] == source


def test_modify_file_diagnoses_the_resulting_whole_file(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text('{"enabled": true}', encoding="utf-8")
    executor = ToolExecutor(workspace=str(tmp_path))
    result = _call(executor, "modify_file", {
        "path": "settings.json",
        "old_text": "true",
        "new_text": "oops",
    })
    assert result["status"] == "modified"
    assert result["diagnostics"][0]["source"] == "json"


def test_diagnostics_can_be_disabled_without_affecting_write(tmp_path):
    executor = ToolExecutor(
        workspace=str(tmp_path),
        per_write_diagnostics=False,
    )
    result = _call(executor, "write_file", {
        "path": "broken.json",
        "content": "{",
    })
    assert result["status"] == "written"
    assert "diagnostics" not in result


def test_findings_are_bounded():
    pytest.importorskip("pyflakes")
    source = "\n".join(f"value_{i} = missing_{i}" for i in range(30))
    findings = diagnose_written_content("many.py", source)
    assert len(findings) == 10
