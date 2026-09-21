"""Same-turn parser and undefined-name feedback for file writes."""

import json

import pytest

from besser.spec_driven_agent.tool_executor import ToolExecutor
from besser.spec_driven_agent.write_diagnostics import diagnose_written_content


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


def test_generated_backend_contract_failures_are_reported_same_turn(tmp_path):
    """The 772298cb failures must reach the editor before a final sweep."""
    executor = ToolExecutor(workspace=str(tmp_path))
    _call(executor, "write_file", {
        "path": "backend/database.py", "content": "DATABASE_URL = 'sqlite:///test.db'\n",
    })
    broken = (
        "from enum import Enum\n"
        "from sqlalchemy import CheckConstraint, Integer\n"
        "from sqlalchemy.orm import mapped_column\n"
        "class Status(Enum):\n    READY = 'READY'\n"
        "class Booking:\n"
        "    id = mapped_column(Integer, primary_key=True)\n"
        "    status = mapped_column(default=Status.ready)\n"
        "    __table_args__ = (CheckConstraint('agreedPrice > 0'),)\n"
        "    __table_args__ = (CheckConstraint('NOT EXISTS (SELECT 1 FROM room)'),)\n"
        "    def validate_capacity(self):\n        return self\n"
        "    def validate_capacity(self):\n        return self\n"
    )
    result = _call(executor, "write_file", {"path": "backend/sql_alchemy.py", "content": broken})
    assert result["status"] == "written"
    findings = result["diagnostics"]
    assert {"invalid-enum-member", "duplicate-declaration", "shadowed-declaration",
            "invalid-sqlite-check"} <= {item["code"] for item in findings}
    assert any("agreedPrice" in item["message"] for item in findings)
    assert any("subqueries prohibited" in item["message"] for item in findings)
    assert any("validate_capacity" in item["message"] for item in findings)
    assert all(item.get("line", 0) > 0 for item in findings)
    assert (tmp_path / "backend" / "sql_alchemy.py").read_text() == broken

    # Replacing the declarations with valid ones clears the immediate errors.
    repaired = broken[:broken.index("    status =")] + (
        "    status = mapped_column(default=Status.READY)\n"
        "    __table_args__ = (CheckConstraint('id > 0'),)\n"
        "    def validate_capacity(self):\n        return self\n"
    )
    result = _call(executor, "modify_file", {
        "path": "backend/sql_alchemy.py", "old_text": broken, "new_text": repaired,
    })
    assert result["status"] == "modified"
    assert not result.get("diagnostics"), result

    schemas = (
        "from pydantic import BaseModel\n"
        "class PersonCreate(BaseModel):\n"
        "    phone: str\n    booking: list[int] = []\n"
        "class EmployeeCreate(PersonCreate):\n    pass\n"
    )
    _call(executor, "write_file", {"path": "backend/pydantic_classes.py", "content": schemas})
    for entity in ("person", "employee"):
        result = _call(executor, "write_file", {
            "path": f"backend/routers/{entity}.py",
            "content": f"def create_{entity}({entity}_data):\n    return {entity}_data.booking\n",
        })
        assert not result.get("diagnostics"), result
    result = _call(executor, "modify_file", {
        "path": "backend/pydantic_classes.py",
        "old_text": "    booking: list[int] = []\n", "new_text": "",
    })
    assert result["status"] == "modified"
    mismatches = [item for item in result["diagnostics"] if item["code"] == "schema-consumer-mismatch"]
    assert len(mismatches) == 2
    assert all("does not define `booking`" in item["message"] for item in mismatches)
    assert any("routers/person.py" in item["message"] for item in mismatches)
    assert any("routers/employee.py" in item["message"] for item in mismatches)

    # The supported verification tool must not report success when unwired,
    # and must surface the orchestrator's actual execution report when wired.
    unavailable = _call(executor, "validate_app", {})
    assert "unavailable" in unavailable["error"]
    report = {"blocker_count": 2, "issues": [item["message"] for item in mismatches]}
    executor.app_validator = lambda: report
    assert _call(executor, "validate_app", {}) == report
