import copy
import json
import logging
import os
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from besser.BUML.metamodel.structural import DomainModel
from besser.generators.alloy_generator.alloy_generator import AlloyGenerator
from besser.generators.alloy_generator.translate_ocl_alloy import EnumReferenceError

logger = logging.getLogger(__name__)

TIMEOUT_CALL_ALLOY = 40


class AlloySolver:

    def __init__(self, model: DomainModel, output_dir: str | None = None, scope: int = 5):
        if output_dir is None:
            output_dir = "output"
        self.scope = scope
        self.model = copy.deepcopy(model)
        self.output_dir = output_dir
        generator = AlloyGenerator(model=self.model, output_dir=output_dir, scope=scope)
        generator.generate()
        self.file = os.path.join(output_dir, "model.als")

    @staticmethod
    def _resolve_alloy_jar_path() -> str | None:
        env_path = os.getenv("BESSER_ALLOY_JAR")
        if env_path:
            candidate = Path(env_path).expanduser().resolve()
            if candidate.exists() and candidate.is_file():
                return str(candidate)
            logger.warning("BESSER_ALLOY_JAR points to a missing file: %s", env_path)

        besser_dir = Path(__file__).resolve().parent.parent.parent.parent  # besser/
        candidate = besser_dir / "BUML" / "notations" / "ocl" / "consistency" / "alloy.jar"
        if candidate.exists() and candidate.is_file():
            return str(candidate)

        logger.warning("Alloy jar not found. Set BESSER_ALLOY_JAR or place alloy.jar in a known location.")
        return None

    @staticmethod
    def _execute_alloy_analyzer(
        als_path: str, exec_output_dir: str, output_type: str = "json"
    ) -> tuple[subprocess.CompletedProcess | None, dict[str, Any] | None]:
        jar_path = AlloySolver._resolve_alloy_jar_path()
        if not jar_path:
            return None, {
                "sat": None,
                "isValid": False,
                "message": "Could not determine satisfiability (Alloy jar not found).",
                "errors": ["Alloy JAR not found. Set BESSER_ALLOY_JAR or place alloy.jar in a known location."],
                "warnings": [],
            }
        try:
            result = subprocess.run(
                [
                    "java", "-jar", jar_path, "exec", "-n", "-f",
                    "-o", exec_output_dir, "-t", output_type, "-r", "5", als_path,
                ],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_CALL_ALLOY,
            )
        except subprocess.TimeoutExpired:
            return None, {
                "sat": None,
                "isValid": False,
                "message": (
                    f"Alloy execution timed out after {TIMEOUT_CALL_ALLOY} seconds "
                    "— model may be unsatisfiable or too complex."
                ),
                "errors": [f"Alloy execution timed out after {TIMEOUT_CALL_ALLOY} seconds."],
                "warnings": [],
            }
        return result, None

    @staticmethod
    def _parse_receipt(
        exec_output_dir: str,
        result: subprocess.CompletedProcess,
        structural_warnings: list[str] | None = None,
    ) -> tuple[tuple[Any, ...] | None, dict[str, Any] | None]:
        warnings = structural_warnings or []
        receipt_path = os.path.join(exec_output_dir, "receipt.json")

        if not os.path.exists(receipt_path):
            output = result.stdout + result.stderr
            logger.warning("Alloy exec produced no receipt.json. Output: %s", output[:500])
            return None, {
                "sat": None,
                "isValid": False,
                "message": "Could not determine satisfiability (no receipt.json produced).",
                "errors": [output[:500]],
                "warnings": warnings,
            }

        with open(receipt_path, "r", encoding="utf-8") as f:
            receipt = json.load(f)

        commands = receipt.get("commands", {})
        if not commands:
            return None, {
                "sat": None,
                "isValid": False,
                "message": "No commands were executed in the Alloy model.",
                "errors": ["The generated .als file contains no run/check commands."],
                "warnings": warnings,
            }

        first_command_name = next(iter(commands))
        first_command = commands[first_command_name]
        solutions = first_command.get("solution", [])
        sat = any(sol.get("instances") for sol in solutions)
        return (sat, first_command_name, solutions), None

    def check_consistency(
        self,
        structural_warnings: list[str] | None = None,
        output_type: str = "json",
        temp_dir: str | None = None,
    ) -> tuple[bool, tuple[Any, ...] | None, dict[str, Any] | None, str]:
        cm = tempfile.TemporaryDirectory() if temp_dir is None else nullcontext(temp_dir)
        with cm as td:
            exec_output_dir = os.path.join(td, "alloy_exec_output")
            try:
                result, error = self._execute_alloy_analyzer(
                    self.file, exec_output_dir, output_type=output_type
                )
            except EnumReferenceError as exc:
                return False, None, {
                    "sat": None,
                    "isValid": False,
                    "message": str(exc),
                    "errors": [str(exc)],
                    "warnings": structural_warnings or [],
                }, exec_output_dir
            except ValueError as exc:
                return False, None, {
                    "sat": None,
                    "isValid": False,
                    "message": str(exc),
                    "errors": [str(exc)],
                    "warnings": structural_warnings or [],
                }, exec_output_dir
            if error:
                return False, None, error, exec_output_dir
            parsed, parse_error = self._parse_receipt(
                exec_output_dir, result, structural_warnings
            )
            if parse_error:
                return False, None, parse_error, exec_output_dir
            sat = parsed[0] if parsed else False
            return sat, parsed, None, exec_output_dir

    def run_sat_validation(
        self,
        structural_warnings: list[str] | None = None,
        output_type: str = "json",
        temp_dir: str | None = None,
    ) -> tuple[tuple[Any, ...] | None, dict[str, Any] | None, str]:
        """Run Alloy satisfiability validation returning a simplified result.

        Wraps :meth:`check_consistency` to collapse the 4-tuple into a 3-tuple
        ``(parsed, error, exec_output_dir)`` for callers that only need the
        parsed result or the error dict.

        Returns:
            On success: ``(parsed, None, exec_output_dir)``.
            On failure: ``(None, error_dict, exec_output_dir)``.
        """
        is_sat, parsed, error, exec_output_dir = self.check_consistency(
            structural_warnings, output_type, temp_dir
        )
        if error:
            return None, error, exec_output_dir
        return parsed, None, exec_output_dir
