import json
import logging
import os
import subprocess
from enum import Enum
from pathlib import Path

from besser.utilities.web_modeling_editor.backend.services.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Seconds the Alloy Analyzer may run before an analysis times out.
_DEFAULT_TIMEOUT_SECONDS = 30


class AlloyResult(Enum):
    SAT = 1
    UNSAT = 2
    TIMEOUT = 3
    # The bounds are too large for the Alloy Analyzer ("Translation capacity exceeded")
    SCOPES_TOO_LARGE = 4
    # The Alloy Analyzer failed for any other reason
    ERROR = 5


class AlloyAnalyzerExecutor:

    def __init__(self):
        self.java_path = None
        self.alloy_jar_path = None
        self._resolve_java_path()
        self._resolve_alloy_jar_path()

    def generate_instances(
        self, spec_als: str, output_dir: str, num_instances: int = 1, timeout: int = _DEFAULT_TIMEOUT_SECONDS
    ) -> tuple[AlloyResult, list[str]]:
        """Generates up to num_instances satisfying instances for spec_als by invoking
        the Alloy Analyzer. Yields instances in XML format, in files named
        output_dir/instance_model-solution-0.xml,
        output_dir/instance_model-solution-1.xml, and so on.
        Returns a pair (result, instance_files) where result is SAT, UNSAT, TIMEOUT,
        SCOPES_TOO_LARGE or ERROR and instance_files is a list of the generated instance files (if any).
        The Alloy Analyzer is stopped, and the result is TIMEOUT, after timeout seconds.
        The result is SCOPES_TOO_LARGE when the bounds of spec_als are too large for the Alloy
        Analyzer, and ERROR when it fails for any other reason."""
        # Clean up any previous instance files and receipt before generating new ones
        for file in Path(output_dir).glob("instance_model-solution-*.xml"):
            file.unlink()
        Path(output_dir, "receipt.json").unlink(missing_ok=True)
        # Execute the Alloy Analyzer via subprocess
        try:
            result = subprocess.run(
                [
                    self.java_path, "-jar", self.alloy_jar_path, "exec", "-n", "-f",
                    "-o", output_dir, "-t", "xml", "-r", str(num_instances), spec_als,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return AlloyResult.TIMEOUT, []

        analysis_result = self._result_in_alloy_receipt_json(result, output_dir)
        if analysis_result == AlloyResult.SAT:
            return AlloyResult.SAT, self._get_instance_xml_files(output_dir)
        else:
            return analysis_result, []

    def _result_in_alloy_receipt_json(self, result: subprocess.CompletedProcess, output_dir: str) -> AlloyResult:
        """Parse the ``receipt.json`` produced by the Alloy Analyzer in output_dir.
        Returns SAT or UNSAT, or, when the Alloy Analyzer produced no receipt, SCOPES_TOO_LARGE
        or ERROR.
        """
        receipt_path = os.path.join(output_dir, "receipt.json")
        if not os.path.exists(receipt_path):
            output = (result.stdout or "") + (result.stderr or "")
            if "Translation capacity exceeded" in output:
                return AlloyResult.SCOPES_TOO_LARGE
            logger.warning("The Alloy Analyzer wrote no receipt (exit code %s): %s", result.returncode, output.strip())
            return AlloyResult.ERROR

        with open(receipt_path, "r", encoding="utf-8") as f:
            receipt = json.load(f)
        commands = receipt.get("commands", {})
        if not commands:
            return AlloyResult.ERROR
        first_command_name = next(iter(commands))
        first_command = commands[first_command_name]
        solutions = first_command.get("solution", [])
        if any(sol.get("instances") for sol in solutions):
            return AlloyResult.SAT
        return AlloyResult.UNSAT

    def _resolve_alloy_jar_path(self) -> None:
        """Locates the ``alloy.jar`` file used to run the Alloy Analyzer,
        using the ``BESSER_ALLOY_JAR`` environment variable.

        Raises:
            ConfigurationError: when ``BESSER_ALLOY_JAR`` is not set or points to a missing file.
        """
        env_path = os.getenv("BESSER_ALLOY_JAR")
        if not env_path:
            raise ConfigurationError("BESSER_ALLOY_JAR is not set.")
        candidate = Path(env_path).expanduser().resolve()
        if candidate.is_file():
            self.alloy_jar_path = str(candidate)

        if not self.alloy_jar_path:
            raise ConfigurationError(f"BESSER_ALLOY_JAR points to a missing file: {env_path}")

    def _resolve_java_path(self) -> None:
        """Locates the ``java`` executable strictly via JAVA_HOME.

        Raises:
            ConfigurationError: when ``JAVA_HOME`` is not set or has no ``java`` executable.
        """
        java_home = os.getenv("JAVA_HOME")
        if not java_home:
            raise ConfigurationError("JAVA_HOME is not set.")
        java_file = "java.exe" if os.name == "nt" else "java"
        candidate = Path(java_home).expanduser() / "bin" / java_file
        if candidate.is_file() and os.access(candidate, os.X_OK):
            self.java_path = str(candidate.resolve())

        if not self.java_path:
            raise ConfigurationError(f"JAVA_HOME points to a directory without a java executable: {java_home}")

    def _get_instance_xml_files(self, output_dir: str) -> list[str]:
        """Returns a list of XML filenames produced by the Alloy Analyzer, which
        are assumed to be called output_dir/instance_model-solution-0.xml,
        output_dir/instance_model-solution-1.xml, ..., in the order of their numbers."""
        xml_files = sorted(
            Path(output_dir).glob("instance_model-solution-*.xml"),
            key=lambda f: int(f.stem.rsplit("-", 1)[1]),
        )
        return [str(f.resolve()) for f in xml_files]
