
import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

from enum import Enum
class AlloyResult(Enum):
    SAT = 1
    UNSAT = 2
    TIMEOUT = 3

class AlloyAnalyzerExecutor():

    GLOBAL_TIMEOUT = 40  # seconds

    def __init__(self):
        self.java_path = None
        self.alloy_jar_path = None
        self._resolve_java_path()
        self._resolve_alloy_jar_path()

    """Generates up to num_instances satisfying instances for spec_als by invoking 
    the Alloy Analyzer. Yields instances in XML format, in files named 
    output_dir/instance_model-solution-0.xml, 
    output_dir/instance_model-solution-1.xml, and so on. 
    Returns a pair (result, instance_files) where result is SAT, UNSAT, or TIMEOUT and 
    instance_files is a list of the generated instance files (if any).
    Raises RuntimeError exceptions if the execution of the Alloy Analyzer fails."""
    def generate_instances(self, spec_als: str, output_dir: str, 
                            num_instances: int = 1, timeout = GLOBAL_TIMEOUT):
        self.output_dir = output_dir
        # Clean up any previous instance files before generating new ones 
        for file in Path(output_dir).glob("instance_model-solution-*.xml"):
            file.unlink()
        # Execute the Alloy Analyzer via subprocess
        try:
            output_type = "xml"
            result = subprocess.run(
                [
                    self.java_path, "-jar", self.alloy_jar_path, "exec", "-n", "-f",
                    "-o", output_dir, "-t", output_type, "-r", str(num_instances), spec_als,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return AlloyResult.TIMEOUT, []

        if self._check_satisfiability_in_alloy_receipt_json(result):
            return AlloyResult.SAT, self._get_instance_xml_files()
        else:
            return AlloyResult.UNSAT, []


    def _check_satisfiability_in_alloy_receipt_json(self, result):
        """Parse the ``receipt.json`` produced by the Alloy Analyzer in output_dir.
        Returns whether the model is satisfiable and the list of solutions (instances) found.
        """
        receipt_path = os.path.join(self.output_dir, "receipt.json")
        if not os.path.exists(receipt_path):
            output = (result.stdout or "") + (result.stderr or "")
            logger.warning("Alloy exec produced no receipt.json. Output: %s", output[:500])
            raise RuntimeError(
                "Alloy execution produced no receipt.json. "
                "Check Alloy output for errors: " + output.strip()[:1000]
            )

        with open(receipt_path, "r", encoding="utf-8") as f:
            receipt = json.load(f)
        commands = receipt.get("commands", {})
        if not commands:
            raise RuntimeError("No commands found in receipt.json. Alloy execution may have failed.")
        first_command_name = next(iter(commands))
        first_command = commands[first_command_name]
        solutions = first_command.get("solution", [])
        return any(sol.get("instances") for sol in solutions)


    def _resolve_alloy_jar_path(self):
        """Locates the ``alloy.jar`` file used to run the Alloy Analyzer, 
        using environment variable``BESSER_ALLOY_JAR`` environment variable.
        """
        env_path = os.getenv("BESSER_ALLOY_JAR")
        if env_path:
            candidate = Path(env_path).expanduser().resolve()
            if candidate.exists() and candidate.is_file():
                self.alloy_jar_path = str(candidate)

        if not self.alloy_jar_path:
            logger.warning("BESSER_ALLOY_JAR points to a missing file: %s", env_path)
            raise RuntimeError("Alloy JAR not found.")


    def _resolve_java_path(self):
        """Locates the ``java`` executable strictly via JAVA_HOME.
        """
        java_home = os.getenv("JAVA_HOME")
        if not java_home:
            logger.error("JAVA_HOME is not set. Cannot run the Alloy Analyzer.")
            return None
        java_file = "java.exe" if os.name == "nt" else "java"
        candidate = Path(java_home).expanduser() / "bin" / java_file
        if candidate.is_file() and os.access(candidate, os.X_OK):
            self.java_path = str(candidate.resolve())

        if not self.java_path:
            logger.error("JAVA_HOME points to a directory without a java executable: %s", java_home)
            raise RuntimeError("JAVA_HOME is not set or does not point to a valid java executable.")


    """Returns a list of XML filenames produced by the Alloy Analyzer, which
    are assumed to be called output_dir/instance_model-solution-0.xml, 
    output_dir/instance_model-solution-1.xml, ..."""
    def _get_instance_xml_files(self):
        output_dir_path = Path(self.output_dir)
        if not output_dir_path.is_dir():
            raise RuntimeError(f"Output directory does not exist: {self.output_dir}")
        xml_files = sorted(output_dir_path.glob("instance_model-solution-*.xml"))
        return [str(f.resolve()) for f in xml_files]
