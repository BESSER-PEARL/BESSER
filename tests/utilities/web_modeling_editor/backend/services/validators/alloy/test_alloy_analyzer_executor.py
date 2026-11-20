import pytest

from besser.utilities.web_modeling_editor.backend.services.exceptions import ConfigurationError
from besser.utilities.web_modeling_editor.backend.services.validators.alloy import alloy_analyzer_executor
from besser.utilities.web_modeling_editor.backend.services.validators.alloy.alloy_analyzer_executor import (
    AlloyAnalyzerExecutor,
)
from tests.utilities.web_modeling_editor.backend.services.validators.alloy.utils_alloy_solver import needs_alloy


def test_missing_java_is_a_configuration_error(monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    with pytest.raises(ConfigurationError, match="JAVA_HOME is not set"):
        AlloyAnalyzerExecutor()


# Java is resolved first: the missing jar is reached only with a valid JAVA_HOME
@needs_alloy
def test_missing_alloy_jar_is_a_configuration_error(monkeypatch):
    monkeypatch.delenv("BESSER_ALLOY_JAR")
    with pytest.raises(ConfigurationError, match="BESSER_ALLOY_JAR is not set"):
        AlloyAnalyzerExecutor()


def test_the_alloy_analyzer_runs_with_the_java4shell_mitigation_flags(monkeypatch, tmp_path):
    """CVE-2026-114938: the JVM that runs the Alloy Analyzer has the Base64 intrinsic off."""
    started = []

    class _NoProcess:
        returncode = 1

        def __init__(self, command, **kwargs):
            started.append(command)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def communicate(self, timeout=None):
            return "", ""

    monkeypatch.setattr(alloy_analyzer_executor.subprocess, "Popen", _NoProcess)
    executor = AlloyAnalyzerExecutor.__new__(AlloyAnalyzerExecutor)
    executor.java_path, executor.alloy_jar_path = "java", "alloy.jar"
    executor._process, executor._stopped = None, False

    executor.generate_instances("spec.als", str(tmp_path))

    assert "-XX:-UseBASE64Intrinsics" in started[0]
    assert started[0].index("-XX:-UseBASE64Intrinsics") < started[0].index("-jar")
