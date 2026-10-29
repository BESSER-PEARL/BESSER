import pytest

from besser.utilities.web_modeling_editor.backend.services.exceptions import ConfigurationError
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
