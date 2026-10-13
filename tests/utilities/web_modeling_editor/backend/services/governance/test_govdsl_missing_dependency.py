"""Without the optional governancedsl package, parsing reports a ConfigurationError."""

import sys

import pytest

from besser.utilities.web_modeling_editor.backend.services.exceptions import ConfigurationError
from besser.utilities.web_modeling_editor.backend.services.governance.govdsl_runtime import (
    summarize_governance,
)


def test_missing_governancedsl_raises_configuration_error(monkeypatch):
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "governancedsl.grammar.govdslLexer", None)
    with pytest.raises(ConfigurationError, match=r"governancedsl==0\.1\.1"):
        summarize_governance("MajorityPolicy p { }")
