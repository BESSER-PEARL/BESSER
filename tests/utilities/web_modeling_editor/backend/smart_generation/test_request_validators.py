"""Tests for SmartGenerateRequest Pydantic validators."""

import pytest
from pydantic import ValidationError

from besser.utilities.web_modeling_editor.backend.constants.constants import (
    LLM_MAX_COST_USD_HARD_CAP,
    LLM_MAX_RUNTIME_SECONDS_HARD_CAP,
)
from besser.utilities.web_modeling_editor.backend.models.smart_generation import (
    SmartGenerateRequest,
)
from tests.utilities.web_modeling_editor.backend.smart_generation.test_runner import (
    _build_request,
)


class TestRequestValidators:
    def test_empty_api_key_rejected(self):
        with pytest.raises(ValidationError, match="api_key"):
            _build_request(api_key="")

    def test_whitespace_api_key_rejected(self):
        with pytest.raises(ValidationError, match="api_key"):
            _build_request(api_key="   ")

    def test_whitespace_instructions_rejected(self):
        with pytest.raises(ValidationError, match="instructions"):
            _build_request(instructions="   \n\t  ")

    def test_nan_max_cost_rejected(self):
        # Pydantic's `gt=0` constraint rejects NaN before our custom
        # validator runs (NaN compares as neither > 0 nor ≤ 0, which
        # Pydantic treats as a `greater_than` failure).
        with pytest.raises(ValidationError):
            _build_request(max_cost_usd=float("nan"))

    def test_inf_max_cost_rejected(self):
        # Positive infinity passes `gt=0` but should be caught by our
        # custom validator as "not finite". If a future pydantic update
        # rejects inf earlier, that's also acceptable.
        with pytest.raises(ValidationError):
            _build_request(max_cost_usd=float("inf"))

    def test_negative_max_cost_rejected(self):
        with pytest.raises(ValidationError):
            _build_request(max_cost_usd=-1.0)

    def test_max_cost_clamped_to_hard_cap(self):
        """Values above the hard cap are silently clamped."""
        req = _build_request(max_cost_usd=9999.0)
        assert req.max_cost_usd == LLM_MAX_COST_USD_HARD_CAP

    def test_max_runtime_clamped_to_hard_cap(self):
        req = _build_request(max_runtime_seconds=999_999)
        assert req.max_runtime_seconds == LLM_MAX_RUNTIME_SECONDS_HARD_CAP

    def test_valid_llm_model_format_accepted(self):
        req = _build_request(llm_model="claude-sonnet-4-6")
        assert req.llm_model == "claude-sonnet-4-6"

    def test_llm_model_with_slashes_accepted(self):
        """Some Anthropic model IDs use vendor/model format."""
        req = _build_request(llm_model="anthropic/claude-3-5-sonnet")
        assert req.llm_model == "anthropic/claude-3-5-sonnet"

    def test_llm_model_with_injection_attempt_rejected(self):
        """A model name with whitespace / quotes / control chars is rejected."""
        with pytest.raises(ValidationError, match="llm_model"):
            _build_request(llm_model="claude-sonnet\"; rm -rf /")

    def test_llm_model_with_newline_rejected(self):
        with pytest.raises(ValidationError, match="llm_model"):
            _build_request(llm_model="claude\nsonnet")

    def test_empty_llm_model_normalised_to_none(self):
        req = _build_request(llm_model="")
        assert req.llm_model is None

    def test_api_key_repr_is_masked(self):
        """repr(request) must not leak the API key value."""
        req = _build_request(api_key="sk-ant-MASK-CANARY-0123456789")
        assert "MASK-CANARY" not in repr(req)
        assert "**********" in repr(req.api_key)


class TestFilenameSanitization:
    """Test the router's _safe_attachment_filename helper directly."""

    def test_plain_filename_passes_through(self):
        from besser.utilities.web_modeling_editor.backend.routers.smart_generation_router import (
            _safe_attachment_filename,
        )
        assert _safe_attachment_filename("output.zip", "fallback.zip") == "output.zip"

    def test_quotes_and_crlf_stripped(self):
        from besser.utilities.web_modeling_editor.backend.routers.smart_generation_router import (
            _safe_attachment_filename,
        )
        injected = 'evil.txt"\r\nX-Injected: header'
        safe = _safe_attachment_filename(injected, "fallback.zip")
        assert '"' not in safe
        assert "\r" not in safe
        assert "\n" not in safe

    def test_empty_after_sanitization_uses_fallback(self):
        from besser.utilities.web_modeling_editor.backend.routers.smart_generation_router import (
            _safe_attachment_filename,
        )
        # Only control chars + quotes → nothing left
        result = _safe_attachment_filename('\r\n"""', "fallback.zip")
        assert result == "fallback.zip"

    def test_very_long_filename_truncated(self):
        from besser.utilities.web_modeling_editor.backend.routers.smart_generation_router import (
            _safe_attachment_filename,
        )
        huge = "a" * 500 + ".zip"
        safe = _safe_attachment_filename(huge, "fallback.zip")
        assert len(safe) <= 120

    def test_null_byte_stripped(self):
        from besser.utilities.web_modeling_editor.backend.routers.smart_generation_router import (
            _safe_attachment_filename,
        )
        safe = _safe_attachment_filename("out\x00put.zip", "fallback.zip")
        assert "\x00" not in safe
