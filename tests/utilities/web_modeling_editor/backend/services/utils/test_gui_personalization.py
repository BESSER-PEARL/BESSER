"""Tests for gui_personalization_utils.personalize_gui_page.

The OpenAI call is mocked, so these run offline and exercise the validation,
model-selection, response-parsing and error-mapping logic in isolation.
"""
from unittest.mock import patch

import pytest

from besser.utilities.web_modeling_editor.backend.services.utils.gui_personalization_utils import (
    personalize_gui_page,
    DEFAULT_OPENAI_MODEL,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ValidationError,
    GenerationError,
)

MODULE = "besser.utilities.web_modeling_editor.backend.services.utils.gui_personalization_utils"

_PAGE = {"components": [{"type": "text", "content": "Hello"}], "css": []}
_PROFILE = {"type": "UserDiagram", "elements": {}, "relationships": {}}


def _mock_llm(return_value=None, side_effect=None):
    """Patch both the LLM call and the profile-document builder."""
    return (
        patch(f"{MODULE}.call_openai_chat", return_value=return_value, side_effect=side_effect),
        patch(f"{MODULE}.generate_user_profile_document", return_value={}),
    )


class TestPersonalizeGuiPageValidation:
    def test_gui_page_must_be_dict(self):
        with pytest.raises(ValidationError):
            personalize_gui_page("not-a-dict", _PROFILE)

    def test_user_profile_must_be_dict(self):
        with pytest.raises(ValidationError):
            personalize_gui_page(_PAGE, "not-a-dict")

    def test_components_must_be_a_list(self):
        with pytest.raises(ValidationError):
            personalize_gui_page({"components": "nope"}, _PROFILE)

    def test_validation_happens_before_any_llm_call(self):
        # A bad payload must never reach call_openai_chat.
        with patch(f"{MODULE}.call_openai_chat") as chat:
            with pytest.raises(ValidationError):
                personalize_gui_page("bad", _PROFILE)
        chat.assert_not_called()


class TestPersonalizeGuiPageBehavior:
    def test_happy_path_parses_llm_response(self):
        llm_out = (
            '{"components": [{"type": "text", "content": "Bonjour"}], '
            '"css": [{"selectorsAdd": "body", "style": {"font-size": "20px !important"}}]}'
        )
        chat, profile = _mock_llm(return_value=llm_out)
        with chat as chat_mock, profile:
            result = personalize_gui_page(_PAGE, _PROFILE, openai_api_key="sk-test")
        assert result["components"][0]["content"] == "Bonjour"
        assert result["css"][0]["selectorsAdd"] == "body"
        assert result["model"] == DEFAULT_OPENAI_MODEL
        # the shared default model is what was actually sent to OpenAI
        assert chat_mock.call_args.kwargs["model"] == DEFAULT_OPENAI_MODEL

    def test_explicit_model_overrides_default(self):
        chat, profile = _mock_llm(return_value='{"components": [], "css": []}')
        with chat as chat_mock, profile:
            result = personalize_gui_page(_PAGE, _PROFILE, model="gpt-5-mini", openai_api_key="sk-test")
        assert result["model"] == "gpt-5-mini"
        assert chat_mock.call_args.kwargs["model"] == "gpt-5-mini"

    def test_missing_css_defaults_to_empty_list(self):
        chat, profile = _mock_llm(return_value='{"components": []}')
        with chat, profile:
            result = personalize_gui_page(_PAGE, _PROFILE, openai_api_key="sk-test")
        assert result["css"] == []


class TestPersonalizeGuiPageErrors:
    def test_missing_api_key_becomes_validation_error(self):
        # call_openai_chat raises RuntimeError when no key is configured.
        chat, profile = _mock_llm(side_effect=RuntimeError("OpenAI API key not found"))
        with chat, profile:
            with pytest.raises(ValidationError):
                personalize_gui_page(_PAGE, _PROFILE)

    def test_unparseable_llm_response_becomes_generation_error(self):
        chat, profile = _mock_llm(return_value="this is not json")
        with chat, profile:
            with pytest.raises(GenerationError):
                personalize_gui_page(_PAGE, _PROFILE, openai_api_key="sk-test")

    def test_llm_response_without_components_list_raises(self):
        chat, profile = _mock_llm(return_value='{"css": []}')
        with chat, profile:
            with pytest.raises(GenerationError):
                personalize_gui_page(_PAGE, _PROFILE, openai_api_key="sk-test")
