"""Tests for gui_personalization_utils.personalize_gui_page.

The OpenAI call is mocked, so these run offline and exercise the validation,
model-selection, response-parsing and error-mapping logic in isolation.
"""
from unittest.mock import patch
import asyncio

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
    def test_summary_only_skips_general_ai_and_preserves_full_text(self):
        from besser.utilities.web_modeling_editor.backend.routers.generation_router import personalize_gui_page as endpoint
        long_text = " ".join(["Keep your appointment details available before you travel."] * 20)
        page = {"components": [{"type": "text", "content": long_text}], "css": []}
        with patch("besser.generators.gui.gui_personalization.call_openai_chat", return_value='{"summaries": [{"heading": "Summary", "text": "Keep your details ready."}]}'), patch(
            f"{MODULE}.call_openai_chat"
        ) as general, patch(f"{MODULE}.generate_user_profile_document") as profile:
            result = asyncio.run(endpoint({
                "guiPage": page, "userProfileModel": _PROFILE, "adaptFromProfile": False,
                "guiPersonalization": {"content": {"summarizeLongContent": True}},
            }))
        assert result["guiPage"]["components"][1]["content"] == long_text
        assert "Keep your details ready." in result["guiPage"]["components"][0]["content"]
        general.assert_not_called()
        profile.assert_not_called()

    def test_summaries_receive_final_personalized_text(self):
        adapted = {"components": [{"type": "text", "content": "Texte adapté"}], "css": []}
        with patch(f"{MODULE}.personalize_gui_text", return_value=adapted), patch(
            f"{MODULE}.summarize_gui_content", return_value=adapted
        ) as summary:
            personalize_gui_page(_PAGE, _PROFILE, adapt_from_profile=False,
                gui_personalization={"presentation": {"language": "french"}, "content": {"summarizeLongContent": True}})
        assert summary.call_args.args[0] == adapted

    def test_router_content_only_uses_full_profile_and_skips_general_llm(self):
        from besser.utilities.web_modeling_editor.backend.routers.generation_router import personalize_gui_page as endpoint
        profile_document = {"mobility": "limited"}
        with patch(f"{MODULE}.generate_user_profile_document", return_value=profile_document) as profile, patch(
            "besser.generators.gui.gui_personalization.call_openai_chat", return_value='{"texts": ["Hi Alice"]}'
        ) as content, patch(f"{MODULE}.call_openai_chat") as general:
            result = asyncio.run(endpoint({
                "guiPage": _PAGE, "userProfileModel": _PROFILE, "adaptFromProfile": False,
                "guiPersonalization": {"content": {"adaptContentToUserProfile": True}},
            }))
        profile.assert_called_once_with(_PROFILE)
        assert '"mobility": "limited"' in content.call_args.args[1]
        assert result["guiPage"]["components"][0]["content"] == "Hi Alice"
        general.assert_not_called()

    def test_router_applies_sentence_length_only_without_general_adaptation(self):
        from besser.utilities.web_modeling_editor.backend.routers.generation_router import personalize_gui_page as endpoint
        with patch("besser.generators.gui.gui_personalization.sentence_length_batch", return_value=["Hi"]) as length, patch(
            "besser.generators.gui.gui_personalization.complexity_text_batch"
        ) as simplify, patch(f"{MODULE}.call_openai_chat") as general:
            result = asyncio.run(endpoint({
                "guiPage": _PAGE, "userProfileModel": _PROFILE, "adaptFromProfile": False,
                "guiPersonalization": {"presentation": {"sentenceLength": "concise"}},
            }))
        assert result["guiPage"]["components"][0]["content"] == "Hi"
        assert length.call_args.args == (["Hello"], "concise")
        simplify.assert_not_called()
        general.assert_not_called()

    def test_router_applies_language_only_without_general_adaptation(self):
        from besser.utilities.web_modeling_editor.backend.routers.generation_router import personalize_gui_page as endpoint
        with patch("besser.generators.gui.gui_personalization.translate_text_batch", return_value=["Bonjour"]) as translate, patch(
            "besser.generators.gui.gui_personalization.complexity_text_batch"
        ) as simplify, patch(f"{MODULE}.call_openai_chat") as general:
            result = asyncio.run(endpoint({
                "guiPage": _PAGE, "userProfileModel": _PROFILE, "adaptFromProfile": False,
                "guiPersonalization": {"presentation": {"language": "french"}},
            }))
        assert result["guiPage"]["components"][0]["content"] == "Bonjour"
        assert translate.call_args.args == (["Hello"], "french")
        simplify.assert_not_called()
        general.assert_not_called()

    def test_router_applies_simple_rule_when_general_adaptation_is_disabled(self):
        from besser.utilities.web_modeling_editor.backend.routers.generation_router import personalize_gui_page as endpoint
        with patch("besser.generators.gui.gui_personalization.complexity_text_batch", return_value=["Hi"]) as text:
            result = asyncio.run(endpoint({
                "guiPage": _PAGE, "userProfileModel": _PROFILE,
                "adaptFromProfile": False,
                "guiPersonalization": {"presentation": {"languageComplexity": "simple"}},
            }))
        assert result["guiPage"]["components"][0]["content"] == "Hi"
        assert text.call_args.args[1] == "simple"

    def test_simple_rule_uses_controlled_text_only_module(self):
        with patch(f"{MODULE}.generate_user_profile_document") as profile, patch(f"{MODULE}.call_openai_chat") as general_llm, patch(
            "besser.generators.gui.gui_personalization.complexity_text_batch", return_value=["Hi"]
        ) as text_llm:
            result = personalize_gui_page(
                _PAGE, _PROFILE, gui_personalization={"presentation": {"languageComplexity": "simple"}},
                adapt_from_profile=False, openai_api_key="test-key",
            )
        assert result["components"][0]["content"] == "Hi"
        assert result["css"] == _PAGE["css"]
        profile.assert_not_called()
        general_llm.assert_not_called()
        assert text_llm.call_args.args == (["Hello"], "simple")

    def test_original_text_rule_needs_no_llm(self):
        with patch(f"{MODULE}.call_openai_chat") as llm, patch(f"{MODULE}.personalize_gui_text") as text:
            result = personalize_gui_page(_PAGE, _PROFILE, adapt_from_profile=False,
                                          gui_personalization={"presentation": {"languageComplexity": "original"}})
        assert result["components"] == _PAGE["components"]
        llm.assert_not_called()
        text.assert_not_called()

    def test_combined_mode_simplifies_the_general_llm_result_last(self):
        chat, profile = _mock_llm(return_value='{"components": [{"content": "Complex adapted wording"}], "css": []}')
        with chat, profile, patch("besser.generators.gui.gui_personalization.complexity_text_batch", return_value=["Easy wording"]) as text:
            result = personalize_gui_page(_PAGE, _PROFILE, gui_personalization={"presentation": {"languageComplexity": "simple"}})
        assert text.call_args.args[0] == ["Complex adapted wording"]
        assert result["components"][0]["content"] == "Easy wording"

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
    def test_invalid_summary_flag_is_rejected_before_llm(self):
        with patch(f"{MODULE}.call_openai_chat") as general, patch(f"{MODULE}.summarize_gui_content") as summary:
            with pytest.raises(ValidationError):
                personalize_gui_page(_PAGE, _PROFILE, gui_personalization={"content": {"summarizeLongContent": "true"}})
        general.assert_not_called()
        summary.assert_not_called()

    @pytest.mark.parametrize("content", [[], {"adaptContentToUserProfile": "true"}, {"adaptContentToUserProfile": 1}])
    def test_invalid_content_flag_is_rejected_before_any_llm(self, content):
        with patch(f"{MODULE}.call_openai_chat") as general, patch(f"{MODULE}.personalize_gui_text") as text:
            with pytest.raises(ValidationError):
                personalize_gui_page(_PAGE, _PROFILE, gui_personalization={"content": content})
        general.assert_not_called()
        text.assert_not_called()

    def test_invalid_sentence_length_is_rejected_before_any_llm(self):
        with patch(f"{MODULE}.call_openai_chat") as general, patch(f"{MODULE}.personalize_gui_text") as text:
            with pytest.raises(ValidationError, match="sentence length"):
                personalize_gui_page(_PAGE, _PROFILE, gui_personalization={"presentation": {"sentenceLength": "unknown"}})
        general.assert_not_called()
        text.assert_not_called()

    def test_invalid_language_is_rejected_before_any_llm(self):
        with patch(f"{MODULE}.call_openai_chat") as general, patch(f"{MODULE}.personalize_gui_text") as text:
            with pytest.raises(ValidationError, match="language"):
                personalize_gui_page(_PAGE, _PROFILE, gui_personalization={"presentation": {"language": "unknown"}})
        general.assert_not_called()
        text.assert_not_called()

    def test_invalid_text_rule_is_rejected_before_the_llm(self):
        with patch(f"{MODULE}.call_openai_chat") as llm:
            with pytest.raises(ValidationError):
                personalize_gui_page(_PAGE, _PROFILE, gui_personalization={"presentation": {"languageComplexity": "unknown"}})
        llm.assert_not_called()

    def test_mismatched_text_batch_is_a_generation_error(self):
        with patch("besser.generators.gui.gui_personalization.complexity_text_batch", return_value=[]):
            with pytest.raises(GenerationError):
                personalize_gui_page(_PAGE, _PROFILE, adapt_from_profile=False,
                                     gui_personalization={"presentation": {"languageComplexity": "simple"}})

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
