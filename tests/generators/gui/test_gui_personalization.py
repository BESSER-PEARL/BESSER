"""Controlled GUI text adaptation, with all LLM calls mocked."""
from copy import deepcopy
from unittest.mock import patch
import json

import pytest

from besser.generators.gui.gui_personalization import GUI_SIMPLE_TEXT_SYSTEM_PROMPT, GUI_CONTENT_SYSTEM_PROMPT, personalize_gui_text
from besser.generators.agents.agent_personalization import complexity_text_batch

MODULE = "besser.generators.gui.gui_personalization"


def test_content_adaptation_reviews_components_with_page_and_profile_context():
    profile = {"disability": {"affects": "mobility"}, "name": "Alice"}
    page = {"components": [
        {"type": "text", "content": '<p class="intro">Take the stairs to the first floor.</p>',
         "attributes": {"id": "directions", "data-binding": "route.name"}, "style": {"font-size": "32px"}},
        {"type": "text", "content": "An elevator near reception also reaches the first floor."},
        {"type": "button", "content": "Contact us", "attributes": {"href": "/contact"}},
        {"type": "table", "attributes": {"columns": [{"field": "name", "label": "Name"}]}, "data": [{"name": "Raw data"}]},
    ], "css": [{"selectors": ["#directions"], "style": {"color": "red"}}]}
    original = deepcopy(page)
    response = {"texts": ["Use the elevator near reception to reach the first floor.",
                          "An elevator near reception also reaches the first floor.", "Contact us", "Name"]}
    with patch(f"{MODULE}.call_openai_chat", return_value=json.dumps(response)) as chat:
        result = personalize_gui_text(page, simplify=False, adapt_content=True, user_profile=profile)
    assert chat.call_args.args[0] == GUI_CONTENT_SYSTEM_PROMPT
    context = json.loads(chat.call_args.args[1])
    assert context["userProfile"] == profile
    assert len(context["components"]) == 4
    assert context["components"][0]["texts"][0]["index"] == 0
    assert "elevator" in context["components"][1]["texts"][0]["text"]
    assert "Never invent" in GUI_CONTENT_SYSTEM_PROMPT
    expected = deepcopy(original)
    expected["components"][0]["content"] = '<p class="intro">Use the elevator near reception to reach the first floor.</p>'
    assert result == expected
    assert page == original


def test_content_adaptation_precedes_explicit_presentation_rules():
    page = {"components": [{"content": "Take the stairs."}], "css": []}
    with patch(f"{MODULE}.adapt_gui_content_batch", return_value=["Use the elevator."]) as content, patch(
        f"{MODULE}.translate_text_batch", return_value=["Utilisez l'ascenseur."]
    ) as translate, patch(f"{MODULE}.complexity_text_batch", return_value=["Prenez l'ascenseur."]) as simplify:
        result = personalize_gui_text(page, language="french", adapt_content=True, user_profile={"mobility": "limited"})
    assert content.call_args.args[0] == ["Take the stairs."]
    assert translate.call_args.args[0] == ["Use the elevator."]
    assert simplify.call_args.args[0] == ["Utilisez l'ascenseur."]
    assert result["components"][0]["content"] == "Prenez l'ascenseur."


@pytest.mark.parametrize("response", ['not JSON', '[]', '{}', '{"texts": []}', '{"texts": [""]}', '{"texts": ["one", "extra"]}'])
def test_invalid_content_results_abort_without_modifying_input(response):
    page = {"components": [{"content": "Hello"}], "css": []}
    original = deepcopy(page)
    with patch(f"{MODULE}.call_openai_chat", return_value=response):
        with pytest.raises(ValueError):
            personalize_gui_text(page, simplify=False, adapt_content=True, user_profile={})
    assert page == original

@pytest.mark.parametrize("preference", ["concise", "verbose"])
def test_sentence_length_only_preserves_model_and_calls_shared_helper(preference):
    page = {"components": [{"content": '<b class="label">Provide your details</b>',
                           "attributes": {"id": "intro", "data-binding": "user.name"},
                           "style": {"font-size": "32px"}}], "css": []}
    original = deepcopy(page)
    with patch(f"{MODULE}.sentence_length_batch", return_value=["Your details"]) as length, patch(
        f"{MODULE}.translate_text_batch"
    ) as translate, patch(f"{MODULE}.complexity_text_batch") as simplify:
        result = personalize_gui_text(page, simplify=False, sentence_length=preference)
    assert length.call_args.args == (["Provide your details"], preference)
    translate.assert_not_called()
    simplify.assert_not_called()
    expected = deepcopy(original)
    expected["components"][0]["content"] = '<b class="label">Your details</b>'
    assert result == expected
    assert page == original


def test_sentence_length_receives_translated_and_simplified_text():
    page = {"components": [{"content": "Provide your particulars"}], "css": []}
    with patch(f"{MODULE}.translate_text_batch", return_value=["Fournissez vos renseignements"]), patch(
        f"{MODULE}.complexity_text_batch", return_value=["Donnez vos informations"]
    ), patch(f"{MODULE}.sentence_length_batch", return_value=["Vos informations"]) as length:
        result = personalize_gui_text(page, language="french", sentence_length="concise")
    assert length.call_args.args == (["Donnez vos informations"], "concise")
    assert result["components"][0]["content"] == "Vos informations"


@pytest.mark.parametrize("response", [[], [""], ["first", "extra"], [None]])
def test_invalid_sentence_length_batch_is_rejected(response):
    with patch(f"{MODULE}.sentence_length_batch", return_value=response):
        with pytest.raises(ValueError, match="sentence length"):
            personalize_gui_text({"components": [{"content": "Hello"}], "css": []}, simplify=False, sentence_length="concise")


def test_translation_only_preserves_markup_structure_styles_and_bindings():
    page = {"components": [{"content": '<strong class="label">Hello</strong>',
                           "attributes": {"id": "greeting", "data-binding": "user.name"},
                           "style": {"font-size": "32px"}}], "css": [{"style": {"color": "red"}}]}
    original = deepcopy(page)
    with patch(f"{MODULE}.translate_text_batch", return_value=["Bonjour"]) as translate, patch(f"{MODULE}.complexity_text_batch") as simplify:
        result = personalize_gui_text(page, language="french", simplify=False, openai_api_key="test-key")
    translate.assert_called_once_with(["Hello"], "french", model=translate.call_args.kwargs["model"], openai_api_key="test-key")
    simplify.assert_not_called()
    expected = deepcopy(original)
    expected["components"][0]["content"] = '<strong class="label">Bonjour</strong>'
    assert result == expected
    assert page == original


def test_translation_runs_before_simplification_in_target_language():
    page = {"components": [{"content": "Provide your particulars"}], "css": []}
    with patch(f"{MODULE}.translate_text_batch", return_value=["Fournissez vos renseignements"]) as translate, patch(
        f"{MODULE}.complexity_text_batch", return_value=["Donnez vos informations"]
    ) as simplify:
        result = personalize_gui_text(page, language="french", simplify=True)
    assert translate.call_args.args == (["Provide your particulars"], "french")
    assert simplify.call_args.args == (["Fournissez vos renseignements"], "simple")
    assert result["components"][0]["content"] == "Donnez vos informations"


def test_original_language_without_simplification_needs_no_llm():
    page = {"components": [{"content": "Hello"}], "css": []}
    with patch(f"{MODULE}.translate_text_batch") as translate, patch(f"{MODULE}.complexity_text_batch") as simplify:
        assert personalize_gui_text(page, language="original", simplify=False) == page
    translate.assert_not_called()
    simplify.assert_not_called()


@pytest.mark.parametrize("response", [[], [""], ["first", "extra"], [None]])
def test_invalid_translation_results_stop_before_simplification(response):
    with patch(f"{MODULE}.translate_text_batch", return_value=response), patch(f"{MODULE}.complexity_text_batch") as simplify:
        with pytest.raises(ValueError, match="translation"):
            personalize_gui_text({"components": [{"content": "Hello"}], "css": []}, language="french")
    simplify.assert_not_called()


def test_accessibility_prompt_reaches_llm_and_preserves_batch_newlines():
    page = {"components": [{"content": "Verify the particulars.\nThen submit your application.", "style": {"font-size": "24px"}}], "css": []}
    with patch("besser.generators.agents.agent_personalization.call_openai_chat", return_value="1. Check your details.[[NL]]Then send your form.") as chat:
        result = personalize_gui_text(page, openai_api_key="test-key")
    assert chat.call_args.args[0] == GUI_SIMPLE_TEXT_SYSTEM_PROMPT
    assert "[[NL]]" in chat.call_args.args[1]
    assert result["components"][0]["content"] == "Check your details.\nThen send your form."
    assert result["components"][0]["style"] == page["components"][0]["style"]


def test_agent_default_prompt_remains_compatible():
    with patch("besser.generators.agents.agent_personalization.call_openai_chat", return_value="1. Send your form.") as chat:
        assert complexity_text_batch(["Submit your application."], "simple") == ["Send your form."]
    assert chat.call_args.args[0] != GUI_SIMPLE_TEXT_SYSTEM_PROMPT
    assert "If a text already fits the A1-A2 complexity level" in chat.call_args.args[0]


def test_simplifies_only_visible_text_and_preserves_rich_markup_and_bindings():
    page = {
        "components": [{
            "type": "text", "tagName": "p",
            "content": '  <strong class="emphasis">Utilize this feature</strong> for assistance.  ',
            "attributes": {"id": "intro", "title": "Additional information", "data-binding": "user.name", "href": "/details"},
            "style": {"font-size": "32px"},
        }, {
            "type": "table", "attributes": {
                "columns": [{"field": "registration_date", "label": "Registration date", "type": "date"}],
                "data-source": "User",
            },
            "data": [{"registration_date": "2026-10-13"}],
        }],
        "css": [{"selectors": ["#intro"], "style": {"color": "red"}}],
    }
    original = deepcopy(page)
    replacements = {"Utilize this feature": "Use this feature", "for assistance.": "for help.", "Additional information": "More information", "Registration date": "Date you joined"}
    with patch(f"{MODULE}.complexity_text_batch", side_effect=lambda texts, *args, **kwargs: [replacements[text] for text in texts]) as llm:
        result = personalize_gui_text(page, openai_api_key="test-key")
    assert result["components"][0]["content"] == '  <strong class="emphasis">Use this feature</strong> for help.  '
    assert result["components"][0]["attributes"] == {"id": "intro", "title": "More information", "data-binding": "user.name", "href": "/details"}
    assert result["components"][0]["style"] == original["components"][0]["style"]
    assert result["components"][1]["attributes"]["columns"] == [{"field": "registration_date", "label": "Date you joined", "type": "date"}]
    assert result["components"][1]["data"] == original["components"][1]["data"]
    assert result["css"] == page["css"]
    assert page == original
    assert llm.call_args.args[1] == "simple"


def test_nested_text_nodes_and_encoded_column_labels_are_supported():
    columns = '[ { "field": "dob", "label": "Date of birth" } ]'
    page = {"components": [{"components": [{"type": "textnode", "content": "Provide your information"}], "attributes": {"columns": columns}}], "css": []}
    with patch(f"{MODULE}.complexity_text_batch", side_effect=lambda texts, *args, **kwargs: ["Your birthday" if text == "Date of birth" else "Give your details" for text in texts]):
        result = personalize_gui_text(page)
    assert result["components"][0]["components"][0]["content"] == "Give your details"
    assert json.loads(result["components"][0]["attributes"]["columns"]) == [{"field": "dob", "label": "Your birthday"}]


def test_skips_code_dynamic_templates_and_data_values_without_an_llm_call():
    page = {"components": [
        {"tagName": "script", "content": "doSomething()"},
        {"tagName": "code", "content": "some_function()"},
        {"type": "text", "content": "Hello {{user.name}}"},
        {"attributes": {"id": "technical", "value": "Do not change", "href": "/url", "data-binding": "user.name"}, "data": [{"text": "Do not change"}]},
    ], "css": []}
    with patch(f"{MODULE}.complexity_text_batch") as llm:
        assert personalize_gui_text(page) == page
    llm.assert_not_called()


@pytest.mark.parametrize("response", [[], [""], ["first", "extra"], [None]])
def test_rejects_malformed_batch_results(response):
    with patch(f"{MODULE}.complexity_text_batch", return_value=response):
        with pytest.raises(ValueError):
            personalize_gui_text({"components": [{"content": "Complex wording"}], "css": []})


def test_escapes_llm_markup_and_keeps_unchanged_encoded_columns_byte_identical():
    columns = '[ { "field": "name", "label": "Name" } ]'
    page = {"components": [{"content": "Visible text", "attributes": {"columns": columns}}], "css": []}
    with patch(f"{MODULE}.complexity_text_batch", side_effect=lambda texts, *args, **kwargs: ["<b>Simple text</b>" if text == "Visible text" else text for text in texts]):
        result = personalize_gui_text(page)
    assert result["components"][0]["content"] == "&lt;b&gt;Simple text&lt;/b&gt;"
    assert result["components"][0]["attributes"]["columns"] == columns
