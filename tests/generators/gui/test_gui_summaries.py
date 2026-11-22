"""Controlled GUI summary insertion, with all LLM calls mocked."""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from besser.generators.gui.gui_personalization import (
    GUI_SUMMARY_SYSTEM_PROMPT, summarize_gui_content,
)

MODULE = "besser.generators.gui.gui_personalization"
LONG_TEXT = " ".join(["Keep your appointment details available before you travel."] * 20)
RESPONSE = json.dumps({"summaries": [{"heading": "Summary", "text": "Keep your appointment details ready before travelling."}]})


def test_summary_is_inserted_above_original_with_markup_and_css_unchanged():
    source = {"type": "text", "tagName": "p", "content": f'<strong class="intro">{LONG_TEXT}</strong>',
              "attributes": {"id": "original"}, "style": {"font-size": "32px", "width": "400px"}}
    page = {"components": [source], "css": [{"selectors": ["#original"], "style": {"color": "red"}}]}
    original = deepcopy(page)
    with patch(f"{MODULE}.call_openai_chat", return_value=RESPONSE) as chat:
        result = summarize_gui_content(page)
    assert chat.call_args.args[0] == GUI_SUMMARY_SYSTEM_PROMPT
    assert json.loads(chat.call_args.args[1])["blocks"] == [LONG_TEXT]
    assert result["components"][1] == source
    summary = result["components"][0]
    assert summary["attributes"] == {"data-besser-gui-summary": "true"}
    assert summary["style"] == {"font-size": "32px"}
    assert summary["content"] == '<strong>Summary</strong><br/>Keep your appointment details ready before travelling.'
    assert result["css"] == original["css"]
    assert page == original


def test_nested_inline_fragments_are_summarized_as_a_whole_without_duplicate_parent_summary():
    page = {"components": [{"tagName": "div", "components": [{"type": "text", "tagName": "p", "components": [
        {"type": "textnode", "content": LONG_TEXT[:len(LONG_TEXT)//2]},
        {"tagName": "strong", "content": LONG_TEXT[len(LONG_TEXT)//2:]},
    ]}]}], "css": []}
    with patch(f"{MODULE}.call_openai_chat", return_value=RESPONSE) as chat:
        result = summarize_gui_content(page)
    assert len(json.loads(chat.call_args.args[1])["blocks"]) == 1
    assert len(result["components"]) == 1
    assert result["components"][0]["components"][0]["attributes"]["data-besser-gui-summary"] == "true"
    assert result["components"][0]["components"][1] == page["components"][0]["components"][0]


def test_short_text_controls_data_templates_and_code_need_no_llm():
    page = {"components": [
        {"type": "text", "content": "Short text"},
        {"type": "button", "content": LONG_TEXT},
        {"type": "table", "data": [{"text": LONG_TEXT}]},
        {"tagName": "script", "content": LONG_TEXT},
        {"type": "text", "content": LONG_TEXT + " {{user.name}}"},
        {"type": "text", "content": f'<input value="name"/>{LONG_TEXT}'},
        {"tagName": "form", "components": [{"type": "text", "content": LONG_TEXT}]},
    ], "css": []}
    with patch(f"{MODULE}.call_openai_chat") as chat:
        assert summarize_gui_content(page) == page
    chat.assert_not_called()


def test_existing_summary_is_not_duplicated():
    page = {"components": [{"type": "text", "content": LONG_TEXT}], "css": []}
    with patch(f"{MODULE}.call_openai_chat", return_value=RESPONSE):
        result = summarize_gui_content(page)
    with patch(f"{MODULE}.call_openai_chat") as chat:
        assert summarize_gui_content(result) == result
    chat.assert_not_called()


def test_multiple_summaries_keep_source_order_and_escape_llm_markup():
    page = {"components": [{"type": "text", "content": LONG_TEXT}, {"type": "text", "content": LONG_TEXT}], "css": []}
    response = json.dumps({"summaries": [{"heading": "Résumé", "text": "<b>Premier résumé.</b>"}, {"heading": "Résumé", "text": "Deuxième résumé."}]})
    with patch(f"{MODULE}.call_openai_chat", return_value=response):
        result = summarize_gui_content(page)
    assert "&lt;b&gt;Premier résumé.&lt;/b&gt;" in result["components"][0]["content"]
    assert result["components"][1] == page["components"][0]
    assert "Deuxième résumé." in result["components"][2]["content"]
    assert result["components"][3] == page["components"][1]


@pytest.mark.parametrize("response", ["not JSON", "[]", "{}", '{"summaries": []}',
    '{"summaries": [{"heading": "Summary", "text": ""}]}',
    json.dumps({"summaries": [{"heading": "Summary", "text": "word " * 81}]}),
])
def test_invalid_results_abort_without_changing_input(response):
    page = {"components": [{"type": "text", "content": LONG_TEXT}], "css": []}
    original = deepcopy(page)
    with patch(f"{MODULE}.call_openai_chat", return_value=response):
        with pytest.raises(ValueError):
            summarize_gui_content(page)
    assert page == original
