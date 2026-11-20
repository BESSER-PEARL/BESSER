"""Personalization of Markdown replies and option buttons.

Markdown formatting must survive personalization: the structure never goes to the
LLM, and a rewrite that loses inline markup is undone. The LLM is faked with a
rule that follows the numbered-list protocol the batch functions use.
"""
import re
from unittest.mock import patch

import pytest

from besser.BUML.metamodel.state_machine.agent import (
    Agent,
    AgentReply,
    WebSocketReplyMarkdown,
    WebSocketReplyOptions,
)
from besser.BUML.metamodel.state_machine.state_machine import Body
from besser.generators.agents.agent_personalization import configure_agent
from besser.generators.agents.markdown_segments import join_markdown, same_inline_markup, split_markdown

SAMPLES = [
    "plain text",
    "**Question 1**\n\nIn which year was Luxembourg founded?\n\n- A. 1914\n- B. 1815\n",
    "# Title\n> quoted *text* here\n1. first item\n2) second item\n- [x] done task\n",
    "Intro\n```python\nprint('keep me')\n```\nOutro with `code` and [a link](https://x.org)",
    "| a | b |\n|---|---|\n| 1 | 2 |\n\n---\nafter",
    "line with trailing break  \nnext line",
    "",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_split_join_round_trip_is_identity(text):
    parts, units = split_markdown(text)
    assert join_markdown(parts, units) == text


def test_structure_never_becomes_a_unit():
    _, units = split_markdown(SAMPLES[1])
    assert units == ["**Question 1**", "In which year was Luxembourg founded?", "A. 1914", "B. 1815"]
    _, units = split_markdown(SAMPLES[3])
    assert units == ["Intro", "Outro with `code` and [a link](https://x.org)"]
    _, units = split_markdown(SAMPLES[4])
    assert units == ["after"]


@pytest.mark.parametrize("before, after, expected", [
    ("**Bold** text", "**Fett** Text", True),
    ("**Bold** text", "Fett Text", False),
    ("see [docs](https://a.b)", "siehe [Doku](https://a.b)", True),
    ("see [docs](https://a.b)", "siehe [Doku](https://c.d)", False),
    ("use `foo()` here", "nutze `foo()` hier", True),
    ("use `foo()` here", "nutze `bar()` hier", False),
    ("hello {name}", "hallo {name}", True),
    ("hello {name}", "hallo {nom}", False),
    ("a snake_case word", "ein snake_case Wort", True),
    ("text", "two\nlines", False),
    ("text", "  ", False),
])
def test_same_inline_markup(before, after, expected):
    assert same_inline_markup(before, after) is expected


def _fake_llm(transform):
    """Fake ``call_openai_chat``: apply ``transform`` to each numbered line."""
    def fake(system_prompt, user_prompt, **kwargs):
        lines = [line for line in user_prompt.splitlines() if re.match(r"\d+\. ", line)]
        return "\n".join(f"{i}. {transform(line.split('. ', 1)[1])}" for i, line in enumerate(lines, 1))
    return fake


def _agent(actions, intents=()):
    agent = Agent("quiz")
    ask = agent.new_state(name="ask", initial=True)
    ask.set_body(Body("ask_body", actions=actions))
    for name, sentences in intents:
        agent.new_intent(name, list(sentences))
    return agent


def _bodies(agent):
    return agent.states[0].body.actions


CONFIG = {"agentLanguage": "German", "openaiApiKey": "k"}


@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_markdown_reply_keeps_its_formatting(mock_chat):
    mock_chat.side_effect = _fake_llm(lambda t: t.upper())
    original = "**Question 1**\n\nIn which year?\n\n- A. one\n- B. two\n"
    agent = _agent([WebSocketReplyMarkdown(message=original)])

    configure_agent(agent, CONFIG)

    assert _bodies(agent)[0].message == "**QUESTION 1**\n\nIN WHICH YEAR?\n\n- A. ONE\n- B. TWO\n"


@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_rewrite_that_drops_markup_is_undone(mock_chat):
    mock_chat.side_effect = _fake_llm(lambda t: t.replace("**", "").replace("see", "SEE"))
    agent = _agent([WebSocketReplyMarkdown(message="**Question 1**\n\n- see [docs](https://a.b)")])

    configure_agent(agent, CONFIG)

    # the bold line lost its markers in the rewrite, so it keeps its original text;
    # the link line kept its markup and is personalized
    assert _bodies(agent)[0].message == "**Question 1**\n\n- SEE [docs](https://a.b)"


@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_code_blocks_and_tables_are_not_sent_to_the_llm(mock_chat):
    sent = []

    def fake(system_prompt, user_prompt, **kwargs):
        sent.append(user_prompt)
        return _fake_llm(lambda t: t.upper())(system_prompt, user_prompt)

    mock_chat.side_effect = fake
    original = "Intro\n```\nprint('x')\n```\n| a | b |\n|---|---|"
    agent = _agent([WebSocketReplyMarkdown(message=original)])

    configure_agent(agent, CONFIG)

    assert _bodies(agent)[0].message == "INTRO\n```\nprint('x')\n```\n| a | b |\n|---|---|"
    assert not any("print" in prompt or "|---|" in prompt for prompt in sent)


@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_plain_replies_still_personalized_next_to_markdown(mock_chat):
    mock_chat.side_effect = _fake_llm(lambda t: t.upper())
    agent = _agent([AgentReply("hello"), WebSocketReplyMarkdown(message="**hi**")])

    configure_agent(agent, CONFIG)

    assert [a.message for a in _bodies(agent)] == ["HELLO", "**HI**"]


@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_option_labels_are_translated_and_intents_follow(mock_chat):
    mock_chat.side_effect = _fake_llm(lambda t: t + "_de")
    agent = _agent(
        [WebSocketReplyOptions(options=["Yes", "No"])],
        intents=[("YesIntent", ["Yes", "yes please"]), ("NoIntent", ["no"])],
    )

    configure_agent(agent, CONFIG)

    assert _bodies(agent)[0].options == ["Yes_de", "No_de"]
    sentences = {i.name: i.training_sentences for i in agent.intents}
    # a sentence equal to a label gets exactly the label's translation (case-insensitive)
    assert sentences["YesIntent"][0] == "Yes_de"
    assert sentences["NoIntent"] == ["No_de"]
    assert sentences["YesIntent"][1] == "yes please_de"


@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_options_are_only_translated_not_restyled(mock_chat):
    mock_chat.side_effect = _fake_llm(lambda t: "REWRITTEN")
    agent = _agent([WebSocketReplyOptions(options=["Yes", "No"])])

    configure_agent(agent, {"agentStyle": "formal", "openaiApiKey": "k"})

    assert _bodies(agent)[0].options == ["Yes", "No"]
    mock_chat.assert_not_called()


@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_options_untouched_without_a_language(mock_chat):
    agent = _agent([WebSocketReplyOptions(options=["A. 1914", "B. 1815"])])

    configure_agent(agent, {"agentStyle": "original"})

    assert _bodies(agent)[0].options == ["A. 1914", "B. 1815"]
    mock_chat.assert_not_called()


@patch("besser.generators.agents.agent_personalization.call_openai_chat")
def test_language_pass_is_one_llm_call_for_intents_and_options(mock_chat):
    mock_chat.side_effect = _fake_llm(lambda t: t + "_de")
    agent = _agent([WebSocketReplyOptions(options=["Yes", "No"])], intents=[("YesIntent", ["Yes", "ja"])])

    configure_agent(agent, CONFIG)

    # one call for training sentences + option labels (there is no plain/Markdown text)
    assert mock_chat.call_count == 1
    assert _bodies(agent)[0].options == ["Yes_de", "No_de"]
