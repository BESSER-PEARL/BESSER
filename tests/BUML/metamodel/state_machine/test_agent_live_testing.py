"""Metamodel tests for the agent live-testing additions: validated input prompt modes,
``Agent.gui_models`` and the GUI reply / form submission constructs."""
import pytest

from besser.BUML.metamodel.gui import GUIModel, Module
from besser.BUML.metamodel.state_machine.agent import (
    VALID_INPUT_PROMPT_MODES,
    Agent,
    DBReply,
    FormSubmitMatcher,
    GUIEvent,
    GUIReplyAction,
    LLMReply,
    RAGReply,
    WebCrawlLLMReply,
)
from besser.BUML.metamodel.state_machine.state_machine import Body


def _gui_model() -> GUIModel:
    return GUIModel(
        name="Signup", package="", versionCode="1.0", versionName="1.0",
        modules={Module(name="SignupModule", screens=set())}, description="Signup form",
    )


# --------------------------------------------------------------------------- #
# input_prompt_mode                                                           #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("make_action", [
    lambda **kw: LLMReply(**kw),
    lambda **kw: RAGReply("docs", **kw),
    lambda **kw: DBReply(**kw),
], ids=["LLMReply", "RAGReply", "DBReply"])
class TestInputPromptMode:

    def test_default_mode(self, make_action):
        action = make_action()
        assert action.input_prompt_mode == "last_user_message"
        assert action.custom_input_prompt is None

    def test_custom_mode_with_prompt(self, make_action):
        action = make_action(input_prompt_mode="custom", custom_input_prompt="Summarise {topic}")
        assert action.input_prompt_mode == "custom"
        assert action.custom_input_prompt == "Summarise {topic}"

    def test_unknown_mode_rejected(self, make_action):
        with pytest.raises(ValueError, match="Unsupported input_prompt_mode 'bogus'"):
            make_action(input_prompt_mode="bogus")

    def test_custom_mode_requires_prompt(self, make_action):
        with pytest.raises(ValueError, match="requires a non-empty custom_input_prompt"):
            make_action(input_prompt_mode="custom")

    def test_setters_validate(self, make_action):
        action = make_action()
        with pytest.raises(ValueError):
            action.input_prompt_mode = "custom"
        with pytest.raises(ValueError):
            action.input_prompt_mode = "other"
        action.custom_input_prompt = "Use {x}"
        action.input_prompt_mode = "custom"
        with pytest.raises(ValueError):
            action.custom_input_prompt = None


def test_valid_input_prompt_modes():
    assert VALID_INPUT_PROMPT_MODES == {"last_user_message", "custom"}


# --------------------------------------------------------------------------- #
# Signatures and representations                                             #
# --------------------------------------------------------------------------- #

def test_web_crawl_keeps_positional_signature_of_development():
    """The live-testing params are appended after llm_name, keeping positional calls valid."""
    action = WebCrawlLLMReply("https://x.org", 1, 5, "html", "https://x.org/docs", False, "none", "prefix", "fast")
    assert action.system_message_prefix == "prefix"
    assert action.llm_name == "fast"
    assert action.system_message_prefix_use_session_vars is False
    assert action.store_in_session is None
    assert action.send_reply is True


def test_reprs_show_live_testing_fields():
    assert "send_reply=False" in repr(LLMReply(send_reply=False))
    assert "store_in_session='k'" in repr(RAGReply("docs", store_in_session="k"))
    assert "send_reply=False" in repr(DBReply(send_reply=False))
    assert "system_message_prefix_use_session_vars=True" in repr(
        WebCrawlLLMReply("u", system_message_prefix_use_session_vars=True)
    )
    assert repr(GUIReplyAction("signup", persist=False, width="300px", is_form=True)) == (
        "GUIReplyAction(gui_id='signup', persist=False, width='300px', is_form=True)"
    )


def test_when_form_submitted_builds_gui_event_and_matcher():
    agent = Agent("FormAgent")
    ask = agent.new_state("ask", initial=True)
    done = agent.new_state("done")
    ask.when_form_submitted(form_id="signup").go_to(done)
    transition = ask.transitions[0]
    assert isinstance(transition.event, GUIEvent)
    assert transition.event.message_id == "signup"
    assert isinstance(transition.conditions[0], FormSubmitMatcher)
    assert transition.conditions[0].form_id == "signup"


# --------------------------------------------------------------------------- #
# Agent.gui_models                                                            #
# --------------------------------------------------------------------------- #

class TestGuiModels:

    def test_defaults_to_empty_dict(self):
        assert Agent("a").gui_models == {}

    def test_add_gui_model(self):
        agent = Agent("a")
        gui = _gui_model()
        assert agent.add_gui_model("signup", gui) is gui
        assert agent.gui_models == {"signup": gui}

    def test_add_duplicate_rejected(self):
        agent = Agent("a")
        agent.add_gui_model("signup", _gui_model())
        with pytest.raises(ValueError, match="already registered"):
            agent.add_gui_model("signup", _gui_model())

    @pytest.mark.parametrize("gui_id", ["", "   ", None, 3])
    def test_invalid_key_rejected(self, gui_id):
        with pytest.raises(ValueError, match="non-empty string"):
            Agent("a").gui_models = {gui_id: _gui_model()}

    def test_raw_json_value_rejected(self):
        with pytest.raises(TypeError, match="must be a GUIModel"):
            Agent("a").gui_models = {"signup": {"pages": []}}
        with pytest.raises(TypeError, match="must be a GUIModel"):
            Agent("a").add_gui_model("signup", {"pages": []})

    def test_non_dict_rejected(self):
        with pytest.raises(TypeError, match="must be a dict"):
            Agent("a").gui_models = [_gui_model()]

    def test_validate_reports_unknown_gui_reference(self):
        agent = Agent("a")
        state = agent.new_state("ask", initial=True)
        state.set_body(Body("ask_body", actions=[GUIReplyAction("missing")]))
        state.set_fallback_body(Body("ask_fallback_body", actions=[GUIReplyAction("missing_too")]))
        result = agent.validate(raise_exception=False)
        assert not result["success"]
        assert any("body GUIReplyAction references GUI 'missing'" in e for e in result["errors"])
        assert any("fallback_body GUIReplyAction references GUI 'missing_too'" in e for e in result["errors"])
        with pytest.raises(ValueError, match="not registered"):
            agent.validate()

    def test_validate_accepts_registered_gui(self):
        agent = Agent("a")
        agent.add_gui_model("signup", _gui_model())
        state = agent.new_state("ask", initial=True)
        state.set_body(Body("ask_body", actions=[GUIReplyAction("signup")]))
        assert agent.validate(raise_exception=False)["success"]
