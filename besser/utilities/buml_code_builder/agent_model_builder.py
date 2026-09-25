"""
Agent Model Builder

This module generates Python code for BUML agent models.
"""

import os
import tempfile
import textwrap
from re import search
from typing import Optional

from besser.BUML.metamodel.state_machine.agent import (
    Agent, AgentReply, LLMReply, LLMChatReply, RAGReply, DBReply,
    WebCrawlLLMReply, ReasoningState, llm_provider_key,
    WebSocketReplyMarkdown, WebSocketReplyHTML, WebSocketReplySpeech,
    WebSocketReplyOptions, WebSocketReplyLocation,
    WebSocketReplyFile, WebSocketReplyImage, WebSocketReplyDataframe, WebSocketReplyPlotly,
    GUIReplyAction, GUIEvent, ReceiveMessageEvent,
)
from besser.BUML.metamodel.state_machine.state_machine import Action, Body, CustomCodeAction, Event
from besser.utilities.buml_code_builder.common import _comment_safe, _escape_python_string, safe_var_name
from besser.utilities.buml_code_builder.gui_model_builder import gui_model_to_code

# Prefix of the module-level functions that build the agent GUIs in the generated code.
# ``agent_diagram_converter`` recognises them through the ``add_gui_model`` call.
GUI_BUILDER_FUNCTION_PREFIX = "_agent_gui_"

# Header ``gui_model_to_code`` writes; dropped when a GUI is embedded in an agent module
# so a project export does not see a GUI MODEL section inside the AGENT MODEL section.
_GUI_SECTION_HEADER = ("###############", "#  GUI MODEL  #", "###############")

_PARAMETERLESS_WS_REPLIES = (
    WebSocketReplyFile, WebSocketReplyImage, WebSocketReplyDataframe, WebSocketReplyPlotly,
)


def _input_prompt_args(action) -> list[str]:
    """Constructor kwargs of the ``input_prompt_mode`` / ``custom_input_prompt*`` fields."""
    args = []
    if action.input_prompt_mode != 'last_user_message':
        args.append(f"input_prompt_mode={action.input_prompt_mode!r}")
    if action.custom_input_prompt:
        args.append(f"custom_input_prompt={action.custom_input_prompt!r}")
    if action.custom_input_prompt_use_session_vars:
        args.append("custom_input_prompt_use_session_vars=True")
    return args


def _output_args(action) -> list[str]:
    """Constructor kwargs of the ``store_in_session`` / ``send_reply`` fields."""
    args = []
    if action.store_in_session:
        args.append(f"store_in_session={action.store_in_session!r}")
    if not action.send_reply:
        args.append("send_reply=False")
    return args


def _llm_reply_expr(action: LLMReply) -> str:
    args = []
    if action.prompt:
        args.append(f"prompt='{_escape_python_string(action.prompt)}'")
    if action.llm_name:
        args.append(f"llm_name={action.llm_name!r}")
    args += _input_prompt_args(action)
    if action.system_prompt_use_session_vars:
        args.append("system_prompt_use_session_vars=True")
    args += _output_args(action)
    return f"LLMReply({', '.join(args)})"


def _llm_chat_reply_expr(action: LLMChatReply) -> str:
    args = []
    if action.prompt:
        args.append(f"prompt='{_escape_python_string(action.prompt)}'")
    if action.llm_name:
        args.append(f"llm_name={action.llm_name!r}")
    if action.system_prompt_use_session_vars:
        args.append("system_prompt_use_session_vars=True")
    args += _output_args(action)
    return f"LLMChatReply({', '.join(args)})"


def _rag_reply_expr(action: RAGReply) -> str:
    args = [f"'{_escape_python_string(action.rag_db_name or '')}'"]
    if action.prompt:
        args.append(f"prompt='{_escape_python_string(action.prompt)}'")
    args += _input_prompt_args(action)
    if action.prompt_use_session_vars:
        args.append("prompt_use_session_vars=True")
    args += _output_args(action)
    return f"RAGReply({', '.join(args)})"


def _db_reply_expr(action: DBReply) -> str:
    args = []
    if action.db_selection_type != 'default':
        args.append(f"db_selection_type={action.db_selection_type!r}")
    if action.db_custom_name:
        args.append(f"db_custom_name={action.db_custom_name!r}")
    if action.db_query_mode != 'llm_query':
        args.append(f"db_query_mode={action.db_query_mode!r}")
    if action.db_operation != 'any':
        args.append(f"db_operation={action.db_operation!r}")
    if action.db_query_mode == 'sql' and action.db_sql_query:
        args.append(f"db_sql_query={action.db_sql_query!r}")
    if action.llm_name:
        args.append(f"llm_name={action.llm_name!r}")
    args += _input_prompt_args(action)
    args += _output_args(action)
    return f"DBReply({', '.join(args)})"


def _web_crawl_reply_expr(action: WebCrawlLLMReply) -> str:
    args = [f"initial_url={action.initial_url!r}"]
    if action.max_depth != 2:
        args.append(f"max_depth={action.max_depth!r}")
    if action.max_pages != 20:
        args.append(f"max_pages={action.max_pages!r}")
    if action.crawl_format != 'markdown':
        args.append(f"crawl_format={action.crawl_format!r}")
    if action.base_url_prefix:
        args.append(f"base_url_prefix={action.base_url_prefix!r}")
    if not action.run_crawl:
        args.append("run_crawl=False")
    if action.no_crawl_error_message != 'No web crawl data is available yet.':
        args.append(f"no_crawl_error_message={action.no_crawl_error_message!r}")
    if action.system_message_prefix:
        args.append(f"system_message_prefix={action.system_message_prefix!r}")
    if action.llm_name:
        args.append(f"llm_name={action.llm_name!r}")
    if action.system_message_prefix_use_session_vars:
        args.append("system_message_prefix_use_session_vars=True")
    args += _output_args(action)
    return f"WebCrawlLLMReply({', '.join(args)})"


def _ws_text_reply_expr(action) -> str:
    args = [f"message={action.message!r}"]
    if isinstance(action, WebSocketReplySpeech) and action.audio_speed is not None:
        args.append(f"audio_speed={action.audio_speed!r}")
    if action.use_session_vars:
        args.append("use_session_vars=True")
    return f"{action.__class__.__name__}({', '.join(args)})"


def _gui_reply_expr(action: GUIReplyAction) -> str:
    args = [repr(action.gui_id)]
    if not action.persist:
        args.append("persist=False")
    if action.width:
        args.append(f"width={action.width!r}")
    if action.is_form:
        args.append("is_form=True")
    return f"GUIReplyAction({', '.join(args)})"


def _agent_reply_expr(action: AgentReply) -> str:
    args = [f"'{_escape_python_string(action.message)}'"]
    if action.use_session_vars:
        args.append("use_session_vars=True")
    return f"AgentReply({', '.join(args)})"


def _action_expr(action: Action) -> Optional[str]:
    """Return the constructor expression of a predefined body action (None if unsupported)."""
    if isinstance(action, LLMReply):
        return _llm_reply_expr(action)
    if isinstance(action, LLMChatReply):
        return _llm_chat_reply_expr(action)
    if isinstance(action, RAGReply):
        return _rag_reply_expr(action)
    if isinstance(action, DBReply):
        return _db_reply_expr(action)
    if isinstance(action, WebCrawlLLMReply):
        return _web_crawl_reply_expr(action)
    if isinstance(action, (WebSocketReplyMarkdown, WebSocketReplyHTML, WebSocketReplySpeech)):
        return _ws_text_reply_expr(action)
    if isinstance(action, WebSocketReplyOptions):
        return f"WebSocketReplyOptions(options={action.options!r})"
    if isinstance(action, WebSocketReplyLocation):
        return f"WebSocketReplyLocation(latitude={action.latitude!r}, longitude={action.longitude!r})"
    if isinstance(action, _PARAMETERLESS_WS_REPLIES):
        return f"{action.__class__.__name__}()"
    if isinstance(action, GUIReplyAction):
        return _gui_reply_expr(action)
    if isinstance(action, AgentReply):
        return _agent_reply_expr(action)
    return None


def _write_body(f, state_var: str, state_name: str, body: Body, fallback: bool) -> None:
    """Write a state's body (or fallback body) and attach it to the state."""
    suffix = "_fallback" if fallback else ""
    body_var = f"{state_var}{suffix}_body"
    body_name = f"{_escape_python_string(state_name)}{suffix}_body"
    custom_action = next((a for a in body.actions if isinstance(a, CustomCodeAction)), None)
    if custom_action is not None:
        # CustomCodeAction is always a singleton in a custom body; its function is
        # emitted before the Body is created.
        f.write(f"{custom_action.to_code()}\n")
        function_match = search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', custom_action.code)
        function_name = (
            function_match.group(1) if function_match
            else f"custom_action_{safe_var_name(custom_action.name)}"
        )
        action_var = f"CustomCodeAction_{state_var}{suffix}"
        f.write(f"{action_var} = CustomCodeAction(callable={function_name})\n")
        f.write(f"{body_var} = Body('{body_name}')\n")
        f.write(f"{body_var}.add_action({action_var})\n")
    else:
        f.write(f"{body_var} = Body('{body_name}')\n")
        for action in body.actions:
            expr = _action_expr(action)
            if expr is not None:
                f.write(f"{body_var}.add_action({expr})\n")
    f.write("\n")
    setter = "set_fallback_body" if fallback else "set_body"
    f.write(f"{state_var}.{setter}({body_var})\n")


def _event_expr(event: Event) -> str:
    """Return the constructor expression of a transition event, keeping its arguments."""
    if isinstance(event, GUIEvent):
        return f"GUIEvent(message_id={event.message_id!r})" if event.message_id else "GUIEvent()"
    if type(event) is ReceiveMessageEvent:
        return f"ReceiveMessageEvent({event.message!r})"
    return f"{event.__class__.__name__}()"


def _write_gui_models(f, model: Agent, model_var_name: str) -> None:
    """Write every agent GUI as a builder function and register its result on the agent.

    The GUI code comes from ``gui_model_to_code``. It is wrapped in a function so the
    variables it declares (screens, components, ``gui_model``) cannot clash with the
    agent's variables or, in a project export, with the other model sections.
    """
    if not model.gui_models:
        return
    f.write("# GUIS\n")
    for index, (gui_id, gui_model) in enumerate(model.gui_models.items()):
        function_name = f"{GUI_BUILDER_FUNCTION_PREFIX}{index}_{safe_var_name(gui_id)}"
        with tempfile.TemporaryDirectory(prefix="besser_agent_gui_") as temp_dir:
            gui_path = os.path.join(temp_dir, "gui_model.py")
            gui_model_to_code(gui_model, gui_path, model_var_name="gui_model")
            with open(gui_path, encoding="utf-8") as gui_file:
                gui_lines = gui_file.read().splitlines()
        gui_code = "\n".join(line for line in gui_lines if line not in _GUI_SECTION_HEADER).strip()
        f.write(f"def {function_name}():\n")
        f.write(textwrap.indent(gui_code, "    "))
        f.write("\n    return gui_model\n\n\n")
        f.write(f"{model_var_name}.add_gui_model({gui_id!r}, {function_name}())\n\n")
    f.write("\n")


def agent_model_to_code(model: Agent, file_path: str, model_var_name: str = "agent"):
    """
    Generates Python code for a B-UML Agent model and writes it to a specified file.

    Parameters:
    model (Agent): The B-UML Agent model object containing states, intents, and transitions.
    file_path (str): The path where the generated code will be saved.
    model_var_name (str, optional): Name of the Agent variable in the generated code. Defaults to "agent".

    Outputs:
    - A Python file containing the code representation of the B-UML agent model.
    """
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not file_path.endswith('.py'):
        file_path += '.py'

    # Build mappings from original names to safe Python variable names
    state_var_names = {state.name: safe_var_name(state.name) for state in model.states}
    intent_var_names = {intent.name: safe_var_name(intent.name) for intent in model.intents}

    with open(file_path, 'w', encoding='utf-8') as f:
        # Write imports
        f.write("###############\n")
        f.write("# AGENT MODEL #\n")
        f.write("###############\n")
        f.write("import datetime\n")
        f.write(
            "from besser.BUML.metamodel.state_machine.state_machine import "
            "Body, Condition, ConfigProperty, CustomCodeAction\n"
        )
        f.write(
            "from besser.BUML.metamodel.state_machine.agent import "
            "Agent, AgentReply, LLMReply, LLMChatReply, RAGReply, DBReply, "
            "WebCrawlLLMReply, GUIReplyAction, "
            "WebSocketReplyMarkdown, WebSocketReplyHTML, WebSocketReplySpeech, "
            "WebSocketReplyOptions, WebSocketReplyLocation, "
            "WebSocketReplyFile, WebSocketReplyImage, WebSocketReplyDataframe, WebSocketReplyPlotly, "
            "LLMOpenAI, LLMHuggingFace, LLMHuggingFaceAPI, LLMReplicate, "
            "LLMMistral, LLMDeepSeek, LLMGoogle, LLMMeta, LLMAnthropic, "
            "LLMQwen, LLMxAI, LLMGroq, LLMTogether, LLMOpenRouter, "
            "RAGVectorStore, RAGTextSplitter, "
            "Tool, Skill, Workspace, ReasoningState, "
            "ReceiveTextEvent, ReceiveFileEvent, ReceiveJSONEvent, "
            "ReceiveMessageEvent, WildcardEvent, DummyEvent, GUIEvent\n"
        )
        f.write("from besser.BUML.metamodel.structural import Metadata\n")
        f.write("import operator\n\n")

        # Create agent with metadata if it exists
        agent_name = _escape_python_string(model.name)
        if model.metadata and model.metadata.description:
            description = _escape_python_string(model.metadata.description)
            f.write(
                f"{model_var_name} = Agent('{agent_name}', "
                f"metadata=Metadata(description=\"{description}\"))\n\n"
            )
        else:
            f.write(f"{model_var_name} = Agent('{agent_name}')\n\n")

        # Write configuration properties
        for prop in model.properties:
            f.write(
                f"{model_var_name}.add_property(ConfigProperty('{_escape_python_string(prop.section)}', "
                f"'{_escape_python_string(prop.name)}', {repr(prop.value)}))\n"
            )
        f.write("\n")

        # Write intents
        f.write("# INTENTS\n")
        for intent in model.intents:
            intent_var = intent_var_names[intent.name]
            f.write(f"{intent_var} = {model_var_name}.new_intent('{_escape_python_string(intent.name)}', [\n")
            for sentence in intent.training_sentences:
                f.write(f"    '{_escape_python_string(sentence)}',\n")
            f.write("],\n")
            if intent.description:
                f.write(f"description=\"{_escape_python_string(intent.description)}\"")
            f.write(")\n")
        f.write("\n")

        # Write tools (reasoning extension)
        if model.tools:
            f.write("# TOOLS\n")
            for tool in model.tools:
                if tool.code:
                    # Emit the source verbatim so the function is defined in
                    # the generated module's namespace; then register it on
                    # the agent. The `code` field carries a complete `def`.
                    f.write(f"{tool.code}\n")
                f.write(
                    f"{model_var_name}.new_tool("
                    f"name={repr(tool.name)}, "
                    f"description={repr(tool.description)}, "
                    f"code={repr(tool.code)})\n"
                )
            f.write("\n")

        # Write skills (reasoning extension)
        if model.skills:
            f.write("# SKILLS\n")
            for skill in model.skills:
                f.write(
                    f"{model_var_name}.new_skill("
                    f"name={repr(skill.name)}, "
                    f"content={repr(skill.content)}, "
                    f"description={repr(skill.description)})\n"
                )
            f.write("\n")

        # Write workspaces (reasoning extension)
        if model.workspaces:
            f.write("# WORKSPACES\n")
            for ws in model.workspaces:
                f.write(
                    f"{model_var_name}.new_workspace("
                    f"name={repr(ws.name)}, "
                    f"path={repr(ws.path)}, "
                    f"description={repr(ws.description)}, "
                    f"writable={ws.writable}, "
                    f"max_read_bytes={ws.max_read_bytes})\n"
                )
            f.write("\n")

        if model.rags:
            f.write("# RAG CONFIGURATIONS\n")
            for index, rag in enumerate(model.rags):
                vector_store = rag.vector_store
                splitter = rag.splitter
                if not vector_store or not splitter:
                    continue
                base_name = f"rag_{index}"
                vector_var = f"{base_name}_vector_store"
                splitter_var = f"{base_name}_splitter"
                rag_var = f"{base_name}_rag"

                f.write(f"{vector_var} = RAGVectorStore(\n")
                f.write(f"    embedding_provider={repr(vector_store.embedding_provider)},\n")
                f.write(f"    embedding_parameters={repr(vector_store.embedding_parameters or {})},\n")
                f.write(f"    persist_directory={repr(vector_store.persist_directory)},\n")
                f.write(")\n")

                f.write(f"{splitter_var} = RAGTextSplitter(\n")
                f.write(f"    splitter_type={repr(splitter.splitter_type)},\n")
                f.write(f"    chunk_size={splitter.chunk_size},\n")
                f.write(f"    chunk_overlap={splitter.chunk_overlap},\n")
                f.write(")\n")

                f.write(f"{rag_var} = {model_var_name}.new_rag(\n")
                f.write(f"    name={repr(rag.name)},\n")
                f.write(f"    vector_store={vector_var},\n")
                f.write(f"    splitter={splitter_var},\n")
                f.write(f"    llm_name={repr(rag.llm_name)},\n")
                f.write(f"    llm_prompt={repr(rag.llm_prompt)},\n")
                f.write(f"    k={rag.k},\n")
                f.write(f"    num_previous_messages={rag.num_previous_messages},\n")
                f.write(f"    use_hybrid_rag={rag.use_hybrid_rag},\n")
                f.write(f"    bm25_weight={rag.bm25_weight},\n")
                f.write(")\n\n")

        # Emit explicit LLM definitions via Agent.new_llm — every consumer
        # (reasoning states, replies, RAG, IC config) references its LLM by
        # name, so the registered list must round-trip exactly.
        llms = model.llms
        if llms:
            f.write("# LLMs\n")
            for llm in llms:
                llm_var = safe_var_name(llm.name)
                if llm_var == "llm":
                    llm_var = f"{llm_var}_{safe_var_name(llm.__class__.__name__)}"
                params = llm.parameters or {}
                provider = llm_provider_key(llm)
                f.write(
                    f"{llm_var} = {model_var_name}.new_llm(\n"
                    f"    name={repr(llm.name)},\n"
                    f"    provider={repr(provider)},\n"
                    f"    parameters={repr(dict(sorted(params.items())))},\n"
                )
                # Only some LLM wrappers keep a conversation window.
                num_prev = getattr(llm, 'num_previous_messages', None)
                if num_prev is not None and num_prev != 1:
                    f.write(f"    num_previous_messages={num_prev},\n")
                if llm.global_context:
                    f.write(f"    global_context={repr(llm.global_context)},\n")
                f.write(")\n")
            if model.default_llm_name:
                f.write(f"{model_var_name}.set_default_llm({repr(model.default_llm_name)})\n")
            f.write("\n")

        if not llms:
            f.write("default_llm = None\n\n")

        # Write the GUIs sent by GUIReplyAction
        _write_gui_models(f, model, model_var_name)

        # Write states
        f.write("# STATES\n")
        for state in model.states:
            state_var = state_var_names[state.name]
            if isinstance(state, ReasoningState):
                # Reasoning states use the dedicated factory; their body is
                # supplied automatically by ``new_reasoning_state`` and the
                # metamodel rejects manual ``set_body`` / ``set_fallback_body``.
                llm_ref = "None"
                if state.llm:
                    llm_ref = repr(state.llm)
                f.write(f"{state_var} = {model_var_name}.new_reasoning_state(\n")
                f.write(f"    name='{_escape_python_string(state.name)}',\n")
                f.write(f"    llm={llm_ref},\n")
                if state.initial:
                    f.write("    initial=True,\n")
                f.write(f"    max_steps={state.max_steps},\n")
                f.write(f"    enable_task_planning={state.enable_task_planning},\n")
                f.write(f"    stream_steps={state.stream_steps},\n")
                if state.system_prompt is not None:
                    f.write(f"    system_prompt={repr(state.system_prompt)},\n")
                if state.fallback_message is not None:
                    f.write(f"    fallback_message={repr(state.fallback_message)},\n")
                f.write(")\n")
            else:
                f.write(f"{state_var} = {model_var_name}.new_state('{_escape_python_string(state.name)}'")
                if state.initial:
                    f.write(", initial=True")
                f.write(")\n")
        f.write("\n")

        # Write state metadata if any states have it
        described_states = [state for state in model.states if state.metadata and state.metadata.description]
        for state in described_states:
            state_var = state_var_names[state.name]
            description = _escape_python_string(state.metadata.description)
            f.write(f"{state_var}.metadata = Metadata(description=\"{description}\")\n")
        if described_states:
            f.write("\n")

        # Write custom conditions so transition chains can reference them.
        predefined_condition_classes = {
            "IntentMatcher",
            "VariableOperationMatcher",
            "FileTypeMatcher",
            "FormSubmitMatcher",
            "Auto",
        }
        written_custom_conditions = set()
        has_custom_conditions = False
        for state in model.states:
            for transition in state.transitions:
                for condition in (transition.conditions or []):
                    condition_class = condition.__class__.__name__
                    if condition_class in predefined_condition_classes:
                        continue

                    condition_name = condition.name
                    if not condition_name or condition_name in written_custom_conditions:
                        continue

                    condition_code = condition.code
                    callable_name = None

                    if isinstance(condition_code, str) and condition_code.strip():
                        f.write(f"{condition_code}\n")
                        function_match = search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', condition_code)
                        callable_name = function_match.group(1) if function_match else None

                    if not callable_name:
                        callable_name = f"{condition_name}_callable"

                    f.write(f"{condition_name} = Condition('{callable_name}', callable={callable_name})\n\n")
                    written_custom_conditions.add(condition_name)
                    has_custom_conditions = True

        if has_custom_conditions:
            f.write("\n")

        # Write bodies for states
        for state in model.states:
            state_var = state_var_names[state.name]
            f.write(f"# {_comment_safe(state.name)} state\n")
            if state.body and state.body.actions:
                _write_body(f, state_var, state.name, state.body, fallback=False)
            if state.fallback_body and state.fallback_body.actions:
                _write_body(f, state_var, state.name, state.fallback_body, fallback=True)

            # Write transitions
            for transition in state.transitions:
                dest_state = transition.dest
                dest_var = state_var_names.get(dest_state.name, safe_var_name(dest_state.name))

                event = transition.event
                conditions = transition.conditions or []
                event_class = event.__class__.__name__ if event else None

                # Predefined transitions are recognized only when there is exactly one condition.
                if len(conditions) == 1:
                    condition = conditions[0]
                    condition_class = condition.__class__.__name__

                    if event_class == "ReceiveTextEvent" and condition_class == "IntentMatcher":
                        intent_name = condition.intent.name
                        intent_var = intent_var_names.get(intent_name, safe_var_name(intent_name))
                        if intent_name == "fallback_intent":
                            f.write(f"{state_var}.when_no_intent_matched().go_to({dest_var})\n")
                        else:
                            f.write(f"{state_var}.when_intent_matched({intent_var}).go_to({dest_var})\n")

                    elif event is None and condition_class == "VariableOperationMatcher":
                        var_name = condition.var_name
                        op_name = condition.operation.__name__
                        target = condition.target
                        f.write(f"{state_var}.when_variable_matches_operation(\n")
                        f.write(f"    var_name='{_escape_python_string(var_name)}',\n")
                        f.write(f"    operation=operator.{op_name},\n")
                        f.write(f"    target='{_escape_python_string(str(target))}'\n")
                        f.write(f").go_to({dest_var})\n")

                    elif event_class == "ReceiveFileEvent" and condition_class == "FileTypeMatcher":
                        file_type = condition.allowed_types
                        if file_type:
                            if isinstance(file_type, list):
                                list_literal = "[" + ", ".join(
                                    f"'{_escape_python_string(t)}'" for t in file_type
                                ) + "]"
                                f.write(f"{state_var}.when_file_received({list_literal}).go_to({dest_var})\n")
                            else:
                                file_type_literal = f"'{_escape_python_string(str(file_type))}'"
                                f.write(f"{state_var}.when_file_received({file_type_literal}).go_to({dest_var})\n")
                        else:
                            f.write(f"{state_var}.when_file_received().go_to({dest_var})\n")

                    elif event_class == "GUIEvent" and condition_class == "FormSubmitMatcher":
                        if condition.form_id:
                            f.write(
                                f"{state_var}.when_form_submitted(form_id={condition.form_id!r})"
                                f".go_to({dest_var})\n"
                            )
                        else:
                            f.write(f"{state_var}.when_form_submitted().go_to({dest_var})\n")

                    elif event is None and condition_class == "Auto":
                        f.write(f"{state_var}.go_to({dest_var})\n")

                    else:
                        # Custom transition with a single condition.
                        if event:
                            transition_chain = f"{state_var}.when_event({_event_expr(event)})"
                            transition_chain += f".with_condition({condition.name})"
                        else:
                            transition_chain = f"{state_var}.when_condition({condition.name})"
                        transition_chain += f".go_to({dest_var})"
                        f.write(f"{transition_chain}\n")

                elif len(conditions) > 1:
                    # Custom transition with multiple conditions.
                    condition_names = [c.name for c in conditions]
                    if event:
                        transition_chain = f"{state_var}.when_event({_event_expr(event)})"
                        for condition_name in condition_names:
                            transition_chain += f".with_condition({condition_name})"
                    else:
                        transition_chain = f"{state_var}.when_condition({condition_names[0]})"
                        for condition_name in condition_names[1:]:
                            transition_chain += f".with_condition({condition_name})"
                    transition_chain += f".go_to({dest_var})"
                    f.write(f"{transition_chain}\n")

                elif event:
                    # Event-only custom transition.
                    f.write(f"{state_var}.when_event({_event_expr(event)}).go_to({dest_var})\n")
                else:
                    # No event and no conditions -> simple transition.
                    f.write(f"{state_var}.go_to({dest_var})\n")

                f.write("\n")

    print(f"Agent model saved to {file_path}")
