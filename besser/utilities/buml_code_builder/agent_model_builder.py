"""
Agent Model Builder

This module generates Python code for BUML agent models.
"""

import os
from re import search
from besser.BUML.metamodel.state_machine.agent import Agent, AgentReply, LLMReply, RAGReply, DBReply
from besser.BUML.metamodel.state_machine.state_machine import CustomCodeAction
from besser.utilities.buml_code_builder.common import _escape_python_string, safe_var_name


def agent_model_to_code(model: Agent, file_path: str, model_var_name: str = "agent"):
    """
    Generates Python code for a B-UML Agent model and writes it to a specified file.

    Parameters:
    model (Agent): The B-UML Agent model object containing states, intents, and transitions.
    file_path (str): The path where the generated code will be saved.
    model_var_name (str, optional): Name of the Agent variable in the generated code.
        Defaults to "agent".

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
            "Agent, AgentReply, LLMReply, RAGReply, DBReply, "
            "LLMOpenAI, LLMHuggingFace, LLMHuggingFaceAPI, LLMReplicate, "
            "RAGVectorStore, RAGTextSplitter, "
            "ReceiveTextEvent, ReceiveFileEvent, ReceiveJSONEvent, "
            "ReceiveMessageEvent, WildcardEvent, DummyEvent\n"
        )
        f.write("from besser.BUML.metamodel.structural import Metadata\n")
        f.write("import operator\n\n")

        # Create agent with metadata if it exists
        if hasattr(model, 'metadata') and model.metadata and model.metadata.description:
            f.write(f"{model_var_name} = Agent('{_escape_python_string(model.name)}', metadata=Metadata(description=\"{_escape_python_string(model.metadata.description)}\"))\n\n")
        else:
            f.write(f"{model_var_name} = Agent('{_escape_python_string(model.name)}')\n\n")

        # Write configuration properties
        for prop in model.properties:
            f.write(f"{model_var_name}.add_property(ConfigProperty('{_escape_python_string(prop.section)}', '{_escape_python_string(prop.name)}', {repr(prop.value)}))\n")
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

        rag_configs = getattr(model, 'rags', []) or []
        if rag_configs:
            f.write("# RAG CONFIGURATIONS\n")
            for index, rag in enumerate(rag_configs):
                vector_store = getattr(rag, 'vector_store', None)
                splitter = getattr(rag, 'splitter', None)
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
                f.write(f"    k={rag.k},\n")
                f.write(f"    num_previous_messages={rag.num_previous_messages},\n")
                f.write(")\n\n")

        # Check if an LLM is necessary
        llm_required = False
        for state in model.states:
            if state.body:
                if hasattr(state.body, 'actions') and isinstance(state.body.actions[0], LLMReply):
                    llm_required = True
                    break

            if state.fallback_body:
                if hasattr(state.fallback_body, 'actions') and isinstance(state.fallback_body.actions[0], LLMReply):
                    llm_required = True
                    break
        if llm_required:
            # Create an LLM instance for use in state bodies
            f.write("# Create LLM instance for use in state bodies\n")
            f.write(f"llm = LLMOpenAI(agent={model_var_name}, name='gpt-4o-mini', parameters={{}})\n\n")

        # Write states
        f.write("# STATES\n")
        for state in model.states:
            state_var = state_var_names[state.name]
            f.write(f"{state_var} = {model_var_name}.new_state('{_escape_python_string(state.name)}'")
            if state.initial:
                f.write(", initial=True")
            f.write(")\n")
        f.write("\n")

        # Write state metadata if any states have it
        for state in model.states:
            state_var = state_var_names[state.name]
            if hasattr(state, 'metadata') and state.metadata and state.metadata.description:
                f.write(f"{state_var}.metadata = Metadata(description=\"{_escape_python_string(state.metadata.description)}\")\n")
        if any(hasattr(state, 'metadata') and state.metadata and state.metadata.description for state in model.states):
            f.write("\n")

        # Write custom conditions so transition chains can reference them.
        predefined_condition_classes = {
            "IntentMatcher",
            "VariableOperationMatcher",
            "FileTypeMatcher",
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

                    condition_name = getattr(condition, "name", None)
                    if not condition_name or condition_name in written_custom_conditions:
                        continue

                    condition_code = getattr(condition, "code", None)
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
            f.write(f"# {state.name} state\n")
            # Write body function if it exists
            if state.body:
                # Check if the body has a messages attribute
                if hasattr(state.body, 'actions') and state.body.actions:
                    if isinstance(state.body.actions[0], AgentReply):
                        f.write(f"{state_var}_body = Body('{_escape_python_string(state.name)}_body')\n")
                        for action in state.body.actions:
                            f.write(f"{state_var}_body.add_action(AgentReply('{_escape_python_string(action.message)}'))\n")
                        f.write("\n")
                        f.write(f"{state_var}.set_body({state_var}_body)\n")
                    elif isinstance(state.body.actions[0], LLMReply):
                        f.write(f"{state_var}_body = Body('{_escape_python_string(state.name)}_body')\n")
                        for action in state.body.actions:
                            prompt = getattr(action, 'prompt', None)
                            if prompt:
                                f.write(f"{state_var}_body.add_action(LLMReply(prompt='{_escape_python_string(prompt)}'))\n")
                            else:
                                f.write(f"{state_var}_body.add_action(LLMReply())\n")
                        f.write("\n")
                        f.write(f"{state_var}.set_body({state_var}_body)\n")
                    elif isinstance(state.body.actions[0], RAGReply):
                        f.write(f"{state_var}_body = Body('{_escape_python_string(state.name)}_body')\n")
                        for action in state.body.actions:
                            rag_name = _escape_python_string(action.rag_db_name or '')
                            prompt = getattr(action, 'prompt', None)
                            if prompt:
                                f.write(
                                    f"{state_var}_body.add_action(RAGReply('{rag_name}', prompt='{_escape_python_string(prompt)}'))\n"
                                )
                            else:
                                f.write(f"{state_var}_body.add_action(RAGReply('{rag_name}'))\n")
                        f.write("\n")
                        f.write(f"{state_var}.set_body({state_var}_body)\n")
                    elif isinstance(state.body.actions[0], DBReply):
                        f.write(f"{state_var}_body = Body('{_escape_python_string(state.name)}_body')\n")
                        for action in state.body.actions:
                            db_args = []
                            if getattr(action, 'db_selection_type', 'default') != 'default':
                                db_args.append(f"db_selection_type={action.db_selection_type!r}")
                            if getattr(action, 'db_custom_name', None):
                                db_args.append(f"db_custom_name={action.db_custom_name!r}")
                            if getattr(action, 'db_query_mode', 'llm_query') != 'llm_query':
                                db_args.append(f"db_query_mode={action.db_query_mode!r}")
                            if getattr(action, 'db_operation', 'any') != 'any':
                                db_args.append(f"db_operation={action.db_operation!r}")
                            if getattr(action, 'db_query_mode', 'llm_query') == 'sql' and getattr(action, 'db_sql_query', None):
                                db_args.append(f"db_sql_query={action.db_sql_query!r}")
                            args = ", ".join(db_args)
                            f.write(f"{state_var}_body.add_action(DBReply({args}))\n" if args else f"{state_var}_body.add_action(DBReply())\n")
                        f.write("\n")
                        f.write(f"{state_var}.set_body({state_var}_body)\n")
                    elif isinstance(state.body.actions[0], CustomCodeAction):
                        action = state.body.actions[0]
                        f.write(f"{action.to_code()}\n")
                        function_match = search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', action.code)
                        if function_match:
                            function_name = function_match.group(1)
                        else:
                            function_name = f"custom_action_{safe_var_name(action.name)}"
                        f.write(
                            f"CustomCodeAction_{state_var} = "
                            f"CustomCodeAction(callable={function_name})\n"
                        )
                        f.write(f"{state_var}_body = Body('{_escape_python_string(state.name)}_body')\n")
                        f.write(f"{state_var}_body.add_action(CustomCodeAction_{state_var})\n")
                        f.write(f"{state_var}.set_body({state_var}_body)\n")

            # Write fallback body function if it exists
            if state.fallback_body:
                # Check if the fallback body has a messages attribute
                if hasattr(state.fallback_body, 'actions') and state.fallback_body.actions:
                    if isinstance(state.fallback_body.actions[0], AgentReply):
                        f.write(f"{state_var}_fallback_body = Body('{_escape_python_string(state.name)}_fallback_body')\n")
                        for action in state.fallback_body.actions:
                            f.write(f"{state_var}_fallback_body.add_action(AgentReply('{_escape_python_string(action.message)}'))\n")
                        f.write("\n")
                        f.write(f"{state_var}.set_fallback_body({state_var}_fallback_body)\n")
                    elif isinstance(state.fallback_body.actions[0], LLMReply):
                        f.write(f"{state_var}_fallback_body = Body('{_escape_python_string(state.name)}_fallback_body')\n")
                        for action in state.fallback_body.actions:
                            prompt = getattr(action, 'prompt', None)
                            if prompt:
                                f.write(
                                    f"{state_var}_fallback_body.add_action(LLMReply(prompt='{_escape_python_string(prompt)}'))\n"
                                )
                            else:
                                f.write(f"{state_var}_fallback_body.add_action(LLMReply())\n")
                        f.write("\n")
                        f.write(f"{state_var}.set_fallback_body({state_var}_fallback_body)\n")
                    elif isinstance(state.fallback_body.actions[0], RAGReply):
                        f.write(f"{state_var}_fallback_body = Body('{_escape_python_string(state.name)}_fallback_body')\n")
                        for action in state.fallback_body.actions:
                            rag_name = _escape_python_string(action.rag_db_name or '')
                            prompt = getattr(action, 'prompt', None)
                            if prompt:
                                f.write(f"{state_var}_fallback_body.add_action(\n")
                                f.write(f"    RAGReply('{rag_name}', prompt='{_escape_python_string(prompt)}')\n")
                                f.write(")\n")
                            else:
                                f.write(f"{state_var}_fallback_body.add_action(RAGReply('{rag_name}'))\n")
                        f.write("\n")
                        f.write(f"{state_var}.set_fallback_body({state_var}_fallback_body)\n")
                    elif isinstance(state.fallback_body.actions[0], DBReply):
                        f.write(f"{state_var}_fallback_body = Body('{_escape_python_string(state.name)}_fallback_body')\n")
                        for action in state.fallback_body.actions:
                            db_args = []
                            if getattr(action, 'db_selection_type', 'default') != 'default':
                                db_args.append(f"db_selection_type={action.db_selection_type!r}")
                            if getattr(action, 'db_custom_name', None):
                                db_args.append(f"db_custom_name={action.db_custom_name!r}")
                            if getattr(action, 'db_query_mode', 'llm_query') != 'llm_query':
                                db_args.append(f"db_query_mode={action.db_query_mode!r}")
                            if getattr(action, 'db_operation', 'any') != 'any':
                                db_args.append(f"db_operation={action.db_operation!r}")
                            if getattr(action, 'db_query_mode', 'llm_query') == 'sql' and getattr(action, 'db_sql_query', None):
                                db_args.append(f"db_sql_query={action.db_sql_query!r}")
                            args = ", ".join(db_args)
                            f.write(f"{state_var}_fallback_body.add_action(DBReply({args}))\n" if args else f"{state_var}_fallback_body.add_action(DBReply())\n")
                        f.write("\n")
                        f.write(f"{state_var}.set_fallback_body({state_var}_fallback_body)\n")
                    elif isinstance(state.fallback_body.actions[0], CustomCodeAction):
                        action = state.fallback_body.actions[0]
                        f.write(f"{action.to_code()}\n")
                        function_match = search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', action.code)
                        if function_match:
                            function_name = function_match.group(1)
                        else:
                            function_name = f"custom_action_{safe_var_name(action.name)}"
                        f.write(
                            f"CustomCodeAction_{state_var}_fallback = "
                            f"CustomCodeAction(callable={function_name})\n"
                        )
                        f.write(f"{state_var}_fallback_body = Body('{_escape_python_string(state.name)}_fallback_body')\n")
                        f.write(f"{state_var}_fallback_body.add_action(CustomCodeAction_{state_var}_fallback)\n")
                        f.write(f"{state_var}.set_fallback_body({state_var}_fallback_body)\n")

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
                            f.write(f"{state_var}.when_file_received('{_escape_python_string(str(file_type))}').go_to({dest_var})\n")
                        else:
                            f.write(f"{state_var}.when_file_received().go_to({dest_var})\n")

                    elif event is None and condition_class == "Auto":
                        f.write(f"{state_var}.go_to({dest_var})\n")

                    else:
                        # Custom transition with a single condition.
                        condition_name = getattr(condition, "name", str(condition))
                        if event:
                            transition_chain = f"{state_var}.when_event({event_class}())"
                            transition_chain += f".with_condition({condition_name})"
                            transition_chain += f".go_to({dest_var})"
                            f.write(f"{transition_chain}\n")
                        else:
                            transition_chain = f"{state_var}.when_condition({condition_name})"
                            transition_chain += f".go_to({dest_var})"
                            f.write(f"{transition_chain}\n")

                elif len(conditions) > 1:
                    # Custom transition with multiple conditions.
                    condition_names = [getattr(c, "name", str(c)) for c in conditions]
                    if event:
                        transition_chain = f"{state_var}.when_event({event_class}())"
                        for condition_name in condition_names:
                            transition_chain += f".with_condition({condition_name})"
                        transition_chain += f".go_to({dest_var})"
                        f.write(f"{transition_chain}\n")
                    else:
                        transition_chain = f"{state_var}.when_condition({condition_names[0]})"
                        for condition_name in condition_names[1:]:
                            transition_chain += f".with_condition({condition_name})"
                        transition_chain += f".go_to({dest_var})"
                        f.write(f"{transition_chain}\n")

                elif event:
                    # Event-only custom transition.
                    f.write(f"{state_var}.when_event({event_class}()).go_to({dest_var})\n")
                else:
                    # No event and no conditions -> simple transition.
                    f.write(f"{state_var}.go_to({dest_var})\n")


                f.write("\n")

    print(f"Agent model saved to {file_path}")
