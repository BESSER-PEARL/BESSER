"""
Agent diagram processing for converting JSON to BUML format.
"""

import logging
import operator
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)
import json as json_lib
from besser.BUML.metamodel.state_machine.state_machine import (
    Body,
    Condition,
    Event,
    ConfigProperty,
    CustomCodeAction,
    TransitionBuilder,
)
from besser.BUML.metamodel.state_machine.agent import (
    Agent,
    Intent,
    Auto,
    IntentMatcher,
    ReceiveTextEvent,
    AgentReply,
    LLMReply,
    RAGReply,
    RAG,
    RAGVectorStore,
    RAGTextSplitter,
)
from besser.BUML.metamodel.structural import Metadata
from besser.utilities.web_modeling_editor.backend.services.converters.parsers import sanitize_text


def _collect_body_messages(body_elements, elements, language, source_language, translate_text):
    """
    Collect and classify body messages from body element IDs.

    Args:
        body_elements: List of body element IDs to process
        elements: Dict of all elements keyed by ID
        language: Target translation language (or None)
        source_language: Source language for translation (or None)
        translate_text: Translation function

    Returns:
        List of classified message strings (prefixed with LLM:/RAG:/CODE: or plain text)
    """
    messages = []
    for body_id in body_elements:
        body_element = elements.get(body_id)
        if not body_element:
            continue

        reply_type = body_element.get("replyType")
        body_content = body_element.get("name", "")

        if reply_type == "text":
            msg = sanitize_text(body_content)
            if language:
                msg = translate_text(msg, language, source_language)
            messages.append(msg)
        elif reply_type == "llm":
            messages.append(f"LLM:{sanitize_text(body_content)}")
        elif reply_type == "rag":
            rag_name = sanitize_text(body_element.get("ragDatabaseName", ""))
            if not rag_name:
                rag_name = sanitize_text(body_content)
            if rag_name:
                messages.append(f"RAG:{rag_name}")
        elif reply_type == "code":
            messages.append(f"CODE:{sanitize_text(body_content)}")

    return messages


def _build_body_from_messages(body_name, messages):
    """
    Build a Body object from classified messages.

    Args:
        body_name: Name for the Body object
        messages: List of classified message strings (from _collect_body_messages)

    Returns:
        A Body object with appropriate actions, or None if messages is empty
    """
    if not messages:
        return None

    has_rag = any(m.startswith("RAG:") for m in messages)
    has_llm = any(m.startswith("LLM:") for m in messages)
    has_code = any(m.startswith("CODE:") for m in messages)

    body = Body(body_name)

    if has_rag:
        rag_names = [m.split(":", 1)[1] for m in messages if m.startswith("RAG:")]
        for rag_db_name in rag_names:
            body.add_action(RAGReply(rag_db_name=rag_db_name))
    elif has_llm:
        body.add_action(LLMReply())
    elif has_code:
        code_contents = [m[5:] for m in messages if m.startswith("CODE:")]
        for code_content in code_contents:
            body.add_action(CustomCodeAction(source=code_content))
    else:
        for message in messages:
            body.add_action(AgentReply(message=message))

    return body


def process_agent_diagram(json_data):
    # Extract language from config if present
    config = json_data.get('config', {})
    lang_value = ""
    language = None
    source_language = None
    if config is not None and config != {}:
        lang_value = config.get('language')
        language = lang_value.lower() if isinstance(lang_value, str) and lang_value else None
        source_language = config.get('source_language')
    def translate_text(text, lang, src_lang=None):
        # Use deep-translator's GoogleTranslator for free translation
        if not lang or lang == 'none':
            return text
        lang_map = {
            'none': 'auto',
            'english': 'en',
            'french': 'fr',
            'german': 'de',
            'spanish': 'es',
            'luxembourgish': 'lb',
            'portuguese': 'pt',
        }
        target_lang = lang_map.get(lang.lower()) if isinstance(lang, str) else None
        if not target_lang:
            return text
        src_code = lang_map.get(src_lang.lower()) if src_lang and isinstance(src_lang, str) else 'auto'
        try:
            translated = GoogleTranslator(source=src_code, target=target_lang).translate(text)
            return translated
        except Exception as e:
            logger.error("Translation error: %s", e)
            return text
    """Process Agent Diagram specific elements and return an Agent model."""
    # Create the agent model
    title = json_data.get('title', 'Generated_Agent')
    if ' ' in title:
        title = title.replace(' ', '_')

    agent = Agent(title)

    # Add default configuration properties
    agent.add_property(ConfigProperty('websocket_platform', 'websocket.host', '0.0.0.0'))
    agent.add_property(ConfigProperty('websocket_platform', 'websocket.port', 8765))
    agent.add_property(ConfigProperty('websocket_platform', 'streamlit.host', '0.0.0.0'))
    agent.add_property(ConfigProperty('websocket_platform', 'streamlit.port', 5000))
    agent.add_property(ConfigProperty('nlp', 'nlp.language', 'en'))
    agent.add_property(ConfigProperty('nlp', 'nlp.region', 'US'))
    agent.add_property(ConfigProperty('nlp', 'nlp.timezone', 'Europe/Madrid'))
    agent.add_property(ConfigProperty('nlp', 'nlp.pre_processing', True))
    agent.add_property(ConfigProperty('nlp', 'nlp.intent_threshold', 0.4))
    agent.add_property(ConfigProperty('nlp', 'nlp.openai.api_key', 'YOUR-API-KEY'))
    agent.add_property(ConfigProperty('nlp', 'nlp.hf.api_key', 'YOUR-API-KEY'))
    agent.add_property(ConfigProperty('nlp', 'nlp.replicate.api_key', 'YOUR-API-KEY'))

    # Get elements and relationships from the JSON data
    elements = json_data.get('model', {}).get('elements', {})
    relationships = json_data.get('model', {}).get('relationships', {})

    # Track states and bodies for later reference
    states_by_id = {}
    bodies_by_id = {}
    fallback_bodies_by_id = {}
    intents_by_id = {}
    rag_dbs_by_id = {}
    rag_dbs_by_name = {}
    
    # Store comments for later processing
    comment_elements = {}  # {comment_id: comment_text}
    comment_links = {}  # {comment_id: [linked_element_ids]}

    # First pass: Process intents and comments
    intent_count = 0
    for element_id, element in elements.items():
        element_type = element.get("type")
        if element_type == "Comments":
            comment_text = element.get("name", "")
            comment_elements[element_id] = comment_text
            continue
        elif element_type == "AgentIntent":
            intent_name = element.get("name")
            training_sentences = []
            intent_description = element.get("intent_description", None)
            # Collect training sentences
            for body_id in element.get("bodies", []):
                body_element = elements.get(body_id)
                if body_element:
                    training_sentence = sanitize_text(body_element.get("name", ""))
                    if language:
                        training_sentence = translate_text(training_sentence, language, source_language)
                    if training_sentence:
                        training_sentences.append(training_sentence)

            # Create intent and add to agent
            intent = Intent(intent_name, training_sentences, description=intent_description)
            agent.add_intent(intent)
            intents_by_id[element_id] = intent
            intent_count += 1
        elif element_type == "AgentRagElement":
            rag_name = sanitize_text((element.get("name") or "").strip())
            if not rag_name:
                continue
            if rag_name in rag_dbs_by_name:
                rag_dbs_by_id[element_id] = rag_dbs_by_name[rag_name]
                continue

            sanitized_slug = rag_name.lower().replace(' ', '_') or "default"
            vector_store = RAGVectorStore(
                embedding_provider="openai",
                embedding_parameters={"api_key_property": "nlp.OPENAI_API_KEY"},
                persist_directory=f"vector_store/{sanitized_slug}",
            )
            splitter = RAGTextSplitter(
                splitter_type="recursive_character",
                chunk_size=1000,
                chunk_overlap=100,
            )
            rag_config = agent.new_rag(
                name=rag_name,
                vector_store=vector_store,
                splitter=splitter,
                llm_name="gpt-4o-mini",
                k=4,
                num_previous_messages=0,
            )
            rag_dbs_by_id[element_id] = rag_config
            rag_dbs_by_name[rag_name] = rag_config

    # First identify the initial state
    initial_state_id = None
    for element_id, element in elements.items():
        if element.get("type") == "AgentState":
            # Check if this is an initial state
            for rel in relationships.values():
                if ((rel.get("type") == "AgentStateTransition" or rel.get("type") == "AgentStateTransitionInit") and
                    rel.get("target", {}).get("element") == element_id and
                    elements.get(rel.get("source", {}).get("element", ""), {}).get("type") == "StateInitialNode"):
                    initial_state_id = element_id
                    break
            if initial_state_id:
                break

    # Process the initial state first if found
    if initial_state_id:
        element = elements.get(initial_state_id)
        state_name = element.get("name", "")

        agent_state = agent.new_state(name=state_name, initial=True)
        states_by_id[initial_state_id] = agent_state

        # Process state bodies
        body_messages = _collect_body_messages(
            element.get("bodies", []), elements, language, source_language, translate_text
        )
        body = _build_body_from_messages(f"{state_name}_body", body_messages)
        if body:
            agent_state.set_body(body)

        # Process fallback bodies
        fallback_messages = _collect_body_messages(
            element.get("fallbackBodies", []), elements, language, source_language, translate_text
        )
        fallback_body = _build_body_from_messages(f"{state_name}_fallback_body", fallback_messages)
        if fallback_body:
            agent_state.set_fallback_body(fallback_body)

    # Now process the rest of the states
    for element_id, element in elements.items():
        if element.get("type") == "AgentState" and element_id != initial_state_id:
            # Create state and add to agent
            state_name = element.get("name", "")

            agent_state = agent.new_state(name=state_name, initial=False)
            states_by_id[element_id] = agent_state

            # Process state bodies
            body_messages = _collect_body_messages(
                element.get("bodies", []), elements, language, source_language, translate_text
            )
            body = _build_body_from_messages(f"{state_name}_body", body_messages)
            if body:
                agent_state.set_body(body)

            # Process fallback bodies
            fallback_messages = _collect_body_messages(
                element.get("fallbackBodies", []), elements, language, source_language, translate_text
            )
            fallback_body = _build_body_from_messages(f"{state_name}_fallback_body", fallback_messages)
            if fallback_body:
                agent_state.set_fallback_body(fallback_body)

    # Third pass: Process transitions and comment links
    transition_count = 0
    for relationship in relationships.values():
        if relationship.get("type") == "Link":
            # Handle comment links
            source_element_id = relationship.get("source", {}).get("element")
            target_element_id = relationship.get("target", {}).get("element")
            
            comment_id = None
            target_id = None
            
            if source_element_id in comment_elements:
                comment_id = source_element_id
                target_id = target_element_id
            elif target_element_id in comment_elements:
                comment_id = target_element_id
                target_id = source_element_id
            
            if comment_id and target_id:
                if comment_id not in comment_links:
                    comment_links[comment_id] = []
                comment_links[comment_id].append(target_id)
        elif relationship.get("type") in ["AgentStateTransition", "AgentStateTransitionInit"]:
            source_id = relationship.get("source", {}).get("element")
            target_id = relationship.get("target", {}).get("element")

            # Skip initial node transitions (already handled when creating states)
            if elements.get(source_id, {}).get("type") == "StateInitialNode":
                continue

            source_state = states_by_id.get(source_id)
            target_state = states_by_id.get(target_id)

            if not source_state or not target_state:
                logger.warning(
                    "Skipping agent transition: source '%s' or target '%s' state not found.",
                    source_id, target_id
                )
                continue

            if source_state and target_state:
                condition_name = relationship.get("condition", "")
                condition_value = relationship.get("conditionValue", "")

                # Create appropriate transition based on condition
                if condition_name == "when_intent_matched":
                    # Find the intent by name
                    intent_to_match = None
                    for intent in agent.intents:
                        if intent.name == condition_value:
                            intent_to_match = intent
                            break

                    if intent_to_match:
                        source_state.when_intent_matched(intent_to_match).go_to(target_state)
                        transition_count += 1
                    elif isinstance(condition_value, str) and condition_value.strip():
                        unresolved_intent = Intent(condition_value.strip())
                        TransitionBuilder(
                            source=source_state,
                            event=ReceiveTextEvent(),
                            conditions=IntentMatcher(unresolved_intent),
                        ).go_to(target_state)
                        transition_count += 1

                elif condition_name == "when_no_intent_matched":
                    source_state.when_no_intent_matched().go_to(target_state)
                    transition_count += 1

                elif condition_name == "when_variable_operation_matched":
                    # Check if condition_value is a dictionary
                    if isinstance(condition_value, dict):
                        variable_name = condition_value.get("variable")
                        operator_value = condition_value.get("operator")
                        target_value = condition_value.get("targetValue")

                        # Map string operators to actual operator functions
                        operator_map = {
                            "<": operator.lt,
                            "<=": operator.le,
                            "==": operator.eq,
                            ">=": operator.ge,
                            ">": operator.gt,
                            "!=": operator.ne
                        }

                        op_func = operator_map.get(operator_value)
                        if op_func:
                            source_state.when_variable_matches_operation(
                                var_name=variable_name,
                                operation=op_func,
                                target=target_value
                            ).go_to(target_state)
                            transition_count += 1
                    else:
                        # If condition_value is not a dictionary, add a simple transition
                        source_state.when_no_intent_matched().go_to(target_state)
                        transition_count += 1

                elif condition_name == "when_file_received":
                    mime_types = {
                        "PDF": "application/pdf",
                        "TXT": "text/plain",
                        "JSON": "application/json"
                    }
                    file_type = mime_types.get(condition_value)
                    if file_type:
                        source_state.when_file_received(file_type).go_to(target_state)
                        transition_count += 1

                elif condition_name == "auto":
                    source_state.go_to(target_state)
                    transition_count += 1

                else:
                    # Default to no_intent_matched if no condition specified
                    source_state.when_no_intent_matched().go_to(target_state)
                    transition_count += 1

    # Process comments
    for comment_id, comment_text in comment_elements.items():
        if comment_id in comment_links:
            # Comment is linked to one or more elements
            for linked_element_id in comment_links[comment_id]:
                if linked_element_id in states_by_id:
                    # Apply comment to state's metadata
                    state = states_by_id[linked_element_id]
                    if state.metadata is None:
                        state.metadata = Metadata(description=comment_text)
                    else:
                        # Append to existing description
                        existing_desc = state.metadata.description or ""
                        state.metadata.description = f"{existing_desc}\n{comment_text}" if existing_desc else comment_text
        else:
            # Unlinked comment - add to Agent metadata
            if agent.metadata is None:
                agent.metadata = Metadata(description=comment_text)
            else:
                # Append to existing description
                existing_desc = agent.metadata.description or ""
                agent.metadata.description = f"{existing_desc}\n{comment_text}" if existing_desc else comment_text

    return agent
