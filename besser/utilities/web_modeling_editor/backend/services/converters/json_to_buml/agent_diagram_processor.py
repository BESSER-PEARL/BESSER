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
    ConfigProperty,
    CustomCodeAction,
    TransitionBuilder,
)
from besser.BUML.metamodel.state_machine.agent import (
    Agent,
    Intent,
    DummyEvent,
    IntentMatcher,
    ReceiveFileEvent,
    ReceiveJSONEvent,
    ReceiveMessageEvent,
    ReceiveTextEvent,
    WildcardEvent,
    AgentReply,
    LLMReply,
    RAGReply,
    DBReply,
    RAGVectorStore,
    RAGTextSplitter,
)
from besser.BUML.metamodel.structural import Metadata
from besser.utilities.web_modeling_editor.backend.services.converters.parsers import sanitize_text


# Maps old informal "replyType" values to new metamodel class names used in "actionType".
_REPLY_TYPE_TO_ACTION_TYPE = {
    "text": "TextReplyAction",
    "llm": "LLMReplyAction",
    "rag": "RAGReplyAction",
    "db_reply": "DBAction",
    "code": "CustomCodeAction",
}


def _resolve_action_type(element: dict) -> str:
    """
    Return the normalized actionType string for an action element.
    Supports both new 'actionType' (metamodel class name) and old 'replyType' (backward compat).
    """
    action_type = element.get("actionType")
    if action_type:
        return action_type
    reply_type = element.get("replyType", "")
    return _REPLY_TYPE_TO_ACTION_TYPE.get(reply_type, "")


def _build_body_from_action_elements(body_name, action_element_ids, elements,
                                     language, source_language, translate_text,
                                     build_db_reply_fn=None):
    """
    Build a Body object from an ordered list of action element IDs using per-element dispatch.

    Each element is classified by its 'actionType' field (new schema, metamodel class name)
    or its legacy 'replyType' field (backward compat). Actions are added to the body in the
    order they appear in action_element_ids, preserving execution order.

    Args:
        body_name: Name for the Body object.
        action_element_ids: Ordered list of action element IDs.
        elements: Dict of all diagram elements keyed by ID.
        language: Target translation language (or None).
        source_language: Source language for translation (or None).
        translate_text: Translation function.
        build_db_reply_fn: Optional callable to build a DBReply from an element dict.

    Returns:
        A Body object with one action per element, or None if no actions were added.
    """
    if not action_element_ids:
        return None

    body = Body(body_name)
    action_added = False

    for element_id in action_element_ids:
        element = elements.get(element_id)
        if not element:
            continue

        action_type = _resolve_action_type(element)
        content = element.get("name", "")

        if action_type == "TextReplyAction":
            msg = sanitize_text(content)
            if language:
                msg = translate_text(msg, language, source_language)
            body.add_action(AgentReply(message=msg))
            action_added = True

        elif action_type == "LLMReplyAction":
            # Prefer dedicated system_message field; fall back to legacy llmPrompt key.
            # Never use name — it is a display label ("LLM Reply"), not the system message.
            prompt_raw = element.get("system_message") or element.get("llmPrompt") or ""
            prompt = sanitize_text(prompt_raw) or None
            # Support "llmName" (new schema key) and "llm_name" (legacy key)
            llm_name_raw = element.get("llm_name") or element.get("llmName") or ""
            llm_name = sanitize_text(llm_name_raw) or None
            body.add_action(LLMReply(prompt=prompt, llm_name=llm_name))
            action_added = True

        elif action_type == "RAGReplyAction":
            rag_name = sanitize_text(element.get("ragDatabaseName", ""))
            if not rag_name:
                rag_name = sanitize_text(content)
            if rag_name:
                body.add_action(RAGReply(rag_db_name=rag_name))
                action_added = True

        elif action_type == "DBAction":
            if build_db_reply_fn:
                body.add_action(build_db_reply_fn(element))
                action_added = True

        elif action_type == "CustomCodeAction":
            # Raw source code must not be sanitized — sanitize_text escapes single quotes
            # which would corrupt string literals inside the user's Python function.
            body.add_action(CustomCodeAction(source=content))
            action_added = True

        else:
            logger.warning("Unknown actionType '%s' on element '%s'; skipping.", action_type, element_id)

    return body if action_added else None


def process_agent_diagram(json_data):
    # Extract language from config if present
    config = json_data.get('config') or {}
    lang_value = config.get('language', '')
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

    def build_db_reply(element: dict) -> DBReply:
        return DBReply(
            db_selection_type=sanitize_text(element.get("dbSelectionType", "default")) or "default",
            db_custom_name=sanitize_text(element.get("dbCustomName", "")) or None,
            db_query_mode=sanitize_text(element.get("dbQueryMode", "llm_query")) or "llm_query",
            db_operation=sanitize_text(element.get("dbOperation", "any")) or "any",
            db_sql_query=element.get("dbSqlQuery") or None,
            llm_name=sanitize_text(element.get("llm_name", "")) or None,
        )

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
    model_data = json_data.get('model') or {}
    elements = model_data.get('elements') or {}
    relationships = model_data.get('relationships') or {}

    # Track states and bodies for later reference
    states_by_id = {}
    intents_by_id = {}
    rag_dbs_by_id = {}
    rag_dbs_by_name = {}

    # Store comments for later processing
    comment_elements = {}  # {comment_id: comment_text}
    comment_links = {}  # {comment_id: [linked_element_ids]}

    # First pass: Process intents, primitives, and comments
    intent_count = 0
    for element_id, element in elements.items():
        element_type = element.get("type")
        if element_type == "Comments":
            comment_text = element.get("name", "")
            comment_elements[element_id] = comment_text
            continue
        elif element_type == "AgentLLM":
            llm_name = sanitize_text((element.get("name") or "").strip())
            if not llm_name:
                continue
            if any(existing.name == llm_name for existing in agent.llms):
                continue
            provider = (element.get("provider") or "openai").lower()
            llm_parameters = element.get("parameters")
            if not isinstance(llm_parameters, dict):
                llm_parameters = {}
            num_prev = element.get("num_previous_messages")
            try:
                num_prev_int = int(num_prev) if num_prev is not None else 1
            except (TypeError, ValueError):
                num_prev_int = 1
            global_ctx = element.get("global_context") or None
            agent.new_llm(
                name=llm_name,
                provider=provider,
                parameters=llm_parameters,
                num_previous_messages=num_prev_int,
                global_context=global_ctx,
            )
            continue
        elif element_type == "AgentTool":
            tool_name = sanitize_text((element.get("name") or "").strip())
            if not tool_name:
                continue
            if any(t.name == tool_name for t in agent.tools):
                continue
            agent.new_tool(
                name=tool_name,
                description=element.get("description", "") or "",
                code=element.get("code", "") or "",
            )
            continue
        elif element_type == "AgentSkill":
            skill_name = sanitize_text((element.get("name") or "").strip())
            if not skill_name:
                continue
            if any(s.name == skill_name for s in agent.skills):
                continue
            agent.new_skill(
                name=skill_name,
                content=element.get("content", "") or "",
                description=element.get("description") or None,
            )
            continue
        elif element_type == "AgentWorkspace":
            ws_name = sanitize_text((element.get("name") or "").strip())
            if not ws_name:
                continue
            if any(w.name == ws_name for w in agent.workspaces):
                continue
            writable = element.get("writable")
            if writable is None:
                writable = True
            max_read_bytes = element.get("max_read_bytes")
            if max_read_bytes is None:
                max_read_bytes = 200_000
            agent.new_workspace(
                name=ws_name,
                path=element.get("path", "") or "",
                description=element.get("description") or None,
                writable=bool(writable),
                max_read_bytes=int(max_read_bytes),
            )
            continue
        elif element_type == "AgentReasoningState":
            # Reasoning states are created in the state-construction passes below.
            continue
        elif element_type == "AgentIntent":
            intent_name = element.get("name")
            training_sentences = []
            intent_description = element.get("intent_description", None)
            # Collect training sentences — AgentIntent still uses "bodies" for sentence IDs.
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
            rag_llm_name = sanitize_text((element.get("llm_name") or "").strip()) or ""
            rag_config = agent.new_rag(
                name=rag_name,
                vector_store=vector_store,
                splitter=splitter,
                llm_name=rag_llm_name,
                k=4,
                num_previous_messages=0,
            )
            rag_dbs_by_id[element_id] = rag_config
            rag_dbs_by_name[rag_name] = rag_config

    def _is_reasoning_element(element: dict) -> bool:
        """Return True if this element represents a ReasoningState (old or new schema)."""
        if element.get("type") == "AgentReasoningState":
            return True
        if element.get("type") == "AgentState" and element.get("stateType") == "reasoning":
            return True
        return False

    def _is_standard_state_element(element: dict) -> bool:
        """Return True if this element represents a standard AgentState."""
        if element.get("type") == "AgentState":
            return element.get("stateType", "standard") != "reasoning"
        return False

    # First identify the initial state
    initial_state_id = None
    for element_id, element in elements.items():
        if element.get("type") in ("AgentState", "AgentReasoningState"):
            # Check if this is an initial state
            for rel in relationships.values():
                rel_type = rel.get("type")
                if rel_type not in ("AgentStateTransition", "AgentStateTransitionInit"):
                    continue
                target_el = rel.get("target") or {}
                source_el = rel.get("source") or {}
                source_elem_id = source_el.get("element", "")
                if (target_el.get("element") == element_id and
                    elements.get(source_elem_id, {}).get("type") == "StateInitialNode"):
                    initial_state_id = element_id
                    break
            if initial_state_id:
                break

    def _build_reasoning_state(element_id: str, element: dict, is_initial: bool):
        state_name = element.get("name", "")
        llm_name = element.get("llm_name") or element.get("llm")
        llm_value = llm_name.strip() if isinstance(llm_name, str) and llm_name.strip() else None
        kwargs = {
            "name": state_name,
            "llm": llm_value,
            "initial": is_initial,
        }
        if element.get("max_steps") is not None:
            kwargs["max_steps"] = int(element.get("max_steps"))
        if element.get("enable_task_planning") is not None:
            kwargs["enable_task_planning"] = bool(element.get("enable_task_planning"))
        if element.get("stream_steps") is not None:
            kwargs["stream_steps"] = bool(element.get("stream_steps"))
        if element.get("system_prompt") is not None:
            kwargs["system_prompt"] = element.get("system_prompt")
        if element.get("fallback_message") is not None:
            kwargs["fallback_message"] = element.get("fallback_message")
        rs = agent.new_reasoning_state(**kwargs)
        # Tools, skills and workspaces are registered at the agent level and
        # shared by every reasoning state; the metamodel has no per-state
        # subset concept, so no per-state ref lists are parsed here.
        states_by_id[element_id] = rs
        return rs

    def _build_standard_state(element_id: str, element: dict, is_initial: bool):
        """Create a standard AgentState and its body/fallback body from the new schema."""
        state_name = element.get("name", "")
        agent_state = agent.new_state(name=state_name, initial=is_initial)
        states_by_id[element_id] = agent_state

        # Resolve action element IDs — new key "actions", backward-compat key "bodies"
        action_ids = element.get("actions", element.get("bodies", []))
        body = _build_body_from_action_elements(
            f"{state_name}_body", action_ids, elements,
            language, source_language, translate_text,
            build_db_reply_fn=build_db_reply,
        )
        if body:
            agent_state.set_body(body)

        # Only attach a fallback body if fallbackBodyEnabled is absent (legacy) or True
        fallback_enabled = element.get("fallbackBodyEnabled", True)
        if fallback_enabled:
            fallback_ids = element.get("fallbackActions", element.get("fallbackBodies", []))
            fallback_body = _build_body_from_action_elements(
                f"{state_name}_fallback_body", fallback_ids, elements,
                language, source_language, translate_text,
                build_db_reply_fn=build_db_reply,
            )
            if fallback_body:
                agent_state.set_fallback_body(fallback_body)

        return agent_state

    # Process the initial state first if found
    if initial_state_id:
        element = elements.get(initial_state_id)
        if _is_reasoning_element(element):
            _build_reasoning_state(initial_state_id, element, is_initial=True)
        else:
            _build_standard_state(initial_state_id, element, is_initial=True)

    # Now process the rest of the states (including reasoning states)
    for element_id, element in elements.items():
        if element_id == initial_state_id:
            continue
        if _is_reasoning_element(element):
            _build_reasoning_state(element_id, element, is_initial=False)
        elif _is_standard_state_element(element):
            _build_standard_state(element_id, element, is_initial=False)

    # Build intent lookup dict for O(1) resolution during transition processing.
    # Intent names are unique case-insensitively in BUML (see Agent._validate_state_intent_name_collisions),
    # so we accept both exact and casefold matches. This protects against frontend personalization
    # variants that emit intentName references with different casing than the intent definitions
    # they were derived from — without this, the lookup misses and a duplicate Intent is created
    # that never appears in agent.intents, leaving the template referencing an undefined variable.
    intent_lookup = {intent.name: intent for intent in agent.intents}
    intent_lookup_casefold = {
        intent.name.casefold(): intent
        for intent in agent.intents
        if isinstance(intent.name, str)
    }

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
                transition_type = relationship.get("transitionType")
                predefined_block = relationship.get("predefined") or {}
                custom_block = relationship.get("custom") or {}

                condition_name = ""
                transition_payload = ""

                is_custom_transition = (
                    transition_type == "custom"
                )

                if is_custom_transition:
                    selected_event = (
                        custom_block.get("event")
                        or relationship.get("event")
                        or relationship.get("customEvent")
                    )
                    custom_conditions = (
                        custom_block.get("condition")
                        if isinstance(custom_block.get("condition"), list)
                        else relationship.get("conditions")
                    )
                    if not isinstance(custom_conditions, list):
                        custom_conditions = relationship.get("customConditions")
                    if not isinstance(custom_conditions, list):
                        custom_conditions = []

                    normalized_event = "None"
                    if isinstance(selected_event, str) and selected_event and selected_event != "None":
                        normalized_event = selected_event
                    condition_name = "custom_transition"
                    transition_payload = {
                        "event": normalized_event,
                        "conditions": custom_conditions if isinstance(custom_conditions, list) else [],
                    }
                else:
                    condition_name = (
                        predefined_block.get("predefinedType")
                        or relationship.get("predefinedType")
                        or ""
                    )
                    if condition_name == "when_intent_matched":
                        transition_payload = (
                            predefined_block.get("intentName")
                            or relationship.get("intentName")
                        )
                    elif condition_name == "when_file_received":
                        transition_payload = (
                            predefined_block.get("fileType")
                            or relationship.get("fileType")
                        )
                    else:
                        transition_payload = predefined_block.get("conditionValue")

                    if transition_payload is None:
                        transition_payload = relationship.get("conditionValue", "")

                # Create appropriate transition based on condition
                if condition_name == "when_intent_matched":
                    # Find the intent by name via O(1) lookup, falling back to a
                    # case-insensitive match so personalization variants whose JSON
                    # uses a different casing than the intent definition still resolve
                    # to the same intent object.
                    intent_to_match = intent_lookup.get(transition_payload)
                    if intent_to_match is None and isinstance(transition_payload, str):
                        intent_to_match = intent_lookup_casefold.get(transition_payload.casefold())

                    if intent_to_match:
                        source_state.when_intent_matched(intent_to_match).go_to(target_state)
                        transition_count += 1
                    elif isinstance(transition_payload, str) and transition_payload.strip():
                        unresolved_intent = Intent(transition_payload.strip())
                        TransitionBuilder(
                            source=source_state,
                            event=ReceiveTextEvent(),
                            conditions=[IntentMatcher(unresolved_intent)],
                        ).go_to(target_state)
                        transition_count += 1

                elif condition_name == "when_no_intent_matched":
                    source_state.when_no_intent_matched().go_to(target_state)
                    transition_count += 1

                elif condition_name == "when_variable_operation_matched":
                    # Check if transition payload is a dictionary
                    if isinstance(transition_payload, dict):
                        variable_name = transition_payload.get("variable")
                        operator_value = transition_payload.get("operator")
                        target_value = transition_payload.get("targetValue")

                        if not variable_name or not operator_value:
                            logger.warning(
                                "Incomplete variable operation condition (variable=%s, operator=%s) "
                                "for transition from '%s' to '%s'. Falling back to no_intent_matched.",
                                variable_name, operator_value,
                                source_state.name, target_state.name,
                            )
                            source_state.when_no_intent_matched().go_to(target_state)
                            transition_count += 1
                        else:
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
                                logger.warning(
                                    "Unknown operator '%s' for variable operation transition from '%s' to '%s'. Skipping.",
                                    operator_value, source_state.name, target_state.name,
                                )
                    else:
                        # If transition_payload is not a dictionary, add a simple transition
                        logger.warning(
                            "Expected dict for when_variable_operation_matched condition but got %s. "
                            "Falling back to no_intent_matched for transition from '%s' to '%s'.",
                            type(transition_payload).__name__, source_state.name, target_state.name,
                        )
                        source_state.when_no_intent_matched().go_to(target_state)
                        transition_count += 1

                elif condition_name == "when_file_received":
                    mime_types = {
                        "PDF": "application/pdf",
                        "TXT": "text/plain",
                        "JSON": "application/json"
                    }
                    if isinstance(transition_payload, str) and "/" in transition_payload:
                        file_type = transition_payload
                    else:
                        file_type = mime_types.get(transition_payload)
                    if file_type:
                        source_state.when_file_received(file_type).go_to(target_state)
                        transition_count += 1
                    else:
                        logger.warning(
                            "Unknown file type '%s' for when_file_received transition from '%s' to '%s'. "
                            "Falling back to when_file_received() without type filter.",
                            transition_payload, source_state.name, target_state.name,
                        )
                        source_state.when_file_received().go_to(target_state)
                        transition_count += 1

                elif condition_name == "auto":
                    source_state.go_to(target_state)
                    transition_count += 1

                elif condition_name == "custom_transition":
                    event_instance = None
                    custom_conditions = []

                    if isinstance(transition_payload, dict):
                        selected_event = transition_payload.get("event")
                        # Backward compatibility for older payloads that used "events": [..]
                        if not selected_event:
                            raw_events = transition_payload.get("events") or []
                            if isinstance(raw_events, list) and raw_events:
                                selected_event = raw_events[0]

                        if selected_event == "ReceiveTextEvent":
                            event_instance = ReceiveTextEvent()
                        elif selected_event == "ReceiveMessageEvent":
                            event_instance = ReceiveMessageEvent("")
                        elif selected_event == "ReceiveJSONEvent":
                            event_instance = ReceiveJSONEvent()
                        elif selected_event == "ReceiveFileEvent":
                            event_instance = ReceiveFileEvent()
                        elif selected_event == "DummyEvent":
                            event_instance = DummyEvent()
                        elif selected_event == "WildcardEvent":
                            event_instance = WildcardEvent()
                        elif selected_event == "None":
                            event_instance = None

                        raw_conditions = transition_payload.get("conditions") or []
                        if isinstance(raw_conditions, list):
                            custom_conditions = [c for c in raw_conditions if isinstance(c, str) and c.strip()]

                    condition_objects = []
                    for condition_index, custom_condition_code in enumerate(custom_conditions, start=1):
                        generated_name = f"condition_{transition_count + 1}_{condition_index}"
                        custom_condition = Condition(name=generated_name, callable=None)
                        custom_condition.code = custom_condition_code
                        condition_objects.append(custom_condition)

                    transition_builder = None
                    if event_instance is not None:
                        transition_builder = source_state.when_event(event_instance)

                    if condition_objects:
                        if transition_builder is None:
                            transition_builder = source_state.when_condition(condition_objects[0])
                            for extra_condition in condition_objects[1:]:
                                transition_builder.with_condition(extra_condition)
                        else:
                            for custom_condition in condition_objects:
                                transition_builder.with_condition(custom_condition)

                    if transition_builder is not None:
                        transition_builder.go_to(target_state)
                        transition_count += 1
                    else:
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

    # Apply default LLM from the customization config block (if set).
    # The customization tab persists which registered LLM is the default;
    # without an explicit pointer the agent already auto-defaulted to the
    # first one registered.
    default_llm_name_cfg = (config or {}).get("default_llm_name")
    if isinstance(default_llm_name_cfg, str) and default_llm_name_cfg.strip():
        if any(existing.name == default_llm_name_cfg for existing in agent.llms):
            agent.set_default_llm(default_llm_name_cfg)

    # Validate the agent model at build time so all callers get validation for free
    agent.validate(raise_exception=True)

    return agent
