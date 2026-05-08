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


def _collect_body_messages(body_elements, elements, language, source_language, translate_text,
                           serialize_db_reply_payload=None):
    """
    Collect and classify body messages from body element IDs.

    Args:
        body_elements: List of body element IDs to process
        elements: Dict of all elements keyed by ID
        language: Target translation language (or None)
        source_language: Source language for translation (or None)
        translate_text: Translation function
        serialize_db_reply_payload: Optional function to serialize DB reply payloads

    Returns:
        List of classified message strings (prefixed with LLM:/RAG:/DB:/CODE: or plain text)
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
        elif reply_type == "db_reply":
            if serialize_db_reply_payload:
                messages.append(serialize_db_reply_payload(body_element))
        elif reply_type == "code":
            messages.append(f"CODE:{sanitize_text(body_content)}")

    return messages


def _build_body_from_messages(body_name, messages, build_db_reply_fn=None):
    """
    Build a Body object from classified messages.

    Args:
        body_name: Name for the Body object
        messages: List of classified message strings (from _collect_body_messages)
        build_db_reply_fn: Optional function to build a DBReply from a deserialized payload dict

    Returns:
        A Body object with appropriate actions, or None if messages is empty
    """
    if not messages:
        return None

    has_db = any(m.startswith("DB:") for m in messages)
    has_rag = any(m.startswith("RAG:") for m in messages)
    has_llm = any(m.startswith("LLM:") for m in messages)
    has_code = any(m.startswith("CODE:") for m in messages)

    body = Body(body_name)

    if has_db and build_db_reply_fn:
        db_replies = [json_lib.loads(m.split(":", 1)[1]) for m in messages if m.startswith("DB:")]
        for db_reply in db_replies:
            body.add_action(build_db_reply_fn(db_reply))
    elif has_rag:
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


def _normalise_agent_transitions(relationships: dict) -> dict:
    """Collapse legacy AgentStateTransition shapes to the canonical v4 form.

    The v4 canonical shape:
        transitionType: 'predefined' | 'custom'
        predefined: { predefinedType, intentName?, fileType?, conditionValue? }
        custom: { event, condition: string[] }

    Five legacy shapes are accepted (see uml-v4-shape.md). Fallthrough
    order:
      1. transitionType=='custom' OR legacy condition=='custom_transition'
         OR custom.event/condition non-empty -> emit canonical custom block.
      2. Otherwise emit canonical predefined block.
    """
    out: dict = {}
    for rid, rel in relationships.items():
        if rel.get("type") != "AgentStateTransition":
            out[rid] = rel
            continue
        nrel = dict(rel)
        predefined = nrel.get("predefined") or {}
        custom = nrel.get("custom") or {}

        # Detect custom path
        is_custom = (
            nrel.get("transitionType") == "custom"
            or nrel.get("condition") == "custom_transition"
            or (isinstance(custom.get("event"), str) and custom.get("event"))
            or (isinstance(custom.get("condition"), list) and any(
                isinstance(c, str) and c.strip() for c in custom["condition"]
            ))
            or (isinstance(nrel.get("conditionValue"), dict) and (
                nrel["conditionValue"].get("events") or nrel["conditionValue"].get("conditions")
            ))
        )

        if is_custom:
            event = (
                custom.get("event")
                or nrel.get("event")
                or nrel.get("customEvent")
                or "None"
            )
            cond = custom.get("condition")
            if not isinstance(cond, list):
                cond = nrel.get("customConditions")
            if not isinstance(cond, list):
                # Legacy nested "conditionValue.events"/"conditions"
                cv = nrel.get("conditionValue")
                if isinstance(cv, dict):
                    events = cv.get("events") or []
                    if isinstance(events, list) and events and not custom.get("event"):
                        event = events[0]
                    cond = cv.get("conditions") or []
            if not isinstance(cond, list):
                cond = []
            nrel["transitionType"] = "custom"
            nrel["custom"] = {"event": event, "condition": cond}
            nrel.pop("predefined", None)
        else:
            predefined_type = (
                predefined.get("predefinedType")
                or nrel.get("predefinedType")
                or (nrel.get("condition") if isinstance(nrel.get("condition"), str) else None)
                or "when_intent_matched"
            )
            block: dict = {"predefinedType": predefined_type}
            intent_name = (
                predefined.get("intentName")
                or nrel.get("intentName")
            )
            if intent_name is not None:
                block["intentName"] = intent_name
            file_type = predefined.get("fileType") or nrel.get("fileType")
            if file_type is not None:
                block["fileType"] = file_type
            cv = predefined.get("conditionValue")
            if cv is None:
                # Legacy flat variable-operation fields
                if nrel.get("variable") is not None or nrel.get("operator") is not None:
                    cv = {
                        "variable": nrel.get("variable", ""),
                        "operator": nrel.get("operator", ""),
                        "targetValue": nrel.get("targetValue", ""),
                    }
                else:
                    cv = nrel.get("conditionValue")
            if cv is not None:
                block["conditionValue"] = cv
            nrel["transitionType"] = "predefined"
            nrel["predefined"] = block
            nrel.pop("custom", None)
        out[rid] = nrel
    return out


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
        )

    def serialize_db_reply_payload(element: dict) -> str:
        payload = {
            "dbSelectionType": element.get("dbSelectionType", "default") or "default",
            "dbCustomName": element.get("dbCustomName", "") or "",
            "dbQueryMode": element.get("dbQueryMode", "llm_query") or "llm_query",
            "dbOperation": element.get("dbOperation", "any") or "any",
            "dbSqlQuery": element.get("dbSqlQuery", "") or "",
        }
        return f"DB:{json_lib.dumps(payload)}"
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

    # v4: translate ``{nodes, edges}`` into the v3-shaped intermediate
    # the rest of this function walks. v3 raw shape is also accepted.
    model_data = json_data.get('model') or {}
    if model_data.get('nodes') is not None or model_data.get('edges') is not None:
        from besser.utilities.web_modeling_editor.backend.services.converters._shape_normalizer import (
            v4_to_v3_model,
        )
        v3_model = v4_to_v3_model(model_data, diagram_type='AgentDiagram')
        elements = v3_model.get('elements') or {}
        relationships = v3_model.get('relationships') or {}
    else:
        elements = model_data.get('elements') or {}
        relationships = model_data.get('relationships') or {}

    # Normalise legacy AgentStateTransition shapes to the canonical v4
    # form (transitionType + predefined/custom blocks). The downstream
    # routing logic only inspects the canonical fields, so this is the
    # one place where legacy fallthrough lives.  See
    # ``docs/source/migrations/uml-v4-shape.md`` "Legacy
    # AgentStateTransition shapes" for the exhaustive list.
    relationships = _normalise_agent_transitions(relationships)

    # Track states and bodies for later reference
    states_by_id = {}
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

    # Process the initial state first if found
    if initial_state_id:
        element = elements.get(initial_state_id)
        state_name = element.get("name", "")

        agent_state = agent.new_state(name=state_name, initial=True)
        states_by_id[initial_state_id] = agent_state

        # Process state bodies
        body_messages = _collect_body_messages(
            element.get("bodies", []), elements, language, source_language, translate_text,
            serialize_db_reply_payload=serialize_db_reply_payload
        )
        body = _build_body_from_messages(f"{state_name}_body", body_messages, build_db_reply_fn=build_db_reply)
        if body:
            agent_state.set_body(body)

        # Process fallback bodies
        fallback_messages = _collect_body_messages(
            element.get("fallbackBodies", []), elements, language, source_language, translate_text,
            serialize_db_reply_payload=serialize_db_reply_payload
        )
        fallback_body = _build_body_from_messages(f"{state_name}_fallback_body", fallback_messages, build_db_reply_fn=build_db_reply)
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
                element.get("bodies", []), elements, language, source_language, translate_text,
                serialize_db_reply_payload=serialize_db_reply_payload
            )
            body = _build_body_from_messages(f"{state_name}_body", body_messages, build_db_reply_fn=build_db_reply)
            if body:
                agent_state.set_body(body)

            # Process fallback bodies
            fallback_messages = _collect_body_messages(
                element.get("fallbackBodies", []), elements, language, source_language, translate_text,
                serialize_db_reply_payload=serialize_db_reply_payload
            )
            fallback_body = _build_body_from_messages(f"{state_name}_fallback_body", fallback_messages, build_db_reply_fn=build_db_reply)
            if fallback_body:
                agent_state.set_fallback_body(fallback_body)

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

    # Validate the agent model at build time so all callers get validation for free
    agent.validate(raise_exception=True)

    return agent
