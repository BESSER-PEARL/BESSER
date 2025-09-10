"""
Agent diagram processing for converting JSON to BUML format.
"""

import operator
from deep_translator import GoogleTranslator
import json as json_lib
from besser.BUML.metamodel.state_machine.state_machine import Body, Condition, Event, ConfigProperty
from besser.BUML.metamodel.state_machine.agent import Agent, Intent, Auto, IntentMatcher, ReceiveTextEvent
from besser.utilities.web_modeling_editor.backend.services.converters.parsers import sanitize_text


def process_agent_diagram(json_data):
    # Extract language from config if present
    config = json_data.get('config', {})
    lang_value = config.get('language')
    language = lang_value.lower() if isinstance(lang_value, str) and lang_value else None

    def translate_text(text, lang):
        # Use deep-translator's GoogleTranslator for free translation
        if not lang or lang == 'none':
            return text
        lang_map = {
            'french': 'fr',
            'german': 'de',
            'spanish': 'es',
            'luxembourgish': 'lb',
            'portuguese': 'pt',
        }
        target_lang = lang_map.get(lang.lower())
        if not target_lang:
            return text
        try:
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    """Process Agent Diagram specific elements and return an Agent model."""
    # Create the agent model
    title = json_data.get('title', 'Generated_Agent')
    if ' ' in title:
        title = title.replace(' ', '_')

    agent = Agent(title)

    # Add default configuration properties
    agent.add_property(ConfigProperty('websocket_platform', 'websocket.host', 'localhost'))
    agent.add_property(ConfigProperty('websocket_platform', 'websocket.port', 8765))
    agent.add_property(ConfigProperty('websocket_platform', 'streamlit.host', 'localhost'))
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

    # First pass: Process intents
    intent_count = 0
    for element_id, element in elements.items():
        if element.get("type") == "AgentIntent":
            intent_name = element.get("name")
            training_sentences = []

            # Collect training sentences
            for body_id in element.get("bodies", []):
                body_element = elements.get(body_id)
                if body_element:
                    training_sentence = sanitize_text(body_element.get("name", ""))
                    if language:
                        training_sentence = translate_text(training_sentence, language)
                    if training_sentence:
                        training_sentences.append(training_sentence)

            # Create intent and add to agent
            intent = Intent(intent_name, training_sentences)
            agent.add_intent(intent)
            intents_by_id[element_id] = intent
            intent_count += 1

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
        body_count = 0
        body_messages = []
        for body_id in element.get("bodies", []):
            body_element = elements.get(body_id)
            if body_element:
                body_name = f"{state_name}_body"
                body_type = body_element.get("replyType")
                body_content = body_element.get("name", "")

                # Collect messages for this body
                if body_type == "text":
                    msg = sanitize_text(body_content)
                    if language:
                        msg = translate_text(msg, language)
                    body_messages.append(msg)
                elif body_type == "llm":
                    # For LLM replies, we need to use llm.predict(session.event.message)
                    body_messages.append(f"LLM:{sanitize_text(body_content)}")
                elif body_type == "code":
                    # For code, store as a special code message
                    body_messages.append(f"CODE:{sanitize_text(body_content)}")

                body_count += 1

        # Create a single body function that combines all messages
        if body_messages:
            # Check if any of the messages are LLM messages
            has_llm = any(message.startswith("LLM:") for message in body_messages)

            # If we have an LLM message, create a function that uses llm.predict
            if has_llm:
                f_name = f"{state_name}_body"
                def create_llm_body_function(name):
                    def body_function(session):
                        session.reply(llm.predict(session.event.message))
                    return body_function

                body = Body(f_name, create_llm_body_function(f_name))
            else:
                # Otherwise, create a regular function with the messages
                def create_body_function(messages):
                    def body_function(session):
                        for message in messages:
                            if message.startswith("CODE:"):
                                # This is code to be executed
                                try:
                                    # Just store the code for later execution
                                    code_content = message[5:]
                                    exec(code_content)
                                except Exception as e:
                                    print(f"Error executing code: {str(e)}")
                            else:
                                session.reply(message)
                    return body_function

                body = Body(f"{state_name}_body", create_body_function(body_messages))

            # Store the messages directly in the Body object for easier extraction
            body.messages = body_messages
            agent_state.set_body(body)

        # Process fallback bodies
        fallback_count = 0
        fallback_messages = []
        for fallback_id in element.get("fallbackBodies", []):
            fallback_element = elements.get(fallback_id)
            if fallback_element:
                fallback_name = f"{state_name}_fallback_body"
                fallback_type = fallback_element.get("replyType")
                fallback_content = fallback_element.get("name", "")

                # Collect messages for this fallback body
                if fallback_type == "text":
                    message = sanitize_text(fallback_content)
                    if language:
                        message = translate_text(message, language)
                    fallback_messages.append(message)
                elif fallback_type == "llm":
                    # For LLM replies, store as a special LLM message
                    fallback_messages.append(f"LLM:{sanitize_text(fallback_content)}")
                elif fallback_type == "code":
                    # For code, store as a special code message
                    fallback_messages.append(f"CODE:{sanitize_text(fallback_content)}")

                fallback_count += 1

        # Create a single fallback body function that combines all messages
        if fallback_messages:
            # Check if any of the messages are LLM messages
            has_llm = any(message.startswith("LLM:") for message in fallback_messages)

            # If we have an LLM message, create a function that uses llm.predict
            if has_llm:
                f_name = f"{state_name}_fallback_body"
                def create_llm_fallback_function(name):
                    def fallback_function(session):
                        session.reply(llm.predict(session.event.message))
                    return fallback_function

                fallback_body = Body(f_name, create_llm_fallback_function(f_name))
            else:
                # Otherwise, create a regular function with the messages
                def create_fallback_function(messages):
                    def fallback_function(session):
                        for message in messages:
                            if message.startswith("CODE:"):
                                # This is code to be executed
                                try:
                                    code_content = message[5:]
                                    exec(code_content)
                                except Exception as e:
                                    print(f"Error executing code: {str(e)}")
                            else:
                                session.reply(message)
                    return fallback_function

                fallback_body = Body(f"{state_name}_fallback_body", create_fallback_function(fallback_messages))

            # Store the messages directly in the Body object for easier extraction
            fallback_body.messages = fallback_messages
            agent_state.set_fallback_body(fallback_body)

    # Now process the rest of the states
    for element_id, element in elements.items():
        if element.get("type") == "AgentState" and element_id != initial_state_id:
            # Create state and add to agent
            state_name = element.get("name", "")

            agent_state = agent.new_state(name=state_name, initial=False)
            states_by_id[element_id] = agent_state

            # Process state bodies
            body_count = 0
            body_messages = []
            for body_id in element.get("bodies", []):
                body_element = elements.get(body_id)
                if body_element:
                    body_name = f"{state_name}_body"
                    body_type = body_element.get("replyType")
                    body_content = body_element.get("name", "")
                    
                    # Collect messages for this body
                    if body_type == "text":
                        msg = sanitize_text(body_content)
                        if language:
                            msg = translate_text(msg, language)
                        body_messages.append(msg)
                    elif body_type == "llm":
                        # For LLM replies, we need to use llm.predict(session.event.message)
                        body_messages.append(f"LLM:{sanitize_text(body_content)}")
                    elif body_type == "code":
                        # For code, store as a special code message
                        body_messages.append(f"CODE:{sanitize_text(body_content)}")

                    body_count += 1

            # Create a single body function that combines all messages
            if body_messages:
                # Check if any of the messages are LLM messages
                has_llm = any(message.startswith("LLM:") for message in body_messages)

                # If we have an LLM message, create a function that uses llm.predict
                if has_llm:
                    f_name = f"{state_name}_body"
                    def create_llm_body_function(name):
                        def body_function(session):
                            session.reply(llm.predict(session.event.message))
                        return body_function

                    body = Body(f_name, create_llm_body_function(f_name))
                else:
                    # Otherwise, create a regular function with the messages
                    def create_body_function(messages):
                        def body_function(session):
                            for message in messages:
                                if message.startswith("CODE:"):
                                    # This is code to be executed
                                    try:
                                        # Just store the code for later execution
                                        code_content = message[5:]
                                        exec(code_content)
                                    except Exception as e:
                                        print(f"Error executing code: {str(e)}")
                                else:
                                    session.reply(message)
                        return body_function

                    body = Body(f"{state_name}_body", create_body_function(body_messages))

                # Store the messages directly in the Body object for easier extraction
                body.messages = body_messages
                agent_state.set_body(body)

            # Process fallback bodies
            fallback_count = 0
            fallback_messages = []
            for fallback_id in element.get("fallbackBodies", []):
                fallback_element = elements.get(fallback_id)
                if fallback_element:
                    fallback_name = f"{state_name}_fallback_body"
                    fallback_type = fallback_element.get("replyType")
                    fallback_content = fallback_element.get("name", "")

                    # Collect messages for this fallback body
                    if fallback_type == "text":
                        msg = sanitize_text(fallback_content)
                        if language:
                            msg = translate_text(msg, language)
                        fallback_messages.append(msg)

                    elif fallback_type == "llm":
                        # For LLM replies, store as a special LLM message
                        fallback_messages.append(f"LLM:{sanitize_text(fallback_content)}")
                    elif fallback_type == "code":
                        # For code, store as a special code message
                        fallback_messages.append(f"CODE:{sanitize_text(fallback_content)}")

                    fallback_count += 1

            # Create a single fallback body function that combines all messages
            if fallback_messages:
                # Check if any of the messages are LLM messages
                has_llm = any(message.startswith("LLM:") for message in fallback_messages)

                # If we have an LLM message, create a function that uses llm.predict
                if has_llm:
                    f_name = f"{state_name}_fallback_body"
                    def create_llm_fallback_function(name):
                        def fallback_function(session):
                            session.reply(llm.predict(session.event.message))
                        return fallback_function

                    fallback_body = Body(f_name, create_llm_fallback_function(f_name))
                else:
                    # Otherwise, create a regular function with the messages
                    def create_fallback_function(messages):
                        def fallback_function(session):
                            for message in messages:
                                if message.startswith("CODE:"):
                                    # This is code to be executed
                                    try:
                                        code_content = message[5:]
                                        exec(code_content)
                                    except Exception as e:
                                        print(f"Error executing code: {str(e)}")
                                else:
                                    session.reply(message)
                        return fallback_function

                    fallback_body = Body(f"{state_name}_fallback_body", create_fallback_function(fallback_messages))

                # Store the messages directly in the Body object for easier extraction
                fallback_body.messages = fallback_messages
                agent_state.set_fallback_body(fallback_body)

    # Third pass: Process transitions
    transition_count = 0
    for relationship in relationships.values():
        if relationship.get("type") in ["AgentStateTransition", "AgentStateTransitionInit"]:
            source_id = relationship.get("source", {}).get("element")
            target_id = relationship.get("target", {}).get("element")

            # Skip initial node transitions (already handled when creating states)
            if elements.get(source_id, {}).get("type") == "StateInitialNode":
                continue

            source_state = states_by_id.get(source_id)
            target_state = states_by_id.get(target_id)

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

    return agent
