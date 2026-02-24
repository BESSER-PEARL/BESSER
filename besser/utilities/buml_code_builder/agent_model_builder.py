"""
Agent Model Builder

This module generates Python code for BUML agent models.
"""

import os
from re import search
from besser.BUML.metamodel.state_machine.agent import Agent, AgentReply, LLMReply
from besser.BUML.metamodel.state_machine.state_machine import CustomCodeAction


def agent_model_to_code(model: Agent, file_path: str):
    """
    Generates Python code for a B-UML Agent model and writes it to a specified file.

    Parameters:
    model (Agent): The B-UML Agent model object containing states, intents, and transitions.
    file_path (str): The path where the generated code will be saved.

    Outputs:
    - A Python file containing the code representation of the B-UML agent model.
    """
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not file_path.endswith('.py'):
        file_path += '.py'

    with open(file_path, 'w', encoding='utf-8') as f:
        # Write imports
        f.write("###############\n")
        f.write("# AGENT MODEL #\n")
        f.write("###############\n")
        f.write("import datetime\n")
        f.write("from besser.BUML.metamodel.state_machine.state_machine import Body, Condition, Event, ConfigProperty\n")
        f.write("from besser.BUML.metamodel.state_machine.agent import Agent, AgentSession, AgentReply, LLMReply, LLMOpenAI, LLMHuggingFace, LLMHuggingFaceAPI, LLMReplicate\n")
        f.write("from besser.BUML.metamodel.structural import Metadata\n")
        f.write("import operator\n\n")

        # Create agent with metadata if it exists
        if hasattr(model, 'metadata') and model.metadata and model.metadata.description:
            desc = model.metadata.description.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            f.write(f"agent = Agent('{model.name}', metadata=Metadata(description=\"{desc}\"))\n\n")
        else:
            f.write(f"agent = Agent('{model.name}')\n\n")
        
        # Write configuration properties
        for prop in model.properties:
            f.write(f"agent.add_property(ConfigProperty('{prop.section}', '{prop.name}', {repr(prop.value)}))\n")
        f.write("\n")

        # Write intents
        f.write("# INTENTS\n")
        for intent in model.intents:
            f.write(f"{intent.name} = agent.new_intent('{intent.name}', [\n")
            for sentence in intent.training_sentences:
                # Escape single quotes for Python string literal
                escaped_sentence = sentence.replace('\\', '\\\\').replace("'", "\\'")
                f.write(f"    '{escaped_sentence}',\n")
            f.write("])\n")
        f.write("\n")
        
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
            f.write("llm = LLMOpenAI(agent=agent, name='gpt-4o-mini', parameters={})\n\n")

        # Write states
        f.write("# STATES\n")
        for state in model.states:
            f.write(f"{state.name} = agent.new_state('{state.name}'")
            if state.initial:
                f.write(", initial=True")
            f.write(")\n")
        f.write("\n")

        # Write state metadata if any states have it
        for state in model.states:
            if hasattr(state, 'metadata') and state.metadata and state.metadata.description:
                desc = state.metadata.description.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                f.write(f"{state.name}.metadata = Metadata(description=\"{desc}\")\n")
        if any(hasattr(state, 'metadata') and state.metadata and state.metadata.description for state in model.states):
            f.write("\n")

        # Write bodies for states
        for state in model.states:
            f.write(f"# {state.name} state\n")
            # Write body function if it exists
            if state.body:
                # Check if the body has a messages attribute
                
                if hasattr(state.body, 'actions') and state.body.actions:
                    if isinstance(state.body.actions[0], AgentReply):
                        f.write(f"{state.name}_body = Body('{state.name}_body')\n")
                        for action in state.body.actions:
                            msg = action.message.replace('\\', '\\\\').replace("'", "\\'")
                            f.write(f"{state.name}_body.add_action(AgentReply('{msg}'))\n")
                        f.write("\n")
                        f.write(f"{state.name}.set_body({state.name}_body)\n")
                    elif isinstance(state.body.actions[0], LLMReply):
                        f.write(f"{state.name}_body = Body('{state.name}_body')\n")
                        f.write(f"{state.name}_body.add_action(LLMReply())\n")
                        f.write(f"{state.name}.set_body({state.name}_body)\n")
                    elif isinstance(state.body.actions[0], CustomCodeAction):
                        action = state.body.actions[0]
                        f.write(f"{action.to_code()}\n")
                        function_match = search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', action.code)
                        f.write(f"CustomCodeAction_{state.name} = CustomCodeAction(callable={function_match.group(1)})\n")
                        f.write(f"{state.name}_body = Body('{state.name}_body')\n")
                        f.write(f"{state.name}_body.add_action(CustomCodeAction_{state.name})\n")
                        f.write(f"{state.name}.set_body({state.name}_body)\n")
                            
                        
                
            # Write fallback body function if it exists
            if state.fallback_body:
                # Check if the fallback body has a messages attribute
                if hasattr(state.fallback_body, 'actions') and state.fallback_body.actions:
                    if isinstance(state.fallback_body.actions[0], AgentReply):
                        f.write(f"{state.name}_fallback_body = Body('{state.name}_fallback_body')\n")
                        for action in state.fallback_body.actions:
                            msg = action.message.replace('\\', '\\\\').replace("'", "\\'")
                            f.write(f"{state.name}_fallback_body.add_action(AgentReply('{msg}'))\n")
                        f.write("\n")
                        f.write(f"{state.name}.set_fallback_body({state.name}_fallback_body)\n")
                    elif isinstance(state.fallback_body.actions[0], LLMReply):
                        f.write(f"{state.name}_fallback_body = Body('{state.name}_fallback_body')\n")
                        f.write(f"{state.name}_fallback_body.add_action(LLMReply())\n")
                        f.write(f"{state.name}.set_fallback_body({state.name}_fallback_body)\n")
                    elif isinstance(state.fallback_body.actions[0], CustomCodeAction):
                        action = state.fallback_body.actions[0]
                        f.write(f"{action.to_code()}\n")
                        function_match = search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', action.code)
                        f.write(f"CustomCodeAction_{state.name}_fallback = CustomCodeAction(callable={function_match.group(1)})\n")
                        f.write(f"{state.name}_fallback_body = Body('{state.name}_fallback_body')\n")
                        f.write(f"{state.name}_fallback_body.add_action(CustomCodeAction_{state.name}_fallback)\n")
                        f.write(f"{state.name}.set_fallback_body({state.name}_fallback_body)\n")
            
            # Write transitions
            for transition in state.transitions:
                dest_state = transition.dest
                
                # Handle different types of transitions
                if transition.conditions:
                    # Check the type of condition
                    condition_class = transition.conditions.__class__.__name__
                    
                    if condition_class == "IntentMatcher":
                        intent_name = transition.conditions.intent.name
                        if intent_name == "fallback_intent":
                            f.write(f"{state.name}.when_no_intent_matched().go_to({dest_state.name})\n")
                        else:
                            f.write(f"{state.name}.when_intent_matched({intent_name}).go_to({dest_state.name})\n")
                    
                    elif condition_class == "VariableOperationMatcher":
                        var_name = transition.conditions.var_name
                        op_name = transition.conditions.operation.__name__
                        target = transition.conditions.target
                        f.write(f"{state.name}.when_variable_matches_operation(\n")
                        f.write(f"    var_name='{var_name}',\n")
                        f.write(f"    operation=operator.{op_name},\n")
                        f.write(f"    target='{target}'\n")
                        f.write(f").go_to({dest_state.name})\n")
                    
                    elif condition_class == "FileTypeMatcher":
                        file_type = transition.conditions.allowed_types
                        f.write(f"{state.name}.when_file_received('{file_type}').go_to({dest_state.name})\n")
                    
                    elif condition_class == "Auto":
                        f.write(f"{state.name}.go_to({dest_state.name})\n")
                    
                    else:
                        # Default case for custom conditions
                        f.write(f"# Custom transition from {state.name} to {dest_state.name}\n")
                        f.write(f"{state.name}.when_no_intent_matched().go_to({dest_state.name})\n")
                
                else:
                    # If no conditions, create a simple transition
                    f.write(f"{state.name}.go_to({dest_state.name})\n")
                
                f.write("\n")

    print(f"Agent model saved to {file_path}")
