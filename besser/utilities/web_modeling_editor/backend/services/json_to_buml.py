import re
import json

from besser.BUML.metamodel.structural import DomainModel, Class, Enumeration, Property, Method, BinaryAssociation, \
    Generalization, PrimitiveDataType, EnumerationLiteral, Multiplicity, UNLIMITED_MAX_MULTIPLICITY, Constraint, AnyType
from besser.utilities.web_modeling_editor.backend.constants.constants import VISIBILITY_MAP, VALID_PRIMITIVE_TYPES
from fastapi import HTTPException


def parse_attribute(attribute_name, domain_model=None):
    """Parse an attribute string to extract visibility, name, and type, removing any colons."""
    # Split the string by colon first to separate name and type
    name_type_parts = attribute_name.split(":")

    if len(name_type_parts) > 1:
        name_part = name_type_parts[0].strip()
        type_part = name_type_parts[1].strip()

        # Check for visibility symbol at start of name
        if name_part[0] in VISIBILITY_MAP:
            visibility = VISIBILITY_MAP[name_part[0]]
            name = name_part[1:].strip()
        else:
            # Existing split logic for space-separated visibility
            name_parts = name_part.split()
            if len(name_parts) > 1:
                visibility_symbol = name_parts[0] if name_parts[0] in VISIBILITY_MAP else "+"
                visibility = VISIBILITY_MAP.get(visibility_symbol, "public")
                name = name_parts[1]
            else:
                visibility = "public"
                name = name_parts[0]

        # Handle the type
        if domain_model and any(isinstance(t, (Enumeration, Class)) and t.name == type_part for t in domain_model.types):
            attr_type = type_part
        else:
            attr_type = VALID_PRIMITIVE_TYPES.get(type_part.lower(), None)
            if attr_type is None:
                raise ValueError(f"Invalid type: {type_part}")
    else:
        # Handle case without type specification
        parts = attribute_name.split()

        if len(parts) == 1:
            part = parts[0].strip()
            if part and part[0] in VISIBILITY_MAP:
                visibility = VISIBILITY_MAP[part[0]]
                name = part[1:].strip()
                attr_type = "str"
            else:
                visibility = "public"
                name = part
                attr_type = "str"
        else:
            visibility_symbol = parts[0] if parts[0] in VISIBILITY_MAP else "+"
            visibility = VISIBILITY_MAP.get(visibility_symbol, "public")
            name = parts[1]
            attr_type = "str"
    if not name:  # Skip if name is empty
        return None, None, None
    return visibility, name, attr_type

def parse_method(method_str, domain_model=None):
    """
    Parse a method string to extract visibility, name, parameters, and return type.
    Examples:
    "+ notify(sms: str = 'message')" -> ("public", "notify", [{"name": "sms", "type": "str", "default": "message"}], None)
    "- findBook(title: str): Book" -> ("private", "findBook", [{"name": "title", "type": "str"}], "Book")
    "validate()" -> ("public", "validate", [], None)
    """

    # Default values
    visibility = "public"
    parameters = []
    return_type = None

    # Check if this is actually a method (contains parentheses)
    if '(' not in method_str:
        return visibility, method_str, parameters, return_type

    # Extract visibility if present
    method_str = method_str.strip()
    if method_str.startswith(tuple(VISIBILITY_MAP.keys())):
        visibility = VISIBILITY_MAP.get(method_str[0], "public")
        method_str = method_str[2:].strip()

    # Parse method using regex
    pattern = r"([^(]+)\((.*?)\)(?:\s*:\s*(.+))?"
    match = re.match(pattern, method_str)

    if not match:
        return visibility, method_str.replace("()", ""), parameters, return_type

    method_name, params_str, return_type = match.groups()
    method_name = method_name.strip()

    # Parse parameters if present
    if params_str:
        # Handle nested parentheses in default values
        param_list = []
        current_param = []
        paren_count = 0

        for char in params_str + ',':
            if char == '(' and paren_count >= 0:
                paren_count += 1
                current_param.append(char)
            elif char == ')' and paren_count > 0:
                paren_count -= 1
                current_param.append(char)
            elif char == ',' and paren_count == 0:
                param_list.append(''.join(current_param).strip())
                current_param = []
            else:
                current_param.append(char)

        for param in param_list:
            if not param:
                continue

            param_dict = {'name': param, 'type': 'any'}

            # Handle parameter with default value
            if '=' in param:
                param_parts = param.split('=', 1)
                param_name_type = param_parts[0].strip()
                default_value = param_parts[1].strip().strip('"\'')

                if ':' in param_name_type:
                    param_name, param_type = [p.strip() for p in param_name_type.split(':')]
                    param_dict.update({
                        'name': param_name,
                        'type': VALID_PRIMITIVE_TYPES.get(param_type.lower(), param_type),
                        'default': default_value
                    })
                else:
                    param_dict.update({
                        'name': param_name_type,
                        'default': default_value
                    })

            # Handle parameter with type annotation
            elif ':' in param:
                param_name, param_type = [p.strip() for p in param.split(':')]

                # Handle the type
                if domain_model and any(isinstance(t, (Enumeration, Class)) and t.name == param_type for t in domain_model.types):
                    type_param = param_type
                else:
                    type_param = VALID_PRIMITIVE_TYPES.get(param_type.lower(), None)
                    if type_param is None:
                        raise ValueError(f"Invalid type '{param_type}' for the parameter '{param_name}'")

                param_dict.update({
                    'name': param_name,
                    'type': type_param
                })
            else:
                param_dict['name'] = param.strip()

            parameters.append(param_dict)

    # Clean up return type if present
    if return_type:
        return_type = return_type.strip()
        # Keep the original return type if it's not a primitive type
        if domain_model and any(isinstance(t, (Enumeration, Class)) and t.name == return_type for t in domain_model.types):
            type_return = return_type
        else:
            type_return = VALID_PRIMITIVE_TYPES.get(return_type.lower(), None)
            if type_return is None:
                raise ValueError(f"Invalid return type '{return_type}' for the method '{method_name}'")

    return visibility, method_name, parameters, return_type

def parse_multiplicity(multiplicity_str):
    """Parse a multiplicity string and return a Multiplicity object with defaults."""
    if not multiplicity_str:
        return Multiplicity(min_multiplicity=1, max_multiplicity=1)

    # Handle single "*" case
    if multiplicity_str == "*":
        return Multiplicity(min_multiplicity=0, max_multiplicity=UNLIMITED_MAX_MULTIPLICITY)

    parts = multiplicity_str.split("..")
    try:
        min_multiplicity = int(parts[0]) if parts[0] and parts[0] != "*" else 0
        max_multiplicity = (
            UNLIMITED_MAX_MULTIPLICITY if len(parts) > 1 and (not parts[1] or parts[1] == "*")
            else int(parts[1]) if len(parts) > 1
            else min_multiplicity
        )
    except ValueError:
        # If parsing fails, return default multiplicity of 1..1
        return Multiplicity(min_multiplicity=1, max_multiplicity=1)

    return Multiplicity(min_multiplicity=min_multiplicity, max_multiplicity=max_multiplicity)

def process_ocl_constraints(ocl_text: str, domain_model: DomainModel, counter: int) -> tuple[list, list]:
    """Process OCL constraints and convert them to BUML Constraint objects."""
    if not ocl_text:
        return [], []

    constraints = []
    warnings = []
    lines = re.split(r'[,]', ocl_text)
    constraint_count = 1

    domain_classes = {cls.name.lower(): cls for cls in domain_model.types}

    for line in lines:

        line = line.strip().replace('\n', '')
        if not line or not line.lower().startswith('context'):
            continue

        # Extract context class name
        parts = line.split()
        if len(parts) < 4:  # Minimum: "context ClassName inv name:"
            continue

        context_class_name = parts[1]
        context_class = domain_classes.get(context_class_name.lower())

        if not context_class:
            warning_msg = f"Warning: Context class {context_class_name} not found"
            warnings.append(warning_msg)
            continue

        constraint_name = f"constraint_{context_class_name}_{counter}_{constraint_count}"
        constraint_count += 1

        constraints.append(
            Constraint(
                name=constraint_name,
                context=context_class,
                expression=line,
                language="OCL"
            )
        )

    return constraints, warnings

def process_class_diagram(json_data):
    """Process Class Diagram specific elements."""
    title = json_data.get('diagramTitle', '')
    if ' ' in title:
        title = title.replace(' ', '_')

    domain_model = DomainModel(title)
    # Get elements and OCL constraints from the JSON data
    elements = json_data.get('elements', {}).get('elements', {})
    relationships = json_data.get('elements', {}).get('relationships', {})

    # FIRST PASS: Process all type declarations (enumerations and classes)
    # 1. First process enumerations
    for element_id, element in elements.items():
        if element.get("type") == "Enumeration":
            element_name = element.get("name", "").strip()
            if not element_name or any(char.isspace() for char in element_name):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid enumeration name: '{element_name}'. Names cannot contain whitespace or be empty."
                )
            literals = set()
            for literal_id in element.get("attributes", []):
                literal = elements.get(literal_id)
                if literal:
                    literal_obj = EnumerationLiteral(name=literal.get("name", ""))
                    literals.add(literal_obj)
            enum = Enumeration(name=element_name, literals=literals)
            domain_model.types.add(enum)
    
    # 2. Then create all class structures without attributes or methods
    for element_id, element in elements.items():
        if element.get("type") in ["Class", "AbstractClass"]:
            class_name = element.get("name", "").strip()
            if not class_name or any(char.isspace() for char in class_name):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid class name: '{class_name}'. Names cannot contain whitespace or be empty."
                )
            is_abstract = element.get("type") == "AbstractClass"
            try:
                cls = Class(name=class_name, is_abstract=is_abstract)
                domain_model.types.add(cls)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

    # SECOND PASS: Now add attributes and methods to classes
    for element_id, element in elements.items():
        if element.get("type") in ["Class", "AbstractClass"]:
            class_name = element.get("name", "").strip()
            cls = domain_model.get_class_by_name(class_name)
            
            if not cls:
                continue  # Skip if class wasn't created successfully in first pass
                
            # Add attributes
            attribute_names = set()
            for attr_id in element.get("attributes", []):
                attr = elements.get(attr_id)
                if attr:
                    visibility, name, attr_type = parse_attribute(attr.get("name", ""), domain_model)
                    if name is None:  # Skip if no name was returned
                        continue
                    if name in attribute_names:
                        raise HTTPException(status_code=400, detail=f"Duplicate attribute name '{name}' found in class '{class_name}'")
                    attribute_names.add(name)
                    
                    # Find the type in the domain model
                    type_obj = None
                    for t in domain_model.types:
                        if isinstance(t, (Enumeration, Class)) and t.name == attr_type:
                            type_obj = t
                            break
                    
                    if type_obj:
                        property_ = Property(name=name, type=type_obj, visibility=visibility)
                    else:
                        property_ = Property(name=name, type=PrimitiveDataType(attr_type), visibility=visibility)
                    cls.attributes.add(property_)

            # Add methods
            for method_id in element.get("methods", []):
                method = elements.get(method_id)
                if method:
                    visibility, name, parameters, return_type = parse_method(method.get("name", ""), domain_model)

                    # Create method parameters
                    method_params = []
                    for param in parameters:
                        param_type_obj = None
                        param_type_name = param['type']
                        
                        # Try to find parameter type in domain model
                        for t in domain_model.types:
                            if isinstance(t, (Enumeration, Class)) and t.name == param_type_name:
                                param_type_obj = t
                                break
                                
                        if not param_type_obj:
                            param_type_obj = PrimitiveDataType(param_type_name)
                            
                        param_obj = Property(
                            name=param['name'],
                            type=param_type_obj,
                            visibility='public'
                        )
                        if 'default' in param:
                            param_obj.default_value = param['default']
                        method_params.append(param_obj)

                    # Create method with parameters and return type
                    method_obj = Method(
                        name=name,
                        visibility=visibility,
                        parameters=method_params
                    )
                    
                    # Handle return type
                    if return_type:
                        return_type_obj = None
                        # Find return type in domain model
                        for t in domain_model.types:
                            if isinstance(t, (Enumeration, Class)) and t.name == return_type:
                                return_type_obj = t
                                break
                                
                        if return_type_obj:
                            method_obj.type = return_type_obj
                        else:
                            method_obj.type = PrimitiveDataType(return_type)
                    
                    cls.methods.add(method_obj)

    # Processing relationships (Associations, Generalizations, and Compositions)
    for rel_id, relationship in relationships.items():
        #print(f"Processing relationship ID: {rel_id} with data: {relationship}")

        rel_type = relationship.get("type")
        source = relationship.get("source")
        target = relationship.get("target")

        if not rel_type or not source or not target:
            print(f"Skipping relationship {rel_id} due to missing data.")
            continue

        # Skip OCL links
        if rel_type == "ClassOCLLink":
            continue

        # Retrieve source and target elements
        source_element = elements.get(source.get("element"))
        target_element = elements.get(target.get("element"))

        if not source_element or not target_element:
            print(f"Skipping relationship {rel_id} due to missing elements.")
            continue

        source_class = domain_model.get_class_by_name(source_element.get("name", ""))
        target_class = domain_model.get_class_by_name(target_element.get("name", ""))

        if not source_class or not target_class:
            print(f"Skipping relationship {rel_id} because classes are missing in the domain model.")
            continue

        # Handle each type of relationship
        if rel_type == "ClassBidirectional" or rel_type == "ClassUnidirectional" or rel_type == "ClassComposition" or rel_type == "ClassAggregation" :
            is_composite = rel_type == "ClassComposition"
            source_navigable = rel_type != "ClassUnidirectional"
            target_navigable = True

            source_multiplicity = parse_multiplicity(source.get("multiplicity", "1"))
            target_multiplicity = parse_multiplicity(target.get("multiplicity", "1"))

            source_role = source.get("role")
            if not source_role:
                source_role = source_class.name.lower()
                existing_roles = {end.name for assoc in domain_model.associations for end in assoc.ends}

                if source_role in existing_roles:
                    counter = 1
                    while f"{source_role}_{counter}" in existing_roles:
                        counter += 1
                    source_role = f"{source_role}_{counter}"


            source_property = Property(
                name=source_role,
                type=source_class,
                multiplicity=source_multiplicity,
                is_navigable=source_navigable
            )

            target_role = target.get("role")
            if not target_role:
                target_role = target_class.name.lower()
                existing_roles = {end.name for assoc in domain_model.associations for end in assoc.ends}

                if target_role in existing_roles:
                    counter = 1
                    while f"{target_role}_{counter}" in existing_roles:
                        counter += 1
                    target_role = f"{target_role}_{counter}"


            target_property = Property(
                name=target_role,
                type=target_class,
                multiplicity=target_multiplicity,
                is_navigable=target_navigable,
                is_composite=is_composite
            )

            association_name = relationship.get("name") or f"{source_class.name}_{target_class.name}"

            # Check if association name already exists and add increment if needed
            if association_name in [assoc.name for assoc in domain_model.associations]:
                counter = 1
                while f"{association_name}_{counter}" in [assoc.name for assoc in domain_model.associations]:
                    counter += 1
                association_name = f"{association_name}_{counter}"

            association = BinaryAssociation(
                name=association_name,
                ends={source_property, target_property}
            )
            domain_model.associations.add(association)

        elif rel_type == "ClassInheritance":
            generalization = Generalization(general=target_class, specific=source_class)
            domain_model.generalizations.add(generalization)

    # Process OCL constraints
    all_constraints = set()
    all_warnings = []
    constraint_counter = 0
    for element_id, element in elements.items():
        if element.get("type") in ["ClassOCLConstraint"]:
            ocl = element.get("constraint")
            if ocl:
                try:
                    new_constraints, warnings = process_ocl_constraints(ocl, domain_model, constraint_counter)
                    all_constraints.update(new_constraints)
                    all_warnings.extend(warnings)
                    constraint_counter += 1
                except Exception as e:
                    error_msg = f"Error processing OCL constraint for element {element_id}: {e}"
                    all_warnings.append(error_msg)
                    continue

    # Attach warnings to domain model for later use
    domain_model.ocl_warnings = all_warnings
    domain_model.constraints = all_constraints

    return domain_model

def process_state_machine(json_data):
    """Process State Machine Diagram specific elements and return Python code as string."""
    code_lines = []
    code_lines.append("import datetime")
    code_lines.append("from besser.BUML.metamodel.state_machine.state_machine import StateMachine, Session, Body, Event\n")
    sm_name = json_data.get("name", "Generated_State_Machine")
    code_lines.append(f"sm = StateMachine(name='{sm_name}')\n")

    elements = json_data.get("elements", {})
    relationships = json_data.get("relationships", {})

    # Track states by ID for later reference
    states_by_id = {}
    body_names = set()
    event_names = set()

    # Collect all body and event names first
    for element in elements.values():
        if element.get("type") == "StateBody":
            body_names.add(element.get("name"))
        elif element.get("type") == "StateFallbackBody":
            body_names.add(element.get("name"))

    # Collect event names from transitions
    for rel in relationships.values():
        if rel.get("type") == "StateTransition" and rel.get("name"):
            event_names.add(rel.get("name"))

    # Write function definitions first
    for element in elements.values():
        if element.get("type") == "StateCodeBlock":
            name = element.get("name", "")
            code_content = element.get("code", {})
            
            # If name is empty, try to extract function name from code content
            if not name:
                # Look for "def function_name(" pattern in the code
                function_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_content)
                if function_match:
                    name = function_match.group(1)
                    
            # Clean up the code content by removing extra newlines
            cleaned_code = "\n".join(line for line in code_content.splitlines() if line.strip())
            # Write the function definition with its code content
            code_lines.append(cleaned_code)  # Write the actual function code
            code_lines.append("")  # Add single blank line after function

            if name in body_names:
                code_lines.append(f"{name} = Body(name='{name}', callable={name})")
            if name in event_names:
                code_lines.append(f"{name} = Event(name='{name}', callable={name})")
            code_lines.append("")  # Add blank line after Body/Event creation

    # Create states
    for element_id, element in elements.items():
        if element.get("type") == "State":
            is_initial = False
            for rel in relationships.values():
                if (rel.get("type") == "StateTransition" and
                    rel.get("target", {}).get("element") == element_id and
                    elements.get(rel.get("source", {}).get("element", ""), {}).get("type") == "StateInitialNode"):
                    is_initial = True
                    break

            state_name = element.get("name", "")
            code_lines.append(f"{state_name}_state = sm.new_state(name='{state_name}', initial={str(is_initial)})")
            states_by_id[element_id] = state_name
    code_lines.append("")

    # Assign bodies to states
    for element_id, element in elements.items():
        if element.get("type") == "State":
            state_name = element.get("name", "")
            for body_id in element.get("bodies", []):
                body_element = elements.get(body_id)
                if body_element:
                    body_name = body_element.get("name")
                    if body_name in body_names:
                        code_lines.append(f"{state_name}_state.set_body(body={body_name})")

            for fallback_id in element.get("fallbackBodies", []):
                fallback_element = elements.get(fallback_id)
                if fallback_element:
                    fallback_name = fallback_element.get("name")
                    if fallback_name in body_names:
                        code_lines.append(f"{state_name}_state.set_fallback_body({fallback_name})")
    code_lines.append("")

    # Write transitions
    for relationship in relationships.values():
        if relationship.get("type") == "StateTransition":
            source_id = relationship.get("source", {}).get("element")
            target_id = relationship.get("target", {}).get("element")

            if elements.get(source_id, {}).get("type") == "StateInitialNode":
                continue

            source_name = states_by_id.get(source_id)
            target_name = states_by_id.get(target_id)

            if source_name and target_name:
                event_name = relationship.get("name", "")
                params = relationship.get("params")

                if event_name:
                    event_params = f"event_params={{ {params} }}" if params else "event_params={}"
                    code_lines.append(f"{source_name}_state.when_event_go_to(")
                    code_lines.append(f"    event={event_name},")
                    code_lines.append(f"    dest={target_name}_state,")
                    code_lines.append(f"    {event_params}")
                    code_lines.append(")")

    return "\n".join(code_lines)


def process_agent_diagram(json_data):
    """Process Agent Diagram specific elements and return Python code as string."""
    code_lines = []
    code_lines.append("import datetime")
    code_lines.append("from besser.BUML.metamodel.state_machine.state_machine import StateMachine, Session, Body, Event")
    code_lines.append("from besser.BUML.metamodel.state_machine.agent import Agent, AgentSession")
    code_lines.append("from besser.BUML.metamodel.state_machine.state_machine import Body, ConfigProperty")
    # code_lines.append("from besser.agent.nlp.llm.llm_openai_api import LLMOpenAI\n") wrong library, i should import from besser not baf
    sm_name = json_data.get("name", "Generated_State_Machine")
    code_lines.append(f"agent = Agent('{sm_name}')\n")
    code_lines.append("agent.add_property(ConfigProperty('websocket_platform', 'websocket.host', 'localhost'))\n")
    code_lines.append("agent.add_property(ConfigProperty('websocket_platform', 'websocket.port', 8765))\n")
    code_lines.append("agent.add_property(ConfigProperty('websocket_platform', 'streamlit.host', 'localhost'))\n")
    code_lines.append("agent.add_property(ConfigProperty('websocket_platform', 'streamlit.port', 5000))\n")
    code_lines.append("agent.add_property(ConfigProperty('nlp', 'nlp.language', 'en'))\n")
    code_lines.append("agent.add_property(ConfigProperty('nlp', 'nlp.region', 'US'))\n")
    code_lines.append("agent.add_property(ConfigProperty('nlp', 'nlp.timezone', 'Europe/Madrid'))\n")
    code_lines.append("agent.add_property(ConfigProperty('nlp', 'nlp.pre_processing', True))\n")
    code_lines.append("agent.add_property(ConfigProperty('nlp', 'nlp.intent_threshold', 0.4))\n")

    code_lines.append("# INTENTS\n")
    elements = json_data.get("elements", {})
    relationships = json_data.get("relationships", {})

    # Track states by ID for later reference
    states_by_id = {}
    body_names = set()
    event_names = set()
    intents = {}
    # Collect all body, event and intents first
    for element in elements.values():
        if element.get("type") == "AgentStateBody":
            body_names.add(element.get("name"))
        elif element.get("type") == "AgentStateFallbackBody":
            body_names.add(element.get("name"))
        elif element.get("type") == "Intent":
            intents[element.get("name")] = []
            for intent_body in element.get("bodies"):
                intents[element.get("name")].append(elements.get(intent_body).get("name"))
    # Collect event names from transitions
    for rel in relationships.values():
        if rel.get("type") == "StateTransition" and rel.get("name"):
            event_names.add(rel.get("name"))
    # Write intents first
    for intent in intents.keys():
        intent_name = intent
        intent_values = intents[intent]
        code_lines.append(f"{intent_name} = agent.new_intent('{intent_name}', [")
        for value in intent_values:
            code_lines.append(f"    '{value}',")
        code_lines.append("])\n")
    # Write function definitions first
    
    print("oueoueoueu")
    try:
        if '"replyType": "llm"' in json.dumps(json_data):
            print("ww")
            # code_lines.append("llm = LLMOpenAI(agent=agent, name='gpt-4o-mini', parameters={})\n")
    except Exception as e:
        print(f"Error: {e}")
    
    for element in elements.values():
        if element.get("type") == "AgentState":
            name = element.get("name")  # throw error if no name
            if element.get("bodies") != []:
                bodyCode = [f"def {name}_body(session: AgentSession):"]
                for body in element.get("bodies"):
                    if elements.get(body).get("replyType") == "text":
                        bodyCode.append(f"    session.reply('{elements.get(body).get('name')}')")
                    elif elements.get(body).get("replyType") == "llm":
                        print("not supported yet")
                        # bodyCode.append(f"    session.reply(llm.predict(session.event.message))") not supported by besser yet
                    elif elements.get(body).get("replyType") == "code":
                        code_lines.append(elements.get(body).get('name').strip())
                        # Extract the function name from the code
                        function_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', elements.get(body).get('name'))
                        if function_match:
                            function_name = function_match.group(1)
                            elements.get(body)["name"] = function_name
                        code_lines.append("")  # Add single blank line after function
                        bodyCode = ""
                code_lines.append("\n".join(bodyCode))
                code_lines.append("")  # Add single blank line after function
            if element.get("fallbackBodies") != []:
                fallbackBodyCode = [f"def {name}_fallback_body(session: AgentSession):"]
                for fallbackBody in element.get("fallbackBodies"):
                    if elements.get(fallbackBody).get("replyType") == "text":
                        fallbackBodyCode.append(f"    session.reply('{elements.get(fallbackBody).get('name')}')")
                    elif elements.get(fallbackBody).get("replyType") == "code":
                        print("todo")
                code_lines.append("\n".join(fallbackBodyCode))
                code_lines.append("")  # Add single blank line after function   
    """
    for element in elements.values():
        if element.get("type") == "StateCodeBlock":
            name = element.get("name", "")
            code_content = element.get("code", {})
            
            # If name is empty, try to extract function name from code content
            if not name:
                # Look for "def function_name(" pattern in the code
                function_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_content)
                if function_match:
                    name = function_match.group(1)
                    
            # Clean up the code content by removing extra newlines
            cleaned_code = "\n".join(line for line in code_content.splitlines() if line.strip())
            # Write the function definition with its code content
            code_lines.append(cleaned_code)  # Write the actual function code
            code_lines.append("")  # Add single blank line after function

            if name in body_names:
                code_lines.append(f"{name} = Body(name='{name}', callable={name})")
            if name in event_names:
                code_lines.append(f"{name} = Event(name='{name}', callable={name})")
            code_lines.append("")  # Add blank line after Body/Event creation
            """
    # Create states
    print("keke")
    for element_id, element in elements.items():
        if element.get("type") == "AgentState":
            is_initial = False
            for rel in relationships.values():
                if (rel.get("type") == "StateTransition" and
                    rel.get("target", {}).get("element") == element_id and
                    elements.get(rel.get("source", {}).get("element", ""), {}).get("type") == "StateInitialNode"):
                    is_initial = True
                    break

            state_name = element.get("name", "")
            code_lines.append(f"{state_name}_state = agent.new_state(name='{state_name}', initial={str(is_initial)})")
            states_by_id[element_id] = state_name
    code_lines.append("")
    # Assign bodies to states
    try:
        for element_id, element in elements.items():
            if element.get("type") == "AgentState":
                state_name = element.get("name", "")
                if element.get("bodies") != []:
                    for body in element.get("bodies"):
                        if elements.get(body).get("replyType") == "code":
                            # Extract the function name from the code
                            code_lines.append(f"{state_name}_state.set_body(Body('{elements.get(body).get('name')}', {elements.get(body).get('name')}))")
                        else:
                            code_lines.append(f"{state_name}_state.set_body(Body('{state_name}_body', {state_name}_body))")
                if element.get("fallbackBodies") != []:
                    for body in element.get("fallbackBodies"):
                        if elements.get(body).get("replyType") == "code":
                            # Extract the function name from the code
                            code_lines.append(f"{state_name}_state.set_fallback_body(Body('{elements.get(body).get('name')}', {elements.get(body).get('name')}))")
                        else:
                            code_lines.append(f"{state_name}_state.set_fallback_body(Body('{state_name}_fallback_body', {state_name}_fallback_body))")
                    
        code_lines.append("")
    except Exception as e:
        print(f"Error: {e}")
    print("lee")
    # Write transitions
    for relationship in relationships.values():
        if relationship.get("type") == "StateTransition":
            source_id = relationship.get("source", {}).get("element")
            target_id = relationship.get("target", {}).get("element")

            if elements.get(source_id, {}).get("type") == "StateInitialNode":
                continue

            source_name = states_by_id.get(source_id)
            target_name = states_by_id.get(target_id)

            if source_name and target_name:
                event_name = relationship.get("name", "")
                params = relationship.get("params")

                if event_name:
                    code_lines.append(f"{source_name}_state.when_intent_matched_go_to(")
                    code_lines.append(f"    {event_name},")
                    code_lines.append(f"    {target_name}_state")
                    code_lines.append(")")
                else:
                    code_lines.append(f"{source_name}_state.when_no_intent_matched_go_to({target_name}_state)")
    return "\n".join(code_lines)