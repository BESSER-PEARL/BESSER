import os
import tempfile
import uuid
import shutil
from besser.BUML.metamodel.structural.structural import DomainModel, AssociationClass, Metadata
from besser.BUML.metamodel.object.object import ObjectModel
from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.state_machine.state_machine import StateMachine
from besser.BUML.metamodel.state_machine.agent import Agent, Intent
from besser.BUML.metamodel.state_machine.state_machine import Body
from besser.utilities import sort_by_timestamp as sort

PRIMITIVE_TYPE_MAPPING = {
    'str': 'StringType',
    'string': 'StringType', 
    'int': 'IntegerType',
    'integer': 'IntegerType',
    'float': 'FloatType',
    'bool': 'BooleanType',
    'boolean': 'BooleanType',
    'time': 'TimeType',
    'date': 'DateType',
    'datetime': 'DateTimeType',
    'timedelta': 'TimeDeltaType',
    'any': 'AnyType'
}

# Reserved names that need special handling
RESERVED_NAMES = ['Class', 'Property', 'Method', 'Parameter', 'Enumeration']

def safe_class_name(name):
    """
    Add a suffix to class names that match reserved keywords or BUML metaclass names.
    If the name already ends with an underscore and would conflict with a reserved name,
    add a numeric suffix instead.
    
    Parameters:
    name (str): The original class name
    
    Returns:
    str: A safe variable name for the class
    """
    if not name:
        return "unnamed_class"
        
    if name.endswith('_'):
        base_name = name[:-1]
        if base_name in RESERVED_NAMES:
            return f"{name}var"
        return name
    elif name in RESERVED_NAMES:
        return f"{name}_"
    else:
        return name

def domain_model_to_code(model: DomainModel, file_path: str, objectmodel: ObjectModel = None):
    """
    Generates Python code for a B-UML model and writes it to a specified file.

    Parameters:
        model (DomainModel): The B-UML model object containing classes, enumerations, 
            associations, and generalizations.
        file_path (str): The path where the generated code will be saved.
        objectmodel (ObjectModel, optional): The B-UML object model to include in the same file.

    Outputs:
        - A Python file containing the base code representation of the B-UML domain model
          and optionally the object model instances.
    """
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not file_path.endswith('.py'):
        file_path += '.py'

    with open(file_path, 'w', encoding='utf-8') as f:
        # Write imports
        f.write("####################\n")
        f.write("# STRUCTURAL MODEL #\n")
        f.write("####################\n\n")
        f.write("from besser.BUML.metamodel.structural import (\n")
        f.write("    Class, Property, Method, Parameter,\n")
        f.write("    BinaryAssociation, Generalization, DomainModel,\n")
        f.write("    Enumeration, EnumerationLiteral, Multiplicity,\n")
        f.write("    StringType, IntegerType, FloatType, BooleanType,\n")
        f.write("    TimeType, DateType, DateTimeType, TimeDeltaType,\n")
        f.write("    AnyType, Constraint, AssociationClass, Metadata\n")
        f.write(")\n")

        # Add object model imports if object model is provided
        if objectmodel:
            f.write("from besser.BUML.metamodel.object import ObjectModel\n")
            f.write("import datetime\n")

        f.write("\n")

        # Write enumerations only if they exist
        enumerations = sort(model.get_enumerations())
        if enumerations:
            f.write("# Enumerations\n")
            for enum in enumerations:
                enum_var_name = safe_class_name(enum.name)
                f.write(f"{enum_var_name}: Enumeration = Enumeration(\n")
                f.write(f"    name=\"{enum.name}\",\n")
                literals_str = ",\n\t\t\t".join([f"EnumerationLiteral(name=\"{lit.name}\")" for lit in sort(enum.literals)])
                f.write(
                    f"    literals={{\n"
                    f"            {literals_str}"
                    f"\n    }}")
                f.write("\n)\n\n")

        # Separate regular classes and association classes
        association_classes = []
        regular_classes = []
        for cls in sort(model.get_classes()):
            if isinstance(cls, AssociationClass):
                association_classes.append(cls)
            else:
                regular_classes.append(cls)

        # Write regular classes
        f.write("# Classes\n")
        for cls in regular_classes:
            cls_var_name = safe_class_name(cls.name)

            # Build class creation parameters
            class_params = [f'name="{cls.name}"']

            if cls.is_abstract:
                class_params.append('is_abstract=True')

            # Add metadata if it exists
            if hasattr(cls, 'metadata') and cls.metadata:
                metadata_params = []
                if cls.metadata.description:
                    # Escape backslashes first, then quotes, then newlines
                    desc = cls.metadata.description.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                    metadata_params.append(f'description="{desc}"')
                if cls.metadata.uri:
                    metadata_params.append(f'uri="{cls.metadata.uri}"')
                if cls.metadata.icon:
                    metadata_params.append(f'icon="{cls.metadata.icon}"')

                if metadata_params:
                    metadata_str = f"Metadata({', '.join(metadata_params)})"
                    class_params.append(f'metadata={metadata_str}')

            f.write(f"{cls_var_name} = Class({', '.join(class_params)})\n")
        f.write("\n")

        # Write class members for regular classes
        for cls in regular_classes:
            cls_var_name = safe_class_name(cls.name)
            f.write(f"# {cls.name} class attributes and methods\n")

            # Write attributes
            for attr in sort(cls.attributes):
                attr_type = PRIMITIVE_TYPE_MAPPING.get(attr.type.name, safe_class_name(attr.type.name))
                visibility_str = f', visibility="{attr.visibility}"' if attr.visibility != "public" else ""
                f.write(f"{cls_var_name}_{attr.name}: Property = Property(name=\"{attr.name}\", "
                       f"type={attr_type}{visibility_str})\n")

            # Write methods
            for method in sort(cls.methods):
                method_type = PRIMITIVE_TYPE_MAPPING.get(method.type.name, safe_class_name(method.type.name)) if method.type else None
                visibility_str = f', visibility="{method.visibility}"' if method.visibility != "public" else ""

                # Build parameters dictionary
                params = {}
                if sort(method.parameters):
                    for param in method.parameters:
                        param_type = PRIMITIVE_TYPE_MAPPING.get(param.type.name, safe_class_name(param.type.name))
                        default_str = f", default_value='{param.default_value}'" if hasattr(param, 'default_value') and param.default_value is not None else ""
                        params[param.name] = f"Parameter(name='{param.name}', type={param_type}{default_str})"

                params_str = "{" + ", ".join(f"{param}" for name, param in params.items()) + "}"

                if method_type:
                    f.write(f"{cls_var_name}_m_{method.name}: Method = Method(name=\"{method.name}\""
                           f"{visibility_str}, parameters={params_str}, type={method_type})\n")
                else:
                    f.write(f"{cls_var_name}_m_{method.name}: Method = Method(name=\"{method.name}\""
                           f"{visibility_str}, parameters={params_str})\n")

            # Write assignments
            if sort(cls.attributes):
                attrs_str = ", ".join([f"{cls_var_name}_{attr.name}" for attr in cls.attributes])
                f.write(f"{cls_var_name}.attributes={{{attrs_str}}}\n")
            if sort(cls.methods):
                methods_str = ", ".join([f"{cls_var_name}_m_{method.name}" for method in cls.methods])
                f.write(f"{cls_var_name}.methods={{{methods_str}}}\n")
            f.write("\n")

        # Get the associations that are not part of association classes
        association_class_associations = [ac.association for ac in association_classes]
        regular_associations = [assoc for assoc in sort(model.associations) if assoc not in association_class_associations]

        # Write regular relationships
        if regular_associations:
            f.write("# Relationships\n")
            for assoc in regular_associations:
                ends_str = []
                for end in sort(assoc.ends):
                    # Determine max value for multiplicity
                    max_value = '"*"' if end.multiplicity.max == "*" else end.multiplicity.max
                    # Use safe class name for the type reference
                    type_var_name = safe_class_name(end.type.name)
                    # Build each property string with all attributes on the same line
                    end_str = (f"Property(name=\"{end.name}\", type={type_var_name}, "
                             f"multiplicity=Multiplicity({end.multiplicity.min}, {max_value})"
                             f"{', is_navigable=' + str(end.is_navigable) if end.is_navigable is not True else ''}"
                             f"{', is_composite=True' if end.is_composite is True else ''})")
                    ends_str.append(end_str)

                # Write the BinaryAssociation with each property on a new line
                f.write(
                    f"{assoc.name}: BinaryAssociation = BinaryAssociation(\n"
                    f"    name=\"{assoc.name}\",\n"
                    "    ends={\n        " + 
                    ",\n        ".join(ends_str) +
                    "\n    }\n"
                    ")\n"
                )
            f.write("\n")

        # Write Association Classes
        if association_classes:
            f.write("# Association Classes\n")
            for ac in association_classes:
                assoc = ac.association
                ac_var_name = safe_class_name(ac.name)
                
                # First write the BinaryAssociation
                ends_str = []
                for end in sort(assoc.ends):
                    # Determine max value for multiplicity
                    max_value = '"*"' if end.multiplicity.max == "*" else end.multiplicity.max
                    # Use safe class name for the type reference
                    type_var_name = safe_class_name(end.type.name)
                    # Build each property string with all attributes on the same line
                    end_str = (f"Property(name=\"{end.name}\", type={type_var_name}, "
                             f"multiplicity=Multiplicity({end.multiplicity.min}, {max_value})"
                             f"{', is_navigable=' + str(end.is_navigable) if end.is_navigable is not True else ''}"
                             f"{', is_composite=True' if end.is_composite is True else ''})")
                    ends_str.append(end_str)

                # Write the BinaryAssociation with each property on a new line
                f.write(
                    f"{assoc.name}: BinaryAssociation = BinaryAssociation(\n"
                    f"    name=\"{assoc.name}\",\n"
                    "    ends={\n        " + 
                    ",\n        ".join(ends_str) +
                    "\n    }\n"
                    ")\n"
                )
                f.write("\n")

                # Now write the attributes for the association class
                for attr in sort(ac.attributes):
                    attr_type = PRIMITIVE_TYPE_MAPPING.get(attr.type.name, safe_class_name(attr.type.name))
                    visibility_str = f', visibility="{attr.visibility}"' if attr.visibility != "public" else ""
                    f.write(f"{ac_var_name}_{attr.name}: Property = Property(name=\"{attr.name}\", "
                           f"type={attr_type}{visibility_str})\n")

                # Write methods for the association class
                for method in sort(ac.methods):
                    method_type = PRIMITIVE_TYPE_MAPPING.get(method.type.name, safe_class_name(method.type.name)) if method.type else None
                    visibility_str = f', visibility="{method.visibility}"' if method.visibility != "public" else ""

                    # Build parameters dictionary
                    params = {}
                    if sort(method.parameters):
                        for param in method.parameters:
                            param_type = PRIMITIVE_TYPE_MAPPING.get(param.type.name, safe_class_name(param.type.name))
                            default_str = f", default_value='{param.default_value}'" if hasattr(param, 'default_value') and param.default_value is not None else ""
                            params[param.name] = f"Parameter(name='{param.name}', type={param_type}{default_str})"
                    
                    params_str = "{" + ", ".join(f"{param}" for name, param in params.items()) + "}"

                    if method_type:
                        f.write(f"{ac_var_name}_m_{method.name}: Method = Method(name=\"{method.name}\""
                               f"{visibility_str}, parameters={params_str}, type={method_type})\n")
                    else:
                        f.write(f"{ac_var_name}_m_{method.name}: Method = Method(name=\"{method.name}\""
                               f"{visibility_str}, parameters={params_str})\n")

                # Create attributes set string if attributes exist
                attributes_str = ""
                if sort(ac.attributes):
                    attrs_str = ", ".join([f"{ac_var_name}_{attr.name}" for attr in ac.attributes])
                    attributes_str = f"attributes={{{attrs_str}}}, "

                # Create methods set string if methods exist
                methods_str = ""
                if sort(ac.methods):
                    methods_list = ", ".join([f"{ac_var_name}_m_{method.name}" for method in ac.methods])
                    methods_str = f", methods={{{methods_list}}}"

                # Now create the association class
                f.write(f"{ac_var_name} = AssociationClass(\n")
                f.write(f"    name=\"{ac.name}\",\n")
                f.write(f"    {attributes_str}association={assoc.name}{methods_str}\n")
                f.write(")\n\n")
            f.write("\n")

        # Write generalizations
        if model.generalizations:
            f.write("# Generalizations\n")
            for gen in sort(model.generalizations):
                specific_var_name = safe_class_name(gen.specific.name)
                general_var_name = safe_class_name(gen.general.name)
                f.write(f"gen_{gen.specific.name}_{gen.general.name} = Generalization"
                        f"(general={general_var_name}, specific={specific_var_name})\n")
            f.write("\n")

        # Write OCL constraints if they exist
        if hasattr(model, 'constraints') and model.constraints:
            f.write("\n# OCL Constraints\n")
            for constraint in sort(model.constraints):
                constraint_name = constraint.name.replace("-", "_")
                context_var_name = safe_class_name(constraint.context.name)
                f.write(f"{constraint_name}: Constraint = Constraint(\n")
                f.write(f"    name=\"{constraint.name}\",\n")
                f.write(f"    context={context_var_name},\n")
                f.write(f"    expression=\"{constraint.expression}\",\n")
                f.write(f"    language=\"{constraint.language}\"\n")
                f.write(")\n")
            f.write("\n")

        # Write domain model
        f.write("# Domain Model\n")
        
        # Write domain model metadata if it exists
        domain_metadata_var = None
        if hasattr(model, 'metadata') and model.metadata:
            domain_metadata_var = "domain_metadata"
            f.write(f"{domain_metadata_var} = Metadata(\n")
            if model.metadata.description:
                # Escape quotes and newlines in description
                desc = model.metadata.description.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                f.write(f'    description="{desc}",\n')
            if model.metadata.uri:
                f.write(f'    uri="{model.metadata.uri}",\n')
            if model.metadata.icon:
                f.write(f'    icon="{model.metadata.icon}"\n')
            f.write(")\n\n")
        
        f.write("domain_model = DomainModel(\n")
        f.write(f"    name=\"{model.name}\",\n")

        # Include all classes (regular and association) and enumerations in types
        class_names = ', '.join(safe_class_name(cls.name) for cls in sort(model.get_classes()))
        enum_names = ', '.join(safe_class_name(enum.name) for enum in sort(model.get_enumerations()))
        types_str = (f"{class_names}, {enum_names}" if class_names and enum_names else
                    class_names or enum_names)
        f.write(f"    types={{{types_str}}},\n")

        # Include both regular associations and those used in association classes
        all_assoc_names = ', '.join([assoc.name for assoc in regular_associations] + 
                                    [ac.association.name for ac in association_classes])
        if all_assoc_names:
            f.write(f"    associations={{{all_assoc_names}}},\n")
        else:
            f.write("    associations={},\n")

        if hasattr(model, 'constraints') and model.constraints:
            constraints_str = ', '.join(c.name.replace("-", "_") for c in sort(model.constraints))
            f.write(f"    constraints={{{constraints_str}}},\n")
        if model.generalizations:
            f.write(f"    generalizations={{{', '.join(f'gen_{gen.specific.name}_{gen.general.name}' for gen in sort(model.generalizations))}}},\n")
        else:
            f.write("    generalizations={},\n")
        
        # Add metadata if it exists
        if domain_metadata_var:
            f.write(f"    metadata={domain_metadata_var}\n")
        else:
            f.write("    metadata=None\n")
        f.write(")\n")

        # Generate object model code if provided
        if objectmodel:
            f.write("\n################\n")
            f.write("# OBJECT MODEL #\n")
            f.write("################\n")

            # Write object instances using fluent API
            for obj in sorted(objectmodel.objects, key=lambda x: x.name_):
                obj_var_name = f"{obj.name_.lower()}_obj"
                classifier_var_name = safe_class_name(obj.classifier.name)

                # Start the fluent API call using the proper syntax: Class("name")
                f.write(f"{obj_var_name} = {classifier_var_name}(\"{obj.name_}\")")

                # Add attributes if the object has slots
                if obj.slots:
                    attributes_dict = {}
                    for slot in obj.slots:
                        attr_name = slot.attribute.name

                        # Format the value based on type
                        if isinstance(slot.value.value, str):
                            value_str = f'"{slot.value.value}"'
                        elif hasattr(slot.value.value, 'isoformat'):  # datetime objects
                            value_str = f'datetime.datetime.fromisoformat("{slot.value.value.isoformat()}")'
                        elif hasattr(slot.value.value, 'owner') and hasattr(slot.value.value.owner, 'name'):
                            # This is an enumeration literal - generate the proper reference
                            enum_name = slot.value.value.owner.name
                            literal_name = slot.value.value.name
                            value_str = f"{enum_name}.{literal_name}"
                        else:
                            value_str = str(slot.value.value)

                        attributes_dict[attr_name] = value_str

                    # Add attributes to the fluent API call
                    if attributes_dict:
                        attr_pairs = [f"{k}={v}" for k, v in attributes_dict.items()]
                        f.write(f".attributes({', '.join(attr_pairs)})")

                # Complete the fluent API call
                f.write(".build()\n")

            f.write("\n")

            # Add links after objects are created (avoiding forward reference issues)
            if hasattr(objectmodel, 'links') and objectmodel.links:

                # Group links by (source_obj_var, end_name)
                grouped_links = {}
                for link in objectmodel.links:
                    if len(link.connections) == 2:
                        end1, end2 = link.connections
                        obj1_var = f"{end1.object.name_.lower()}_obj"
                        obj2_var = f"{end2.object.name_.lower()}_obj"
                        end2_name = end2.association_end.name

                        key = (obj1_var, end2_name)
                        grouped_links.setdefault(key, set()).add(obj2_var)

                # Write assignments for each group
                for (obj_var, end_name), targets in grouped_links.items():
                    if len(targets) == 1:
                        [single_target] = targets
                        f.write(f"{obj_var}.{end_name} = {single_target}\n")
                    else:
                        target_str = ", ".join(sorted(targets))  # sorted for consistency
                        f.write(f"{obj_var}.{end_name} = {{{target_str}}}\n")

                f.write("\n")

            # Create the object model instance
            f.write("# Object Model instance\n")
            objects_str = ", ".join([f"{obj.name_.lower()}_obj" for obj in sorted(objectmodel.objects, key=lambda x: x.name_)])
            f.write(f"object_model: ObjectModel = ObjectModel(\n")
            f.write(f"    name=\"{objectmodel.name}\",\n")
            f.write(f"    objects={{{objects_str}}}")
            
            # Add metadata if it exists
            if hasattr(objectmodel, 'metadata') and objectmodel.metadata:
                if objectmodel.metadata.description:
                    # Escape quotes and newlines in description
                    desc = objectmodel.metadata.description.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                    f.write(",\n")
                    f.write(f'    metadata=Metadata(description="{desc}")\n')
                else:
                    f.write("\n")
            else:
                f.write("\n")
            f.write(")\n")

    print(f"BUML model saved to {file_path}")

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
        f.write("from besser.BUML.metamodel.state_machine.agent import Agent, AgentSession, LLMOpenAI, LLMHuggingFace, LLMHuggingFaceAPI, LLMReplicate\n")
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
                f.write(f"    '{sentence}',\n")
            f.write("])\n")
        f.write("\n")
        
        # Check if an LLM is necessary
        llm_required = False
        for state in model.states:
            if state.body:
                if hasattr(state.body, 'messages') and isinstance(state.body.messages, list) and state.body.messages:
                    # Check if any of the messages are LLM messages
                    if any(message.startswith("LLM:") for message in state.body.messages):
                        llm_required = True
                        break
            elif state.fallback_body:
                if hasattr(state.fallback_body, 'messages') and isinstance(state.fallback_body.messages, list) and state.fallback_body.messages:
                    # Check if any of the messages are LLM messages
                    if any(message.startswith("LLM:") for message in state.fallback_body.messages):
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
                if hasattr(state.body, 'messages') and isinstance(state.body.messages, list) and state.body.messages:
                    # Check if any of the messages are code messages
                    has_code = any(message.startswith("CODE:") for message in state.body.messages)
                    
                    if has_code:
                        # For code messages, write them directly as functions
                        for message in state.body.messages:
                            if message.startswith("CODE:"):
                                # Extract the code content
                                code_content = message[5:]
                                # Write the code directly
                                f.write(f"{code_content}\n\n")
                        
                        # Set the body to the function defined in the code
                        # Look for function name in the code
                        import re
                        function_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_content)
                        if function_match:
                            function_name = function_match.group(1)
                            f.write(f"{state.name}.set_body(Body('{function_name}', {function_name}))\n")
                        else:
                            # If no function name found, use state_name_body
                            f.write(f"def {state.name}_body(session: AgentSession):\n")
                            f.write(f"    session.reply('Code body for {state.name}')\n\n")
                            f.write(f"{state.name}.set_body(Body('{state.name}_body', {state.name}_body))\n")
                    else:
                        # Check if any of the messages are LLM messages
                        has_llm = any(message.startswith("LLM:") for message in state.body.messages)
                        
                        if has_llm:
                            # Generate a function that uses llm.predict
                            f.write(f"def {state.name}_body(session: AgentSession):\n")
                            f.write(f"    session.reply(llm.predict(session.event.message))\n\n")
                        else:
                            # Write a function that outputs all messages
                            f.write(f"def {state.name}_body(session: AgentSession):\n")
                            for message in state.body.messages:
                                # Escape single quotes and backslashes
                                escaped_message = message.replace('\\', '\\\\').replace("'", "\\'")
                                f.write(f"    session.reply('{escaped_message}')\n")
                            f.write("\n")
                        
                        # Set the body on the state
                        f.write(f"{state.name}.set_body(Body('{state.name}_body', {state.name}_body))\n")
                
            # Write fallback body function if it exists
            if state.fallback_body:
                # Check if the fallback body has a messages attribute
                if hasattr(state.fallback_body, 'messages') and isinstance(state.fallback_body.messages, list) and state.fallback_body.messages:
                    # Check if any of the messages are code messages
                    has_code = any(message.startswith("CODE:") for message in state.fallback_body.messages)
                    
                    if has_code:
                        # For code messages, write them directly as functions
                        for message in state.fallback_body.messages:
                            if message.startswith("CODE:"):
                                # Extract the code content
                                code_content = message[5:]
                                # Write the code directly
                                f.write(f"{code_content}\n\n")
                    
                        # Set the fallback body to the function defined in the code
                        # Look for function name in the code
                        import re
                        function_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_content)
                        if function_match:
                            function_name = function_match.group(1)
                            f.write(f"{state.name}.set_fallback_body(Body('{function_name}', {function_name}))\n")
                        else:
                            # If no function name found, use state_name_fallback_body
                            f.write(f"def {state.name}_fallback_body(session: AgentSession):\n")
                            f.write(f"    session.reply('Code fallback body for {state.name}')\n\n")
                            f.write(f"{state.name}.set_fallback_body(Body('{state.name}_fallback_body', {state.name}_fallback_body))\n")
                    else:
                        # Check if any of the messages are LLM messages
                        has_llm = any(message.startswith("LLM:") for message in state.fallback_body.messages)
                        
                        if has_llm:
                            # Generate a function that uses llm.predict
                            f.write(f"def {state.name}_fallback_body(session: AgentSession):\n")
                            f.write(f"    session.reply(llm.predict(session.event.message))\n\n")
                        else:
                            # Write a function that outputs all messages
                            f.write(f"def {state.name}_fallback_body(session: AgentSession):\n")
                            for message in state.fallback_body.messages:
                                # Escape single quotes and backslashes
                                escaped_message = message.replace('\\', '\\\\').replace("'", "\\'")
                                f.write(f"    session.reply('{escaped_message}')\n")
                            f.write("\n")
                        
                        # Set the fallback body on the state
                        f.write(f"{state.name}.set_fallback_body(Body('{state.name}_fallback_body', {state.name}_fallback_body))\n")
            
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


def project_to_code(project: Project, file_path: str, sm: str = ""):
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not file_path.endswith('.py'):
        file_path += '.py'

    domain_model = None
    objectmodel = None
    agent_model = None
    state_machine = None

    for model in project.models:
        if isinstance(model, DomainModel):
            domain_model = model
        if isinstance(model, ObjectModel):
            objectmodel = model
        if isinstance(model, Agent):
            agent_model = model

    models = []
    with open(file_path, 'w', encoding='utf-8') as f:
        # Models
        temp_dir = tempfile.mkdtemp(prefix=f"besser_{uuid.uuid4().hex}_")

        if objectmodel and domain_model:
            output_file_path = os.path.join(temp_dir, "domain_model.py")
            domain_model_to_code(model=domain_model, file_path=output_file_path, objectmodel=objectmodel)
            with open(output_file_path, "r") as m:
                file_content = m.read()
            content_str = file_content
            f.write(content_str)
            f.write("\n\n")
            models.append("domain_model")
            models.append("object_model")
        elif domain_model:
            output_file_path = os.path.join(temp_dir, "domain_model.py")
            domain_model_to_code(model=domain_model, file_path=output_file_path)
            with open(output_file_path, "r") as m:
                file_content = m.read()
            content_str = file_content
            f.write(content_str)
            f.write("\n\n")
            models.append("domain_model")
        if agent_model:
            output_file_path = os.path.join(temp_dir, "agent_model.py")
            agent_model_to_code(model=agent_model, file_path=output_file_path)
            with open(output_file_path, "r") as m:
                file_content = m.read()
            content_str = file_content
            f.write(content_str)
            f.write("\n\n")
            models.append("agent")

        if sm != "":
            f.write(sm)
            models.append("sm")
            f.write("\n\n")

        shutil.rmtree(temp_dir, ignore_errors=True)

        # Project
        # Write imports
        f.write("######################\n")
        f.write("# PROJECT DEFINITION #\n")
        f.write("######################\n\n")
        f.write("from besser.BUML.metamodel.project import Project\n")
        f.write("from besser.BUML.metamodel.structural.structural import Metadata\n\n")

        # Write metadata and project
        f.write(f'metadata = Metadata(description="{project.metadata.description}")\n')
        f.write("project = Project(\n")
        f.write(f'    name="{project.name}",\n')
        f.write(f'    models=[{", ".join(models)}],\n')
        f.write(f'    owner="{project.owner}",\n')
        f.write("    metadata=metadata\n")
        f.write(")\n")
