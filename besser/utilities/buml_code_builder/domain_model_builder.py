"""
Domain Model Builder

This module generates Python code for BUML domain models and object models.
"""

import os
from besser.BUML.metamodel.structural.structural import DomainModel, AssociationClass, Metadata
from besser.BUML.metamodel.object.object import ObjectModel
from besser.utilities import sort_by_timestamp as sort
from besser.utilities.buml_code_builder.common import PRIMITIVE_TYPE_MAPPING, safe_class_name


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
                # Extract just the method name (before any parenthesis) for the variable name
                method_var_name = method.name.split('(')[0] if '(' in method.name else method.name
                
                method_type = PRIMITIVE_TYPE_MAPPING.get(method.type.name, safe_class_name(method.type.name)) if method.type else None
                visibility_str = f', visibility="{method.visibility}"' if method.visibility != "public" else ""
                
                # Handle code attribute - escape special characters
                code_str = ""
                if hasattr(method, 'code') and method.code:
                    # Escape backslashes first, then quotes, then newlines
                    escaped_code = method.code.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                    code_str = f', code="{escaped_code}"'

                # Build parameters dictionary
                params = {}
                if sort(method.parameters):
                    for param in method.parameters:
                        param_type = PRIMITIVE_TYPE_MAPPING.get(param.type.name, safe_class_name(param.type.name))
                        default_str = f", default_value='{param.default_value}'" if hasattr(param, 'default_value') and param.default_value is not None else ""
                        params[param.name] = f"Parameter(name='{param.name}', type={param_type}{default_str})"

                params_str = "{" + ", ".join(f"{param}" for name, param in params.items()) + "}"

                if method_type:
                    f.write(f"{cls_var_name}_m_{method_var_name}: Method = Method(name=\"{method.name}\""
                           f"{visibility_str}, parameters={params_str}, type={method_type}{code_str})\n")
                else:
                    f.write(f"{cls_var_name}_m_{method_var_name}: Method = Method(name=\"{method.name}\""
                           f"{visibility_str}, parameters={params_str}{code_str})\n")

            # Write assignments
            if sort(cls.attributes):
                attrs_str = ", ".join([f"{cls_var_name}_{attr.name}" for attr in cls.attributes])
                f.write(f"{cls_var_name}.attributes={{{attrs_str}}}\n")
            if sort(cls.methods):
                # Extract just method names for variable references
                methods_str = ", ".join([f"{cls_var_name}_m_{method.name.split('(')[0] if '(' in method.name else method.name}" for method in cls.methods])
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
                    # Extract just the method name (before any parenthesis) for the variable name
                    method_var_name = method.name.split('(')[0] if '(' in method.name else method.name
                    
                    method_type = PRIMITIVE_TYPE_MAPPING.get(method.type.name, safe_class_name(method.type.name)) if method.type else None
                    visibility_str = f', visibility="{method.visibility}"' if method.visibility != "public" else ""
                    
                    # Handle code attribute - escape special characters
                    code_str = ""
                    if hasattr(method, 'code') and method.code:
                        # Escape backslashes first, then quotes, then newlines
                        escaped_code = method.code.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                        code_str = f', code="{escaped_code}"'

                    # Build parameters dictionary
                    params = {}
                    if sort(method.parameters):
                        for param in method.parameters:
                            param_type = PRIMITIVE_TYPE_MAPPING.get(param.type.name, safe_class_name(param.type.name))
                            default_str = f", default_value='{param.default_value}'" if hasattr(param, 'default_value') and param.default_value is not None else ""
                            params[param.name] = f"Parameter(name='{param.name}', type={param_type}{default_str})"
                    
                    params_str = "{" + ", ".join(f"{param}" for name, param in params.items()) + "}"

                    if method_type:
                        f.write(f"{ac_var_name}_m_{method_var_name}: Method = Method(name=\"{method.name}\""
                               f"{visibility_str}, parameters={params_str}, type={method_type}{code_str})\n")
                    else:
                        f.write(f"{ac_var_name}_m_{method_var_name}: Method = Method(name=\"{method.name}\""
                               f"{visibility_str}, parameters={params_str}{code_str})\n")

                # Create attributes set string if attributes exist
                attributes_str = ""
                if sort(ac.attributes):
                    attrs_str = ", ".join([f"{ac_var_name}_{attr.name}" for attr in ac.attributes])
                    attributes_str = f"attributes={{{attrs_str}}}, "

                # Create methods set string if methods exist
                methods_str = ""
                if sort(ac.methods):
                    # Extract just method names for variable references
                    methods_list = ", ".join([f"{ac_var_name}_m_{method.name.split('(')[0] if '(' in method.name else method.name}" for method in ac.methods])
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
