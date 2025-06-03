import os
from besser.BUML.metamodel.structural.structural import DomainModel, AssociationClass, Metadata
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

def domain_model_to_code(model: DomainModel, file_path: str):
    """
    Generates Python code for a B-UML model and writes it to a specified file.

    Parameters:
    model (DomainModel): The B-UML model object containing classes, enumerations, 
        associations, and generalizations.
    file_path (str): The path where the generated code will be saved.

    Outputs:
    - A Python file containing the base code representation of the B-UML domain model.
    """
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not file_path.endswith('.py'):
        file_path += '.py'

    with open(file_path, 'w', encoding='utf-8') as f:
        # Write imports
        f.write("# Generated B-UML Model\n")
        f.write("from besser.BUML.metamodel.structural import (\n")
        f.write("    Class, Property, Method, Parameter,\n")
        f.write("    BinaryAssociation, Generalization, DomainModel,\n")
        f.write("    Enumeration, EnumerationLiteral, Multiplicity,\n")
        f.write("    StringType, IntegerType, FloatType, BooleanType,\n")
        f.write("    TimeType, DateType, DateTimeType, TimeDeltaType,\n")
        f.write("    AnyType, Constraint, AssociationClass, Metadata\n")
        f.write(")\n\n")

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
                    metadata_params.append(f'description="{cls.metadata.description}"')
                if cls.metadata.uri:
                    metadata_params.append(f'uri="{cls.metadata.uri}"')
                
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
            f.write(f"    generalizations={{{', '.join(f'gen_{gen.specific.name}_{gen.general.name}' for gen in sort(model.generalizations))}}}\n")
        else:
            f.write("    generalizations={}\n")
        f.write(")\n")

    print(f"BUML model saved to {file_path}")