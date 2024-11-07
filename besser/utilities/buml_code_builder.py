import os
from besser.BUML.metamodel.structural.structural import (DomainModel, StringType, IntegerType,
                                                          FloatType, BooleanType, TimeType,
                                                          DateType, DateTimeType, TimeDeltaType)

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
    'timedelta': 'TimeDeltaType'
}

def domain_model_to_code(model: DomainModel, file_path: str):
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, 'w', encoding='utf-8') as f:
        # Write imports
        f.write("# Generated B-UML Model\n")
        f.write("from besser.BUML.metamodel.structural import (\n")
        f.write("    Class, Property, Method, Parameter,\n")
        f.write("    BinaryAssociation, Generalization, DomainModel,\n")
        f.write("    Enumeration, EnumerationLiteral, Multiplicity,\n")
        f.write("    StringType, IntegerType, FloatType, BooleanType,\n")
        f.write("    TimeType, DateType, DateTimeType, TimeDeltaType\n")
        f.write(")\n\n")

        # Write enumerations
        f.write("# Enumerations\n")
        for enum in model.get_enumerations():
            f.write(f"{enum.name}: Enumeration = Enumeration(\n")
            f.write(f"    name=\"{enum.name}\",\n")
            literals_str = ", ".join([f"EnumerationLiteral(name=\"{lit.name}\")" for lit in enum.literals])
            f.write(f"    literals={{{literals_str}}}\n")
            f.write(")\n\n")

        # Write classes
        f.write("# Classes\n")
        for cls in model.get_classes():
            f.write(f"{cls.name} = Class(name=\"{cls.name}\")\n")
        f.write("\n")

        # Write class members
        for cls in model.get_classes():
            f.write(f"# {cls.name} class attributes and methods\n")

            # Write attributes
            for attr in cls.attributes:
                attr_type = PRIMITIVE_TYPE_MAPPING.get(attr.type.name, attr.type.name)
                visibility_str = f', visibility="{attr.visibility}"' if attr.visibility != "public" else ""
                f.write(f"{cls.name}_{attr.name}: Property = Property(name=\"{attr.name}\", "
                       f"type={attr_type}{visibility_str})\n")

            # Write methods
            for method in cls.methods:
                method_type = PRIMITIVE_TYPE_MAPPING.get(method.type.name, method.type.name) if method.type else None
                visibility_str = f', visibility="{method.visibility}"' if method.visibility != "public" else ""
                
                if method_type:
                    f.write(f"{cls.name}_m_{method.name}: Method = Method(name=\"{method.name}\""
                           f"{visibility_str}, parameters={{}}, type={method_type})\n")
                else:
                    f.write(f"{cls.name}_m_{method.name}: Method = Method(name=\"{method.name}\""
                           f"{visibility_str}, parameters={{}})\n")

            # Write assignments
            if cls.attributes:
                attrs_str = ", ".join([f"{cls.name}_{attr.name}" for attr in cls.attributes])
                f.write(f"{cls.name}.attributes={{{attrs_str}}}\n")
            if cls.methods:
                methods_str = ", ".join([f"{cls.name}_m_{method.name}" for method in cls.methods])
                f.write(f"{cls.name}.methods={{{methods_str}}}\n")
            f.write("\n")

# Write relationships
        if model.associations:
            f.write("# Relationships\n")
            for assoc in model.associations:
                ends_str = []
                for end in assoc.ends:
                    # Determine max value for multiplicity
                    max_value = '"*"' if end.multiplicity.max == "*" else end.multiplicity.max
                    # Build each property string with all attributes on the same line
                    end_str = (f"Property(name=\"{end.name}\", type={end.type.name}, "
                             f"multiplicity=Multiplicity({end.multiplicity.min}, {max_value})"
                             f"{', is_navigable=' + str(end.is_navigable) if end.is_navigable is not True else ''}"
                             f"{', is_composite=True' if end.is_composite is True else ''})")
                    ends_str.append(end_str)
        
                # Write the BinaryAssociation with each property on a new line
                f.write(
                    f"{assoc.name}: BinaryAssociation = BinaryAssociation(\n"
                    f"    name=\"{assoc.name}\",\n"
                    f"    ends={{\n        {',\n        '.join(ends_str)}\n    }}\n"
                    f")\n"
                )
            f.write("\n")
        

        # Write generalizations
        if model.generalizations:
            f.write("# Generalizations\n")
            for gen in model.generalizations:
                f.write(f"gen_{gen.specific.name}_{gen.general.name} = Generalization"
                        f"(general={gen.general.name}, specific={gen.specific.name})\n")
            f.write("\n")

        # Write domain model
        f.write("# Domain Model\n")
        f.write("domain_model = DomainModel(\n")
        f.write("    name=\"Generated Model\",\n")
        class_names = ', '.join(cls.name for cls in model.get_classes())
        enum_names = ', '.join(enum.name for enum in model.get_enumerations())
        types_str = (f"{class_names}, {enum_names}" if class_names and enum_names else 
                    class_names or enum_names)
        f.write(f"    types={{{types_str}}},\n")
        if model.associations:
            f.write(f"    associations={{{', '.join(assoc.name for assoc in model.associations)}}},\n")
        else:
            f.write("    associations={},\n")
        if model.generalizations:
            f.write(f"    generalizations={{{', '.join(f'gen_{gen.specific.name}_{gen.general.name}' for gen in model.generalizations)}}}\n")
        else:
            f.write("    generalizations={}\n")
        f.write(")\n")

    print(f"BUML model saved to {file_path}")
