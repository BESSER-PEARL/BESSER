import os
from besser.BUML.metamodel.structural.structural import DomainModel


def domain_model_to_code(model: DomainModel, file_path: str):
    """
    Save B-UML model to a Python file.

    Args:
        model: The DomainModel to save
        file_path: Path where the file should be saved
    """
    # Create output directory if needed
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
            literals_str = ", ".join([f"EnumerationLiteral(name=\"{lit.name}\")" \
                                      for lit in enum.literals])
            f.write(f"    literals={{{literals_str}}}\n")
            f.write(")\n\n")

        # Write classes (but skip enums that were incorrectly detected as classes)
        f.write("# Classes\n")
        for cls in model.get_classes():
            if not cls.name.startswith("<<Enum>>"):  # Skip enum classes
                f.write(f"{cls.name} = Class(name=\"{cls.name}\")\n")
        f.write("\n")

        # Write class members
        for cls in model.get_classes():
            f.write(f"# {cls.name} class attributes and methods\n")

            # Write attributes
            for attr in cls.attributes:
                f.write(f"{cls.name}_{attr.name}: Property = Property(name=\"{attr.name}\", "
                       f"type={attr.type}, visibility=\"{attr.visibility}\")\n")

            # Write methods
            for method in cls.methods:
                f.write(f"{cls.name}_m_{method.name}: Method = Method(name=\"{method.name}\", "
                    f"visibility=\"{method.visibility}\", parameters={{}}, type={method.type})\n")

            # Write assignments
            if cls.attributes:
                attrs_str = ", ".join([f"{cls.name}_{attr.name}" for attr in cls.attributes])
                f.write(f"{cls.name}.attributes={{{attrs_str}}}\n")
            if cls.methods:
                methods_str = ", ".join([f"{cls.name}_m_{method.name}" for method in cls.methods])
                f.write(f"{cls.name}.methods={{{methods_str}}}\n")
            f.write("\n")

        # Write associations
        if model.associations:
            f.write("# Relationships\n")
            for assoc in model.associations:
                ends_str = []
                for end in assoc.ends:
                    max_value = '"*"' if end.multiplicity.max == "*" else end.multiplicity.max
                    ends_str.append(f"Property(name=\"{end.name}\", type={end.type.name}, "
                                f"multiplicity=Multiplicity({end.multiplicity.min}, {max_value}),"
                              f"is_navigable={end.is_navigable}, is_composite={end.is_composite})")
                f.write(f"{assoc.name}: BinaryAssociation = BinaryAssociation(name=\"{assoc.name}\""
                       f",ends={{{', '.join(ends_str)}}})\n")
            f.write("\n")


         # Write generalizations
        if model.generalizations:
            f.write("# Generalizations\n")
            for gen in model.generalizations:
                f.write(f"gen_{gen.specific.name}_{gen.general.name} = Generalization"
                        f"(general={gen.general.name},"
                       f"specific={gen.specific.name})\n")
            f.write("\n")

        # Write domain model
        f.write("# Domain Model\n")
        f.write("domain_model = DomainModel(\n")
        f.write("    name=\"Generated Model\",\n")
        class_names = ', '.join(cls.name for cls in model.get_classes())
        enum_names = ', '.join(enum.name for enum in model.get_enumerations())
        types_str = f"{class_names}{', ' + enum_names if enum_names else ''}"
        f.write(f"    types={{{types_str}}},\n")
        if model.associations:
            f.write(f"    associations={{{', '.join(assoc.name for assoc in model.associations)}}},\n")
        else:
            f.write("    associations={},\n")
        if model.generalizations:
            f.write(f"    generalizations={{{', '.join(f'gen_{gen.specific.name}_{gen.general.name}' \
                                                   for gen in model.generalizations)}}}\n")
        else:
            f.write("    generalizations={}\n")
        f.write(")\n")

    print(f"BUML model saved to {file_path}")