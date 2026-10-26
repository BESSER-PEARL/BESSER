"""
Domain Model Builder

This module generates Python code for BUML domain models and object models.
"""

import keyword
import logging
import os
from typing import Optional
from besser.BUML.metamodel.structural.structural import (
    Class,
    DomainModel,
    AssociationClass,
    Enumeration,
    MethodImplementationType,
    UNLIMITED_MAX_MULTIPLICITY,
)
from besser.BUML.metamodel.object.object import ObjectModel
from besser.utilities import sort_by_timestamp as sort
from besser.utilities.buml_code_builder.common import (
    PRIMITIVE_TYPE_MAPPING,
    RESERVED_NAMES,
    NameDispenser,
    _comment_safe,
    buml_header,
    _escape_python_string,
    attribute_var_name,
    identifier_part,
    method_var_name,
    safe_class_name,
    safe_var_name,
    timestamp_expr,
    timestamp_kwarg,
    id_kwarg,
)

logger = logging.getLogger(__name__)


def _set_literal(items: str) -> str:
    """Python source for a set of the comma-separated ``items``.

    ``{}`` is an empty dict, not an empty set, so an empty collection is
    written as ``set()``.
    """
    return f"{{{items}}}" if items else "set()"


def _timestamp_line(element) -> str:
    """``,\n    timestamp=...`` for a multi-line constructor call, or ``""``."""
    expr = timestamp_expr(element)
    return f",\n    timestamp={expr}" if expr else ""


def _id_line(element, emit_ids) -> str:
    """``,\n    id=...`` for a multi-line constructor call, or ``""``."""
    kwarg = id_kwarg(element, emit_ids)
    return f",\n    {kwarg[2:]}" if kwarg else ""


def _attribute_kwargs(attr) -> str:
    """The keyword arguments of an attribute's ``Property(...)`` after ``type=``."""
    kwargs = ""
    if attr.visibility != "public":
        kwargs += f', visibility="{attr.visibility}"'
    multiplicity = attr.multiplicity
    if (multiplicity.min, multiplicity.max) != (1, 1):
        max_value = '"*"' if multiplicity.max == UNLIMITED_MAX_MULTIPLICITY else multiplicity.max
        kwargs += f", multiplicity=Multiplicity({multiplicity.min}, {max_value})"
    if attr.is_optional:
        kwargs += ", is_optional=True"
    if attr.is_id:
        kwargs += ", is_id=True"
    if attr.is_external_id:
        kwargs += ", is_external_id=True"
    if attr.is_read_only:
        kwargs += ", is_read_only=True"
    if attr.is_derived:
        kwargs += ", is_derived=True"
    if attr.default_value is not None:
        if isinstance(attr.default_value, str):
            kwargs += f', default_value="{_escape_python_string(attr.default_value)}"'
        else:
            kwargs += f', default_value={attr.default_value}'
    return kwargs


# Every name the generated script binds or reads besides its own element
# variables: the imports (``RESERVED_NAMES``), the ``set()`` builtin and the
# names read inside the ``except NameError`` guards. An association or
# constraint variable with one of these names would shadow it.
_SCRIPT_NAMES = frozenset(RESERVED_NAMES) | {"set", "NameError", "sm", "qc", "nn_model"}


def _generalization_var_name(gen) -> str:
    """Variable for a generalization: ``gen_<specific>_<general>``."""
    return f"gen_{identifier_part(gen.specific.name)}_{identifier_part(gen.general.name)}"


def _object_var_name(obj) -> str:
    """Python variable for an object instance: ``<safe name>_obj``.

    Object names are user-controlled and may hold spaces, colons or other
    characters that are not valid in an identifier, so they go through
    ``safe_var_name``. Every reference to the object's variable must use
    this helper so definitions and references stay in sync.
    """
    return f"{safe_var_name(obj.name_)}_obj"


def _require_identifier(name: str, what: str) -> str:
    """Return ``name`` if it can be written as a Python keyword argument or
    attribute; raise ``ValueError`` otherwise, rather than emit it raw."""
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"{what} '{name}' is not a valid Python identifier and cannot be written to B-UML code.")
    return name


_IMPLEMENTATION_TYPE_VALUE_TO_NAME = {
    MethodImplementationType.NONE.value: MethodImplementationType.NONE.name,
    MethodImplementationType.CODE.value: MethodImplementationType.CODE.name,
    MethodImplementationType.BAL.value: MethodImplementationType.BAL.name,
    MethodImplementationType.STATE_MACHINE.value: MethodImplementationType.STATE_MACHINE.name,
    MethodImplementationType.QUANTUM_CIRCUIT.value: MethodImplementationType.QUANTUM_CIRCUIT.name,
    MethodImplementationType.NEURAL_NETWORK.value: MethodImplementationType.NEURAL_NETWORK.name,
}


def _get_impl_type_name(impl_type) -> str | None:
    """Normalize enum and string values to valid MethodImplementationType enum names."""
    if impl_type is None:
        return None

    if isinstance(impl_type, MethodImplementationType):
        return impl_type.name

    impl_text = str(impl_type).strip()
    if impl_text.startswith("MethodImplementationType."):
        impl_text = impl_text.split(".", maxsplit=1)[1]

    upper_name = impl_text.upper()
    if upper_name in MethodImplementationType.__members__:
        return upper_name

    return _IMPLEMENTATION_TYPE_VALUE_TO_NAME.get(impl_text.lower())


def _format_method_code_literal(code: str) -> str:
    """Return a multiline-safe Python string literal for method code."""
    escaped_code = code.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
    return f'"""{escaped_code}"""'


def contains_user_class(model) -> bool:
    """Return True if the supplied model exposes a Class literally named 'User'."""
    if not model:
        return False
    get_classes = getattr(model, "get_classes", None)
    if not callable(get_classes):
        return False
    classes = get_classes() or []
    for cls in classes:
        class_name = (getattr(cls, "name", "") or "").strip().lower()
        if class_name == "user":
            return True
    return False


def is_user_object_model(obj_model) -> bool:
    """Detect whether an ObjectModel belongs to the user reference domain."""
    if not obj_model:
        return False
    domain_model = getattr(obj_model, "domain_model", None)
    if contains_user_class(domain_model):
        return True

    objects = getattr(obj_model, "objects", None) or []
    for obj in objects:
        classifier = getattr(obj, "classifier", None)
        classifier_name = (getattr(classifier, "name", "") or "").strip().lower()
        if classifier_name == "user":
            return True
    return False


def domain_model_to_code(
    model: DomainModel,
    file_path: str,
    objectmodel: ObjectModel = None,
    model_var_name: str = "domain_model",
    metadata_var_name: Optional[str] = None,
    object_model_var_name: str = "object_model",
    emit_ids: bool = False,
):
    """
    Generates Python code for a B-UML model and writes it to a specified file.

    Parameters:
        model (DomainModel): The B-UML model object containing classes, enumerations,
            associations, and generalizations.
        file_path (str): The path where the generated code will be saved.
        objectmodel (ObjectModel, optional): The B-UML object model to include in the same file.
        model_var_name (str, optional): Name of the DomainModel variable in the generated code.
        metadata_var_name (Optional[str], optional): Name for the metadata helper variable. Defaults
            to "<model_var_name>_metadata" when not provided.
        object_model_var_name (str, optional): Name of the ObjectModel variable when an object model
            is included. Defaults to "object_model".
        emit_ids (bool, optional): Write each element's ``id`` as an ``id="..."`` argument.
            Off by default: BESSER releases before ``Element.id`` cannot load such a file.

    Outputs:
        - A Python file containing the base code representation of the B-UML domain model
          and optionally the object model instances.
    """
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not file_path.endswith('.py'):
        file_path += '.py'

    metadata_var_name = metadata_var_name or f"{model_var_name}_metadata"
    object_model_var_name = object_model_var_name or "object_model"

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(buml_header())
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
        f.write("    AnyType, Constraint, AssociationClass, Metadata, MethodImplementationType\n")
        f.write(")\n")
        # ``datetime`` rebuilds element timestamps (and object slot values).
        f.write("import datetime\n")

        # Add object model imports if object model is provided
        if objectmodel:
            f.write("from besser.BUML.metamodel.object import ObjectModel\n")

        f.write("\n")

        # Write enumerations only if they exist
        enumerations = sort(model.get_enumerations())
        if enumerations:
            f.write("# Enumerations\n")
            for enum in enumerations:
                enum_var_name = safe_class_name(enum.name)
                f.write(f"{enum_var_name}: Enumeration = Enumeration(\n")
                f.write(f"    name=\"{_escape_python_string(enum.name)}\",\n")
                literals_str = ",\n\t\t\t".join([
                    f"EnumerationLiteral(name=\"{_escape_python_string(lit.name)}\"{timestamp_kwarg(lit)}{id_kwarg(lit, emit_ids)})"
                    for lit in sort(enum.literals)
                ])
                f.write(
                    f"    literals={{\n"
                    f"            {literals_str}"
                    f"\n    }}")
                f.write(_timestamp_line(enum))
                f.write(_id_line(enum, emit_ids))
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
            class_params = [f'name="{_escape_python_string(cls.name)}"']

            if cls.is_abstract:
                class_params.append('is_abstract=True')

            # Add metadata if it exists
            if hasattr(cls, 'metadata') and cls.metadata:
                metadata_params = []
                if cls.metadata.description:
                    metadata_params.append(f'description="{_escape_python_string(cls.metadata.description)}"')
                if cls.metadata.uri:
                    metadata_params.append(f'uri="{_escape_python_string(cls.metadata.uri)}"')
                if cls.metadata.icon:
                    metadata_params.append(f'icon="{_escape_python_string(cls.metadata.icon)}"')

                if metadata_params:
                    metadata_str = f"Metadata({', '.join(metadata_params)})"
                    class_params.append(f'metadata={metadata_str}')

            f.write(f"{cls_var_name} = Class({', '.join(class_params)}{timestamp_kwarg(cls)}{id_kwarg(cls, emit_ids)})\n")
        f.write("\n")

        # Write class members for regular classes
        for cls in regular_classes:
            cls_var_name = safe_class_name(cls.name)
            f.write(f"# {_comment_safe(cls.name)} class attributes and methods\n")

            # Write attributes
            for attr in sort(cls.attributes):
                attr_type = PRIMITIVE_TYPE_MAPPING.get(attr.type.name, safe_class_name(attr.type.name))
                f.write(f"{attribute_var_name(cls.name, attr.name)}: Property = Property(name=\"{_escape_python_string(attr.name)}\", "
                       f"type={attr_type}{_attribute_kwargs(attr)}{timestamp_kwarg(attr)}{id_kwarg(attr, emit_ids)})\n")

            # Write methods
            for method in sort(cls.methods):
                method_var = method_var_name(cls.name, method.name)

                method_type = PRIMITIVE_TYPE_MAPPING.get(method.type.name, safe_class_name(method.type.name)) if method.type else None
                visibility_str = f', visibility="{method.visibility}"' if method.visibility != "public" else ""

                method_code = method.code if hasattr(method, "code") and method.code else ""

                impl_str = ""
                impl_type = getattr(method, "implementation_type", None)
                impl_name = _get_impl_type_name(impl_type)
                if impl_name:
                    impl_str = f", implementation_type=MethodImplementationType.{impl_name}"

                # Build parameters dictionary
                params = {}
                if sort(method.parameters):
                    for param in method.parameters:
                        param_type = PRIMITIVE_TYPE_MAPPING.get(param.type.name, safe_class_name(param.type.name))
                        default_str = f", default_value='{_escape_python_string(str(param.default_value))}'" if hasattr(param, 'default_value') and param.default_value is not None else ""
                        params[param.name] = f"Parameter(name='{_escape_python_string(param.name)}', type={param_type}{default_str}{timestamp_kwarg(param)}{id_kwarg(param, emit_ids)})"

                # A list literal: parameter order is the method signature.
                params_str = "[" + ", ".join(f"{param}" for name, param in params.items()) + "]"

                if method_type:
                    f.write(f"{method_var}: Method = Method(name=\"{_escape_python_string(method.name)}\""
                           f"{visibility_str}, parameters={params_str}, type={method_type}{impl_str}{timestamp_kwarg(method)}{id_kwarg(method, emit_ids)})\n")
                else:
                    f.write(f"{method_var}: Method = Method(name=\"{_escape_python_string(method.name)}\""
                           f"{visibility_str}, parameters={params_str}{impl_str}{timestamp_kwarg(method)}{id_kwarg(method, emit_ids)})\n")
                if method_code:
                    code_literal = _format_method_code_literal(method_code)
                    f.write(f"{method_var}.code = {code_literal}\n")
                if impl_type == MethodImplementationType.STATE_MACHINE:
                    f.write("try:\n")
                    f.write(f"    {method_var}.state_machine = sm\n")
                    f.write("except NameError:\n")
                    f.write("    pass\n")
                if impl_type == MethodImplementationType.QUANTUM_CIRCUIT:
                    f.write("try:\n")
                    f.write(f"    {method_var}.quantum_circuit = qc\n")
                    f.write("except NameError:\n")
                    f.write("    pass\n")
                if impl_type == MethodImplementationType.NEURAL_NETWORK:
                    f.write("try:\n")
                    f.write(f"    {method_var}.neural_network = nn_model\n")
                    f.write("except NameError:\n")
                    f.write("    pass\n")

            # Write assignments
            if sort(cls.attributes):
                attrs_str = ", ".join(sorted([attribute_var_name(cls.name, attr.name) for attr in cls.attributes]))
                f.write(f"{cls_var_name}.attributes={{{attrs_str}}}\n")
            if sort(cls.methods):
                methods_str = ", ".join(sorted(
                    method_var_name(cls.name, method.name)
                    for method in cls.methods
                ))
                f.write(f"{cls_var_name}.methods={{{methods_str}}}\n")
            f.write("\n")

        # Association and constraint variables must not shadow anything else
        # the script binds or reads.
        taken = set(_SCRIPT_NAMES) | {model_var_name, metadata_var_name, object_model_var_name}
        for type_ in model.types:
            if isinstance(type_, Enumeration):
                taken.add(safe_class_name(type_.name))
            elif isinstance(type_, Class):
                taken.add(safe_class_name(type_.name))
                taken.update(attribute_var_name(type_.name, attr.name) for attr in type_.attributes)
                taken.update(method_var_name(type_.name, method.name) for method in type_.methods)
        taken.update(_generalization_var_name(gen) for gen in model.generalizations)
        element_vars = NameDispenser(taken, type_prefix=False, lowercase=False)

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
                    end_str = (f"Property(name=\"{_escape_python_string(end.name)}\", type={type_var_name}, "
                             f"multiplicity=Multiplicity({end.multiplicity.min}, {max_value})"
                             f"{', is_navigable=' + str(end.is_navigable) if end.is_navigable is not True else ''}"
                             f"{', is_composite=True' if end.is_composite is True else ''}"
                             f"{timestamp_kwarg(end)}{id_kwarg(end, emit_ids)})")
                    ends_str.append(end_str)

                # Write the BinaryAssociation with each property on a new line
                f.write(
                    f"{element_vars.name_for(assoc)}: BinaryAssociation = BinaryAssociation(\n"
                    f"    name=\"{_escape_python_string(assoc.name)}\",\n"
                    "    ends={\n        " +
                    ",\n        ".join(ends_str) +
                    "\n    }" + _timestamp_line(assoc) + _id_line(assoc, emit_ids) + "\n"
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
                    end_str = (f"Property(name=\"{_escape_python_string(end.name)}\", type={type_var_name}, "
                             f"multiplicity=Multiplicity({end.multiplicity.min}, {max_value})"
                             f"{', is_navigable=' + str(end.is_navigable) if end.is_navigable is not True else ''}"
                             f"{', is_composite=True' if end.is_composite is True else ''}"
                             f"{timestamp_kwarg(end)}{id_kwarg(end, emit_ids)})")
                    ends_str.append(end_str)

                # Write the BinaryAssociation with each property on a new line
                f.write(
                    f"{element_vars.name_for(assoc)}: BinaryAssociation = BinaryAssociation(\n"
                    f"    name=\"{_escape_python_string(assoc.name)}\",\n"
                    "    ends={\n        " +
                    ",\n        ".join(ends_str) +
                    "\n    }" + _timestamp_line(assoc) + _id_line(assoc, emit_ids) + "\n"
                    ")\n"
                )
                f.write("\n")

                # Now write the attributes for the association class
                for attr in sort(ac.attributes):
                    attr_type = PRIMITIVE_TYPE_MAPPING.get(attr.type.name, safe_class_name(attr.type.name))
                    f.write(f"{attribute_var_name(ac.name, attr.name)}: Property = Property(name=\"{_escape_python_string(attr.name)}\", "
                           f"type={attr_type}{_attribute_kwargs(attr)}{timestamp_kwarg(attr)}{id_kwarg(attr, emit_ids)})\n")

                # Write methods for the association class
                for method in sort(ac.methods):
                    method_var = method_var_name(ac.name, method.name)

                    method_type = PRIMITIVE_TYPE_MAPPING.get(method.type.name, safe_class_name(method.type.name)) if method.type else None
                    visibility_str = f', visibility="{method.visibility}"' if method.visibility != "public" else ""

                    method_code = method.code if hasattr(method, "code") and method.code else ""

                    impl_str = ""
                    impl_type = getattr(method, "implementation_type", None)
                    impl_name = _get_impl_type_name(impl_type)
                    if impl_name:
                        impl_str = f", implementation_type=MethodImplementationType.{impl_name}"

                    # Build parameters dictionary
                    params = {}
                    if sort(method.parameters):
                        for param in method.parameters:
                            param_type = PRIMITIVE_TYPE_MAPPING.get(param.type.name, safe_class_name(param.type.name))
                            default_str = f", default_value='{_escape_python_string(str(param.default_value))}'" if hasattr(param, 'default_value') and param.default_value is not None else ""
                            params[param.name] = f"Parameter(name='{_escape_python_string(param.name)}', type={param_type}{default_str}{timestamp_kwarg(param)}{id_kwarg(param, emit_ids)})"

                    # A list literal: parameter order is the method signature.
                    params_str = "[" + ", ".join(f"{param}" for name, param in params.items()) + "]"

                    if method_type:
                        f.write(f"{method_var}: Method = Method(name=\"{_escape_python_string(method.name)}\""
                               f"{visibility_str}, parameters={params_str}, type={method_type}{impl_str}{timestamp_kwarg(method)}{id_kwarg(method, emit_ids)})\n")
                    else:
                        f.write(f"{method_var}: Method = Method(name=\"{_escape_python_string(method.name)}\""
                               f"{visibility_str}, parameters={params_str}{impl_str}{timestamp_kwarg(method)}{id_kwarg(method, emit_ids)})\n")
                    if method_code:
                        code_literal = _format_method_code_literal(method_code)
                        f.write(f"{method_var}.code = {code_literal}\n")
                    if impl_type == MethodImplementationType.STATE_MACHINE:
                        f.write("try:\n")
                        f.write(f"    {method_var}.state_machine = sm\n")
                        f.write("except NameError:\n")
                        f.write("    pass\n")
                    if impl_type == MethodImplementationType.QUANTUM_CIRCUIT:
                        f.write("try:\n")
                        f.write(f"    {method_var}.quantum_circuit = qc\n")
                        f.write("except NameError:\n")
                        f.write("    pass\n")
                    if impl_type == MethodImplementationType.NEURAL_NETWORK:
                        f.write("try:\n")
                        f.write(f"    {method_var}.neural_network = nn_model\n")
                        f.write("except NameError:\n")
                        f.write("    pass\n")

                # ``attributes`` is a required AssociationClass argument, so an
                # association class without attributes gets an empty set.
                attributes_str = "attributes=set(), "
                if sort(ac.attributes):
                    attrs_str = ", ".join(sorted([attribute_var_name(ac.name, attr.name) for attr in ac.attributes]))
                    attributes_str = f"attributes={{{attrs_str}}}, "

                # Create methods set string if methods exist
                methods_str = ""
                if sort(ac.methods):
                    methods_list = ", ".join(sorted(
                        method_var_name(ac.name, method.name)
                        for method in ac.methods
                    ))
                    methods_str = f", methods={{{methods_list}}}"

                # Now create the association class
                f.write(f"{ac_var_name} = AssociationClass(\n")
                f.write(f"    name=\"{_escape_python_string(ac.name)}\",\n")
                f.write(f"    {attributes_str}association={element_vars.name_for(assoc)}{methods_str}{timestamp_kwarg(ac)}{id_kwarg(ac, emit_ids)}\n")
                f.write(")\n\n")
            f.write("\n")

        # Write generalizations
        if model.generalizations:
            f.write("# Generalizations\n")
            for gen in sort(model.generalizations):
                specific_var_name = safe_class_name(gen.specific.name)
                general_var_name = safe_class_name(gen.general.name)
                f.write(f"{_generalization_var_name(gen)} = Generalization"
                        f"(general={general_var_name}, specific={specific_var_name}{timestamp_kwarg(gen)}{id_kwarg(gen, emit_ids)})\n")
            f.write("\n")

        # Helper: emit a single ``Constraint(...)`` declaration. Reused for
        # invariants (top-level) and for method ``pre`` / ``post`` contracts
        # (which are then attached via ``Method.add_pre`` / ``add_post``).
        def _emit_constraint_decl(constraint, var_name: str) -> None:
            context_var_name = safe_class_name(constraint.context.name)
            f.write(f"{var_name}: Constraint = Constraint(\n")
            f.write(f"    name=\"{_escape_python_string(constraint.name)}\",\n")
            f.write(f"    context={context_var_name},\n")
            f.write(f"    expression=\"{_escape_python_string(constraint.expression)}\",\n")
            f.write(f"    language=\"{_escape_python_string(constraint.language)}\"")
            description = getattr(constraint, 'description', None)
            if description:
                f.write(",\n")
                f.write(f"    description=\"{_escape_python_string(description)}\"")
            f.write(_timestamp_line(constraint))
            f.write("\n)\n")

        # Collect every method's pre / post contracts. Emit them alongside
        # invariants so the generated file round-trips back to the editor
        # with the same routing — without this loop, BUML export would
        # silently drop method contracts (a regression we hit in
        # https://github.com/BESSER-PEARL/BESSER-Web-Modeling-Editor/pull/124).
        # Tuple shape: (constraint, method, kind, cls).
        method_contracts: list[tuple] = []
        for cls in sorted(
            (t for t in model.types if isinstance(t, Class)),
            key=lambda t: t.name,
        ):
            for method in cls.methods:
                for c in method.pre:
                    method_contracts.append((c, method, 'pre', cls))
                for c in method.post:
                    method_contracts.append((c, method, 'post', cls))
        # Emit deterministically: same model → same byte-stable output.
        method_contracts.sort(
            key=lambda t: (t[3].name, t[1].name, t[2], t[0].name)
        )

        invariants = list(model.constraints) if hasattr(model, 'constraints') else []
        if invariants or method_contracts:
            f.write("\n# OCL Constraints\n")
            for constraint in sort(invariants):
                # ``constraint.name`` is user-typed and lands as the Python
                # identifier on the LHS — must go through ``safe_var_name``
                # to neutralize any character that would let the value
                # escape into executable code at ``exec()`` time.
                _emit_constraint_decl(constraint, element_vars.name_for(constraint))
            for constraint, method, kind, cls in method_contracts:
                method_var = method_var_name(cls.name, method.name)
                contract_var = element_vars.name_for(constraint)
                _emit_constraint_decl(constraint, contract_var)
                f.write(f"{method_var}.add_{kind}({contract_var})\n")
            f.write("\n")

        # Write domain model
        f.write("# Domain Model\n")

        # Write domain model metadata if it exists
        domain_metadata_var = None
        if hasattr(model, 'metadata') and model.metadata:
            domain_metadata_var = metadata_var_name
            f.write(f"{domain_metadata_var} = Metadata(\n")
            if model.metadata.description:
                f.write(f'    description="{_escape_python_string(model.metadata.description)}",\n')
            if model.metadata.uri:
                f.write(f'    uri="{_escape_python_string(model.metadata.uri)}",\n')
            if model.metadata.icon:
                f.write(f'    icon="{_escape_python_string(model.metadata.icon)}"\n')
            f.write(")\n\n")

        f.write(f"{model_var_name} = DomainModel(\n")
        f.write(f"    name=\"{_escape_python_string(model.name)}\",\n")

        # Include all classes (regular and association) and enumerations in types
        class_names = ', '.join(safe_class_name(cls.name) for cls in sort(model.get_classes()))
        enum_names = ', '.join(safe_class_name(enum.name) for enum in sort(model.get_enumerations()))
        types_str = (f"{class_names}, {enum_names}" if class_names and enum_names else
                    class_names or enum_names)
        f.write(f"    types={_set_literal(types_str)},\n")

        # Include both regular associations and those used in association classes
        all_assoc_names = ', '.join([element_vars.name_for(assoc) for assoc in regular_associations] +
                                    [element_vars.name_for(ac.association) for ac in association_classes])
        f.write(f"    associations={_set_literal(all_assoc_names)},\n")

        if hasattr(model, 'constraints') and model.constraints:
            constraints_str = ', '.join(element_vars.name_for(c) for c in sort(model.constraints))
            f.write(f"    constraints={{{constraints_str}}},\n")
        gen_names = ', '.join(_generalization_var_name(gen) for gen in sort(model.generalizations))
        f.write(f"    generalizations={_set_literal(gen_names)},\n")
        model_timestamp = timestamp_expr(model)
        if model_timestamp:
            f.write(f"    timestamp={model_timestamp},\n")

        # Add metadata if it exists
        if domain_metadata_var:
            f.write(f"    metadata={domain_metadata_var}\n")
        else:
            f.write("    metadata=None\n")
        f.write(")\n")

        # Generate object model code if provided
        if objectmodel:
            _write_object_model_section(f, objectmodel, object_model_var_name)

    logger.info("BUML model saved to %s", file_path)


def _write_object_model_section(f, objectmodel: ObjectModel, object_model_var_name: str = "object_model",
                                header_label: str = None):
    """Write the object-model portion of the builder output to an open file handle.

    Emits the ``# OBJECT MODEL #`` banner, the object instances (via the fluent
    API), the links between objects, and the final ``ObjectModel(...)`` binding.
    This section assumes the referenced domain classes and enumerations are
    already defined earlier in the same (concatenated) file.

    Parameters:
        f: An already-open, writable file handle.
        objectmodel (ObjectModel): The B-UML object model to serialize.
        object_model_var_name (str): Name of the ObjectModel variable to bind.
        header_label (str, optional): Custom text for the banner's middle line
            (e.g. ``OBJECT MODEL 2: "Orders"``) — the single section marker the
            project importer keys on. Defaults to the plain ``OBJECT MODEL`` banner.
    """
    object_model_var_name = object_model_var_name or "object_model"

    # This banner's middle line is the single section marker the project importer
    # keys on. When a custom label is given (numbered + titled, for multi-object
    # projects) we size the box border to it; otherwise keep the plain, historic
    # "# OBJECT MODEL #" banner. Callers must NOT emit a second header alongside
    # this one, or the importer splits each model into an extra empty section.
    if header_label:
        border = "#" * (len(header_label) + 4)
        f.write(f"\n{border}\n# {header_label} #\n{border}\n")
    else:
        f.write("\n################\n")
        f.write("# OBJECT MODEL #\n")
        f.write("################\n")

    # Write object instances using fluent API
    for obj in sorted(objectmodel.objects, key=lambda x: x.name_):
        obj_var_name = _object_var_name(obj)
        classifier_var_name = safe_class_name(obj.classifier.name)

        # Start the fluent API call using the proper syntax: Class("name")
        f.write(f"{obj_var_name} = {classifier_var_name}(\"{_escape_python_string(obj.name_)}\")")

        # Add attributes if the object has slots
        if obj.slots:
            attributes_dict = {}
            for slot in obj.slots:
                attr_name = _require_identifier(slot.attribute.name, "Attribute")

                # Format the value based on type
                if isinstance(slot.value.value, str):
                    value_str = f'"{_escape_python_string(slot.value.value)}"'
                elif hasattr(slot.value.value, 'isoformat'):  # datetime objects
                    value_str = f'datetime.datetime.fromisoformat("{slot.value.value.isoformat()}")'
                elif hasattr(slot.value.value, 'owner') and hasattr(slot.value.value.owner, 'name'):
                    # This is an enumeration literal - generate the proper reference
                    enum_var = safe_class_name(slot.value.value.owner.name)
                    literal_name = _require_identifier(slot.value.value.name, "Enumeration literal")
                    value_str = f"{enum_var}.{literal_name}"
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
                obj1_var = _object_var_name(end1.object)
                obj2_var = _object_var_name(end2.object)
                end2_name = _require_identifier(end2.association_end.name, "Association end")

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
    objects_str = ", ".join([_object_var_name(obj) for obj in sorted(objectmodel.objects, key=lambda x: x.name_)])
    f.write(f"{object_model_var_name}: ObjectModel = ObjectModel(\n")
    f.write(f"    name=\"{_escape_python_string(objectmodel.name)}\",\n")
    f.write(f"    objects={_set_literal(objects_str)}")

    # Add metadata if it exists
    if hasattr(objectmodel, 'metadata') and objectmodel.metadata:
        if objectmodel.metadata.description:
            f.write(",\n")
            f.write(f'    metadata=Metadata(description="{_escape_python_string(objectmodel.metadata.description)}")\n')
        else:
            f.write("\n")
    else:
        f.write("\n")
    f.write(")\n")


def object_model_to_code(
    objectmodel: ObjectModel,
    file_path: str,
    object_model_var_name: str = "object_model",
    header_label: str = None,
):
    """Generate Python code for a standalone B-UML object model.

    The generated file is self-contained apart from the domain classes and
    enumerations it references: it only imports ``ObjectModel`` and ``datetime``
    and then emits the object-model section. It is meant to be concatenated
    *after* the corresponding domain model (whose classes/enums it references)
    so the combined file executes cleanly.

    Parameters:
        objectmodel (ObjectModel): The B-UML object model to serialize.
        file_path (str): The path where the generated code will be saved.
        object_model_var_name (str): Name of the ObjectModel variable in the
            generated code. Defaults to "object_model".
        header_label (str, optional): Custom banner text for the section header
            (numbered + titled, e.g. ``OBJECT MODEL 2: "Orders"``). Defaults to
            the plain ``OBJECT MODEL`` banner.

    Outputs:
        - A Python file containing the object-model instances and their links.
    """
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not file_path.endswith('.py'):
        file_path += '.py'

    object_model_var_name = object_model_var_name or "object_model"

    with open(file_path, 'w', encoding='utf-8') as f:
        # Object-model imports. The domain classes/enums are assumed to be
        # defined earlier in the concatenated output.
        f.write("from besser.BUML.metamodel.object import ObjectModel\n")
        f.write("import datetime\n")

        _write_object_model_section(f, objectmodel, object_model_var_name, header_label)

    logger.info("BUML object model saved to %s", file_path)
