from __future__ import annotations

from typing import Dict, List

from besser.BUML.metamodel.structural import DomainModel, AssociationClass


def _sorted_association_ends(association) -> list:
    """Return association ends in a deterministic order."""
    return sorted(association.ends, key=lambda end: (end.type.name, end.name or ""))


def get_foreign_keys(model: DomainModel) -> Dict[str, List[str]]:
    """
    Return a mapping of association name -> [class_name_with_fk, fk_property_name].
    """
    fkeys: Dict[str, List[str]] = {}

    for association in model.associations:
        ends = _sorted_association_ends(association)
        if len(ends) != 2:
            continue

        end0, end1 = ends[0], ends[1]

        # One-to-one
        if end0.multiplicity.max == 1 and end1.multiplicity.max == 1:
            if end0.multiplicity.min > 0 and end1.multiplicity.min == 0:
                fkeys[association.name] = [end1.type.name, end0.name]
            elif end1.multiplicity.min > 0 and end0.multiplicity.min == 0:
                fkeys[association.name] = [end0.type.name, end1.name]
            else:
                fkeys[association.name] = [end0.type.name, end1.name]

        # Many-to-one
        elif end0.multiplicity.max > 1 and end1.multiplicity.max <= 1:
            fkeys[association.name] = [end0.type.name, end1.name]

        elif end0.multiplicity.max <= 1 and end1.multiplicity.max > 1:
            fkeys[association.name] = [end1.type.name, end0.name]

    return fkeys


# SQLAlchemy type mapping
SQLALCHEMY_TYPES = {
    "int": "Integer",
    "str": "String(100)",
    "float": "Float",
    "bool": "Boolean",
    "time": "Time",
    "date": "Date",
    "datetime": "DateTime",
}


def get_sqlalchemy_types(model: DomainModel) -> dict:
    """Return the TYPES dict with enum types added."""
    types = dict(SQLALCHEMY_TYPES)
    for enum in model.get_enumerations():
        types[enum.name] = f"Enum('{enum.name}')"
    return types


def get_ids(model: DomainModel) -> dict:
    """Return a dict mapping class names to their id attribute names."""
    ids_dict = {}
    for cls in model.get_classes():
        id_attr = next((attr.name for attr in cls.attributes if attr.is_id), None)
        if not id_attr:
            id_attr = next((attr.name for attr in cls.attributes if attr.name == "id"), None)
        if id_attr:
            ids_dict[cls.name] = id_attr
    return ids_dict


def separate_classes(model: DomainModel) -> tuple:
    """Separate regular classes from association classes.

    Returns:
        tuple: (regular_classes, association_classes)
    """
    classes_list = model.classes_sorted_by_inheritance()
    classes = []
    asso_classes = []
    for class_item in classes_list:
        if isinstance(class_item, AssociationClass):
            asso_classes.append(class_item)
        else:
            classes.append(class_item)
    return classes, asso_classes


def get_concrete_table_inheritance(model: DomainModel) -> list:
    """Return class names that use concrete table inheritance (abstract, no parents, no associations)."""
    concrete_parents = []
    for class_ in model.get_classes():
        if class_.is_abstract and not class_.parents() and not class_.association_ends():
            concrete_parents.append(class_.name)
    return concrete_parents


def used_enums_for_class(class_obj, enumerations) -> list:
    """Return sorted list of enum names used by a class's attributes."""
    enum_names = {e.name for e in enumerations}
    used = set()
    for attr in class_obj.attributes:
        if attr.type.name in enum_names or attr.type.__class__.__name__ == "Enumeration":
            used.add(attr.type.name)
    return sorted(used)
