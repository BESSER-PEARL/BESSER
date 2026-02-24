__all__ = ["csv_to_domain_model"]

import csv
import os
import re
from besser.BUML.metamodel.structural.structural import (
    DomainModel, Class, Property, StringType, IntegerType,
    FloatType, BooleanType, BinaryAssociation, Multiplicity,
    UNLIMITED_MAX_MULTIPLICITY
)

def infer_type(values):
    """Infer the BUML type from a list of values."""
    is_int = True
    is_float = True
    is_bool = True
    for v in values:
        if v == '' or v is None:
            continue
        try:
            int(v)
        except ValueError:
            is_int = False
        try:
            float(v)
        except ValueError:
            is_float = False
        if str(v).lower() not in {'true', 'false', '0', '1'}:
            is_bool = False
    if is_int:
        return IntegerType
    if is_float:
        return FloatType
    if is_bool:
        return BooleanType
    return StringType

def normalize_name(name: str) -> str:
    name = name.replace('-', '_').replace(' ', '_')
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    if name.endswith('s') and not name.endswith('ss'):
        name = name[:-1]
    return name

def to_camel_case(s: str) -> str:
    parts = re.split(r'[_\-\s]+', s)
    return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])

def pluralize(s: str) -> str:
    if s.endswith('s'):
        return s
    return s + 's'

def csv_to_domain_model(csv_paths, model_name=None) -> DomainModel:
    """
    Reverse engineer one or more CSV files into a BUML DomainModel with classes and associations.
    Args:
        csv_paths: List of CSV file paths or file-like objects.
        model_name: Optional name for the DomainModel.
    Returns:
        DomainModel instance.
    """
    class_map = {}  # class_name -> Class
    field_types = {}  # class_name -> {field: type}
    field_props = {}  # class_name -> {field: Property}
    id_fields = {}  # class_name -> id field name
    fk_candidates = {}  # class_name -> set of fields that are foreign keys
    # 1. First pass: collect id fields for all classes
    for csv_path in csv_paths:
        if isinstance(csv_path, str):
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = list(csv.DictReader(f))
        else:
            reader = list(csv.DictReader(csv_path))
        if not reader:
            continue
        fieldnames = list(reader[0].keys())
        class_name = os.path.splitext(os.path.basename(csv_path if isinstance(csv_path, str) else getattr(csv_path, 'name', 'CSVTable')))[0]
        first_field = fieldnames[0]
        id_field = None
        for field in fieldnames:
            if field == first_field and (field.lower() == "id" or field.lower().endswith("_id")):
                id_field = field
        if id_field:
            id_fields[class_name] = id_field
        fk_candidates[class_name] = {field for field in fieldnames if (field.endswith('_id') or field.lower().endswith('id')) and field != id_field}

    # 2. Second pass: create classes and properties, but always create Property for all fields (add as attribute only if not a true association)
    for csv_path in csv_paths:
        if isinstance(csv_path, str):
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = list(csv.DictReader(f))
        else:
            reader = list(csv.DictReader(csv_path))
        if not reader:
            continue
        fieldnames = list(reader[0].keys())
        columns = {field: [row[field] for row in reader] for field in fieldnames}
        types = {field: infer_type(values) for field, values in columns.items()}
        class_name = os.path.splitext(os.path.basename(csv_path if isinstance(csv_path, str) else getattr(csv_path, 'name', 'CSVTable')))[0]
        buml_class = Class(class_name)
        class_map[class_name] = buml_class
        field_types[class_name] = types
        field_props[class_name] = {}
        first_field = fieldnames[0]
        id_field = id_fields.get(class_name)
        for field in fieldnames:
            is_id = False
            if field == first_field and (field.lower() == "id" or field.lower().endswith("_id")):
                is_id = True
            prop = Property(name=field, type=types[field], owner=buml_class, is_id=is_id)
            field_props[class_name][field] = prop
            # Only add as attribute if not a true association (foreign key to another class)
            if not (
                field in fk_candidates[class_name]
                and any(
                    normalize_name(target_class)
                    == normalize_name(field[:-3] if field.lower().endswith('_id') else field[:-2] if field.lower().endswith('id') else field)
                    for target_class in id_fields
                )
            ):
                buml_class.add_attribute(prop)

    # 3. Detect foreign keys and create BinaryAssociations
    associations = set()
    for class_name, buml_class in class_map.items():
        for field, prop in field_props[class_name].items():
            if field in fk_candidates[class_name]:
                ref_raw = field[:-3] if field.lower().endswith('_id') else field[:-2] if field.lower().endswith('id') else field
                ref_name = normalize_name(ref_raw)
                for target_class in id_fields:
                    if normalize_name(target_class) == ref_name:
                        target_cls = class_map[target_class]
                        fk_role = to_camel_case(target_class)
                        ref_role = pluralize(class_name.lower())
                        end_fk = Property(name=fk_role, type=target_cls, owner=None, multiplicity=Multiplicity(0, 1))
                        end_target = Property(name=ref_role, type=buml_class, owner=None, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY))
                        assoc_name = f"{class_name}_{field}_to_{target_class}"
                        assoc = BinaryAssociation(assoc_name, ends={end_fk, end_target})
                        associations.add(assoc)
                        break
    model = DomainModel(model_name or "CSVModel", types=set(class_map.values()), associations=associations)
    return model
