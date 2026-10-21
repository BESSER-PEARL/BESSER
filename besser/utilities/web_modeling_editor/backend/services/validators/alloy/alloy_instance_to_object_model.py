"""
Conversion of Alloy instances into B-UML object models.

An instance is the XML file the Alloy Analyzer writes for a solution of the specification
:class:`AlloyGenerator` generated from a domain model. The specification owns the naming
scheme from B-UML elements to Alloy names, so the conversion inverts it: it asks the
specification for the Alloy name of every class, field and value of the domain model, instead
of parsing the names found in the instance.
"""

import datetime
import logging
import xml.etree.ElementTree as ET
from typing import Any

from besser.BUML.metamodel.object import AttributeLink, DataValue, Link, LinkEnd, Object, ObjectModel
from besser.BUML.metamodel.structural import BinaryAssociation, Class, DomainModel, Enumeration, Property
from besser.generators.alloy.alloy_specification import AlloySpecification

logger = logging.getLogger(__name__)

# Prefix of the signatures (and fields) declared by the specification's own module.
_MODULE_PREFIX = "this/"


class AlloyInstanceToObjectModel:
    """Converts the Alloy instances of *specification*, generated from *domain_model*, into object models.

    Each atom of a class signature becomes an object of the most specific class containing it;
    each tuple of an attribute field becomes a value of that attribute, and each tuple of an
    association end field becomes a link (once per pair of objects, although bidirectional
    associations are navigated by two fields).
    """

    def __init__(self, domain_model: DomainModel, specification: AlloySpecification):
        self.domain_model = domain_model
        self.specification = specification
        # Signature label (as in the instance) -> class
        self._classes: dict[str, Class] = {}
        # (signature label, field name) -> attribute or association end
        self._fields: dict[tuple[str, str], Property] = {}
        # Name of the signature of a value (an atom without its "$n" suffix) -> value
        self._values: dict[str, Any] = {"boolean/True": True, "boolean/False": False}
        self._index_classes()
        self._index_fields()
        self._index_values()

    def convert(self, xml_file: str, name: str = "Object_Diagram") -> ObjectModel:
        """Returns the object model *name* of the Alloy instance stored in *xml_file*."""
        root = ET.parse(xml_file).getroot()
        sig_labels = {sig.get("ID"): sig.get("label") for sig in root.iter("sig")}

        objects = self._create_objects(root)
        links: set[tuple] = set()
        for field in root.iter("field"):
            prop = self._fields.get((sig_labels.get(field.get("parentID")), field.get("label")))
            if prop is None:
                continue
            for tuple_elem in field.findall("tuple"):
                atoms = [atom.get("label") for atom in tuple_elem.findall("atom")]
                if len(atoms) != 2 or atoms[0] not in objects:
                    continue
                source, target = objects[atoms[0]], atoms[1]
                if isinstance(prop.owner, BinaryAssociation):
                    self._add_link(source, prop, objects.get(target), links)
                else:
                    self._add_value(source, prop, target)

        object_model = ObjectModel(name=name, objects=set(objects.values()))
        validation = object_model.validate(raise_exception=False)
        for error in validation["errors"]:
            logger.warning("Object model of the Alloy instance '%s' is invalid: %s", xml_file, error)
        return object_model

    # ------------------------------------------------------------------
    # Reverse naming tables
    # ------------------------------------------------------------------

    def _index_classes(self) -> None:
        for cls in self.domain_model.get_classes():
            self._classes[self._sig_label(cls.name)] = cls

    def _index_fields(self) -> None:
        """Indexes the fields as :class:`BumlToAlloy` declares them: a field per attribute of a class,
        and a field per navigable association end in the class at the opposite end."""
        spec = self.specification
        for cls in self.domain_model.get_classes():
            for attr in cls.attributes:
                if spec.has_field(cls.name, attr.name):
                    self._fields[(self._sig_label(cls.name), spec.field_name(cls.name, attr.name))] = attr
        for association in self.domain_model.associations:
            if not isinstance(association, BinaryAssociation):
                continue
            for end in association.ends:
                opposite = end.opposite_end()
                if end.is_navigable and spec.has_field(opposite.type.name, end.name):
                    owner = self._sig_label(opposite.type.name)
                    self._fields[(owner, spec.field_name(opposite.type.name, end.name))] = end

    def _index_values(self) -> None:
        """Indexes the singleton signatures standing for values: enumeration literals, strings and dates."""
        spec = self.specification
        for enumeration in self.domain_model.get_enumerations():
            for literal in enumeration.literals:
                value_name = spec.enumeration_value(enumeration.name, literal.name)
                self._values[spec.signature_name(value_name)] = literal
        # Str and Date are declared by library modules; the specification declares their singletons
        for label in spec.subsignatures("Str"):
            self._values[spec.signature_name(label)] = label.removeprefix("Str_")
        for label in spec.subsignatures("Date"):
            self._values[spec.signature_name(label)] = _date_from_label(label)

    def _sig_label(self, label: str) -> str:
        """The label, in an instance, of the signature the specification declares for *label*."""
        return _MODULE_PREFIX + self.specification.signature_name(label)

    # ------------------------------------------------------------------
    # Objects, values and links
    # ------------------------------------------------------------------

    def _create_objects(self, root: ET.Element) -> dict[str, Object]:
        """Creates an object per atom of a class signature, keyed by atom, instance of the most
        specific class containing the atom."""
        containing: dict[str, list[Class]] = {}
        for sig in root.iter("sig"):
            cls = self._classes.get(sig.get("label"))
            if cls is None:
                continue
            for atom in sig.findall("atom"):
                containing.setdefault(atom.get("label"), []).append(cls)

        objects = {}
        for atom in sorted(containing):
            classes = sorted(containing[atom], key=lambda cls: (-len(cls.all_parents()), cls.name))
            leaf = classes[0]
            if any(cls is not leaf and cls not in leaf.all_parents() for cls in classes):
                logger.warning(
                    "Atom '%s' belongs to unrelated classes %s; it is converted as an instance of '%s'",
                    atom, sorted(cls.name for cls in classes), leaf.name,
                )
            objects[atom] = Object(name=atom.replace("$", "_"), classifier=leaf)
        return objects

    def _add_value(self, obj: Object, attr: Property, atom: str) -> None:
        """Sets the value of the attribute *attr* of *obj* to the value of *atom*."""
        value = self._decode(atom, attr)
        if value is None:
            logger.warning(
                "Skipping attribute '%s' of object '%s': cannot decode the Alloy atom '%s' as a value of type '%s'",
                attr.name, obj.name_, atom, attr.type.name,
            )
            return
        obj.add_slot(AttributeLink(value=DataValue(classifier=attr.type, value=value), attribute=attr))

    def _decode(self, atom: str, attr: Property) -> Any:
        """The value of *atom* as a value of the type of *attr*, or ``None`` when it has none."""
        value = self._values.get(atom.rsplit("$", 1)[0])
        if isinstance(attr.type, Enumeration):
            return value if value in attr.type.literals else None
        type_name = attr.type.name
        if type_name in ("int", "integer", "float"):
            try:
                number = int(atom)
            except ValueError:
                return None
            return float(number) if type_name == "float" else number
        if type_name == "bool":
            return value if isinstance(value, bool) else None
        if type_name in ("str", "string"):
            return value if isinstance(value, str) else None
        if type_name == "date":
            return value if isinstance(value, datetime.date) else None
        return None

    def _add_link(self, source: Object, end: Property, target: Object | None, links: set[tuple]) -> None:
        """Links *source* to *target* through the association end *end*, unless they are already
        linked through it (by the field of the opposite end)."""
        if target is None:
            return
        opposite = end.opposite_end()
        key = (end.owner, frozenset({(opposite, source), (end, target)}))
        if key in links:
            return
        links.add(key)
        Link(
            name=f"{source.name_}_{end.name}_{target.name_}",
            association=end.owner,
            connections=[
                LinkEnd(name=opposite.name, association_end=opposite, object=source),
                LinkEnd(name=end.name, association_end=end, object=target),
            ],
        )


def _date_from_label(label: str) -> datetime.date:
    """The date of the date signature *label*, ``Date_YYYY_MM_DD``."""
    return datetime.datetime.strptime(label, "Date_%Y_%m_%d").date()
