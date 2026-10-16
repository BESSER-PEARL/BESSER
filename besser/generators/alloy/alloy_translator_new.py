"""
Translation of B-UML domain models into Alloy specifications.

:class:`BumlToAlloy` builds the :class:`AlloySpecification` of a domain model, translating
its OCL invariants with :class:`OclToAlloy`.  Signatures, fields and enumeration values are
declared with the names of the B-UML elements they come from; the specification derives
their Alloy names.

The metamodel stores classes, attributes, literals, associations, association ends and
constraints in sets, so every collection is iterated in sorted order to make the output
deterministic.
"""

from besser.BUML.metamodel.structural import (
    UNLIMITED_MAX_MULTIPLICITY,
    BinaryAssociation,
    Class,
    DomainModel,
    Enumeration,
    Multiplicity,
    Property,
    Type,
)
from besser.generators.alloy.alloy_specification import AlloySpecification
from besser.generators.alloy.ocl_translator_new import Literals, OclToAlloy


class BumlToAlloy:
    """Builds the :class:`AlloySpecification` of a B-UML domain model.

    The specification has a signature per class (fields for its attributes and navigable
    association ends), a signature hierarchy per enumeration, facts for the association
    multiplicities that field declarations cannot express, facts relating the two fields of
    bidirectional associations, a fact per OCL invariant, and an ``instance_model`` predicate
    with its ``run`` command.
    """

    def __init__(self, model: DomainModel, scope: int):
        self.model = model
        self.scope = scope
        self.classes = sorted(model.get_classes(), key=lambda cls: cls.name)
        self.enumerations = sorted(model.get_enumerations(), key=lambda enum: enum.name)
        self.associations = sorted(
            model.associations,
            key=lambda assoc: (assoc.name, [end.name for end in self._ends(assoc)]),
        )
        self.constraints = sorted(model.constraints, key=lambda constraint: constraint.name)
        # Alloy types of the attributes, except enumerations (declared by their own signatures)
        self.attribute_types = sorted({
            self._type(attr.type)
            for cls in self.classes for attr in cls.attributes
            if not isinstance(attr.type, Enumeration)
        })

    def translate(self) -> AlloySpecification:
        """Returns the Alloy specification of the model."""
        spec = AlloySpecification("model")
        for enumeration in self.enumerations:
            self._add_enumeration(spec, enumeration)
        for cls in self.classes:
            self._add_class(spec, cls)
        for association in self.associations:
            self._add_multiplicity_facts(spec, association)
            self._add_inverse_fact(spec, association)
        literals = self._add_invariants(spec)
        self._add_instance_predicate(spec)
        self._set_run_command(spec, literals)
        return spec

    # ------------------------------------------------------------------
    # Parts of the specification
    # ------------------------------------------------------------------

    def _add_enumeration(self, spec: AlloySpecification, enumeration: Enumeration) -> None:
        """Declares *enumeration*: an abstract signature extended by a singleton per literal."""
        literals = sorted(literal.name for literal in enumeration.literals)
        spec.add_enumeration(enumeration.name, literals)

    def _add_class(self, spec: AlloySpecification, cls: Class) -> None:
        """Declares the signature of *cls*, with its attributes and navigable association ends as fields."""
        parents = sorted(parent.name for parent in cls.parents())
        spec.add_signature(
            cls.name,
            is_abstract=cls.is_abstract,
            extends=parents[0] if len(parents) == 1 else None,
            subset_of=parents if len(parents) > 1 else (),
        )
        for attr in sorted(cls.attributes, key=lambda attr: attr.name):
            spec.add_field(cls.name, attr.name, self._type(attr.type))
        # A navigable end is a field of the class at the opposite end.
        for association in self.associations:
            first, second = self._ends(association)
            for end, opposite in ((first, second), (second, first)):
                if end.is_navigable and opposite.type is cls:
                    spec.add_field(cls.name, end.name, end.type.name, self._field_multiplicity(end.multiplicity))

    def _add_multiplicity_facts(self, spec: AlloySpecification, association: BinaryAssociation) -> None:
        """Adds facts bounding the number of objects reachable through each end of *association*.

        The bounds of an end are checked from the class at the opposite end, navigating the
        end's field, or the opposite end's field backwards when the end itself is not
        navigable.  Lower bounds of 0 and unlimited upper bounds are skipped, as are navigable
        1..1 ends, whose ``one`` field already enforces them.
        """
        first, second = self._ends(association)
        for end, opposite in ((first, second), (second, first)):
            bounds = end.multiplicity
            if end.is_navigable and bounds.min == 1 and bounds.max == 1:
                continue
            if end.is_navigable:
                navigation = f"a.{spec.field_name(opposite.type.name, end.name)}"
            elif opposite.is_navigable:
                navigation = f"{spec.field_name(end.type.name, opposite.name)}.a"
            else:
                continue
            source = spec.signature_name(opposite.type.name)
            if bounds.min >= 1:
                spec.add_fact(f"all a: {source} | #({navigation}) >= {bounds.min}")
            if bounds.max < UNLIMITED_MAX_MULTIPLICITY:
                spec.add_fact(f"all a: {source} | #({navigation}) <= {bounds.max}")

    def _add_inverse_fact(self, spec: AlloySpecification, association: BinaryAssociation) -> None:
        """Adds a fact stating that the fields navigating to the two ends of *association* are inverses."""
        first, second = self._ends(association)
        if first.is_navigable and second.is_navigable:
            to_first = spec.field_name(second.type.name, first.name)
            to_second = spec.field_name(first.type.name, second.name)
            spec.add_fact(f"{to_first} = ~{to_second}")

    def _add_invariants(self, spec: AlloySpecification) -> Literals:
        """Adds a fact per OCL invariant of the model; returns the literals of the invariants."""
        ocl = OclToAlloy(spec, self.model)
        for constraint in self.constraints:
            ocl.add_invariant(constraint)
        return ocl.literals

    def _add_instance_predicate(self, spec: AlloySpecification) -> None:
        """Adds the ``instance_model`` predicate, requiring an instance of every signature."""
        spec.add_predicate(
            "instance_model", [f"some {spec.signature_name(label)}" for label in self._instance_signatures()]
        )

    def _set_run_command(self, spec: AlloySpecification, literals: Literals) -> None:
        """Sets the ``run`` command of ``instance_model``.

        Strings are sequences of characters, so ``seq`` bounds their length: the scope, or the
        length of the longest OCL string literal when longer.  The integer bitwidth must be
        able to index it and to hold the largest OCL integer literal.
        """
        scopes = {label: self.scope for label in self._instance_signatures()}
        bitwidth = max(self.scope, literals.largest_integer().bit_length() + 1)
        if "Str" in self.attribute_types or literals.strings():
            maxseq = max(self.scope, literals.longest_string())
            scopes.update({"Int": max(bitwidth, maxseq.bit_length() + 1), "Str": self.scope, "seq": maxseq})
        elif "Int" in self.attribute_types or literals.integers():
            scopes["Int"] = bitwidth
        spec.set_run("instance_model", scopes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _instance_signatures(self) -> list[str]:
        """Signatures that must have instances: classes and dates."""
        labels = [cls.name for cls in self.classes]
        if "Date" in self.attribute_types:
            labels.append("Date")
        return labels

    def _type(self, buml_type: Type) -> str:
        """The label of the Alloy type of a B-UML type."""
        if isinstance(buml_type, (Class, Enumeration)):
            return buml_type.name
        if buml_type.name not in _PRIMITIVE_TYPES:
            raise NotImplementedError(f"BumlToAlloy does not support attributes of type '{buml_type.name}'")
        return _PRIMITIVE_TYPES[buml_type.name]

    def _field_multiplicity(self, multiplicity: Multiplicity) -> str:
        """The multiplicity keyword of an association end field."""
        if multiplicity.min == 0 and multiplicity.max == 1:
            return "lone"
        if multiplicity.max > 1:
            return "set"
        return "one"

    def _ends(self, association: BinaryAssociation) -> tuple[Property, Property]:
        """The two ends of *association*, ordered by name."""
        first, second = sorted(association.ends, key=lambda end: end.name)
        return first, second


# Alloy types of B-UML primitive types; the others (``time``, ``timedelta`` and ``any``) are not supported.
_PRIMITIVE_TYPES = {
    "int": "Int",
    "integer": "Int",
    "float": "Int",
    "bool": "Bool",
    "str": "Str",
    "string": "Str",
    "date": "Date",
    "datetime": "Date",
}