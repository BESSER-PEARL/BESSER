"""
An Alloy specification, built by adding its components and written as ``.als`` files.

:class:`AlloySpecification` is the entry point: its components (:class:`Signature`,
:class:`Field`, :class:`Enumeration`, :class:`Function`, :class:`Fact`, :class:`Predicate` and
:class:`RunCommand`) are created through its ``add_*`` methods.  It knows nothing about
B-UML: its signatures, fields and enumeration values are declared with *labels* (the names
of the elements they come from), and it owns the naming scheme that turns labels into Alloy
names.
Translators ask it for those names, so they all agree on them:

- a signature is named after its label: ``Team``;
- a field is prefixed by its signature: ``Team_players``;
- an enumeration value is prefixed by ``ENUM_`` and its enumeration: ``ENUM_Position_FORWARD``.

Formulas, expressions and parameters are Alloy text, written with those names.
"""

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

from jinja2 import Environment, FileSystemLoader

_TEMPLATE = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
).get_template("alloy_specification.j2")

# Reserved words of Alloy 6, which cannot be used as identifiers.
_ALLOY_KEYWORDS = frozenset({
    "abstract", "after", "all", "always", "and", "as", "assert", "before", "but", "check",
    "disj", "else", "enabled", "enum", "event", "eventually", "exactly", "expect", "extends",
    "fact", "for", "fun", "historically", "iden", "iff", "implies", "in", "Int", "int", "let",
    "lone", "module", "no", "none", "not", "once", "one", "open", "or", "pred", "private",
    "releases", "run", "seq", "set", "sig", "since", "some", "steps", "String", "sum", "this",
    "triggered", "univ", "until", "var",
})

# Characters that cannot be part of an Alloy identifier.
INVALID_IDENTIFIER_CHARACTERS = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class AlloyBounds:
    """What the ``run`` command of a specification searches: *objects* atoms per bounded signature, and
    the values: *bitwidth* bits for ``Int``, and *strings* ``Str`` and *dates* ``Date`` atoms, each ``None``
    when the specification has no such values.  The integers are always bounded, since the multiplicity
    facts and OCL ``size()`` count objects."""

    objects: int
    bitwidth: int
    strings: int | None = None
    dates: int | None = None

    @property
    def integers(self) -> tuple[int, int]:
        """The range of the integers."""
        return -2 ** (self.bitwidth - 1), 2 ** (self.bitwidth - 1) - 1

    def __str__(self) -> str:
        """The bounds in words, e.g. ``5 objects per class, integers in [-16, 15], 5 strings and 5 dates``."""
        parts = [f"{_count(self.objects, 'object')} per class", f"integers in [{self.integers[0]}, {self.integers[1]}]"]
        if self.strings is not None:
            parts.append(_count(self.strings, "string"))
        if self.dates is not None:
            parts.append(_count(self.dates, "date"))
        return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _count(number: int, noun: str) -> str:
    """*number* followed by *noun*, in the plural unless *number* is 1."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _identifier(name: str) -> str:
    """A valid Alloy identifier for *name*.

    Alloy identifiers start with a letter and contain letters, digits and underscores, and
    cannot be reserved words: invalid characters are dropped, and names that do not start
    with a letter or that are reserved words are prefixed with ``id_``.
    """
    identifier = INVALID_IDENTIFIER_CHARACTERS.sub("", name)
    if not identifier:
        raise ValueError(f"Cannot create a valid Alloy identifier from '{name}'")
    if not identifier[0].isalpha() or identifier in _ALLOY_KEYWORDS:
        return f"id_{identifier}"
    return identifier


class AlloySpecification:
    """An Alloy module named *name*, stored in the file ``<name>.als``.

    Components are added with the ``add_*`` methods and printed in the order they were added.
    The library modules declaring the types the components use (see :class:`LibraryFactory`)
    are opened, and written together with the specification, when it is written.
    """

    def __init__(self, name: str):
        self.name = name
        self._opens: list[str] = []
        self._signatures: list[Signature] = []
        self._signatures_by_label: dict[str, Signature] = {}
        self._enumerations: dict[str, Enumeration] = {}
        self._functions: list[Function] = []
        self._facts: list[Fact] = []
        self._predicates: list[Predicate] = []
        self._run: RunCommand | None = None

    @property
    def file_name(self) -> str:
        """The name of the file storing the specification."""
        return f"{self.name}.als"

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_signature(
        self,
        label: str,
        *,
        is_abstract: bool = False,
        multiplicity: str = "",
        extends: str | None = None,
    ) -> None:
        """Declares the signature *label*, a subsignature of *extends* if given."""
        self._declare(Signature(self, [label], is_abstract, multiplicity, extends))

    def add_enumeration(self, label: str, values: Iterable[str]) -> None:
        """Declares the abstract signature *label*, extended by a singleton signature per value."""
        enumeration = Enumeration(label, values)
        self._enumerations[label] = enumeration
        self._declare(Signature(self, [label], is_abstract=True))
        self._declare(Signature(self, enumeration.value_names(), multiplicity="one", extends=label))

    def add_field(self, signature: str, label: str, type: str, multiplicity: str = "") -> None:
        """Adds the field *label* of type *type* (a label) to the declared *signature*."""
        self.signature(signature).add_field(label, type, multiplicity)

    def add_function(self, name: str, parameters: dict[str, str], result: str, expression: str) -> None:
        """Adds the function ``name[parameters]: result`` computing *expression*."""
        self._functions.append(Function(name, parameters, result, expression))

    def add_fact(self, formula: str, types: Iterable[str] = ()) -> None:
        """Adds a fact stating *formula*.

        *types* are the labels of the types the formula refers to by name (e.g. ``Bool`` for
        ``True``), besides those of the fields it navigates.
        """
        self._facts.append(Fact(formula, types))

    def add_predicate(self, name: str, formulas: Iterable[str], parameters: dict[str, str] | None = None) -> None:
        """Adds the predicate ``name[parameters]``, the conjunction of *formulas*."""
        self._predicates.append(Predicate(name, parameters or {}, formulas))

    def set_run(self, predicate: str, signatures: Iterable[str], bounds: AlloyBounds) -> None:
        """Sets the ``run`` command of *predicate*, bounding each of the *signatures* (labels) by
        ``bounds.objects``, and ``Str``, ``Date`` and ``Int`` by the *bounds* that give them a value."""
        self._run = RunCommand(self, predicate, signatures, bounds)

    def get_scopes(self) -> AlloyBounds | None:
        """The bounds of the ``run`` command, ``None`` before it is set."""
        return self._run.bounds if self._run else None

    # ------------------------------------------------------------------
    # Names
    # ------------------------------------------------------------------

    def signature_name(self, label: str) -> str:
        """The Alloy name of the signature *label*; other types (e.g. ``Int``) keep their label."""
        return _identifier(label) if label in self._signatures_by_label else label

    def field_name(self, signature: str, label: str) -> str:
        """The Alloy name of the field *label* of *signature*, possibly inherited."""
        return self._field(signature, label).name

    def field_type(self, signature: str, label: str) -> str:
        """The type (a label) of the field *label* of *signature*, possibly inherited."""
        return self._field(signature, label).type

    def field_multiplicity(self, signature: str, label: str) -> str:
        """The multiplicity of the field *label* of *signature*, possibly inherited."""
        return self._field(signature, label).multiplicity

    def has_field(self, signature: str, label: str) -> bool:
        """Whether *signature* has the field *label*, possibly inherited."""
        return signature in self._signatures_by_label and self.signature(signature).find_field(label) is not None

    def has_signature(self, label: str) -> bool:
        """Whether the signature *label* is declared."""
        return label in self._signatures_by_label

    def enumeration_value(self, enumeration: str, value: str) -> str:
        """The Alloy name of the value *value* of the enumeration *enumeration*."""
        if enumeration not in self._enumerations:
            raise KeyError(f"'{enumeration}' is not an enumeration of '{self.name}'")
        return self._enumerations[enumeration].value_name(value)

    def signature(self, label: str) -> "Signature":
        """The declared signature *label*."""
        if label not in self._signatures_by_label:
            raise KeyError(f"Signature '{label}' is not declared in '{self.name}'")
        return self._signatures_by_label[label]

    def subsignatures(self, label: str) -> list[str]:
        """Labels of the signatures extending *label*, directly or indirectly."""
        children = [child for signature in self._signatures if signature.is_below(label) for child in signature.labels]
        return children + [grandchild for child in children for grandchild in self.subsignatures(child)]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def render(self) -> str:
        """The Alloy text of the specification."""
        return _TEMPLATE.render(
            name=self.name,
            opens=self._opens,
            signatures=self._signatures,
            functions=self._functions,
            facts=self._facts,
            predicates=self._predicates,
            run=self._run,
        )

    def write(self, directory: str) -> str:
        """Writes the specification and the modules it opens to *directory*; returns its path."""
        os.makedirs(directory, exist_ok=True)
        for library in self._open_libraries():
            library.write(directory)
        path = os.path.join(directory, self.file_name)
        with open(path, mode="w", encoding="utf-8") as f:
            f.write(self.render())
        return path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_libraries(self) -> list["AlloySpecification"]:
        """Opens, and returns, the library modules declaring the types used by the fields, the facts
        and the run command."""
        used_types = {field.type for signature in self._signatures for field in signature.fields}
        used_types.update(type_name for fact in self._facts for type_name in fact.types)
        if self._run:
            used_types.update(self._run.labels)
        factory = LibraryFactory()
        libraries = [
            factory.create(type_name) for type_name in sorted(used_types)
            if type_name not in self._signatures_by_label and factory.declares(type_name)
        ]
        self._opens += [library.name for library in libraries if library.name not in self._opens]
        return libraries

    def _declare(self, signature: "Signature") -> None:
        for label in signature.labels:
            if label in self._signatures_by_label:
                raise ValueError(f"Signature '{label}' is already declared in '{self.name}'")
            self._signatures_by_label[label] = signature
        self._signatures.append(signature)

    def _field(self, signature: str, label: str) -> "Field":
        found = self.signature(signature).find_field(label)
        if found is None:
            raise KeyError(f"Signature '{signature}' has no field '{label}'")
        return found


class Signature:
    """A signature declaration of a specification, possibly declaring several signatures at once."""

    def __init__(
        self,
        specification: AlloySpecification,
        labels: Iterable[str],
        is_abstract: bool = False,
        multiplicity: str = "",
        extends: str | None = None,
    ):
        self.specification = specification
        self.labels = list(labels)
        self.is_abstract = is_abstract
        self.multiplicity = multiplicity
        self.extends = extends
        self._fields: dict[str, Field] = {}

    def add_field(self, label: str, type: str, multiplicity: str) -> None:
        self._fields[label] = Field(self, label, type, multiplicity)

    def find_field(self, label: str) -> "Field | None":
        """The field *label* of the signature or of its parents."""
        if label in self._fields:
            return self._fields[label]
        for parent in self.parents():
            found = self.specification.signature(parent).find_field(label)
            if found:
                return found
        return None

    def parents(self) -> list[str]:
        """Labels of the signatures this one extends."""
        return [self.extends] if self.extends else []

    def is_below(self, label: str) -> bool:
        """Whether this signature extends *label*."""
        return label in self.parents()

    @property
    def name(self) -> str:
        """The Alloy name of the (first) declared signature."""
        return self.specification.signature_name(self.labels[0])

    @property
    def header(self) -> str:
        """The declaration without its body, e.g. ``one sig A, B extends E``."""
        names = self.specification.signature_name
        parts = ["abstract"] if self.is_abstract else []
        if self.multiplicity:
            parts.append(self.multiplicity)
        parts += ["sig", ", ".join(names(label) for label in self.labels)]
        if self.extends:
            parts += ["extends", names(self.extends)]
        return " ".join(parts)

    @property
    def fields(self) -> list["Field"]:
        return list(self._fields.values())


class Field:
    """A field of a signature, named after the signature and its label."""

    def __init__(self, signature: Signature, label: str, type: str, multiplicity: str):
        self.signature = signature
        self.name = _identifier(f"{signature.name}_{label}")
        self.type = type
        self.multiplicity = multiplicity

    def __str__(self) -> str:
        type_name = self.signature.specification.signature_name(self.type)
        if self.multiplicity:
            return f"{self.name}: {self.multiplicity} {type_name}"
        return f"{self.name}: {type_name}"


class Enumeration:
    """The values of an enumeration, named ``ENUM_<enumeration>_<value>``."""

    def __init__(self, label: str, values: Iterable[str]):
        self.label = label
        self._names = {value: _identifier(f"ENUM_{_identifier(label)}_{value}") for value in values}

    def value_names(self) -> list[str]:
        return list(self._names.values())

    def value_name(self, value: str) -> str:
        if value not in self._names:
            raise KeyError(f"'{value}' is not a value of '{self.label}' (values: {', '.join(self._names)})")
        return self._names[value]


class Function:
    """A function ``name[parameters]: result`` computing an expression."""

    def __init__(self, name: str, parameters: dict[str, str], result: str, expression: str):
        self.name = name
        self.parameters = dict(parameters)
        self.result = result
        self.body = [expression]

    @property
    def header(self) -> str:
        parameters = ", ".join(f"{name}: {type}" for name, type in self.parameters.items())
        return f"{self.name}[{parameters}]: {self.result}"


class Fact:
    """A fact stating a formula, with the types (labels) the formula refers to by name."""

    def __init__(self, formula: str, types: Iterable[str]):
        self.formula = formula
        self.types = list(types)

    def __str__(self) -> str:
        return self.formula


class Predicate:
    """A predicate ``name[parameters]``, the conjunction of its formulas."""

    def __init__(self, name: str, parameters: dict[str, str], formulas: Iterable[str]):
        self.name = name
        self.parameters = dict(parameters)
        self.body = list(formulas)

    @property
    def header(self) -> str:
        if not self.parameters:
            return self.name
        return f"{self.name}[{', '.join(f'{name}: {type}' for name, type in self.parameters.items())}]"


class RunCommand:
    """A ``run`` command of a predicate, bounding signatures (labels) by their scopes."""

    def __init__(
        self, specification: AlloySpecification, predicate: str, signatures: Iterable[str], bounds: AlloyBounds
    ):
        self.specification = specification
        self.predicate = predicate
        self.bounds = bounds
        self._scopes = {label: bounds.objects for label in signatures}
        for label, bound in (("Str", bounds.strings), ("Date", bounds.dates)):
            if bound is not None:
                self._scopes[label] = bound
        self._scopes["Int"] = bounds.bitwidth

    @property
    def labels(self) -> list[str]:
        """Labels of the bounded signatures."""
        return list(self._scopes)

    @property
    def scopes(self) -> list[str]:
        return [f"{bound} {self.specification.signature_name(label)}" for label, bound in self._scopes.items()]


class BooleanLibrary(AlloySpecification):
    """Booleans: the two values of the abstract signature Bool."""

    def __init__(self):
        super().__init__("boolean")
        self.add_signature("Bool", is_abstract=True)
        self.add_signature("True", multiplicity="one", extends="Bool")
        self.add_signature("False", multiplicity="one", extends="Bool")


class UninterpretedDateLibrary(AlloySpecification):
    """Dates: totally ordered atoms, with (in)equality and order comparisons.

    OCL translations use (in)equality only; the order comparisons state the chronological
    order of the dates declared by the specification using the library.
    """

    def __init__(self):
        super().__init__("date")
        self._opens.append("util/ordering[Date]")
        self.add_signature("Date")
        dates = {"a": "Date", "b": "Date"}
        self.add_predicate("dateEq", ["a = b"], dates)
        self.add_predicate("dateNe", ["a != b"], dates)
        self.add_predicate("dateLt", ["a in prevs[b]"], dates)
        self.add_predicate("dateLte", ["a = b or a in prevs[b]"], dates)
        self.add_predicate("dateGt", ["a in nexts[b]"], dates)
        self.add_predicate("dateGte", ["a = b or a in nexts[b]"], dates)


class UninterpretedStringsLibrary(AlloySpecification):
    """Strings: uninterpreted atoms, with (in)equality only.

    ``Str`` is abstract: every string is a singleton ``Str`` declared by the specification
    using the library, so two strings are equal exactly when they are the same singleton.
    Strings have no characters, hence no operations.
    """

    def __init__(self):
        super().__init__("strings")
        self.add_signature("Str", is_abstract=True)
        strings = {"a": "Str", "b": "Str"}
        self.add_predicate("strEq", ["a = b"], strings)
        self.add_predicate("strNe", ["a != b"], strings)


class LibraryFactory:
    """Creates the library modules declaring the types Alloy lacks, to be opened by the
    specifications using them: ``Bool``, ``Date`` and ``Str`` (uninterpreted strings)."""

    def declares(self, type_name: str) -> bool:
        """Whether a library module declares the type *type_name*."""
        return type_name in ("Bool", "Date", "Str")

    def create(self, type_name: str) -> AlloySpecification:
        """A new library module declaring the type *type_name*."""
        if type_name == "Bool":
            return BooleanLibrary()
        elif type_name == "Date":
            return UninterpretedDateLibrary()
        elif type_name == "Str":
            return UninterpretedStringsLibrary()
        else:
            raise ValueError(f"Library module for type '{type_name}' is not implemented")
