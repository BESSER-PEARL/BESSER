"""The domain model builder keeps element timestamps (GitHub issue #153).

A generated script must recreate every structural element with the timestamp
stored on the source model, and building the same model twice must give the
same source (the timestamp is read from the model, never from the clock).
"""
from datetime import datetime, timedelta

from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, Class, Constraint, DateType, DomainModel,
    Enumeration, EnumerationLiteral, Generalization, IntegerType, Method,
    Multiplicity, Parameter, Property, StringType,
)
from besser.utilities.buml_code_builder.common import RESERVED_NAMES, timestamp_kwarg
from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
    parse_buml_content,
)

_BASE = datetime(2024, 3, 1, 9, 30, 0, 123456)


def _ts(offset: int) -> datetime:
    return _BASE + timedelta(seconds=offset)


def _build_model() -> DomainModel:
    """A model touching every element kind the builder emits, with fixed timestamps."""
    status = Enumeration(name="Status", timestamp=_ts(1), literals={
        EnumerationLiteral(name="OPEN", timestamp=_ts(2)),
        EnumerationLiteral(name="CLOSED", timestamp=_ts(3)),
    })
    renew = Method(
        name="renew", type=StringType, timestamp=_ts(4),
        parameters=[Parameter(name="days", type=IntegerType, timestamp=_ts(5))],
    )
    library = Class(name="Library", timestamp=_ts(6), methods={renew},
                    attributes={Property(name="name", type=StringType, timestamp=_ts(7))})
    book = Class(name="Book", timestamp=_ts(8),
                 attributes={Property(name="status", type=status, timestamp=_ts(9))})
    novel = Class(name="Novel", timestamp=_ts(10))
    generalization = Generalization(general=book, specific=novel, timestamp=_ts(11))
    holds = BinaryAssociation(name="Library_Book", timestamp=_ts(12), ends={
        Property(name="library", type=library, multiplicity=Multiplicity(1, 1), timestamp=_ts(13)),
        Property(name="books", type=book, multiplicity=Multiplicity(0, "*"), timestamp=_ts(14)),
    })
    loaned = BinaryAssociation(name="Library_Novel", timestamp=_ts(15), ends={
        Property(name="lender", type=library, timestamp=_ts(16)),
        Property(name="novels", type=novel, multiplicity=Multiplicity(0, "*"), timestamp=_ts(17)),
    })
    loan = AssociationClass(name="Loan", association=loaned, timestamp=_ts(18),
                            attributes={Property(name="since", type=DateType, timestamp=_ts(19))})
    named = Constraint(name="named", context=library, language="OCL", timestamp=_ts(20),
                       expression="context Library inv named: self.name <> ''")
    return DomainModel(
        name="Lending", timestamp=_ts(21), types={status, library, book, novel, loan},
        associations={holds, loaned}, generalizations={generalization}, constraints={named},
    )


def _timestamps(model: DomainModel) -> dict:
    """Map a stable key per structural element to its timestamp."""
    found = {("model",): model.timestamp}
    for type_ in model.types:
        if not isinstance(type_, (Class, Enumeration)):
            continue  # shared primitive types, not part of the model
        found[("type", type_.name)] = type_.timestamp
        for literal in getattr(type_, "literals", None) or ():
            found[("literal", type_.name, literal.name)] = literal.timestamp
        for attribute in getattr(type_, "attributes", None) or ():
            found[("attribute", type_.name, attribute.name)] = attribute.timestamp
        for method in getattr(type_, "methods", None) or ():
            found[("method", type_.name, method.name)] = method.timestamp
            for parameter in method.parameters:
                found[("parameter", type_.name, method.name, parameter.name)] = parameter.timestamp
    for association in model.associations:
        found[("association", association.name)] = association.timestamp
        for end in association.ends:
            found[("end", association.name, end.name)] = end.timestamp
    for generalization in model.generalizations:
        found[("generalization", generalization.specific.name)] = generalization.timestamp
    for constraint in model.constraints:
        found[("constraint", constraint.name)] = constraint.timestamp
    return found


def _generate(model: DomainModel, tmp_path, name: str = "model.py") -> str:
    path = tmp_path / name
    domain_model_to_code(model, str(path))
    return path.read_text(encoding="utf-8")


def test_generated_code_recreates_every_timestamp(tmp_path):
    model = _build_model()
    namespace: dict = {}
    exec(_generate(model, tmp_path), namespace)
    expected = _timestamps(model)
    assert len(expected) == 21
    assert _timestamps(namespace["domain_model"]) == expected


def test_timestamps_are_emitted_as_iso_constructor_arguments(tmp_path):
    code = _generate(_build_model(), tmp_path)
    assert "import datetime\n" in code
    assert ('Library = Class(name="Library", '
            'timestamp=datetime.datetime.fromisoformat("2024-03-01T09:30:06.123456"))') in code
    # Never the generation time: every emitted timestamp is one of the model's.
    assert code.count("datetime.datetime.fromisoformat(") == 21


def test_same_model_gives_identical_source(tmp_path):
    model = _build_model()
    assert _generate(model, tmp_path, "a.py") == _generate(model, tmp_path, "b.py")


def test_rebuilt_model_regenerates_identical_source(tmp_path):
    """BUML -> code -> BUML -> code is a fixed point once timestamps are kept."""
    first = _generate(_build_model(), tmp_path, "a.py")
    namespace: dict = {}
    exec(first, namespace)
    assert _generate(namespace["domain_model"], tmp_path, "b.py") == first


def test_editor_importer_keeps_timestamps(tmp_path):
    """The WME's sandboxed loader accepts the emitted ``datetime`` calls."""
    model = _build_model()
    parsed = parse_buml_content(_generate(model, tmp_path))
    assert _timestamps(parsed) == _timestamps(model)


def test_element_without_datetime_timestamp_gets_no_argument():
    cls = Class(name="Untimed")
    cls.timestamp = None
    assert timestamp_kwarg(cls) == ""
    assert timestamp_kwarg(object()) == ""


def test_class_named_datetime_does_not_shadow_the_module(tmp_path):
    assert "datetime" in RESERVED_NAMES
    clash = Class(name="datetime", timestamp=_ts(1),
                  attributes={Property(name="value", type=StringType, timestamp=_ts(2))})
    model = DomainModel(name="Clash", types={clash}, timestamp=_ts(3))
    namespace: dict = {}
    exec(_generate(model, tmp_path), namespace)
    rebuilt = next(iter(namespace["domain_model"].types))
    assert rebuilt.name == "datetime"
    assert rebuilt.timestamp == _ts(1)


def _clash_model(assoc_names, constraint_names) -> DomainModel:
    """A model whose association and constraint names collide with names the
    generated script itself relies on (imports, builtins, class variables)."""
    a = Class(name="A", timestamp=_ts(1))
    b = Class(name="B", timestamp=_ts(2))
    associations = {
        BinaryAssociation(name=name, timestamp=_ts(10 + i), ends={
            Property(name=f"a{i}", type=a, timestamp=_ts(30 + i)),
            Property(name=f"b{i}", type=b, multiplicity=Multiplicity(0, "*"), timestamp=_ts(50 + i)),
        })
        for i, name in enumerate(assoc_names)
    }
    constraints = {
        Constraint(name=name, context=a, language="OCL", timestamp=_ts(70 + i),
                   expression="context A inv: true")
        for i, name in enumerate(constraint_names)
    }
    return DomainModel(name="Clash", types={a, b}, associations=associations,
                       constraints=constraints, timestamp=_ts(90))


_CLASHING_NAMES = ["datetime", "Multiplicity", "Property", "set", "class", "A", "domain_model", "sm"]


def test_association_and_constraint_names_do_not_shadow_script_names(tmp_path):
    """Association and constraint variables used the raw element name, so an
    association called ``datetime`` replaced the module before the next
    ``datetime.datetime.fromisoformat(...)`` and the script failed to run."""
    model = _clash_model(_CLASHING_NAMES, ["datetime", "Constraint", "DomainModel", "None"])
    code = _generate(model, tmp_path)
    namespace: dict = {}
    exec(code, namespace)
    rebuilt = namespace["domain_model"]
    assert _timestamps(rebuilt) == _timestamps(model)
    assert {assoc.name for assoc in rebuilt.associations} == set(_CLASHING_NAMES)
    assert {c.name for c in rebuilt.constraints} == {"datetime", "Constraint", "DomainModel", "None"}
    # Deterministic, and a fixed point once rebuilt.
    assert _generate(model, tmp_path, "again.py") == code
    assert _generate(rebuilt, tmp_path, "rebuilt.py") == code


def test_association_named_datetime_round_trips_through_the_editor_loader(tmp_path):
    model = _clash_model(["datetime"], [])
    parsed = parse_buml_content(_generate(model, tmp_path))
    assert _timestamps(parsed) == _timestamps(model)


def test_defaulted_timestamps_are_not_emitted(tmp_path):
    """A timestamp left to the creation-time default is not written out, so a
    model built without explicit timestamps generates the same code as before."""
    library = Class(name="Library", attributes={Property(name="name", type=StringType)})
    model = DomainModel(name="Plain", types={library})
    assert not library.has_explicit_timestamp
    code = _generate(model, tmp_path)
    assert "fromisoformat" not in code
    assert 'Library = Class(name="Library")\n' in code


def test_explicit_timestamp_flag():
    assert Class(name="A", timestamp=_ts(1)).has_explicit_timestamp
    defaulted = Class(name="B")
    assert not defaulted.has_explicit_timestamp
    defaulted.timestamp = _ts(2)
    assert defaulted.has_explicit_timestamp


def test_flag_survives_deepcopy_and_pickle():
    import copy
    import pickle

    explicit, defaulted = Class(name="A", timestamp=_ts(1)), Class(name="B")
    for clone in (copy.deepcopy, lambda obj: pickle.loads(pickle.dumps(obj))):
        assert clone(explicit).has_explicit_timestamp
        assert not clone(defaulted).has_explicit_timestamp
