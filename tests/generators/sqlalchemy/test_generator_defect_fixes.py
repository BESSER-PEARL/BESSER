"""Regression tests for SQLAlchemy generator defect fixes.

Covers:
1. ``timedelta`` / ``any`` primitives map to ``Interval`` / ``PickleType`` and
   the generated module is importable and usable end-to-end.
2. Unknown attribute types fail loudly (ValueError naming the attribute)
   instead of rendering an empty string that crashes on import.
3. Non-sqlite DBMS output reads DATABASE_URL from the environment, drops
   ``echo=True``, and defers ``create_all`` behind ``if __name__ == "__main__":``
   so importing the module never touches the database.
4. The sqlite default URL matches the FastAPI backend's ``init_db`` default
   (one database for the composite backend).
5. N-ary associations emit an explicit warning comment instead of being
   silently dropped.
"""

import datetime
import importlib.util
import os
import sys

import pytest

from besser.BUML.metamodel.structural import (
    Association, AnyType, Class, DomainModel, Multiplicity, Property,
    StringType, TimeDeltaType,
)
from besser.generators.sql_alchemy import SQLAlchemyGenerator


def _generate(model: DomainModel, tmpdir, dbms: str = "sqlite", subdir: str = "out"):
    """Run the generator and return (path, generated_code)."""
    output_dir = tmpdir.mkdir(subdir)
    SQLAlchemyGenerator(model=model, output_dir=str(output_dir)).generate(dbms=dbms)
    path = os.path.join(str(output_dir), "sql_alchemy.py")
    with open(path, "r", encoding="utf-8") as f:
        return path, f.read()


def _import_module(path: str, module_name: str):
    """Import the generated file directly — it must be side-effect free."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _single_class_model(model_name: str = "Catalog", class_name: str = "Item") -> DomainModel:
    cls = Class(name=class_name)
    cls.attributes = {Property(name="name", type=StringType)}
    model = DomainModel(name=model_name)
    model.add_type(cls)
    return model


# ---------------------------------------------------------------------------
# 1. timedelta / any primitives
# ---------------------------------------------------------------------------

def test_timedelta_and_any_attributes_generate_importable_code(tmpdir):
    task = Class(name="Task")
    task.attributes = {
        Property(name="title", type=StringType),
        Property(name="duration", type=TimeDeltaType),
        Property(name="payload", type=AnyType),
    }
    model = DomainModel(name="Scheduler")
    model.add_type(task)

    path, code = _generate(model, tmpdir, subdir="timedelta_any")

    # Column types and Python annotations are rendered (no empty strings)
    assert "duration: Mapped_[dt_timedelta] = mapped_column(Interval_)" in code
    assert "payload: Mapped_[object] = mapped_column(PickleType_)" in code
    # ... and the names referenced are actually imported
    assert "Interval as Interval_" in code
    assert "PickleType as PickleType_" in code
    assert "timedelta as dt_timedelta" in code

    module = _import_module(path, "timedelta_any_module")

    # End-to-end round trip through an in-memory sqlite database
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    engine = create_engine("sqlite:///:memory:")
    module.Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(module.Task(
            title="backup",
            duration=datetime.timedelta(hours=2, minutes=30),
            payload={"retries": [1, 2, 3]},
        ))
        session.commit()
    with Session(engine) as session:
        row = session.query(module.Task).one()
        assert row.duration == datetime.timedelta(hours=2, minutes=30)
        assert row.payload == {"retries": [1, 2, 3]}


# ---------------------------------------------------------------------------
# 2. Unknown attribute types fail loudly
# ---------------------------------------------------------------------------

def test_unknown_attribute_type_raises_value_error(tmpdir):
    """A type that is neither a known primitive nor a model enumeration must
    raise instead of rendering an empty column type."""
    other = Class(name="Other")
    thing = Class(name="Thing")
    thing.attributes = {Property(name="ref", type=other)}
    model = DomainModel(name="Unknowns")
    model.add_type(other)
    model.add_type(thing)

    generator = SQLAlchemyGenerator(model=model, output_dir=str(tmpdir))
    with pytest.raises(ValueError) as exc_info:
        generator.generate(dbms="sqlite")

    message = str(exc_info.value)
    assert "Thing.ref" in message
    assert "'Other'" in message
    # The generator must fail before writing a broken file
    assert not os.path.isfile(os.path.join(str(tmpdir), "sql_alchemy.py"))


def test_enum_entries_do_not_leak_between_generator_instances():
    """Enum names registered for one model must not silence the unknown-type
    validation for a different model generated in the same process."""
    from besser.BUML.metamodel.structural import Enumeration, EnumerationLiteral

    color = Enumeration(name="Color", literals={EnumerationLiteral(name="RED")})
    first = DomainModel(name="First")
    first.add_type(color)
    SQLAlchemyGenerator(model=first)  # registers 'Color' for THIS instance only

    assert "Color" not in SQLAlchemyGenerator.TYPES
    assert "Color" not in SQLAlchemyGenerator(model=DomainModel(name="Second")).TYPES


# ---------------------------------------------------------------------------
# 3. Non-sqlite DBMS output is import-safe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dbms", ["postgresql", "mysql", "mariadb", "mssql", "oracle"])
def test_non_sqlite_output_reads_url_from_env_and_defers_create_all(tmpdir, dbms):
    model = _single_class_model()
    path, code = _generate(model, tmpdir, dbms=dbms, subdir=f"dbms_{dbms}")

    # Must at least be valid Python (no DB driver needed for this check)
    compile(code, path, "exec")

    assert 'DATABASE_URL = os.getenv("DATABASE_URL", ' in code
    assert "echo=True" not in code
    assert 'if __name__ == "__main__":' in code
    # create_all only appears after (inside) the __main__ guard
    assert code.index("Base.metadata.create_all") > code.index('if __name__ == "__main__":')


def test_importing_sqlite_module_has_no_side_effects(tmpdir, monkeypatch):
    """Importing the generated sqlite module must not create the database
    file or the data/ folder — table creation only happens when the module
    is executed directly."""
    model = _single_class_model(model_name="WidgetStore", class_name="Widget")
    path, _code = _generate(model, tmpdir, subdir="sqlite_side_effects")

    workdir = tmpdir.mkdir("cwd")
    monkeypatch.chdir(workdir)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    module = _import_module(path, "sqlite_side_effect_module")

    assert os.listdir(str(workdir)) == []
    assert module.DATABASE_URL == "sqlite:///./data/WidgetStore.db"
    # engine stays importable for other modules (main_api does `from sql_alchemy import *`)
    assert hasattr(module, "engine")


# ---------------------------------------------------------------------------
# 4. sqlite default matches the FastAPI backend default
# ---------------------------------------------------------------------------

def test_sqlite_default_url_matches_backend_init_db_default(tmpdir):
    model = _single_class_model(model_name="MyShop", class_name="Item")
    _path, code = _generate(model, tmpdir, subdir="sqlite_default")

    # Same default as main_api.py's init_db: sqlite:///./data/<model_name>.db
    assert 'DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/MyShop.db")' in code


# ---------------------------------------------------------------------------
# 5. N-ary associations emit an explicit warning comment
# ---------------------------------------------------------------------------

def test_nary_association_emits_warning_comment(tmpdir):
    a = Class(name="A")
    b = Class(name="B")
    c = Class(name="C")
    nary = Association(
        name="TripleLink",
        ends={
            Property(name="a_end", type=a, multiplicity=Multiplicity(0, "*")),
            Property(name="b_end", type=b, multiplicity=Multiplicity(0, "*")),
            Property(name="c_end", type=c, multiplicity=Multiplicity(0, "*")),
        },
    )
    model = DomainModel(name="NaryModel", types={a, b, c}, associations={nary})

    path, code = _generate(model, tmpdir, subdir="nary")

    assert ("# WARNING: n-ary association 'TripleLink' is not supported by "
            "this generator — model it manually.") in code

    # The generated module must still import cleanly, with the binary-only tables
    module = _import_module(path, "nary_module")
    for class_name in ("A", "B", "C"):
        assert hasattr(module, class_name)


def test_binary_associations_do_not_emit_nary_warning(tmpdir):
    """Two-ended associations must not trigger the n-ary warning."""
    a = Class(name="A")
    b = Class(name="B")
    from besser.BUML.metamodel.structural import BinaryAssociation

    binary = BinaryAssociation(
        name="PairLink",
        ends={
            Property(name="a_end", type=a, multiplicity=Multiplicity(0, "*")),
            Property(name="b_end", type=b, multiplicity=Multiplicity(0, "*")),
        },
    )
    model = DomainModel(name="BinaryModel", types={a, b}, associations={binary})

    _path, code = _generate(model, tmpdir, subdir="binary_no_warning")
    assert "WARNING: n-ary association" not in code


# ----------------------------------------------------------------------
# 6. The generated module binds BOTH the aliased and plain typing names.
#
# Every typing/sqlalchemy import is aliased with a trailing underscore so a
# modelled class called List or Optional cannot shadow it. That is correct for
# generated code, but this file is routinely edited afterwards by an LLM, which
# writes ordinary Python. A nullable FK column genuinely wants
# ``Mapped_[Optional[int]]``, and with only the aliases bound that raised
# "undefined name 'Optional'" and the app would not import at all — observed in
# a generated app on 2026-09-11.
# ----------------------------------------------------------------------


def _optional_fk_model() -> DomainModel:
    """Team/Player with an OPTIONAL team end, which emits a nullable FK."""
    from besser.BUML.metamodel.structural import BinaryAssociation, IntegerType

    team = Class(name="Team")
    team.attributes = {
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="name", type=StringType),
    }
    player = Class(name="Player")
    player.attributes = {
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="name", type=StringType),
    }
    assoc = BinaryAssociation(name="team_players", ends={
        Property(name="team", type=team, multiplicity=Multiplicity(0, 1)),
        Property(name="players", type=player, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(name="Squad", types={team, player}, associations={assoc})


def test_generated_module_binds_plain_and_aliased_typing_names(tmp_path):
    SQLAlchemyGenerator(model=_optional_fk_model(), output_dir=str(tmp_path)).generate()
    src = (tmp_path / "sql_alchemy.py").read_text(encoding="utf-8")
    line = next(l for l in src.splitlines() if l.startswith("from typing import"))
    for name in ("List", "Optional", "List as List_", "Optional as Optional_"):
        assert name in line, f"{name!r} not bound: {line!r}"


def test_a_bare_Optional_annotation_still_imports(tmp_path):
    """Simulates the LLM's edit: annotate the nullable FK as Optional[int].

    Before this fix that produced an undefined name and the backend could not
    be imported, so the whole generated app was dead on arrival.
    """
    import re

    SQLAlchemyGenerator(model=_optional_fk_model(), output_dir=str(tmp_path)).generate()
    path = tmp_path / "sql_alchemy.py"
    src = path.read_text(encoding="utf-8")
    edited, n = re.subn(
        r"Mapped_\[int\] = mapped_column\(ForeignKey_",
        "Mapped_[Optional[int]] = mapped_column(ForeignKey_",
        src, count=1,
    )
    assert n == 1, "expected a nullable foreign-key column to rewrite"

    namespace: dict = {}
    exec(compile(edited, str(path), "exec"), namespace)   # must not NameError
    assert "Team" in namespace and "Player" in namespace


# ----------------------------------------------------------------------
# 7. The same applies to the SQLAlchemy column types.
#
# Fixing only List/Optional above left the identical trap set for the thirteen
# column types, which are aliased the same way. Verification run on 2026-09-11
# walked straight into it: the LLM added a User class for JWT auth and wrote
#
#     created_at: Mapped_[dt_datetime] = mapped_column(DateTime(), ...)
#                                                      ^^^^^^^^
#
# F821 Undefined name `DateTime` — one blocker, app dead on arrival, the same
# failure as 'Optional' one fix earlier. Bind every aliased name both ways so
# the class of bug is closed rather than its latest instance.
# ----------------------------------------------------------------------

_ALIASED_COLUMN_TYPES = (
    "Boolean", "Column", "Date", "DateTime", "Float", "ForeignKey", "Integer",
    "Interval", "PickleType", "String", "Table", "Text", "Time",
)


def _names_imported_from(src: str, module: str) -> set:
    """Names the module actually binds from ``module`` (alias-aware)."""
    import ast

    bound = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            for alias in node.names:
                bound.add(alias.asname or alias.name)
    return bound


def test_every_aliased_column_type_is_bound_under_its_plain_name(tmp_path):
    SQLAlchemyGenerator(model=_optional_fk_model(), output_dir=str(tmp_path)).generate()
    src = (tmp_path / "sql_alchemy.py").read_text(encoding="utf-8")
    bound = _names_imported_from(src, "sqlalchemy")
    for name in _ALIASED_COLUMN_TYPES:
        assert f"{name}_" in bound, f"alias {name}_ missing — generated code needs it"
        assert name in bound, f"plain {name} not bound — an LLM edit using it cannot import"


def test_bare_Mapped_is_bound_too(tmp_path):
    """A class the LLM adds is annotated Mapped[...], not Mapped_[...]."""
    SQLAlchemyGenerator(model=_optional_fk_model(), output_dir=str(tmp_path)).generate()
    src = (tmp_path / "sql_alchemy.py").read_text(encoding="utf-8")
    bound = _names_imported_from(src, "sqlalchemy.orm")
    assert "Mapped_" in bound and "Mapped" in bound


def test_the_live_auth_class_blocker_now_imports(tmp_path):
    """Verbatim shape of the blocker from the 2026-09-11 verification run.

    An LLM adding JWT auth appends a User model written in ordinary Python:
    bare DateTime, String, Integer and Mapped. Before this fix the module
    raised NameError on import and the generated app could not start.
    """
    SQLAlchemyGenerator(model=_optional_fk_model(), output_dir=str(tmp_path)).generate()
    path = tmp_path / "sql_alchemy.py"
    src = path.read_text(encoding="utf-8")

    appended = src + (
        "\n\n"
        "class User(Base):\n"
        '    __tablename__ = "user"\n'
        "    id: Mapped[int] = mapped_column(Integer, primary_key=True)\n"
        "    email: Mapped[str] = mapped_column(String(255), unique=True)\n"
        "    hashed_password: Mapped[str] = mapped_column(String(255))\n"
        "    is_active: Mapped[bool] = mapped_column(Boolean, default=True)\n"
        "    created_at: Mapped[dt_datetime] = mapped_column(\n"
        "        DateTime(), default=dt_datetime.utcnow\n"
        "    )\n"
    )

    namespace: dict = {}
    exec(compile(appended, str(path), "exec"), namespace)   # must not NameError
    assert "User" in namespace
    assert namespace["User"].__tablename__ == "user"
