"""Server-owned audit timestamps must not appear in backend *Create* schemas.

A modeled ``createdAt`` / ``updatedAt`` (any case, with or without an underscore)
is filled in by the server (the SQLAlchemy generator stamps it via
``default=``/``onupdate=``). Leaving it in the ``*Create`` schema would force the
client to supply a value the server owns and break the generated app — the P4/P5
pilot failure. The full (read) schema keeps the fields, and a declared ``is_id``
primary key is always client-supplied and stays in Create.
"""

import os
import tempfile

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Property, IntegerType, StringType, DateTimeType,
)
from besser.generators.pydantic_classes import PydanticGenerator


def _generate(model, backend):
    out = tempfile.mkdtemp()
    PydanticGenerator(model=model, output_dir=out, backend=backend).generate()
    with open(os.path.join(out, "pydantic_classes.py"), encoding="utf-8") as f:
        return f.read()


def _class_block(code, class_name):
    start = code.find(f"class {class_name}(")
    assert start != -1, f"{class_name} not found in generated code"
    end = code.find("\n\n", start)
    return code[start:end if end != -1 else None]


def _event_model():
    event = Class(name="Event", attributes={
        Property(name="id", type=IntegerType, is_id=True),
        Property(name="title", type=StringType),
        Property(name="created_at", type=DateTimeType),
        Property(name="updatedAt", type=DateTimeType),
    })
    return DomainModel(name="M", types={event})


def test_backend_create_excludes_audit_timestamps():
    code = _generate(_event_model(), backend=True)
    block = _class_block(code, "EventCreate")
    assert "title" in block
    assert "id" in block  # declared is_id PK stays client-supplied
    assert "created_at" not in block
    assert "updatedAt" not in block


def test_full_schema_keeps_audit_timestamps():
    code = _generate(_event_model(), backend=False)
    block = _class_block(code, "Event")
    # The read model must still expose the server-owned fields.
    assert "created_at" in block
    assert "updatedAt" in block


def test_declared_id_timestamp_is_kept():
    """A timestamp named `id` that IS the declared PK is client-supplied, so it stays."""
    ts_pk = Class(name="Snapshot", attributes={
        Property(name="createdAt", type=DateTimeType, is_id=True),
        Property(name="label", type=StringType),
    })
    code = _generate(DomainModel(name="M2", types={ts_pk}), backend=True)
    block = _class_block(code, "SnapshotCreate")
    assert "createdAt" in block  # it's the PK -> client-supplied despite the name
