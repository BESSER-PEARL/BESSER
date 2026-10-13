"""Default timestamps must follow creation order, so declaration order is stable.

Timestamps used to be ``now()`` plus a sub-millisecond perf-counter offset, which
is not monotonic: elements created in the same millisecond sorted nearly at
random, and rebuilding one model in one process flipped ``Book.__init__`` from
``(authors, publisher)`` to ``(publisher, authors)``, breaking positional callers.
"""
from datetime import datetime

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, IntegerType, Multiplicity, Property, StringType,
)
from besser.generators.python_classes import PythonGenerator


def test_default_timestamps_strictly_increase():
    stamps = [Property(name=f"p{i}", type=StringType).timestamp for i in range(5000)]
    assert all(a < b for a, b in zip(stamps, stamps[1:]))


def test_explicit_timestamp_is_kept():
    stamp = datetime(2020, 1, 1)
    assert Property(name="p", type=StringType, timestamp=stamp).timestamp == stamp


def _build():
    author = Class(name="Author", attributes={Property(name="name", type=StringType)})
    publisher = Class(name="Publisher", attributes={Property(name="name", type=StringType)})
    book = Class(name="Book", attributes={Property(name="title", type=StringType),
                                          Property(name="pages", type=IntegerType)})
    writes = BinaryAssociation(name="writes", ends={
        Property(name="authors", type=author, multiplicity=Multiplicity(1, "*")),
        Property(name="books", type=book, multiplicity=Multiplicity(0, "*"))})
    publishes = BinaryAssociation(name="publishes", ends={
        Property(name="publisher", type=publisher, multiplicity=Multiplicity(1, 1)),
        Property(name="published", type=book, multiplicity=Multiplicity(0, "*"))})
    return DomainModel(name="Lib", types={author, publisher, book}, associations={writes, publishes})


def test_rebuilt_model_generates_identical_code(tmp_path):
    outputs = set()
    for i in range(20):
        out = tmp_path / str(i)
        PythonGenerator(_build(), output_dir=str(out)).generate()
        outputs.add((out / "classes.py").read_text(encoding="utf-8"))
    assert len(outputs) == 1
    code = outputs.pop()
    assert ('def __init__(self, title: str, pages: int, authors: set["Author"] = None, '
            'publisher: "Publisher" = None):') in code
