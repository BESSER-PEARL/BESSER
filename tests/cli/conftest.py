"""Model files and a runner for the ``besser`` command-line tests."""

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass

import pytest

LIBRARY_MODEL = '''
from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, IntegerType, Multiplicity, Property, StringType,
)

library = Class(name="Library", attributes={Property(name="name", type=StringType)})
book = Class(name="Book", attributes={
    Property(name="title", type=StringType), Property(name="pages", type=IntegerType),
})
holds = BinaryAssociation(name="holds", ends={
    Property(name="library", type=library, multiplicity=Multiplicity(1, 1)),
    Property(name="books", type=book, multiplicity=Multiplicity(0, "*")),
})
library_model = DomainModel(name="LibraryModel", types={library, book}, associations={holds})
'''

# The library model with one more attribute on Book.
LIBRARY_MODEL_WITH_ISBN = LIBRARY_MODEL.replace(
    'Property(name="pages", type=IntegerType),',
    'Property(name="pages", type=IntegerType), Property(name="isbn", type=StringType),',
)

# Two domain models and no Project: a generator cannot tell which one to read.
TWO_DOMAIN_MODELS = LIBRARY_MODEL + '''
draft = DomainModel(name="Draft", types={Class(name="Idea", attributes={Property(name="text", type=StringType)})})
'''

# validate() reports an error: the attribute's type is not in the model.
INVALID_MODEL = '''
from besser.BUML.metamodel.structural import Class, DomainModel, Property

book = Class(name="Book", attributes={Property(name="title", type="Text")})
broken_model = DomainModel(name="Broken", types={book})
'''

# validate() reports a warning and no error: a class with nothing in it.
WARNING_MODEL = LIBRARY_MODEL + '''
library_model.add_type(Class(name="Shelf"))
'''

# One class diagram as the editor exports it.
CLASS_DIAGRAM = {
    "title": "Library",
    "model": {
        "type": "ClassDiagram",
        "elements": {
            "class-1": {"type": "Class", "name": "Book", "attributes": ["attr-1"], "methods": []},
            "attr-1": {"type": "Attribute", "name": "+ title: str"},
        },
        "relationships": {},
    },
}

PROJECT_EXPORT = {
    "project": {
        "id": "project-1",
        "type": "Project",
        "name": "library",
        "createdAt": "2026-11-10T00:00:00Z",
        "currentDiagramType": "ClassDiagram",
        "diagrams": {"ClassDiagram": [{"id": "diagram-1", **CLASS_DIAGRAM}]},
    },
    "exportedAt": "2026-11-10T00:00:00Z",
    "version": "1.0.0",
}


@dataclass
class Run:
    """A finished ``besser`` process."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def json(self) -> dict:
        return json.loads(self.stdout)


@pytest.fixture
def besser(tmp_path):
    """Run ``python -m besser`` in ``tmp_path`` with the interpreter running the tests.

    ``stdin`` is the text the process reads on standard input. ``output`` is ``BESSER_OUTPUT``: the
    tests read the full report unless they ask for another; None runs with the default, which is
    terse since standard output is a pipe.
    """
    def run(*args, stdin: str = "", output: str | None = "full") -> Run:
        env = {key: value for key, value in os.environ.items() if key != "BESSER_OUTPUT"}
        if output is not None:
            env["BESSER_OUTPUT"] = output
        process = subprocess.run(
            [sys.executable, "-m", "besser", *map(str, args)],
            cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False, input=stdin, env=env,
        )
        return Run(process.returncode, process.stdout, process.stderr)
    return run


@pytest.fixture
def write(tmp_path):
    """Write a file under ``tmp_path`` (a dict as JSON) and return its name."""
    def write_file(name: str, content) -> str:
        text = json.dumps(content) if isinstance(content, dict) else textwrap.dedent(content)
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return name
    return write_file


@pytest.fixture
def model(write):
    """A valid B-UML model file: Library 1 --- 0..* Book."""
    return write("library.py", LIBRARY_MODEL)
