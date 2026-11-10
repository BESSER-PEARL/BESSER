"""The model files ``besser new`` writes: one small, commented B-UML script per model kind.

Each ``<kind>.py`` beside this file is read as text and written to the user's file; none is
imported from here. ``MODEL_FILE`` in a starter stands for the name of the file it is written to.
``tests/cli/test_new.py`` validates every starter and runs its generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Starter:
    """One starter model file."""

    # What it models, as ``besser new --list`` says it.
    summary: str
    # A generator that runs on it, for the next steps ``besser new`` prints; ``None`` when none reads the kind.
    generator: str | None = None


# Model kind (see ``besser.cli.loading.MODEL_KINDS``) -> its starter; ``project`` is a file of several models.
STARTERS: dict[str, Starter] = {
    "domain": Starter("a class diagram: a library, its books and their authors", "python"),
    "object": Starter("an object diagram: instances of a small domain model", "jsonobject"),
    "state_machine": Starter("a state machine: the life of an order"),
    "agent": Starter("a conversational agent: intents, states and replies", "agent"),
    "gui": Starter("a GUI model over a domain model: screens that list its classes", "web_app"),
    "nn": Starter("a neural network: a small image classifier", "pytorch"),
    "quantum": Starter("a quantum circuit: a Bell state", "qiskit"),
    "bpmn": Starter("a BPMN process: check an order, then ship or reject it", "bpmn"),
    # No generator to suggest: terraform also reads a cluster.conf file, which the starter's header explains.
    "deployment": Starter("a cloud deployment: a containerized application on a cluster"),
    "uml_deployment": Starter("a UML deployment diagram: where an application's parts run", "docker_compose"),
    "component": Starter("a UML component diagram: components and their interfaces"),
    "kg": Starter("a knowledge graph: classes, individuals and the edges between them"),
    "feature": Starter("a feature model: the features of a product line"),
    "project": Starter("a project: several models in one file", "python"),
}
DEFAULT_KIND = "domain"
_MODEL_FILE = "MODEL_FILE"


def starter_source(kind: str, file_name: str) -> str:
    """The starter of ``kind`` as the text of a model file called ``file_name``."""
    source = Path(__file__).with_name(f"{kind}.py").read_text(encoding="utf-8")
    return source.replace(_MODEL_FILE, file_name)
