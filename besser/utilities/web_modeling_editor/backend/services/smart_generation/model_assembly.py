"""Assemble BUML models from a ``ProjectInput`` for smart generation.

Picks the active ``ClassDiagram`` (preferring the one referenced by the
active ``GUINoCodeDiagram`` when present) as the required domain model,
then best-effort processes every other editor diagram type the LLM can
benefit from: GUI, agent, object (instance data), state machines,
quantum circuits.

Any optional processor failure degrades gracefully to ``None`` for that
model — the smart-generation run continues with whatever succeeded.

Lives in its own module (not inside ``generation_router``) so that the
smart-generation router does not import from the generation router and
vice-versa — no circular deps between routers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from besser.utilities.web_modeling_editor.backend.models.diagram import DiagramInput
from besser.utilities.web_modeling_editor.backend.models.project import ProjectInput
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.agent_diagram_processor import (
    process_agent_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_diagram_processor import (
    process_gui_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.object_diagram_processor import (
    process_object_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.quantum_diagram_processor import (
    process_quantum_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.state_machine_processor import (
    process_state_machine,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssembledModels:
    """BUML models ready to be passed to ``LLMOrchestrator``.

    ``domain_model`` is required. Everything else is optional and degrades
    to ``None`` / empty when the matching editor diagram is absent or fails
    to process.
    """

    domain_model: Any                                   # DomainModel — always present
    gui_model: Optional[Any] = None                     # GUIModel or None
    agent_model: Optional[Any] = None                   # AgentModel or None
    agent_config: Optional[dict] = None                 # per-diagram or project-level agent config
    object_model: Optional[Any] = None                  # ObjectModel — instance data / fixtures
    state_machines: List[Any] = field(default_factory=list)  # list of StateMachine
    quantum_circuit: Optional[Any] = None               # QuantumCircuit or None


def assemble_models_from_project(project: ProjectInput) -> AssembledModels:
    """Build BUML models from a ``ProjectInput``.

    The class diagram is required — it is the anchor for every other model.
    Every additional diagram type is processed best-effort: a processor
    failure logs and continues rather than blocking the whole run.

    Supported diagram types:
      * ``ClassDiagram``            — required, produces ``DomainModel``.
      * ``GUINoCodeDiagram``        — optional, produces ``GUIModel``.
      * ``AgentDiagram``            — optional, produces ``AgentModel``
        (only wired when the GUI contains agent components).
      * ``ObjectDiagram``           — optional, produces ``ObjectModel``
        (instance data — great for seeders / test fixtures).
      * ``StateMachineDiagram``     — optional, produces a list of
        ``StateMachine`` objects (all state-machine diagrams in the
        project are collected, not just the active one).
      * ``QuantumCircuitDiagram``   — optional, produces ``QuantumCircuit``.

    Raises
    ------
    ValueError
        If the project does not contain a ClassDiagram.
    """
    class_diagram = _pick_class_diagram(project)
    if class_diagram is None:
        raise ValueError(
            "Smart generation requires a ClassDiagram in the project"
        )

    domain_model = process_class_diagram(class_diagram.model_dump())

    gui_model, agent_model, agent_config = _assemble_gui_and_agent(
        project, class_diagram, domain_model
    )
    object_model = _assemble_object_model(project, domain_model)
    state_machines = _assemble_state_machines(project)
    quantum_circuit = _assemble_quantum_circuit(project)

    return AssembledModels(
        domain_model=domain_model,
        gui_model=gui_model,
        agent_model=agent_model,
        agent_config=agent_config,
        object_model=object_model,
        state_machines=state_machines,
        quantum_circuit=quantum_circuit,
    )


def _assemble_gui_and_agent(
    project: ProjectInput,
    class_diagram: DiagramInput,
    domain_model: Any,
) -> tuple[Optional[Any], Optional[Any], Optional[dict]]:
    """Process GUI and (conditionally) agent diagrams.

    Agent processing is only attempted when the GUI contains at least one
    AgentComponent — matches the existing generation_router behaviour.
    """
    gui_diagram = project.get_active_diagram("GUINoCodeDiagram")
    gui_model: Optional[Any] = None
    agent_model: Optional[Any] = None
    agent_config: Optional[dict] = None

    if gui_diagram is None:
        return gui_model, agent_model, agent_config

    try:
        gui_model = process_gui_diagram(
            gui_diagram.model, class_diagram.model, domain_model
        )
    except Exception:
        # GUI processing can fail if the class diagram and GUI are
        # out of sync. Degrade gracefully to class-only rather than
        # blocking the whole smart-generation run.
        logger.exception(
            "Failed to process GUINoCodeDiagram for smart generation; "
            "continuing with class diagram only"
        )
        gui_model = None

    if gui_model is not None and _check_for_agent_components(gui_model):
        agent_diagram = project.get_referenced_diagram(gui_diagram, "AgentDiagram")
        if agent_diagram is None:
            # Fall back to the active AgentDiagram if the GUI doesn't
            # explicitly reference one.
            agent_diagram = project.get_active_diagram("AgentDiagram")
        if agent_diagram is not None and agent_diagram.model:
            try:
                agent_diagram_dict = agent_diagram.model_dump()
                agent_model = process_agent_diagram(agent_diagram_dict)
                project_settings = project.settings if isinstance(project.settings, dict) else {}
                project_config = project_settings.get("config") if isinstance(project_settings, dict) else None
                project_agent_config = (
                    project_config.get("agentConfig")
                    if isinstance(project_config, dict) else None
                )
                agent_config = (
                    agent_diagram_dict.get("config")
                    or agent_diagram.config
                    or project_agent_config
                    or project_config
                )
            except Exception:
                logger.exception(
                    "Failed to process AgentDiagram for smart generation; "
                    "continuing without agent model"
                )
                agent_model = None
                agent_config = None

    return gui_model, agent_model, agent_config


def _assemble_object_model(
    project: ProjectInput, domain_model: Any
) -> Optional[Any]:
    """Process the active ObjectDiagram into an ObjectModel.

    Object instances need the DomainModel to resolve their classifiers,
    so this is called after the class diagram is built.
    """
    object_diagram = project.get_active_diagram("ObjectDiagram")
    if object_diagram is None or not getattr(object_diagram, "model", None):
        return None
    try:
        return process_object_diagram(object_diagram.model_dump(), domain_model)
    except Exception:
        logger.exception(
            "Failed to process ObjectDiagram for smart generation; "
            "continuing without object model"
        )
        return None


def _assemble_state_machines(project: ProjectInput) -> List[Any]:
    """Process every StateMachineDiagram in the project.

    Unlike the other single-active-diagram types, state machines are
    collected as a list because a project can legitimately declare
    several independent behavioural specs (one per entity).
    """
    diagrams = _collect_diagrams_of_type(project, "StateMachineDiagram")
    results: List[Any] = []
    for diagram in diagrams:
        if not getattr(diagram, "model", None):
            continue
        try:
            sm = process_state_machine(diagram.model_dump())
        except Exception:
            logger.exception(
                "Failed to process StateMachineDiagram %r; skipping",
                getattr(diagram, "title", "<unnamed>"),
            )
            continue
        if sm is not None:
            results.append(sm)
    return results


def _assemble_quantum_circuit(project: ProjectInput) -> Optional[Any]:
    """Process the active QuantumCircuitDiagram into a QuantumCircuit."""
    diagram = project.get_active_diagram("QuantumCircuitDiagram")
    if diagram is None or not getattr(diagram, "model", None):
        return None
    try:
        return process_quantum_diagram(diagram.model_dump())
    except Exception:
        logger.exception(
            "Failed to process QuantumCircuitDiagram for smart generation; "
            "continuing without quantum circuit"
        )
        return None


def _collect_diagrams_of_type(
    project: ProjectInput, diagram_type: str
) -> List[DiagramInput]:
    """Return every DiagramInput of a given type in the project, preferring
    the active-first ordering so the primary diagram dominates the prompt.

    ``ProjectInput.diagrams`` is a ``Dict[str, List[DiagramInput]]`` after
    the backward-compatibility model validator runs, so we can iterate it
    directly. Falls back to the active-only diagram when the dict shape
    isn't what we expect (defensive — shouldn't happen in practice).
    """
    diagrams_by_type = getattr(project, "diagrams", None)
    if not isinstance(diagrams_by_type, dict):
        active = project.get_active_diagram(diagram_type)
        return [active] if active is not None else []

    entries = diagrams_by_type.get(diagram_type) or []
    if not entries:
        return []

    active_idx = 0
    indices = getattr(project, "currentDiagramIndices", None)
    if isinstance(indices, dict):
        maybe_idx = indices.get(diagram_type)
        if isinstance(maybe_idx, int) and 0 <= maybe_idx < len(entries):
            active_idx = maybe_idx

    ordered = [entries[active_idx]] + [
        d for i, d in enumerate(entries) if i != active_idx
    ]
    return ordered


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _pick_class_diagram(project: ProjectInput) -> Optional[DiagramInput]:
    """Resolve the ClassDiagram to use as the domain input.

    Preference order:
      1. The ClassDiagram referenced by the active GUINoCodeDiagram (if any).
      2. The active ClassDiagram.
    """
    gui = project.get_active_diagram("GUINoCodeDiagram")
    if gui is not None:
        referenced = project.get_referenced_diagram(gui, "ClassDiagram")
        if referenced is not None:
            return referenced
    return project.get_active_diagram("ClassDiagram")


def _check_for_agent_components(gui_model) -> bool:
    """Return True if the GUI model contains any AgentComponent.

    Copied from ``generation_router._check_for_agent_components`` so that
    this module does not import from the router (avoiding a
    router→router dependency).
    """
    from besser.BUML.metamodel.gui.dashboard import AgentComponent
    from besser.BUML.metamodel.gui import ViewContainer

    if not gui_model or not gui_model.modules:
        return False

    for module in gui_model.modules:
        if not module.screens:
            continue
        for screen in module.screens:
            if not screen.view_elements:
                continue
            for element in screen.view_elements:
                if isinstance(element, AgentComponent):
                    return True
                if isinstance(element, ViewContainer):
                    if _check_container_for_agent_components(element):
                        return True
    return False


def _check_container_for_agent_components(container) -> bool:
    """Recursively check a ViewContainer for AgentComponents."""
    from besser.BUML.metamodel.gui.dashboard import AgentComponent
    from besser.BUML.metamodel.gui import ViewContainer

    if not container.view_elements:
        return False

    for element in container.view_elements:
        if isinstance(element, AgentComponent):
            return True
        if isinstance(element, ViewContainer):
            if _check_container_for_agent_components(element):
                return True
    return False
