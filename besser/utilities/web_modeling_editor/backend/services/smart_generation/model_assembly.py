"""Assemble BUML models from a ``ProjectInput`` for smart generation.

Mirrors the logic used by ``generation_router._handle_web_app_project_generation``:
pick the active ``ClassDiagram`` (preferring the one referenced by the
active ``GUINoCodeDiagram`` when present), then optionally build a
``GUIModel`` and an ``AgentModel`` when those diagrams are present.

Lives in its own module (not inside ``generation_router``) so that the
smart-generation router does not import from the generation router and
vice-versa — no circular deps between routers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssembledModels:
    """BUML models ready to be passed to ``LLMOrchestrator``."""

    domain_model: Any            # DomainModel — always present
    gui_model: Optional[Any]     # GUIModel or None
    agent_model: Optional[Any]   # AgentModel or None
    agent_config: Optional[dict] # per-diagram or project-level agent config


def assemble_models_from_project(project: ProjectInput) -> AssembledModels:
    """Build BUML models from a ``ProjectInput``.

    Supported shapes:
    #TODO: we should take as input all types of models from the wme editor, not just these three — but this is a good start and we can expand as needed.
    * ClassDiagram only — most common. LLM works off the domain model alone.
    * ClassDiagram + GUINoCodeDiagram — full frontend+backend input.
    * ClassDiagram + GUINoCodeDiagram + AgentDiagram — when the GUI
      contains agent components and an agent diagram is referenced.

    Agent-only projects are rejected: the smart generator needs a domain
    model to reason about.

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

    gui_diagram = project.get_active_diagram("GUINoCodeDiagram")
    gui_model: Optional[Any] = None
    agent_model: Optional[Any] = None
    agent_config: Optional[dict] = None

    if gui_diagram is not None:
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

    return AssembledModels(
        domain_model=domain_model,
        gui_model=gui_model,
        agent_model=agent_model,
        agent_config=agent_config,
    )


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
