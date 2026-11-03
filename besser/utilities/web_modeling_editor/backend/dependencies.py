"""FastAPI dependencies that hand routes the state built in the app's lifespan."""

import threading
from typing import Annotated

from fastapi import Depends, Request
from starlette.requests import HTTPConnection

from besser.utilities.web_modeling_editor.backend.services.agent_simulation import (
    AgentSimulatorState,
)
from besser.utilities.web_modeling_editor.backend.services.deployment.session_store import (
    GitHubSessions,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runtime import (
    SpecDrivenRuntime,
)
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings, get_settings


def get_spec_driven_runtime(request: Request) -> SpecDrivenRuntime:
    """The spec-driven runtime the lifespan stored on ``app.state.spec_driven``."""
    return request.app.state.spec_driven


# ``HTTPConnection`` rather than ``Request``: WebSocket routes take these too.

def get_github_sessions(connection: HTTPConnection) -> GitHubSessions:
    """The GitHub sign-in stores the lifespan stored on ``app.state.github_sessions``."""
    return connection.app.state.github_sessions


def get_agent_simulator(connection: HTTPConnection) -> AgentSimulatorState:
    """The agent simulator's limiter and sessions, on ``app.state.agent_simulator``."""
    return connection.app.state.agent_simulator


def get_kg_consistency_slots(connection: HTTPConnection) -> threading.BoundedSemaphore:
    """The cap on concurrent KG consistency checks, on ``app.state.kg_consistency_slots``."""
    return connection.app.state.kg_consistency_slots


def get_sdmx_import_slots(connection: HTTPConnection) -> threading.BoundedSemaphore:
    """The cap on concurrent SDMX imports, on ``app.state.sdmx_import_slots``."""
    return connection.app.state.sdmx_import_slots


SpecDrivenRuntimeDep = Annotated[SpecDrivenRuntime, Depends(get_spec_driven_runtime)]
GitHubSessionsDep = Annotated[GitHubSessions, Depends(get_github_sessions)]
AgentSimulatorDep = Annotated[AgentSimulatorState, Depends(get_agent_simulator)]
KgConsistencySlotsDep = Annotated[threading.BoundedSemaphore, Depends(get_kg_consistency_slots)]
SdmxImportSlotsDep = Annotated[threading.BoundedSemaphore, Depends(get_sdmx_import_slots)]

# The process settings read once (``get_settings``); tests swap them through
# ``app.dependency_overrides[get_settings]``. Fields read per call stay
# ``BackendSettings()`` reads.
SettingsDep = Annotated[BackendSettings, Depends(get_settings)]
