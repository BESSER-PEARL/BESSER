"""FastAPI dependencies that hand routes the state built in the app's lifespan."""

from typing import Annotated

from fastapi import Depends, Request

from besser.utilities.web_modeling_editor.backend.services.spec_driven.runtime import (
    SpecDrivenRuntime,
)
from besser.utilities.web_modeling_editor.backend.settings import BackendSettings, get_settings


def get_spec_driven_runtime(request: Request) -> SpecDrivenRuntime:
    """The spec-driven runtime the lifespan stored on ``app.state.spec_driven``."""
    return request.app.state.spec_driven


SpecDrivenRuntimeDep = Annotated[SpecDrivenRuntime, Depends(get_spec_driven_runtime)]

# The process settings read once (``get_settings``); tests swap them through
# ``app.dependency_overrides[get_settings]``. Fields read per call stay
# ``BackendSettings()`` reads.
SettingsDep = Annotated[BackendSettings, Depends(get_settings)]
