"""FastAPI dependencies that hand routes the state built in the app's lifespan."""

from typing import Annotated

from fastapi import Depends, Request

from besser.utilities.web_modeling_editor.backend.services.spec_driven.runtime import (
    SpecDrivenRuntime,
)


def get_spec_driven_runtime(request: Request) -> SpecDrivenRuntime:
    """The spec-driven runtime the lifespan stored on ``app.state.spec_driven``."""
    return request.app.state.spec_driven


SpecDrivenRuntimeDep = Annotated[SpecDrivenRuntime, Depends(get_spec_driven_runtime)]
