"""
Runtime Router

Exposes the besser4DT runtime kernel over HTTP so the web modeling
editor can execute user-authored methods on Object instances without
generating a platform first. ``POST /runtime/invoke`` is the general
entry point — any method on any instance with any args.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

# Centralized error handling — maps Conversion/Validation/Generation
# exceptions to HTTP statuses. Runtime exceptions (KeyError,
# AttributeError, TypeError) are raised manually below so the frontend
# gets targeted statuses instead of generic 500s.
from besser.utilities.web_modeling_editor.backend.routers.error_handler import (
    handle_endpoint_errors,
)
from besser.utilities.web_modeling_editor.backend.services.runtime_executor import (
    invoke_method_on_diagram,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/besser_api", tags=["runtime"])


class InvokeRequest(BaseModel):
    """Per-method invocation payload from the editor's object diagram.

    The editor sends the *current* class + object diagrams on every call
    so the executor stays stateless — no session, no server-side store
    to drift out of sync with what's on screen.
    """

    classDiagram: Dict[str, Any] = Field(
        ...,
        description="Class diagram JSON ({title, model}) — provides class and method definitions.",
    )
    objectDiagram: Dict[str, Any] = Field(
        ...,
        description="Object diagram JSON ({title, model}) — current instance state.",
    )
    className: str = Field(..., description="Domain class of the target instance")
    instanceName: str = Field(..., description="Object name as shown on the diagram")
    methodName: str = Field(..., description="Method to dispatch on the instance")
    args: Dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments forwarded to the method"
    )


class InvokeResponse(BaseModel):
    """Outcome of a single invocation — the frontend overlays
    ``updated_attributes`` onto its in-diagram object representation."""

    instance_name: str
    class_name: str
    method_name: str
    args: Dict[str, Any]
    result: Any = None
    updated_attributes: Dict[str, Any]


@router.post("/runtime/invoke", response_model=InvokeResponse)
@handle_endpoint_errors("runtime_invoke")
async def invoke(payload: InvokeRequest):
    """Execute a single method on an Object instance.

    The flow: parse the class diagram → materialize Python classes →
    extract instance state from the object diagram → call the method →
    return mutated attributes. Errors from each stage map to the
    HTTP statuses the frontend uses to decide how to surface them.
    """
    try:
        return invoke_method_on_diagram(
            class_diagram_json=payload.classDiagram,
            object_diagram_json=payload.objectDiagram,
            class_name=payload.className,
            instance_name=payload.instanceName,
            method_name=payload.methodName,
            args=payload.args,
        )
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except AttributeError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
    except TypeError as e:
        # Wrong kwargs / missing / extra args.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )
