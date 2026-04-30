"""Standardized API response models."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DiagramExportResponse(BaseModel):
    """Response for diagram export/conversion endpoints."""
    title: str
    model: Dict[str, Any]
    diagramType: str
    exportedAt: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0.0"
    warnings: Optional[List[Dict[str, Any]]] = None


class ProjectExportResponse(BaseModel):
    """Response for project export/conversion endpoints."""
    project: Dict[str, Any]
    exportedAt: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0.0"


class ValidationResponse(BaseModel):
    """Response for validation endpoints."""
    isValid: bool
    errors: List[str] = []
    warnings: List[str] = []
    message: str = ""
    valid_constraints: List[str] = []
    invalid_constraints: List[str] = []
    ocl_message: str = ""


class ApiInfoResponse(BaseModel):
    """Response for the API root endpoint."""
    message: str
    version: str
    supported_generators: List[str]
    endpoints: Dict[str, str]


class FeedbackResponse(BaseModel):
    """Response for feedback submission."""
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class KGResolutionOptionModel(BaseModel):
    """One resolution choice the frontend can present to the user."""
    key: str
    label: str
    parametersSchema: Dict[str, str] = Field(default_factory=dict)


class KGIssueModel(BaseModel):
    """A preflight finding emitted by ``/analyze-kg-for-buml-conversion``."""
    id: str
    code: str
    severity: str  # "blocking" | "advisory"
    message: str
    affectedNodeIds: List[str] = []
    affectedEdgeIds: List[str] = []
    context: Dict[str, Any] = Field(default_factory=dict)
    suggestedResolutions: List[KGResolutionOptionModel] = []


class KGPreflightResponse(BaseModel):
    """Response for ``POST /besser_api/analyze-kg-for-buml-conversion``."""
    kgSignature: str
    blockingCount: int
    advisoryCount: int
    issues: List[KGIssueModel] = []


class KGResolutionInputModel(BaseModel):
    """A resolution submitted by the frontend back to ``/kg-to-class-diagram``."""
    issueId: str
    choice: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
