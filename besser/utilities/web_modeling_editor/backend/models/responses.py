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


class IncrementalPreanalysisItem(BaseModel):
    """Events that may violate one invariant, addressed to its editor box."""
    name: str
    elementId: Optional[str] = None
    source: Optional[str] = None
    warnings: List[str] = []
    events: List[Dict[str, Any]] = []


class IncrementalPreanalysisResponse(BaseModel):
    """On-demand chapter-4 pre-analysis of the invariants in a class diagram."""
    constraints: List[IncrementalPreanalysisItem] = []
    warnings: List[str] = []


class ConstraintSimplificationItem(BaseModel):
    """One OCL box after chapter-3 simplification."""
    elementId: Optional[str] = None
    changed: bool = False
    expression: Optional[str] = None
    original: Optional[str] = None


class ConstraintSimplificationResponse(BaseModel):
    """On-demand simplification of the invariants in a class diagram.

    ``original`` is present only when the normalized expression differs from
    the text in the box. The editor keeps that text as a comment linked to
    the constraint.
    """
    constraints: List[ConstraintSimplificationItem] = []
    warnings: List[str] = []


class ApiInfoResponse(BaseModel):
    """Response for the API root endpoint."""
    message: str
    version: str
    besser_version: str
    besser_build: Optional[str] = None
    supported_generators: List[str]
    endpoints: Dict[str, str]


class FeedbackResponse(BaseModel):
    """Response for feedback submission."""
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SimulationSessionResponse(BaseModel):
    """Response for creating a live agent simulation session."""
    sessionId: str
    eventList: List[str]


class SimulationValidationResponse(BaseModel):
    """Response for validating that an agent can be simulated."""
    valid: bool
    agentCode: str
    eventList: List[str]
    errors: List[str]


class SimulationSessionFile(BaseModel):
    """One file produced inside a simulation session's workspace."""
    path: str
    content: str


class SimulationSessionFilesResponse(BaseModel):
    """Files and directories produced inside a simulation session's workspace."""
    files: List[SimulationSessionFile]
    directories: List[str] = Field(default_factory=list)


class SimulationSessionStopResponse(BaseModel):
    """Response for stopping a simulation session."""
    ok: bool


class SimulationLimitsResponse(BaseModel):
    """Resource limits and quota settings that apply to a simulation session."""
    memoryMb: Optional[int] = None
    cpuCores: Optional[float] = None
    diskMb: Optional[int] = None
    sessionLifetimeSeconds: Optional[int] = None
    editorQuotaEnabled: bool = False
