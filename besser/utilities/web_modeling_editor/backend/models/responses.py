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
