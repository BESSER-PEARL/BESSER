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
    # The project trace (trace file JSON) when the B-UML file has a TRACEABILITY section.
    trace: Optional[Dict[str, Any]] = None


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


# ---------------------------------------------------------------------------
# Knowledge Graph (KG) endpoints
#
# These pin the wire shape the editor already consumes, field for field, so
# the frontend contract is declared in one place and visible in /openapi.json.
# All fields are camelCase, like the rest of the API; the Python dataclasses
# behind them (e.g. ``ConsistencyIssue``) stay snake_case and are mapped
# explicitly in ``services/kg/payloads.py``.
# ---------------------------------------------------------------------------


class KGActionResponse(BaseModel):
    """A pre-filled action attached to a preflight issue."""
    key: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    label: str = ""


class KGIssueResponse(BaseModel):
    """One KG element with no clean 1-to-1 B-UML mapping."""
    id: str
    code: str
    description: str
    affectedNodeIds: List[str] = Field(default_factory=list)
    affectedEdgeIds: List[str] = Field(default_factory=list)
    recommendedAction: Optional[KGActionResponse] = None
    skipAction: Optional[KGActionResponse] = None


class KGPreflightReportResponse(BaseModel):
    """Response of ``/kg/preflight``, ``/kg/llm-clean`` and
    ``/kg/classify-orphans``."""
    kgSignature: str
    diagramType: str
    issueCount: int
    issues: List[KGIssueResponse] = Field(default_factory=list)


class KGConsistencyIssueResponse(BaseModel):
    """One OWL/SHACL finding."""
    id: str
    code: str
    severity: str
    message: str
    affectedNodeIds: List[str] = Field(default_factory=list)
    affectedEdgeIds: List[str] = Field(default_factory=list)
    constraintNodeId: Optional[str] = None
    specKind: Optional[str] = None
    pathIri: Optional[str] = None
    constraintLabel: Optional[str] = None


class KGConsistencyReportResponse(BaseModel):
    """Response of ``/kg/consistency``."""
    issues: List[KGConsistencyIssueResponse] = Field(default_factory=list)
    issueCount: int
    severityCounts: Dict[str, int]
    kgSignature: str
    inferenceUsed: str
    exportedAt: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0.0"


class KGConfigResponse(BaseModel):
    """Response of ``GET /kg/config``: server-owned limits the KG editor must respect."""
    llmRequestTimeoutSeconds: int = Field(
        description="Overall server deadline, in seconds, for one KG LLM request.",
    )
    maxOrphanClassifyNodes: int = Field(
        description="Most node ids one /kg/classify-orphans request may carry.",
    )


class KGPendingOrphanClassification(BaseModel):
    """Orphan nodes the user deferred to the per-node LLM classifier."""
    nodeIds: List[str]
    kgSignature: str


class KGRefinementResponse(BaseModel):
    """Response of ``/kg/apply-refinement``: the cleaned KG plus its new signature."""
    title: str
    model: Dict[str, Any]
    diagramType: str
    kgSignature: str
    pendingOrphanClassification: Optional[KGPendingOrphanClassification] = None
    exportedAt: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0.0"
