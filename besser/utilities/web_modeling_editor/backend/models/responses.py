"""Standardized API response models."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class ResponseModel(BaseModel):
    """Base of the API's response models.

    Every field is sent, defaults included (no ``exclude_none``), so the OpenAPI
    schema marks them all as required. Not for a model that is also part of a
    request body: its schema would split into ``-Input`` / ``-Output``.
    """
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


# What ``GitHubService.commit_tree`` reports for a commit it made.
CommitStatus = Literal["pushed", "merged", "conflict"]


class DiagramExportResponse(ResponseModel):
    """Response for diagram export/conversion endpoints."""
    title: str
    model: Dict[str, Any]
    diagramType: str
    exportedAt: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0.0"
    warnings: Optional[List[Dict[str, Any]]] = None


class ProjectExportResponse(ResponseModel):
    """Response for project export/conversion endpoints."""
    project: Dict[str, Any]
    exportedAt: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0.0"
    warnings: Optional[List[Dict[str, Any]]] = None


class ValidationResponse(ResponseModel):
    """Response for validation endpoints."""
    isValid: bool
    errors: List[str] = []
    warnings: List[str] = []
    message: str = ""
    valid_constraints: List[str] = []
    invalid_constraints: List[str] = []
    ocl_message: str = ""


class ApiInfoResponse(ResponseModel):
    """Response for the API root endpoint."""
    message: str
    version: str
    besser_version: str
    besser_build: Optional[str] = None
    supported_generators: List[str]
    endpoints: Dict[str, str]


class FeedbackResponse(ResponseModel):
    """Response for feedback submission."""
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SimulationSessionResponse(ResponseModel):
    """Response for creating a live agent simulation session."""
    sessionId: str
    eventList: List[str]


class SimulationValidationResponse(ResponseModel):
    """Response for validating that an agent can be simulated."""
    valid: bool
    agentCode: str
    eventList: List[str]
    errors: List[str]


class SimulationSessionFile(ResponseModel):
    """One file produced inside a simulation session's workspace."""
    path: str
    content: str


class SimulationSessionFilesResponse(ResponseModel):
    """Files and directories produced inside a simulation session's workspace."""
    files: List[SimulationSessionFile]
    directories: List[str] = Field(default_factory=list)


class SimulationSessionStopResponse(ResponseModel):
    """Response for stopping a simulation session."""
    ok: bool


class SimulationLimitsResponse(ResponseModel):
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


# KGActionResponse and KGIssueResponse are also part of the KGResolutionInput
# request body, so they stay plain BaseModels (see ResponseModel).
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


class KGPreflightReportResponse(ResponseModel):
    """Response of ``/kg/preflight``, ``/kg/llm-clean`` and
    ``/kg/classify-orphans``."""
    kgSignature: str
    diagramType: str
    issueCount: int
    issues: List[KGIssueResponse] = Field(default_factory=list)


class KGConsistencyIssueResponse(ResponseModel):
    """One OWL/SHACL finding."""
    id: str
    code: str
    # The severities of ``check_kg_consistency`` (its ``_SEVERITY_MAP``).
    severity: Literal["info", "warning", "violation"]
    message: str
    affectedNodeIds: List[str] = Field(default_factory=list)
    affectedEdgeIds: List[str] = Field(default_factory=list)
    constraintNodeId: Optional[str] = None
    specKind: Optional[str] = None
    pathIri: Optional[str] = None
    constraintLabel: Optional[str] = None


class KGSeverityCountsResponse(ResponseModel):
    """How many issues of each severity a consistency report holds."""
    info: int
    warning: int
    violation: int


class KGConsistencyReportResponse(ResponseModel):
    """Response of ``/kg/consistency``."""
    issues: List[KGConsistencyIssueResponse] = Field(default_factory=list)
    issueCount: int
    severityCounts: KGSeverityCountsResponse
    kgSignature: str
    # The ``inference`` of ``check_kg_consistency``.
    inferenceUsed: Literal["rdfs", "owlrl"]
    exportedAt: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0.0"


class KGConfigResponse(ResponseModel):
    """Response of ``GET /kg/config``: server-owned limits the KG editor must respect."""
    llmRequestTimeoutSeconds: int = Field(
        description="Overall server deadline, in seconds, for one KG LLM request.",
    )
    maxOrphanClassifyNodes: int = Field(
        description="Most node ids one /kg/classify-orphans request may carry.",
    )


class KGPendingOrphanClassification(ResponseModel):
    """Orphan nodes the user deferred to the per-node LLM classifier."""
    nodeIds: List[str]
    kgSignature: str


class KGRefinementResponse(ResponseModel):
    """Response of ``/kg/apply-refinement``: the cleaned KG plus its new signature."""
    title: str
    model: Dict[str, Any]
    diagramType: str
    kgSignature: str
    pendingOrphanClassification: Optional[KGPendingOrphanClassification] = None
    exportedAt: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0.0"
