from typing import Dict, Any, List, Literal, Optional
from datetime import datetime
from pydantic import BaseModel

class DjangoConfig(BaseModel):
    project_name: str
    app_name: str
    containerization: bool = False

class SQLConfig(BaseModel):
    dialect: Literal["sqlite", "postgresql", "mysql", "mssql", "mariadb"] = "sqlite"

class SQLAlchemyConfig(BaseModel):
    dbms: Literal["sqlite", "postgresql", "mysql", "mssql", "mariadb"] = "sqlite"


class DiagramInput(BaseModel):
    id: Optional[str] = None
    title: str
    model: Dict[str, Any]
    lastUpdate: Optional[datetime] = None
    generator: Optional[str] = None
    config: Optional[dict] = None
    configYaml: Optional[str] = None
    referenceDiagramData: Optional[Dict[str, Any]] = None
    references: Optional[Dict[str, str]] = None  # Per-diagram cross-references by ID (e.g. {"ClassDiagram": "uuid-..."})
    # KG → BUML preflight integration. ``resolutions`` lets the frontend
    # send back user choices for issues surfaced by
    # ``/analyze-kg-for-buml-conversion``; ``kgSignature`` is the value
    # echoed from the analyze response so the backend can detect a
    # stale resolution payload (KG mutated between analyze & convert).
    resolutions: Optional[List[Dict[str, Any]]] = None
    kgSignature: Optional[str] = None
    # LLM-cleanup round-trip: ``llmIssues`` carries the issue list returned
    # by ``/llm-clean-kg`` so ``/apply-kg-refinement`` can reconstruct the
    # KGIssue / KGAction objects without re-calling the LLM. (The
    # natural-language system description is not part of this model: it
    # arrives as a multipart form field on the endpoints that need it.)
    llmIssues: Optional[List[Dict[str, Any]]] = None
    # KG refinement (unified Refine KG modal): ``source`` selects which
    # apply path the unified ``/apply-kg-refinement`` endpoint takes —
    # ``"static"`` re-runs the static analyzer and dispatches accept/skip
    # decisions, ``"llm"`` reconstructs LLM-issue objects from
    # ``llmIssues`` and dispatches them.
    source: Optional[Literal["static", "llm"]] = None
    # KG RDF export: which constraint vocabularies to emit. ``"owl"`` emits
    # only ``owl:Restriction`` blank nodes; ``"shacl"`` emits only
    # ``sh:NodeShape`` / ``sh:PropertyShape``; ``"both"`` (default) emits
    # both where each spec has the required semantics.
    vocab: Optional[Literal["owl", "shacl", "both"]] = None


class FeedbackSubmission(BaseModel):
    """Model for user feedback submissions."""
    satisfaction: Literal["happy", "neutral", "sad"]
    category: str = ""
    feedback: str
    email: Optional[str] = None
    timestamp: str
    user_agent: str
