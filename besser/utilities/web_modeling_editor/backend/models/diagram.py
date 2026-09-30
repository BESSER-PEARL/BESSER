from typing import Dict, Any, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, model_validator

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

    @model_validator(mode="after")
    def reject_legacy_v3_model(self) -> "DiagramInput":
        """Reject a UML model saved in the legacy v3 editor format.

        Raises ``LegacyDiagramFormatError`` (a ``ConversionError``) rather
        than ``ValueError`` on purpose: pydantic lets non-``ValueError``
        exceptions propagate, so the app-level ``ConversionError`` handler
        answers HTTP 400 with this message instead of a generic 422. It also
        fires for diagrams nested in a ``ProjectInput``.
        """
        # Lazy import: importing the services package at module load would
        # cycle back here (services.deployment -> routers -> models).
        from besser.utilities.web_modeling_editor.backend.services.validators.legacy_format import (
            ensure_diagram_not_legacy,
        )
        ensure_diagram_not_legacy(self.model, title=self.title, reference=self.referenceDiagramData)
        return self


class FeedbackSubmission(BaseModel):
    """Model for user feedback submissions."""
    satisfaction: Literal["happy", "neutral", "sad"]
    category: str = ""
    feedback: str
    email: Optional[str] = None
    timestamp: str
    user_agent: str
