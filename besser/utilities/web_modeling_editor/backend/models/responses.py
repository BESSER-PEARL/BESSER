"""Standardized API response models."""
from datetime import datetime, timezone
from typing import Any, Dict, List
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
