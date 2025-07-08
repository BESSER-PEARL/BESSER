from typing import Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

class DjangoConfig(BaseModel):
    project_name: str
    app_name: str
    containerization: bool = False

class SQLConfig(BaseModel):
    dialect: Literal["sqlite", "postgresql", "mysql", "mssql", "mariadb"] = "sqlite"

class SQLAlchemyConfig(BaseModel):
    dbms: Literal["sqlite", "postgresql", "mysql", "mssql", "mariadb"] = "sqlite"

class SizeInput(BaseModel):
    width: int
    height: int

class InteractiveInput(BaseModel):
    elements: Dict[str, Any] = Field(default_factory=dict)
    relationships: Dict[str, Any] = Field(default_factory=dict)

class ModelInput(BaseModel):
    version: str
    type: str
    size: SizeInput
    elements: Dict[str, Any] = Field(default_factory=dict)
    relationships: Dict[str, Any] = Field(default_factory=dict)
    interactive: InteractiveInput
    assessments: Dict[str, Any] = Field(default_factory=dict)

class DiagramInput(BaseModel):
    id: str
    title: str
    model: ModelInput
    lastUpdate: datetime
