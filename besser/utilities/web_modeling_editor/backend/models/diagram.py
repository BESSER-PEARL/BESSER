from typing import Dict, Any, Literal, Optional
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


class DiagramInput(BaseModel):
    id: Optional[str] = None
    title: str
    model: Dict[str, Any]
    lastUpdate: Optional[datetime] = None
    generator: Optional[str] = None
    config: Optional[dict] = None
    referenceDiagramData: Optional[Dict[str, Any]] = None
