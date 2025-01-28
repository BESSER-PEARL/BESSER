from pydantic import BaseModel
from typing import Optional, Dict, Any

class DjangoConfig(BaseModel):
    project_name: str
    app_name: str
    containerization: bool = False

class ClassDiagramInput(BaseModel):
    elements: Dict[str, Any]
    generator: Optional[str] = None
    config: Optional[DjangoConfig] = None
