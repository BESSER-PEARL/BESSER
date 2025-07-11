from typing import Optional, Dict, Any
from pydantic import BaseModel
from besser.utilities.web_modeling_editor.backend.models import DiagramInput

class ProjectInput(BaseModel):
    id: str
    type: str
    name: str
    description: Optional[str]
    owner: Optional[str]
    createdAt: str
    currentDiagramType: Optional[str]
    diagrams: Dict[str, DiagramInput]
    settings: Optional[Dict[str, Any]]
