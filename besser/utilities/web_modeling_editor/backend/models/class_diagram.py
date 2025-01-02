from pydantic import BaseModel
from typing import Optional, Dict, Any

class ClassDiagramInput(BaseModel):
    elements: Dict[str, Any]
    generator: str
    ocl: Optional[str] = None
