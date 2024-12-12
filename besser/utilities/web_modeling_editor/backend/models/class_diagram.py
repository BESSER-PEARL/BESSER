from pydantic import BaseModel

class ClassDiagramInput(BaseModel):
    elements: dict
    generator: str
