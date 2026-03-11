from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, model_validator
from besser.utilities.web_modeling_editor.backend.models import DiagramInput

class ProjectInput(BaseModel):
    id: str
    type: str
    name: str
    description: Optional[str] = None
    owner: Optional[str] = None
    createdAt: str
    currentDiagramType: Optional[str] = None
    currentDiagramIndices: Optional[Dict[str, int]] = None
    diagrams: Dict[str, List[DiagramInput]]
    settings: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_diagrams(cls, data: Any) -> Any:
        """Accept both old format (single DiagramInput per type) and new format (arrays)."""
        if isinstance(data, dict):
            diagrams = data.get("diagrams")
            if isinstance(diagrams, dict):
                normalized = {}
                for key, value in diagrams.items():
                    if isinstance(value, list):
                        normalized[key] = value
                    elif isinstance(value, dict):
                        # Old format: single diagram → wrap in list
                        normalized[key] = [value]
                    else:
                        normalized[key] = value
                data["diagrams"] = normalized
        return data

    def get_active_diagram(self, diagram_type: str) -> Optional[DiagramInput]:
        """Get the active diagram for a type using currentDiagramIndices."""
        diagrams = self.diagrams.get(diagram_type, [])
        if not diagrams:
            return None
        index = (self.currentDiagramIndices or {}).get(diagram_type, 0)
        safe_index = min(index, len(diagrams) - 1) if diagrams else 0
        return diagrams[safe_index]

    def get_referenced_diagram(
        self,
        from_diagram: Optional[DiagramInput],
        ref_type: str,
    ) -> Optional[DiagramInput]:
        """Get the diagram that `from_diagram` references for `ref_type`.

        Looks up by diagram ID (stable across deletions/reordering).
        Falls back to `currentDiagramIndices[ref_type]` when no
        per-diagram reference exists or the referenced ID is not found.
        """
        diagrams = self.diagrams.get(ref_type, [])
        if not diagrams:
            return None

        # Look up by ID (string-based reference)
        if from_diagram and from_diagram.references:
            ref_id = from_diagram.references.get(ref_type)
            if ref_id:
                # Could be an ID string or a legacy integer index
                if isinstance(ref_id, str):
                    found = next((d for d in diagrams if d.id == ref_id), None)
                    if found:
                        return found
                elif isinstance(ref_id, int):
                    # Legacy index-based reference
                    safe_index = min(ref_id, len(diagrams) - 1)
                    return diagrams[safe_index]

        # Fallback: use global active index
        fallback_index = (self.currentDiagramIndices or {}).get(ref_type, 0)
        safe_index = min(fallback_index, len(diagrams) - 1)
        return diagrams[safe_index]
