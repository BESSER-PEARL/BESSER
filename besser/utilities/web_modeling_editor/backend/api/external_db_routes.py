from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from besser.utilities.web_modeling_editor.backend.services.external_db_service import get_database_metadata
from besser.utilities.web_modeling_editor.backend.services.converters.db_metadata_to_buml import parse_metadata_to_buml
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import class_buml_to_json

router = APIRouter()

class ConnectionRequest(BaseModel):
    connection_url: str

@router.post("/external-db/metadata")
async def fetch_external_db_metadata(request: ConnectionRequest):
    """
    Endpoint to retrieve database metadata and return a class diagram JSON.
    """
    try:
        metadata = get_database_metadata(request.connection_url)
        domain_model = parse_metadata_to_buml(metadata, model_name="ExternalDB_DomainModel")
        diagram_json = class_buml_to_json(domain_model)
        return diagram_json
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
