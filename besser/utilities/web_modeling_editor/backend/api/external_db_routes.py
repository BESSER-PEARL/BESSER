from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from besser.utilities.web_modeling_editor.backend.services.external_db_service import get_database_metadata

router = APIRouter()

class ConnectionRequest(BaseModel):
    connection_url: str

@router.post("/external-db/metadata")
async def fetch_external_db_metadata(request: ConnectionRequest):
    """
    Endpoint to retrieve database metadata using a connection URL.
    """
    try:
        metadata = get_database_metadata(request.connection_url)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
