from typing import Optional

from pydantic import BaseModel, Field


class SdmxImportRequest(BaseModel):
    """Import a class diagram from the structure of an SDMX dataflow.

    Give either ``provider`` (a preset key, see ``GET /sdmx-providers``) or ``baseUrl``
    (the https base URL of an SDMX 2.1 REST API).
    """
    dataflow: str = Field(..., min_length=1, max_length=200, description="ID, AGENCY,ID or AGENCY,ID,VERSION")
    provider: Optional[str] = Field(None, max_length=40)
    baseUrl: Optional[str] = Field(None, max_length=500)
    agency: Optional[str] = Field(None, max_length=80)
    version: Optional[str] = Field(None, max_length=20)
    modelName: Optional[str] = Field(None, max_length=100)


class SdmxProvider(BaseModel):
    """A preset SDMX provider the editor offers."""
    key: str
    name: str
    baseUrl: str
    agency: str


class SdmxProvidersResponse(BaseModel):
    """The preset providers, and whether a custom SDMX API URL is accepted."""
    providers: list[SdmxProvider]
    customUrlAllowed: bool
