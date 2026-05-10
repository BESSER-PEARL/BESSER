"""Pydantic models for the database reverse-engineering endpoints."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


SupportedDialect = Literal["postgresql", "mysql", "sqlite", "mssql", "oracle"]


class DbConnectionParams(BaseModel):
    """Connection details for an external database.

    ``raw_url`` (a SQLAlchemy URL) takes precedence over the structured
    fields when provided. For SQLite, set ``database`` to either an
    absolute server-side path or — preferably — a ``database_token``
    obtained from ``/db-upload-sqlite``.
    """
    dialect: Optional[SupportedDialect] = None
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    raw_url: Optional[str] = None
    database_token: Optional[str] = Field(
        default=None,
        description="Token returned by /db-upload-sqlite for an uploaded SQLite file.",
    )
    options: Dict[str, str] = Field(default_factory=dict)


class DbIntrospectRequest(BaseModel):
    connection: DbConnectionParams


class DbImportRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    connection: DbConnectionParams
    selection: Dict[str, List[str]] = Field(
        ...,
        description="Mapping of schema name to list of table names to import.",
    )
    model_name: str = "ClassDiagram"
