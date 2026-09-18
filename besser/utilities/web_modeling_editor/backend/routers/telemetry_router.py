"""Pilot-experiment telemetry router.

Two endpoints:

* ``POST /besser_api/telemetry/event`` — the collector. Producers (the
  modeling agent's fire-and-forget prompt events, the frontend's delivery
  clicks) post ``{session, participant, kind, payload}``. Responds 204 on
  acceptance AND when telemetry is disabled — the response never reveals
  whether collection is active. 422 is reserved for malformed input.

* ``GET /besser_api/telemetry/report`` — the aggregated pilot report,
  gated by the ``X-Telemetry-Token`` header matching
  ``BESSER_TELEMETRY_ADMIN_TOKEN``. When no admin token is configured the
  endpoint answers 404 as if it did not exist.

Run summaries (kind ``run_summary``) are recorded in-process by the
spec-driven runner, never over HTTP — the collector rejects that kind by
schema.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from typing import Any, Literal, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from besser.utilities.web_modeling_editor.backend.services.spec_driven.telemetry import (
    MAX_PAYLOAD_BYTES,
    build_report,
    record_event,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/besser_api", tags=["telemetry"])

_SESSION_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
_PARTICIPANT_PATTERN = r"^[A-Za-z0-9_-]{1,16}$"


class TelemetryEventRequest(BaseModel):
    """Body of ``POST /besser_api/telemetry/event``."""

    session: str = Field(..., pattern=_SESSION_PATTERN)
    participant: str = Field(..., pattern=_PARTICIPANT_PATTERN)
    kind: Literal["prompt", "agent_action", "delivery", "friction"]
    payload: dict[str, Any] = Field(default_factory=dict)


@router.post("/telemetry/event", status_code=204)
async def telemetry_event(event: TelemetryEventRequest) -> Response:
    """Accept one pilot telemetry event.

    Always 204 for well-formed input, whether or not the event was
    stored — a disabled deployment is indistinguishable from an enabled
    one, so the endpoint cannot be used to probe server configuration.
    422 only for malformed input (invalid session/participant/kind, or a
    payload whose serialized form exceeds the size cap).
    """
    try:
        serialized = json.dumps(event.payload, ensure_ascii=False, default=str)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail="payload is not JSON-serializable"
        ) from exc
    if len(serialized.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"payload exceeds the {MAX_PAYLOAD_BYTES // 1024}KB limit",
        )

    # Best-effort by contract: record_event applies the master switch and
    # participant gate internally and never raises.
    record_event(event.session, event.participant, event.kind, event.payload)
    return Response(status_code=204)


@router.get("/telemetry/report")
async def telemetry_report(
    participant: Optional[str] = Query(default=None, pattern=_PARTICIPANT_PATTERN),
    format: Literal["md", "csv"] = Query(default="md"),
    token: Optional[str] = Header(default=None, alias="X-Telemetry-Token"),
) -> PlainTextResponse:
    """Aggregated pilot report (Markdown default, CSV alternative).

    Requires ``X-Telemetry-Token`` to match ``BESSER_TELEMETRY_ADMIN_TOKEN``.
    Answers 404 when no admin token is configured (the endpoint should be
    indistinguishable from absent on non-pilot deployments) and 403 for a
    missing or wrong token.
    """
    admin_token = os.environ.get("BESSER_TELEMETRY_ADMIN_TOKEN", "").strip()
    if not admin_token:
        raise HTTPException(status_code=404, detail="Not Found")
    if not token or not secrets.compare_digest(token, admin_token):
        raise HTTPException(
            status_code=403, detail="Invalid or missing telemetry token"
        )

    try:
        text = build_report(participant=participant, fmt=format)
    except Exception as exc:
        logger.exception("Failed to build telemetry report")
        raise HTTPException(
            status_code=500, detail="Failed to build the telemetry report"
        ) from exc

    media_type = "text/markdown" if format == "md" else "text/csv"
    return PlainTextResponse(text, media_type=f"{media_type}; charset=utf-8")
