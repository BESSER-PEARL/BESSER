"""
Generation Router

Handles all code generation endpoints for the BESSER web modeling editor backend.
Generator dispatch lives in ``services/generation``; the routes here parse the
request, call the service and turn its result into the HTTP response.
"""

import asyncio
import json
import logging
import os
import re
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Header, HTTPException
from fastapi.responses import StreamingResponse, Response

from besser.utilities.web_modeling_editor.backend.routers.github_oauth import (
    get_user_token,
)

# BESSER utilities
from besser.generators.agents.agent_personalization import call_openai_chat

# Backend models
from besser.utilities.web_modeling_editor.backend.models import (
    DiagramInput,
    ProjectInput,
)

# Backend services
from besser.utilities.web_modeling_editor.backend.services.generation import (
    GeneratedOutput,
    generate_from_diagram,
    generate_from_project,
)
from besser.utilities.web_modeling_editor.backend.services.utils.agent_generation_utils import (
    extract_openai_api_key,
)
from besser.utilities.web_modeling_editor.backend.services.utils.agent_config_recommendation_utils import (
    RECOMMENDATION_ALLOWED_VALUES,
    load_default_agent_recommendation_config,
    extract_json_object,
    normalize_recommended_agent_config,
)
from besser.utilities.web_modeling_editor.backend.services.utils.agent_config_manual_mapping_utils import (
    get_manual_agent_config_mapping,
    build_manual_mapping_recommendation,
)
from besser.utilities.web_modeling_editor.backend.services.utils.user_profile_utils import (
    generate_user_profile_document as _generate_user_profile_document,
)
from besser.utilities.web_modeling_editor.backend.services.utils.gui_personalization_utils import (
    personalize_gui_page as run_gui_personalization,
)

# Centralized error handling and shared auth gate
from besser.utilities.web_modeling_editor.backend.routers.auth import require_github_session
from besser.utilities.web_modeling_editor.backend.dependencies import GitHubSessionsDep
from besser.utilities.web_modeling_editor.backend.services.deployment.session_store import (
    GitHubSessions,
)
from besser.utilities.web_modeling_editor.backend.routers.error_handler import (
    handle_endpoint_errors,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    GenerationError,
    ValidationError,
)
from besser.utilities.web_modeling_editor.backend.services.validators.legacy_format import (
    ensure_not_legacy_payload,
)

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/besser_api", tags=["generation"])


def _require_github_session(github_sessions: GitHubSessions, github_session: Optional[str]) -> None:
    """Verify a GitHub OAuth session is present and active (HTTP 401 otherwise).

    Delegates to the shared ``routers.auth.require_github_session`` and passes
    this module's ``get_user_token`` so tests can keep patching it here.
    """
    require_github_session(github_sessions, github_session, token_lookup=get_user_token)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@router.post("/recommend-agent-config-llm")
@handle_endpoint_errors("recommend_agent_config_llm")
async def recommend_agent_config_llm(
    github_sessions: GitHubSessionsDep,
    payload: Dict[str, Any] = Body(...),
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Recommend a structured agent configuration from a user profile using an LLM.

    Requires authenticated GitHub session.
    """
    _require_github_session(github_sessions, github_session)
    user_profile_model = payload.get("userProfileModel")
    if not isinstance(user_profile_model, dict):
        raise ValidationError("userProfileModel is required and must be a JSON object")
    ensure_not_legacy_payload(user_profile_model, "UserDiagram", "userProfileModel")

    user_profile_name = payload.get("userProfileName") if isinstance(payload.get("userProfileName"), str) else None
    current_config = payload.get("currentConfig") if isinstance(payload.get("currentConfig"), dict) else {}
    requested_model = payload.get("model")
    llm_model = (
        requested_model
        if isinstance(requested_model, str) and requested_model.strip()
        else "gpt-5.5"
    )

    profile_document = _generate_user_profile_document(user_profile_model)
    default_config = load_default_agent_recommendation_config()

    # Provider choice and every per-provider model list are excluded: this prompt
    # flow does not recommend the LLM provider, so shipping the model catalogues
    # would only add noise. Matching on the "Models" suffix keeps this correct as
    # new providers are added, instead of a hand-maintained key list.
    allowed_values_payload = {
        key: value
        for key, value in RECOMMENDATION_ALLOWED_VALUES.items()
        if key != "llmProvider" and not key.endswith("Models")
    }

    system_prompt = (
        "You are an assistant that recommends a valid agent configuration JSON for a user profile. "
        "Return ONLY a JSON object with this exact shape: "
        "{presentation:{...}, modality:{...}, behavior:{...}, content:{...}, system:{...}}. "
        "Use only allowed values and keep output concise and deterministic. "
        "Do not include markdown, comments, or explanatory text."
    )
    user_prompt = (
        "Create a recommended configuration adapted to this user profile.\n\n"
        f"Selected profile name: {user_profile_name or 'N/A'}\n\n"
        f"Default config baseline:\n{json.dumps(default_config, ensure_ascii=False, indent=2)}\n\n"
        f"Allowed values:\n{json.dumps(allowed_values_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Current config context (optional):\n{json.dumps(current_config, ensure_ascii=False, indent=2)}\n\n"
        f"User profile document:\n{json.dumps(profile_document, ensure_ascii=False, indent=2)}\n\n"
        "Return only the JSON object."
    )

    try:
        recommendation_text = await asyncio.to_thread(
            call_openai_chat,
            system_prompt,
            user_prompt,
            model=llm_model,
            openai_api_key=extract_openai_api_key(payload if isinstance(payload, dict) else {}),
            config=payload,
        )
        parsed_recommendation = extract_json_object(recommendation_text)
        normalized_config = normalize_recommended_agent_config(parsed_recommendation, user_profile_name)

        return {
            "config": normalized_config,
            "source": "openai",
            "model": llm_model,
            "generatedAt": _utc_now_iso(),
        }
    except RuntimeError as runtime_error:
        raise ValidationError(str(runtime_error)) from runtime_error
    except ValueError as parse_error:
        logger.exception("Failed to parse LLM recommendation response")
        raise GenerationError("Failed to parse LLM recommendation response") from parse_error
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to generate LLM recommendation")
        raise GenerationError("Failed to generate LLM recommendation") from exc


@router.post("/personalize-gui-page")
@handle_endpoint_errors("personalize_gui_page")
async def personalize_gui_page(payload: Dict[str, Any] = Body(...)):
    """Personalize a GUI page (GrapesJS) for a user profile using an LLM.

    Open endpoint (no GitHub session required). The OpenAI API key is resolved
    from the payload or the server's ``OPENAI_API_KEY`` environment variable.

    Request body:
        guiPage: {"components": [...], "css": [...]}  -- a GrapesJS page snapshot
        userProfileModel: <UserDiagram UML model JSON>
        pageName: str (optional)
        model: str (optional OpenAI model id)

    Returns the same ``{components, css}`` shape adapted in style and content,
    ready to be imported back into the editor as a page variant.
    """
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object")

    # All validation, prompt building, the LLM call and response parsing live in
    # gui_personalization_utils (mirroring agent_personalization). The router
    # only resolves the API key, delegates off the event loop, and shapes the
    # HTTP response; @handle_endpoint_errors maps ValidationError/GenerationError.
    ensure_not_legacy_payload(payload.get("userProfileModel"), "UserDiagram", "userProfileModel")
    result = await asyncio.to_thread(
        run_gui_personalization,
        payload.get("guiPage"),
        payload.get("userProfileModel"),
        page_name=payload.get("pageName"),
        model=payload.get("model"),
        openai_api_key=extract_openai_api_key(payload),
    )
    return {
        "guiPage": {"components": result["components"], "css": result["css"]},
        "source": "openai",
        "model": result["model"],
        "generatedAt": _utc_now_iso(),
    }


@router.get("/agent-config-manual-mapping")
@handle_endpoint_errors("get_agent_config_manual_mapping")
async def get_agent_config_manual_mapping(
    github_sessions: GitHubSessionsDep,
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Return the complete rule mapping used for deterministic recommendations.

    Requires authenticated GitHub session.
    """
    _require_github_session(github_sessions, github_session)
    return {
        "mapping": get_manual_agent_config_mapping(),
        "source": "manual_mapping",
        "generatedAt": _utc_now_iso(),
    }


@router.post("/recommend-agent-config-mapping")
@handle_endpoint_errors("recommend_agent_config_mapping")
async def recommend_agent_config_mapping(
    github_sessions: GitHubSessionsDep,
    payload: Dict[str, Any] = Body(...),
    github_session: Optional[str] = Header(None, alias="X-GitHub-Session"),
):
    """Recommend a structured agent configuration using deterministic mapping rules.

    Requires authenticated GitHub session.
    """
    _require_github_session(github_sessions, github_session)
    user_profile_model = payload.get("userProfileModel")
    if not isinstance(user_profile_model, dict):
        raise ValidationError("userProfileModel is required and must be a JSON object")
    ensure_not_legacy_payload(user_profile_model, "UserDiagram", "userProfileModel")

    user_profile_name = payload.get("userProfileName") if isinstance(payload.get("userProfileName"), str) else None
    current_config = payload.get("currentConfig") if isinstance(payload.get("currentConfig"), dict) else {}

    profile_document = _generate_user_profile_document(user_profile_model)
    recommendation = build_manual_mapping_recommendation(
        user_profile_document=profile_document,
        user_profile_name=user_profile_name,
        current_config=current_config,
    )

    return {
        "config": recommendation["config"],
        "matchedRules": recommendation.get("matchedRules", []),
        "signals": recommendation.get("signals", {}),
        "source": "manual_mapping",
        "generatedAt": _utc_now_iso(),
    }


@router.post("/generate-output-from-project")
@handle_endpoint_errors("generate_code_output_from_project")
async def generate_code_output_from_project(input_data: ProjectInput):
    """
    Generate code output from a complete project.
    This endpoint handles generators that require multiple diagrams (e.g., Web App).
    """
    return _download_response(await generate_from_project(input_data))


@router.post("/generate-output")
@handle_endpoint_errors("generate_code_output")
async def generate_code_output(input_data: DiagramInput):
    """
    Generate code from UML diagram data using specified generator.

    Args:
        input_data: DiagramInput containing diagram data and generator type

    Returns:
        StreamingResponse or Response: Generated code file or ZIP

    Raises:
        HTTPException: If generator is not supported or generation fails
    """
    return _download_response(await generate_from_diagram(input_data))


def _download_response(output: GeneratedOutput) -> Response:
    """A ZIP is streamed; a single file is sent whole."""
    if output.encode_file_name:
        disposition = _attachment(output.file_name)
    else:
        disposition = f'attachment; filename="{output.file_name}"'
    headers = {"Content-Disposition": disposition}
    if output.is_zip:
        return StreamingResponse(output.content, media_type=output.media_type, headers=headers)
    return Response(content=output.content, media_type=output.media_type, headers=headers)


def _attachment(file_name: str) -> str:
    """The Content-Disposition of a download. HTTP headers are latin-1: a name
    outside ASCII (a model called ``数据``) is sent as RFC 6266's UTF-8
    ``filename*``, with an ASCII ``filename`` for older clients."""
    if file_name.isascii():
        return f'attachment; filename="{file_name}"'
    stem, ext = os.path.splitext(file_name)
    fallback = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "download"
    return (f'attachment; filename="{fallback}{ext}"; '
            f"filename*=UTF-8''{urllib.parse.quote(file_name)}")
