"""
Gap analysis for the LLM orchestrator.

Compares user instructions against generator capabilities to determine
what customizations the LLM needs to implement. Uses LLM-based analysis
(cheap Haiku call) with keyword matching as a fallback.
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Keywords used in gap analysis
_AUTH_KEYWORDS = {"auth", "jwt", "login", "register", "password", "token", "oauth", "session"}
_DB_KEYWORDS = {"postgres", "postgresql", "mysql", "mongo", "database", "db"}
_DOCKER_KEYWORDS = {"docker", "container", "deploy", "dockerfile", "compose"}
_TEST_KEYWORDS = {"test", "pytest", "unittest", "jest", "testing"}
_SEARCH_KEYWORDS = {"search", "filter", "query", "find"}
_PAGINATION_KEYWORDS = {"pagination", "paginate", "paging", "limit", "offset", "page"}
_ROLE_KEYWORDS = {"role", "rbac", "permission", "admin", "authorization", "access control"}


def analyze_gaps(
    instructions: str,
    generator_used: str | None,
    domain_model,
    llm_client,
) -> list[str]:
    """
    Compare user instructions against generator capabilities.

    Returns a list of specific tasks the LLM should do.

    If a generator was used, attempts LLM-based gap analysis using a
    cheap Haiku call. Falls back to keyword matching if the call fails.

    Args:
        instructions: User's natural language instructions.
        generator_used: Name of the generator used in Phase 1, or None.
        domain_model: The BUML domain model.
        llm_client: The Claude LLM client (for Haiku gap analysis).

    Returns:
        List of task strings.
    """
    if not generator_used:
        # No generator -- LLM builds everything from scratch
        return [
            "No BESSER generator was used. Write the entire application from scratch "
            "using the domain model as your specification. Every entity, attribute, "
            "type, and relationship in your code must match the model."
        ]

    # Try LLM-based gap analysis first
    llm_tasks = _analyze_gaps_with_llm(instructions, generator_used, domain_model, llm_client)
    if llm_tasks is not None:
        return llm_tasks

    # Fall back to keyword matching
    return _analyze_gaps_keyword(instructions, generator_used)


def _analyze_gaps_with_llm(
    instructions: str,
    generator_used: str,
    domain_model,
    llm_client,
) -> list[str] | None:
    """
    Use a cheap Haiku call to analyze what customizations are needed.

    Returns a list of task strings, or None if the call fails
    (in which case the caller falls back to keyword matching).
    """
    try:
        model_summary = build_model_summary(domain_model)
        prompt = (
            f"Model: {model_summary}. "
            f"Generator used: {generator_used}. "
            f"User wants: {instructions}\n\n"
            "The generator already provides: CRUD endpoints, ORM, schemas, basic pages.\n"
            "List ONLY what's missing as a JSON array of task strings. "
            "Each task should be a specific, actionable instruction for a developer. "
            "Include a README task. Respond with ONLY the JSON array, no other text."
        )

        # Skip if client doesn't look like a real provider (mock in tests)
        if not hasattr(llm_client, '_client'):
            return None

        # Use the main LLM client (whatever provider — Claude, OpenAI, etc.)
        response = llm_client.chat(
            system="You analyze what customizations are needed for a generated app. "
                   "Respond with ONLY a JSON array of task strings.",
            messages=[{"role": "user", "content": prompt}],
            tools=[],
        )

        # Parse the response
        text = ""
        for block in response.get("content", []):
            if hasattr(block, "text"):
                text = block.text.strip()
            elif isinstance(block, dict) and block.get("text"):
                text = block["text"].strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        tasks = json.loads(text)
        if isinstance(tasks, list) and all(isinstance(t, str) for t in tasks):
            logger.info("LLM gap analysis returned %d tasks", len(tasks))
            return tasks

        logger.warning("LLM gap analysis returned unexpected format: %s", type(tasks))
        return None

    except Exception as e:
        logger.warning("LLM gap analysis failed, falling back to keywords: %s", e)
        return None


def _analyze_gaps_keyword(instructions: str, generator_used: str) -> list[str]:
    """
    Keyword-based gap analysis fallback.

    Compare user instructions against generator capabilities using
    keyword matching. Returns a list of specific tasks.
    """
    tasks = []
    lower = instructions.lower()

    def _has_keyword(keywords: set[str]) -> bool:
        """Check if any keyword appears as a word boundary in the text."""
        pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b'
        return bool(re.search(pattern, lower))

    # The generator handled: CRUD, ORM, schemas, basic pages
    # Check what the user asked for that the generator doesn't provide:

    if _has_keyword(_AUTH_KEYWORDS):
        tasks.append(
            "Add authentication: Write a NEW backend/auth.py (or web_app/backend/auth.py) "
            "with JWT token creation, password hashing (bcrypt), and a get_current_user dependency. "
            "Then use modify_file to add 'from auth import get_current_user' to main_api.py imports, "
            "and add Depends(get_current_user) to protected endpoints. "
            "Add python-jose and passlib to requirements.txt."
        )

    if _has_keyword(_DB_KEYWORDS):
        db_type = "PostgreSQL" if "postgres" in lower else "the requested database"
        tasks.append(
            f"Switch to {db_type}: Use modify_file on sql_alchemy.py to change the "
            f"create_engine() connection string. Add the appropriate driver to requirements.txt "
            f"(e.g. psycopg2-binary for PostgreSQL)."
        )

    if _has_keyword(_PAGINATION_KEYWORDS):
        tasks.append(
            "Add pagination: Use modify_file on main_api.py to add skip/limit query parameters "
            "to GET list endpoints."
        )

    if _has_keyword(_SEARCH_KEYWORDS):
        tasks.append(
            "Add search: Use modify_file on main_api.py to add search query parameters "
            "to relevant GET endpoints (filter by name, email, etc.)."
        )

    if _has_keyword(_ROLE_KEYWORDS):
        tasks.append(
            "Add role-based access: Extend auth.py with role checking. "
            "Use modify_file to add role-based guards to protected endpoints."
        )

    if _has_keyword(_DOCKER_KEYWORDS):
        tasks.append(
            "Update Docker config: modify docker-compose.yml for the requested services "
            "(e.g., add PostgreSQL container). Update Dockerfiles if needed. "
            "Write .env.example with all configuration variables."
        )
    elif not generator_used.endswith("web_app"):
        # web_app generator already produces Docker; others don't
        tasks.append(
            "Write Docker deployment: docker-compose.yml, backend Dockerfile, "
            "frontend Dockerfile (if frontend exists), .env.example."
        )

    if _has_keyword(_TEST_KEYWORDS):
        tasks.append("Write tests for the key endpoints and business logic.")

    # Always add README
    tasks.append(
        "Write README.md with setup instructions: how to install deps, "
        "configure env vars, run the app (with and without Docker)."
    )

    # If user mentions custom pages/features not in the generator
    custom_features = []
    for keyword in ["dashboard", "kanban", "pipeline", "analytics", "chart"]:
        if keyword in lower:
            custom_features.append(keyword)
    if custom_features:
        tasks.append(
            f"Write custom frontend pages for: {', '.join(custom_features)}. "
            "Create new .tsx files in the pages directory. "
            "The generated pages handle basic CRUD -- write NEW pages for these custom views."
        )

    return tasks


def build_model_summary(domain_model) -> str:
    """Build a concise model summary for the gap analysis prompt."""
    classes = [c.name for c in domain_model.get_classes()]
    enums = [e.name for e in domain_model.get_enumerations()]
    parts = [f"Classes: {', '.join(classes)}"]
    if enums:
        parts.append(f"Enums: {', '.join(enums)}")
    parts.append(f"Associations: {len(domain_model.associations)}")
    return "; ".join(parts)
