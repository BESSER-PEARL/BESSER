"""GUI page personalization via an LLM.

Mirrors the ``agent_personalization`` module pattern: all prompt construction,
the OpenAI call and response parsing live here so the router stays a thin HTTP
layer. The router imports :func:`personalize_gui_page` and delegates to it.
"""
import json
import logging

from besser.generators.agents.agent_personalization import (
    DEFAULT_OPENAI_MODEL,
    call_openai_chat,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    GenerationError,
    ValidationError,
)
from besser.utilities.web_modeling_editor.backend.services.utils.agent_config_recommendation_utils import (
    extract_json_object,
)
from besser.utilities.web_modeling_editor.backend.services.utils.user_profile_utils import (
    generate_user_profile_document,
)

logger = logging.getLogger(__name__)


# LLM instruction for adapting a single GrapesJS page to a user profile.
# Kept as a module constant so it can be versioned, reviewed and tested in
# isolation rather than being buried inside a request handler.
GUI_PERSONALIZATION_SYSTEM_PROMPT = (
    "You are a UI personalization assistant for a GrapesJS-based no-code GUI editor. "
    "You receive a single GUI page as GrapesJS data: a 'components' array (the component "
    "tree, each node carrying fields such as type, tagName, attributes, classes, components, "
    "style, and text content) and a 'css' array (GrapesJS style-rule objects with 'selectors', "
    "'selectorsAdd', 'style', and 'pageId'). You also receive a user profile model.\n\n"
    "Your job is to adapt the page's PRESENTATION to this user, making decisions based on the "
    "information in the profile. Presentation covers both visual styling AND the wording of the "
    "page — its language, tone, and verbosity. Preserve the page's MEANING (its message, "
    "purpose, and the information each element conveys), but you MAY re-express that meaning in "
    "different words when the profile calls for it. When the profile clearly implies a "
    "preference — for example a single language the user reads, a reading level, or a tone — "
    "act on it decisively across the WHOLE page rather than leaving the wording unchanged.\n\n"
    "HOW TO APPLY STYLE — TWO TIERS:\n"
    "Tier 1 (PREFER THIS): express broad, consistent adaptations as high-level CSS rules keyed "
    "by semantic category, appended to the 'css' array. Target a category with the "
    "'selectorsAdd' field (a raw CSS selector string), NOT the 'selectors' array. Useful "
    "categories: 'body' (base font-size and line-height — most text inherits from here), "
    "headings ('h1, h2, h3, h4, h5, h6'), buttons ('button, .btn, a.button'), form fields "
    "('input, textarea, select'), 'label', links ('a'). Many components in this editor ship "
    "with baked-in inline/id-level styles that would otherwise win the cascade, so EVERY "
    "declaration in a Tier-1 rule MUST end with '!important' (e.g. {\"font-size\": \"20px "
    "!important\"}). A single 'body' rule with a larger font-size and line-height is the "
    "correct way to enlarge text everywhere — do NOT resize each text node individually.\n"
    "Tier 2 (USE SPARINGLY): for an adaptation a category rule cannot express (e.g. emphasizing "
    "one specific call-to-action), you MAY edit the 'style' object of individual components in "
    "the 'components' tree. Keep this to a small, deliberate set of nodes, and append "
    "'!important' when overriding an existing baked-in value.\n\n"
    "STRICT OUTPUT RULES: "
    "1) Return ONLY a single JSON object with EXACTLY this shape: "
    '{"components": [...], "css": [...]}. No markdown, no comments, no prose. '
    "2) Preserve the overall structure: keep each component's 'type', 'tagName', and "
    "'attributes.id' unchanged, and do not add, remove, or reorder components. "
    "3) You MAY (a) append new high-level rules to the 'css' array, (b) change the 'style' "
    "objects of existing css rules, (c) change class lists and 'style' objects of individual "
    "components (Tier 2), and (d) rewrite the user-visible wording of the page — text-node "
    "content and text-bearing attributes/properties (placeholder, title, alt, aria-label, link "
    "text, button and field labels) — to translate it or adjust its tone/verbosity. Every "
    "OTHER value must be byte-for-byte identical to the input. "
    "3a) Preserve MEANING, not exact words: when you rewrite wording (per 3d) the message, "
    "purpose, and information of every element must stay identical — translation and "
    "tone/verbosity changes are allowed, but do NOT add, drop, or alter what the page tells the "
    "user. A VISUAL adaptation such as enlarging the font is made ENTIRELY through "
    "'style'/class/CSS-rule changes and must not touch wording; never rewrite text into a "
    "description of the adaptation (e.g. do NOT turn a heading into 'bigger text for "
    "readability'). "
    "4) Keep every EXISTING css rule's 'selectors', 'selectorsAdd', and 'pageId' values exactly "
    "as given (you may still edit its 'style'). For each NEW Tier-1 rule you add, set "
    "'selectors' to an empty array and put the category selector in 'selectorsAdd'; you may "
    "omit 'pageId'. "
    "5) Return the input page verbatim if the profile gives no basis to adapt the presentation. "
    "6) The result must be valid JSON, parseable as-is, and remain a working GrapesJS page."
)


def build_user_prompt(page_name: str, profile_document, components, css) -> str:
    """Build the per-request user prompt for GUI page personalization."""
    return (
        f'Personalize the GUI page named "{page_name}" for the following user profile model.\n\n'
        f"User profile model:\n{json.dumps(profile_document, ensure_ascii=False, indent=2)}\n\n"
        "GUI page to adapt (GrapesJS):\n"
        f"{json.dumps({'components': components, 'css': css}, ensure_ascii=False)}\n\n"
        "Make decisions based on the information in the profile. Only adapt something when the "
        "profile gives a clear reason to. You can assume the profile contains the most important "
        "information about the user, and anything missing can be assumed absent.\n\n"
        "Prefer Tier-1 category CSS rules (every declaration ending with '!important') for anything "
        "that should apply broadly and consistently — e.g. if the user needs larger text, add a "
        "'body' rule with a larger font-size and line-height rather than resizing individual "
        "nodes. Use Tier-2 per-component edits only for targeted exceptions. Presentation includes "
        "textual presentation too (tone, verbosity, language) as long as the core meaning is "
        "preserved, as well as colors, contrast, font sizing, spacing, and forms. Your goal is the "
        "best possible experience for this user with all their needs met.\n\n"
        'Return only the resulting JSON object {"components": [...], "css": [...]}.'
    )


def personalize_gui_page(
    gui_page,
    user_profile_model,
    *,
    page_name: str = "page",
    model=None,
    openai_api_key=None,
) -> dict:
    """Personalize a single GrapesJS page for a user profile using an LLM.

    Args:
        gui_page: a GrapesJS page snapshot ``{"components": [...], "css": [...]}``.
        user_profile_model: a UserDiagram UML model JSON.
        page_name: the page name (used in the prompt only).
        model: optional OpenAI model id; falls back to ``DEFAULT_OPENAI_MODEL``.
        openai_api_key: resolved OpenAI API key (the caller resolves it).

    Returns:
        ``{"components": [...], "css": [...], "model": <id>}`` — the adapted page
        plus the model actually used.

    Raises:
        ValidationError: on malformed input or a missing OpenAI API key.
        GenerationError: when the LLM response cannot be parsed or is malformed.
    """
    if not isinstance(gui_page, dict):
        raise ValidationError(
            "guiPage is required and must be a JSON object with 'components' and 'css'"
        )
    if not isinstance(user_profile_model, dict):
        raise ValidationError("userProfileModel is required and must be a JSON object")

    components = gui_page.get("components")
    if not isinstance(components, list):
        raise ValidationError("guiPage.components must be a list")
    css = gui_page.get("css")
    if not isinstance(css, list):
        css = []

    if not isinstance(page_name, str) or not page_name.strip():
        page_name = "page"
    llm_model = model if isinstance(model, str) and model.strip() else DEFAULT_OPENAI_MODEL

    # Transform the user profile (UserDiagram) into its JSON object representation.
    profile_document = generate_user_profile_document(user_profile_model)
    user_prompt = build_user_prompt(page_name, profile_document, components, css)

    try:
        response_text = call_openai_chat(
            GUI_PERSONALIZATION_SYSTEM_PROMPT,
            user_prompt,
            model=llm_model,
            openai_api_key=openai_api_key,
        )
    except RuntimeError as runtime_error:
        # Raised by call_openai_chat when no OpenAI API key is configured.
        raise ValidationError(str(runtime_error)) from runtime_error

    try:
        parsed = extract_json_object(response_text)
    except ValueError as parse_error:
        logger.exception("Failed to parse LLM personalization response")
        raise GenerationError("Failed to parse LLM personalization response") from parse_error

    personalized_components = parsed.get("components")
    if not isinstance(personalized_components, list):
        raise GenerationError("LLM response did not include a valid 'components' list")
    personalized_css = parsed.get("css", [])
    if not isinstance(personalized_css, list):
        personalized_css = []

    return {
        "components": personalized_components,
        "css": personalized_css,
        "model": llm_model,
    }
