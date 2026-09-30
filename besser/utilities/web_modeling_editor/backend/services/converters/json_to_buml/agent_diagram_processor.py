"""
Agent diagram processing for converting v4 JSON to BUML format.

Reads the v4 wire shape natively: canvas ``nodes`` / ``edges`` plus the
off-canvas ``components`` map (intents, LLMs, RAG databases, tools, skills,
workspaces, GUIs). ``AgentState`` action rows (``data.bodies`` /
``data.fallbackBodies``) are dispatched one by one on their ``actionType``
(metamodel class name) or legacy ``replyType``, preserving their order.
``_normalise_agent_transitions`` collapses the historical
``AgentStateTransition`` shapes (see ``docs/source/migrations/uml-v4-shape.md``)
to the canonical ``transitionType + predefined|custom`` form.
"""

import logging
import operator

from deep_translator import GoogleTranslator

from besser.BUML.metamodel.state_machine.state_machine import (
    Body,
    Condition,
    CustomCodeAction,
    TransitionBuilder,
)
from besser.BUML.metamodel.state_machine.agent import (
    Agent,
    Intent,
    DummyEvent,
    IntentMatcher,
    ReceiveFileEvent,
    ReceiveJSONEvent,
    ReceiveMessageEvent,
    ReceiveTextEvent,
    WildcardEvent,
    AgentReply,
    LLMReply,
    LLMChatReply,
    RAGReply,
    DBReply,
    WebCrawlLLMReply,
    WebSocketReplyMarkdown,
    WebSocketReplyHTML,
    WebSocketReplySpeech,
    WebSocketReplyOptions,
    WebSocketReplyLocation,
    WebSocketReplyFile,
    WebSocketReplyImage,
    WebSocketReplyDataframe,
    WebSocketReplyPlotly,
    RAGVectorStore,
    RAGTextSplitter,
    GUIReplyAction,
    GUIEvent,
)
from besser.BUML.metamodel.structural import DomainModel, Metadata
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml._node_helpers import (
    node_data,
)
from besser.utilities.web_modeling_editor.backend.services.converters.parsers import sanitize_text
from besser.utilities.web_modeling_editor.backend.services.validators.python_code_validator import (
    validate_custom_code_action,
)

logger = logging.getLogger(__name__)

_DEFAULT_INPUT_PROMPT_MODE = "last_user_message"


# Maps old informal "replyType" values to new metamodel class names used in "actionType".
_REPLY_TYPE_TO_ACTION_TYPE = {
    "text": "TextReplyAction",
    "llm": "LLMReplyAction",
    "llm_chat": "LLMChatAction",
    "rag": "RAGReplyAction",
    "db_reply": "DBAction",
    "code": "CustomCodeAction",
    "web_crawl_llm": "WebCrawlLLMAction",
    "ws_markdown": "WebSocketReplyMarkdownAction",
    "ws_html": "WebSocketReplyHTMLAction",
    "ws_speech": "WebSocketReplySpeechAction",
    "ws_options": "WebSocketReplyOptionsAction",
    "ws_location": "WebSocketReplyLocationAction",
    "ws_file": "WebSocketReplyFileAction",
    "ws_image": "WebSocketReplyImageAction",
    "ws_dataframe": "WebSocketReplyDataframeAction",
    "ws_plotly": "WebSocketReplyPlotlyAction",
    "gui_reply": "GUIReplyAction",
}


def _resolve_action_type(element: dict) -> str:
    """
    Return the normalized actionType string for an action element.
    Supports both new 'actionType' (metamodel class name) and old 'replyType' (backward compat).
    """
    action_type = element.get("actionType")
    if action_type:
        return action_type
    reply_type = element.get("replyType", "")
    return _REPLY_TYPE_TO_ACTION_TYPE.get(reply_type, "")


def _input_prompt_kwargs(element: dict) -> dict:
    """Return the ``input_prompt_mode`` / ``custom_input_prompt*`` kwargs of an action element."""
    return {
        "input_prompt_mode": sanitize_text(element.get("inputPromptMode")) or _DEFAULT_INPUT_PROMPT_MODE,
        "custom_input_prompt": element.get("customInputPrompt") or None,
        "custom_input_prompt_use_session_vars": bool(element.get("customInputPromptUseSessionVars", False)),
    }


def _build_body_from_action_elements(body_name, action_element_ids, elements,
                                     language, source_language, translate_text,
                                     build_db_reply_fn=None, gui_defs_by_id=None):
    """
    Build a Body object from an ordered list of action element IDs using per-element dispatch.

    Each element is classified by its 'actionType' field (new schema, metamodel class name)
    or its legacy 'replyType' field (backward compat). Actions are added to the body in the
    order they appear in action_element_ids, preserving execution order.

    Args:
        body_name: Name for the Body object.
        action_element_ids: Ordered list of action element IDs.
        elements: Dict of all diagram elements keyed by ID.
        language: Target translation language (or None).
        source_language: Source language for translation (or None).
        translate_text: Translation function.
        build_db_reply_fn: Optional callable to build a DBReply from an element dict.
        gui_defs_by_id: AgentGUI definitions (persist/width/is_form) keyed by gui id.

    Returns:
        A Body object with one action per element, or None if no actions were added.
    """
    if not action_element_ids:
        return None

    body = Body(body_name)
    action_added = False

    for element_id in action_element_ids:
        element = elements.get(element_id)
        if not element:
            continue

        action_type = _resolve_action_type(element)
        content = element.get("name", "")

        if action_type == "TextReplyAction":
            msg = sanitize_text(content)
            if language:
                msg = translate_text(msg, language, source_language)
            use_session_vars = bool(element.get("useSessionVars", False))
            body.add_action(AgentReply(message=msg, use_session_vars=use_session_vars))
            action_added = True

        elif action_type == "LLMReplyAction":
            # Prefer dedicated system_message field; fall back to legacy llmPrompt key.
            # Never use name — it is a display label ("LLM Reply"), not the system message.
            prompt_raw = element.get("system_message") or element.get("llmPrompt") or ""
            prompt = sanitize_text(prompt_raw) or None
            # Support "llmName" (new schema key) and "llm_name" (legacy key)
            llm_name_raw = element.get("llm_name") or element.get("llmName") or ""
            llm_name = sanitize_text(llm_name_raw) or None
            system_prompt_use_session_vars = bool(element.get("systemPromptUseSessionVars", False))
            store_in_session = sanitize_text(element.get("storeInSession", "")) or None
            send_reply = bool(element.get("sendReply", True))
            body.add_action(LLMReply(
                prompt=prompt,
                llm_name=llm_name,
                **_input_prompt_kwargs(element),
                system_prompt_use_session_vars=system_prompt_use_session_vars,
                store_in_session=store_in_session,
                send_reply=send_reply,
            ))
            action_added = True

        elif action_type == "LLMChatAction":
            # Keep the same payload keys as LLMReply for frontend symmetry.
            prompt_raw = element.get("system_message") or element.get("llmPrompt") or ""
            prompt = sanitize_text(prompt_raw) or None
            llm_name_raw = element.get("llm_name") or element.get("llmName") or ""
            llm_name = sanitize_text(llm_name_raw) or None
            system_prompt_use_session_vars = bool(element.get("systemPromptUseSessionVars", False))
            store_in_session = sanitize_text(element.get("storeInSession", "")) or None
            send_reply = bool(element.get("sendReply", True))
            body.add_action(LLMChatReply(
                prompt=prompt,
                llm_name=llm_name,
                system_prompt_use_session_vars=system_prompt_use_session_vars,
                store_in_session=store_in_session,
                send_reply=send_reply,
            ))
            action_added = True

        elif action_type == "RAGReplyAction":
            rag_name = sanitize_text(element.get("ragDatabaseName", ""))
            if not rag_name:
                rag_name = sanitize_text(content)
            rag_prompt_raw = element.get("prompt") or ""
            rag_prompt = sanitize_text(rag_prompt_raw) or None
            prompt_use_session_vars = bool(element.get("promptUseSessionVars", False))
            store_in_session = sanitize_text(element.get("storeInSession", "")) or None
            send_reply = bool(element.get("sendReply", True))
            if rag_name:
                body.add_action(RAGReply(
                    rag_db_name=rag_name,
                    prompt=rag_prompt,
                    **_input_prompt_kwargs(element),
                    prompt_use_session_vars=prompt_use_session_vars,
                    store_in_session=store_in_session,
                    send_reply=send_reply,
                ))
                action_added = True

        elif action_type == "DBAction":
            if build_db_reply_fn:
                body.add_action(build_db_reply_fn(element))
                action_added = True

        elif action_type == "WebCrawlLLMAction":
            initial_url = sanitize_text(element.get("initial_url", ""))
            max_depth_raw = element.get("max_depth", 2)
            max_pages_raw = element.get("max_pages", 20)
            try:
                max_depth = int(max_depth_raw) if max_depth_raw is not None else 2
            except (TypeError, ValueError):
                max_depth = 2
            try:
                max_pages = int(max_pages_raw) if max_pages_raw is not None else 20
            except (TypeError, ValueError):
                max_pages = 20
            crawl_format = sanitize_text(element.get("crawl_format", "markdown")) or "markdown"
            base_url_prefix_raw = element.get("base_url_prefix") or ""
            base_url_prefix = sanitize_text(base_url_prefix_raw) or None
            run_crawl_raw = element.get("run_crawl", True)
            run_crawl = bool(run_crawl_raw) if run_crawl_raw is not None else True
            no_crawl_error_message = (
                sanitize_text(element.get("no_crawl_error_message", "No web crawl data is available yet."))
                or "No web crawl data is available yet."
            )
            system_message_prefix_raw = element.get("system_message_prefix") or ""
            system_message_prefix = sanitize_text(system_message_prefix_raw) or None
            system_message_prefix_use_session_vars = bool(element.get("systemMessagePrefixUseSessionVars", False))
            llm_name_raw = element.get("llm_name") or ""
            llm_name = sanitize_text(llm_name_raw) or None
            store_in_session_raw = element.get("storeInSession") or ""
            store_in_session = sanitize_text(store_in_session_raw) or None
            send_reply = bool(element.get("sendReply", True))
            if initial_url:
                body.add_action(WebCrawlLLMReply(
                    initial_url=initial_url,
                    max_depth=max_depth,
                    max_pages=max_pages,
                    crawl_format=crawl_format,
                    base_url_prefix=base_url_prefix,
                    run_crawl=run_crawl,
                    no_crawl_error_message=no_crawl_error_message,
                    system_message_prefix=system_message_prefix,
                    llm_name=llm_name,
                    system_message_prefix_use_session_vars=system_message_prefix_use_session_vars,
                    store_in_session=store_in_session,
                    send_reply=send_reply,
                ))
                action_added = True

        elif action_type == "WebSocketReplyMarkdownAction":
            msg = sanitize_text(element.get("ws_message", ""))
            use_session_vars = bool(element.get("useSessionVars", False))
            body.add_action(WebSocketReplyMarkdown(message=msg, use_session_vars=use_session_vars))
            action_added = True

        elif action_type == "WebSocketReplyHTMLAction":
            msg = sanitize_text(element.get("ws_message", ""))
            use_session_vars = bool(element.get("useSessionVars", False))
            body.add_action(WebSocketReplyHTML(message=msg, use_session_vars=use_session_vars))
            action_added = True

        elif action_type == "WebSocketReplySpeechAction":
            msg = sanitize_text(element.get("ws_message", ""))
            speed_raw = element.get("ws_audio_speed")
            try:
                audio_speed = float(speed_raw) if speed_raw not in (None, "") else None
            except (TypeError, ValueError):
                audio_speed = None
            use_session_vars = bool(element.get("useSessionVars", False))
            body.add_action(WebSocketReplySpeech(
                message=msg, audio_speed=audio_speed, use_session_vars=use_session_vars,
            ))
            action_added = True

        elif action_type == "WebSocketReplyOptionsAction":
            opts_raw = element.get("ws_options", "")
            options = [o.strip() for o in opts_raw.split('\n') if o.strip()]
            body.add_action(WebSocketReplyOptions(options=options))
            action_added = True

        elif action_type == "WebSocketReplyLocationAction":
            try:
                lat = float(element.get("ws_latitude", 0.0))
            except (TypeError, ValueError):
                lat = 0.0
            try:
                lon = float(element.get("ws_longitude", 0.0))
            except (TypeError, ValueError):
                lon = 0.0
            body.add_action(WebSocketReplyLocation(latitude=lat, longitude=lon))
            action_added = True

        elif action_type == "WebSocketReplyFileAction":
            body.add_action(WebSocketReplyFile())
            action_added = True

        elif action_type == "WebSocketReplyImageAction":
            body.add_action(WebSocketReplyImage())
            action_added = True

        elif action_type == "WebSocketReplyDataframeAction":
            body.add_action(WebSocketReplyDataframe())
            action_added = True

        elif action_type == "WebSocketReplyPlotlyAction":
            body.add_action(WebSocketReplyPlotly())
            action_added = True

        elif action_type == "GUIReplyAction":
            gui_id = sanitize_text(element.get("guiId", "") or element.get("gui_id", "")) or None
            # width/persist/is_form live on the AgentGUI component definition, not on the
            # action element itself (the action only carries guiId). Look up the definition
            # and fall back to element-level values for backward compatibility. A guiId
            # without AgentGUI definition is reported by Agent.validate().
            gui_def = (gui_defs_by_id or {}).get(gui_id, {}) if gui_id else {}
            persist = gui_def.get("persist", bool(element.get("persist", True)))
            width = gui_def.get("width") or sanitize_text(element.get("width", "")) or None
            is_form = gui_def.get("is_form", bool(element.get("is_form", False)))
            if gui_id:
                body.add_action(GUIReplyAction(
                    gui_id=gui_id,
                    persist=persist,
                    width=width,
                    is_form=is_form,
                ))
                action_added = True

        elif action_type == "CustomCodeAction":
            # Raw source code is kept verbatim (no unicode normalization or control-char
            # stripping). Structural and semantic validation is enforced by
            # validate_custom_code_action().
            validate_custom_code_action(content)
            body.add_action(CustomCodeAction(source=content))
            action_added = True

        else:
            logger.warning("Unknown actionType '%s' on element '%s'; skipping.", action_type, element_id)

    return body if action_added else None


# ---------------------------------------------------------------------------
# v4 input helpers
# ---------------------------------------------------------------------------

# The React Flow frontend creates ``comment`` nodes tethered by
# ``CommentLink`` edges; ``Comments`` / ``Link`` are the legacy spellings.
COMMENT_NODE_TYPES = ("comment", "Comments")
COMMENT_LINK_TYPES = ("CommentLink", "Link")

# Off-canvas agent component types (``model.components``). Nodes of these
# types are a legacy input only (pre-Components-page React Flow models).
AGENT_COMPONENT_TYPES = frozenset({
    "AgentIntent", "AgentIntentBody", "AgentRagElement", "AgentTool",
    "AgentSkill", "AgentWorkspace", "AgentLLM", "AgentGUI",
})

# Placeholder label older v4 writers put on the ``name`` of an LLM row.
_LLM_ROW_PLACEHOLDER = "AI response 🪄"


def _normalise_body_row(row: dict) -> dict:
    """Lift legacy v4 row conventions onto the canonical action-row fields.

    * Older v4 LLM / LLM-chat rows carried the system prompt on ``name``;
      the canonical field is ``system_message``.
    * v4 code rows carry the source on ``code`` (``name`` may be a label);
      the dispatcher reads the source from ``name``.
    """
    out = dict(row)
    action_type = _resolve_action_type(out)
    if action_type in ("LLMReplyAction", "LLMChatAction") and "system_message" not in out:
        legacy_prompt = out.get("name") or ""
        if legacy_prompt and legacy_prompt != _LLM_ROW_PLACEHOLDER:
            out["system_message"] = legacy_prompt
    elif action_type == "CustomCodeAction":
        code_source = out.get("code")
        if isinstance(code_source, str) and code_source:
            out["name"] = code_source
    return out


def _build_body_from_action_rows(body_name, rows, language, source_language, translate_text,
                                 build_db_reply_fn=None, gui_defs_by_id=None):
    """Build a Body from the v4 inline action rows of an AgentState (order preserved)."""
    if not isinstance(rows, list) or not rows:
        return None
    index: dict = {}
    ids: list = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        row_id = row.get("id") or f"{body_name}_{position}"
        if row_id in index:
            row_id = f"{row_id}_{position}"
        index[row_id] = _normalise_body_row(row)
        ids.append(row_id)
    return _build_body_from_action_elements(
        body_name, ids, index, language, source_language, translate_text,
        build_db_reply_fn=build_db_reply_fn, gui_defs_by_id=gui_defs_by_id,
    )


def _node_to_components(node: dict) -> list:
    """Flatten a legacy v4 component node into ``model.components`` entries."""
    data = dict(node_data(node))
    node_id = node.get("id")
    node_type = node.get("type")
    # node.data never carries geometry, but drop stray canvas keys (``width`` is a GUI field).
    for key in ("bounds", "position", "measured", "parentId"):
        data.pop(key, None)
    if node_type == "AgentIntent":
        rows = data.pop("training_phrases", None)
        if not isinstance(rows, list) or not rows:
            rows = data.get("bodies") or []
        entries = []
        body_ids = []
        for position, row in enumerate(rows):
            if isinstance(row, str):
                body_ids.append(row)
                continue
            if not isinstance(row, dict):
                continue
            body_id = row.get("id") or f"{node_id}-body-{position}"
            body_ids.append(body_id)
            entries.append({"id": body_id, "type": "AgentIntentBody", "name": row.get("name", ""),
                            "owner": node_id})
        intent = {**data, "id": node_id, "type": "AgentIntent", "name": data.get("name", ""),
                  "owner": None, "bodies": body_ids}
        return [intent] + entries
    if node_type == "AgentIntentBody":
        return [{**data, "id": node_id, "type": "AgentIntentBody", "name": data.get("name", ""),
                 "owner": node.get("parentId") or data.get("owner")}]
    return [{**data, "id": node_id, "type": node_type, "name": data.get("name", ""), "owner": None}]


def collect_agent_components(json_data: dict) -> dict:
    """Merge every location agent components may be stored in, keyed by id.

    Precedence (later wins): legacy component nodes in ``model.nodes`` <
    diagram-level ``agentComponents`` < ``model.agentComponents`` <
    ``model.components`` (canonical). v3-style component elements in
    ``model.elements`` are accepted as the lowest-precedence source.
    """
    model_data = json_data.get("model") or {}
    merged: dict = {}
    legacy_elements = model_data.get("elements")
    if isinstance(legacy_elements, dict):
        for element_id, element in legacy_elements.items():
            if isinstance(element, dict) and element.get("type") in AGENT_COMPONENT_TYPES:
                merged[element_id] = {**element, "id": element.get("id") or element_id}
    for node in model_data.get("nodes") or []:
        if isinstance(node, dict) and node.get("type") in AGENT_COMPONENT_TYPES:
            for entry in _node_to_components(node):
                merged[entry["id"]] = entry
    for source in (json_data.get("agentComponents"), model_data.get("agentComponents"),
                   model_data.get("components")):
        if isinstance(source, dict):
            for component_id, component in source.items():
                if isinstance(component, dict):
                    merged[component_id] = {**component, "id": component.get("id") or component_id}
    return merged


def _normalise_agent_transitions(edges: list[dict]) -> list[dict]:
    """Collapse legacy AgentStateTransition shapes to the canonical v4 form.

    The v4 canonical shape is on ``edge.data``:
        transitionType: 'predefined' | 'custom'
        predefined: { predefinedType, intentName?, fileType?, formGuiId?, conditionValue? }
        custom: { event, condition: string[], guiEventGuiId? }

    Legacy shapes are accepted (see uml-v4-shape.md). Fallthrough order:
      1. transitionType=='custom' OR legacy condition=='custom_transition'
         OR custom.event/condition non-empty -> emit canonical custom block.
      2. Otherwise emit canonical predefined block.

    Returns a NEW list of edges; input is not mutated.
    """
    out: list[dict] = []
    for edge in edges:
        if edge.get("type") != "AgentStateTransition":
            out.append(edge)
            continue
        nedge = dict(edge)
        ndata = dict(edge.get("data") or {})
        predefined = ndata.get("predefined") or {}
        custom = ndata.get("custom") or {}

        is_custom = (
            ndata.get("transitionType") == "custom"
            or ndata.get("condition") == "custom_transition"
            or (isinstance(custom.get("event"), str) and custom.get("event"))
            or (isinstance(custom.get("condition"), list) and any(
                isinstance(c, str) and c.strip() for c in custom["condition"]
            ))
            or (isinstance(ndata.get("conditionValue"), dict) and (
                ndata["conditionValue"].get("events") or ndata["conditionValue"].get("conditions")
            ))
        )
        if ndata.get("transitionType") == "predefined" and not ndata.get("condition") == "custom_transition":
            is_custom = False

        if is_custom:
            event = (
                custom.get("event")
                or ndata.get("event")
                or ndata.get("customEvent")
                or "None"
            )
            cond = custom.get("condition")
            if not isinstance(cond, list):
                cond = ndata.get("customConditions")
            if not isinstance(cond, list):
                cv = ndata.get("conditionValue")
                if isinstance(cv, dict):
                    events = cv.get("events") or []
                    if isinstance(events, list) and events and not custom.get("event"):
                        event = events[0]
                    cond = cv.get("conditions") or []
            if not isinstance(cond, list):
                cond = []
            block: dict = {"event": event, "condition": cond}
            gui_event_gui_id = custom.get("guiEventGuiId") or ndata.get("guiEventGuiId")
            if gui_event_gui_id:
                block["guiEventGuiId"] = gui_event_gui_id
            ndata["transitionType"] = "custom"
            ndata["custom"] = block
            ndata.pop("predefined", None)
        else:
            predefined_type = (
                predefined.get("predefinedType")
                or ndata.get("predefinedType")
                or (ndata.get("condition") if isinstance(ndata.get("condition"), str) else None)
                or "when_intent_matched"
            )
            block = {"predefinedType": predefined_type}
            intent_name = predefined.get("intentName") or ndata.get("intentName")
            if intent_name is not None:
                block["intentName"] = intent_name
            file_type = predefined.get("fileType") or ndata.get("fileType")
            if file_type is not None:
                block["fileType"] = file_type
            form_gui_id = predefined.get("formGuiId") or ndata.get("formGuiId")
            if form_gui_id:
                block["formGuiId"] = form_gui_id
            cv = predefined.get("conditionValue")
            if cv is None:
                if ndata.get("variable") is not None or ndata.get("operator") is not None:
                    cv = {
                        "variable": ndata.get("variable", ""),
                        "operator": ndata.get("operator", ""),
                        "targetValue": ndata.get("targetValue", ""),
                    }
                else:
                    cv = ndata.get("conditionValue")
            if cv is not None:
                block["conditionValue"] = cv
            ndata["transitionType"] = "predefined"
            ndata["predefined"] = block
            ndata.pop("custom", None)
        nedge["data"] = ndata
        out.append(nedge)
    return out


_FILE_MIME_MAP = {
    "pdf": "application/pdf",
    "txt": "text/plain",
    "json": "application/json",
    "csv": "text/csv",
    "xml": "text/xml",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "mp3": "audio/mpeg",
    "mp4": "video/mp4",
}


def process_agent_diagram(json_data):
    """Process an Agent Diagram in the v4 wire shape and return an Agent.

    Canvas: ``model.nodes`` (``AgentState`` with inline ``data.bodies`` /
    ``data.fallbackBodies`` action rows, ``comment``) and ``model.edges``
    (``AgentStateTransition``, ``CommentLink``). Off-canvas components come
    from ``model.components`` and the legacy locations merged by
    :func:`collect_agent_components`.
    """
    config = json_data.get('config') or {}
    lang_value = config.get('language', '')
    language = lang_value.lower() if isinstance(lang_value, str) and lang_value else None
    source_language = config.get('source_language')

    def translate_text(text, lang, src_lang=None):
        # Use deep-translator's GoogleTranslator for free translation
        if not lang or lang == 'none':
            return text
        lang_map = {
            'none': 'auto',
            'english': 'en',
            'french': 'fr',
            'german': 'de',
            'spanish': 'es',
            'luxembourgish': 'lb',
            'portuguese': 'pt',
        }
        target_lang = lang_map.get(lang.lower()) if isinstance(lang, str) else None
        if not target_lang:
            return text
        src_code = lang_map.get(src_lang.lower()) if src_lang and isinstance(src_lang, str) else 'auto'
        try:
            return GoogleTranslator(source=src_code, target=target_lang).translate(text)
        except Exception as e:
            logger.error("Translation error: %s", e)
            return text

    def build_db_reply(element: dict) -> DBReply:
        return DBReply(
            db_selection_type=sanitize_text(element.get("dbSelectionType", "default")) or "default",
            db_custom_name=sanitize_text(element.get("dbCustomName", "")) or None,
            db_query_mode=sanitize_text(element.get("dbQueryMode", "llm_query")) or "llm_query",
            db_operation=sanitize_text(element.get("dbOperation", "any")) or "any",
            db_sql_query=element.get("dbSqlQuery") or None,
            llm_name=sanitize_text(element.get("llm_name", "")) or None,
            **_input_prompt_kwargs(element),
            store_in_session=sanitize_text(element.get("storeInSession", "")) or None,
            send_reply=bool(element.get("sendReply", True)),
        )

    title = json_data.get('title', 'Generated_Agent') or 'Generated_Agent'
    if ' ' in title:
        title = title.replace(' ', '_')

    agent = Agent(title)

    model_data = json_data.get('model') or {}
    nodes = model_data.get('nodes') or []
    edges = model_data.get('edges') or []
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    edges = _normalise_agent_transitions(edges)
    nodes_by_id = {n.get("id"): n for n in nodes if isinstance(n, dict) and n.get("id")}
    components = collect_agent_components(json_data)

    states_by_id = {}
    intents_by_id = {}
    rag_dbs_by_id = {}
    rag_dbs_by_name = {}
    gui_defs_by_id = {}  # gui_id -> {persist, width, is_form} of its AgentGUI component
    comment_nodes = {}
    comment_links = {}

    for node in nodes:
        if isinstance(node, dict) and node.get("type") in COMMENT_NODE_TYPES:
            comment_nodes[node.get("id")] = node_data(node).get("name", "")

    # Components: LLMs, tools, skills, workspaces, intents, GUIs, RAG databases.
    for element_id, element in components.items():
        element_type = element.get("type")
        if element_type == "AgentLLM":
            llm_name = sanitize_text((element.get("name") or "").strip())
            if not llm_name or any(existing.name == llm_name for existing in agent.llms):
                continue
            provider = (element.get("provider") or "openai").lower()
            llm_parameters = element.get("parameters")
            if not isinstance(llm_parameters, dict):
                llm_parameters = {}
            num_prev = element.get("num_previous_messages")
            try:
                num_prev_int = int(num_prev) if num_prev is not None else 1
            except (TypeError, ValueError):
                num_prev_int = 1
            agent.new_llm(
                name=llm_name,
                provider=provider,
                parameters=llm_parameters,
                num_previous_messages=num_prev_int,
                global_context=element.get("global_context") or None,
            )
        elif element_type == "AgentTool":
            tool_name = sanitize_text((element.get("name") or "").strip())
            if not tool_name or any(t.name == tool_name for t in agent.tools):
                continue
            agent.new_tool(
                name=tool_name,
                description=element.get("description", "") or "",
                code=element.get("code", "") or "",
            )
        elif element_type == "AgentSkill":
            skill_name = sanitize_text((element.get("name") or "").strip())
            if not skill_name or any(s.name == skill_name for s in agent.skills):
                continue
            agent.new_skill(
                name=skill_name,
                content=element.get("content", "") or "",
                description=element.get("description") or None,
            )
        elif element_type == "AgentWorkspace":
            ws_name = sanitize_text((element.get("name") or "").strip())
            if not ws_name or any(w.name == ws_name for w in agent.workspaces):
                continue
            writable = element.get("writable")
            if writable is None:
                writable = True
            max_read_bytes = element.get("max_read_bytes")
            if max_read_bytes is None:
                max_read_bytes = 200_000
            agent.new_workspace(
                name=ws_name,
                path=element.get("path", "") or "",
                description=element.get("description") or None,
                writable=bool(writable),
                max_read_bytes=int(max_read_bytes),
            )
        elif element_type == "AgentIntent":
            intent_name = element.get("name")
            training_sentences = []
            # Training sentences: ids of AgentIntentBody components under "bodies"
            # (canonical) or "ownedElements" (legacy panel builds).
            body_id_list = element.get("bodies") or element.get("ownedElements") or []
            for body_ref in body_id_list:
                body_element = components.get(body_ref) if isinstance(body_ref, str) else body_ref
                if isinstance(body_element, dict):
                    training_sentence = sanitize_text(body_element.get("name", ""))
                    if language:
                        training_sentence = translate_text(training_sentence, language, source_language)
                    if training_sentence:
                        training_sentences.append(training_sentence)
            intent = Intent(intent_name, training_sentences, description=element.get("intent_description", None))
            agent.add_intent(intent)
            intents_by_id[element_id] = intent
        elif element_type == "AgentGUI":
            resolved_gui_id = sanitize_text((element.get("gui_id") or "").strip())
            if not resolved_gui_id:
                continue
            gui_defs_by_id[resolved_gui_id] = {
                "persist": bool(element.get("persist", True)),
                "width": sanitize_text(element.get("width", "") or "") or None,
                "is_form": bool(element.get("is_form", False)),
            }
            # The GrapesJS design becomes a B-UML GUIModel. Agent GUIs are not bound to a
            # class diagram, hence the empty domain model. A GUI that has not been designed
            # yet (guiModel null) becomes an empty GUIModel so replies referencing it resolve.
            from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.gui_diagram_processor import (  # noqa: E501 — lazy: avoids an import cycle
                process_gui_diagram,
            )
            agent.add_gui_model(
                resolved_gui_id,
                process_gui_diagram(element.get("guiModel") or {}, None, DomainModel("AgentGUIDomain")),
            )
        elif element_type == "AgentRagElement":
            rag_name = sanitize_text((element.get("name") or "").strip())
            if not rag_name:
                continue
            if rag_name in rag_dbs_by_name:
                rag_dbs_by_id[element_id] = rag_dbs_by_name[rag_name]
                continue
            sanitized_slug = rag_name.lower().replace(' ', '_') or "default"
            embedding_provider = (element.get("embedding_provider") or "openai").lower()
            if embedding_provider == "ollama":
                embedding_parameters = {
                    "base_url": element.get("embedding_base_url") or "http://localhost:11434",
                    "model": element.get("embedding_model") or "nomic-embed-text",
                }
            else:
                embedding_parameters = {"api_key_property": "nlp.OPENAI_API_KEY"}
            vector_store = RAGVectorStore(
                embedding_provider=embedding_provider,
                embedding_parameters=embedding_parameters,
                persist_directory=f"vector_store/{sanitized_slug}",
            )
            splitter = RAGTextSplitter(
                splitter_type="recursive_character",
                chunk_size=1000,
                chunk_overlap=100,
            )
            # The LLM is referenced by name; an empty value means "resolve the
            # agent default at codegen time".
            rag_llm_name = sanitize_text((element.get("llm_name") or element.get("llm") or "").strip()) or ""
            rag_llm_prompt_raw = element.get("llm_prompt") or element.get("llmPrompt") or ""
            rag_llm_prompt = sanitize_text(rag_llm_prompt_raw.strip()) or None
            raw_k = element.get("k")
            try:
                rag_k = int(raw_k) if raw_k is not None else 4
            except (TypeError, ValueError):
                rag_k = 4
            if rag_k <= 0:
                rag_k = 4
            raw_npm = element.get("num_previous_messages")
            if raw_npm is None:
                raw_npm = element.get("numPreviousMessages")
            try:
                rag_num_previous_messages = int(raw_npm) if raw_npm is not None else 0
            except (TypeError, ValueError):
                rag_num_previous_messages = 0
            if rag_num_previous_messages < 0:
                rag_num_previous_messages = 0
            rag_use_hybrid = bool(element.get("use_hybrid_rag", False))
            raw_bm25 = element.get("bm25_weight")
            try:
                rag_bm25_weight = float(raw_bm25) if raw_bm25 is not None else 0.6
            except (TypeError, ValueError):
                rag_bm25_weight = 0.6
            if not (0 < rag_bm25_weight < 1):
                rag_bm25_weight = 0.6
            rag_config = agent.new_rag(
                name=rag_name,
                vector_store=vector_store,
                splitter=splitter,
                llm_name=rag_llm_name,
                llm_prompt=rag_llm_prompt,
                k=rag_k,
                num_previous_messages=rag_num_previous_messages,
                use_hybrid_rag=rag_use_hybrid,
                bm25_weight=rag_bm25_weight,
            )
            rag_dbs_by_id[element_id] = rag_config
            rag_dbs_by_name[rag_name] = rag_config

    def _is_state_node(node: dict) -> bool:
        return isinstance(node, dict) and node.get("type") in ("AgentState", "AgentReasoningState")

    def _is_reasoning_node(node: dict, data: dict) -> bool:
        """Reasoning states: ``AgentState`` + ``data.stateType == 'reasoning'`` (canonical)
        or the legacy ``AgentReasoningState`` node type."""
        if node.get("type") == "AgentReasoningState":
            return True
        return node.get("type") == "AgentState" and data.get("stateType") == "reasoning"

    # Initial state: the ``data.initial`` flag (canonical v4); fall back to the
    # legacy ``StateInitialNode`` marker + init edge.
    initial_state_id = None
    for node in nodes:
        if _is_state_node(node) and node_data(node).get("initial") is True:
            initial_state_id = node.get("id")
            break
    if initial_state_id is None:
        for edge in edges:
            if edge.get("type") not in ("AgentStateTransition", "AgentStateTransitionInit"):
                continue
            source_node = nodes_by_id.get(edge.get("source")) or {}
            target_node = nodes_by_id.get(edge.get("target")) or {}
            if source_node.get("type") == "StateInitialNode" and _is_state_node(target_node):
                initial_state_id = target_node.get("id")
                break

    def _build_reasoning_state(node_id: str, data: dict, is_initial: bool):
        state_name = data.get("name", "") or ""
        llm_name = data.get("llm_name") or data.get("llm")
        llm_value = llm_name.strip() if isinstance(llm_name, str) and llm_name.strip() else None
        kwargs = {"name": state_name, "llm": llm_value, "initial": is_initial}
        if data.get("max_steps") is not None:
            kwargs["max_steps"] = int(data.get("max_steps"))
        if data.get("enable_task_planning") is not None:
            kwargs["enable_task_planning"] = bool(data.get("enable_task_planning"))
        if data.get("stream_steps") is not None:
            kwargs["stream_steps"] = bool(data.get("stream_steps"))
        if data.get("system_prompt") is not None:
            kwargs["system_prompt"] = data.get("system_prompt")
        if data.get("fallback_message") is not None:
            kwargs["fallback_message"] = data.get("fallback_message")
        # Tools, skills and workspaces are registered at the agent level and
        # shared by every reasoning state.
        rs = agent.new_reasoning_state(**kwargs)
        states_by_id[node_id] = rs
        return rs

    def _build_agent_state(node_id: str, data: dict, is_initial: bool):
        state_name = data.get("name", "") or ""
        agent_state = agent.new_state(name=state_name, initial=is_initial)
        states_by_id[node_id] = agent_state
        body = _build_body_from_action_rows(
            f"{state_name}_body", data.get("actions", data.get("bodies")),
            language, source_language, translate_text,
            build_db_reply_fn=build_db_reply, gui_defs_by_id=gui_defs_by_id,
        )
        if body:
            agent_state.set_body(body)
        # Only attach a fallback body if fallbackBodyEnabled is absent (legacy) or True.
        if data.get("fallbackBodyEnabled", True):
            fallback_body = _build_body_from_action_rows(
                f"{state_name}_fallback_body", data.get("fallbackActions", data.get("fallbackBodies")),
                language, source_language, translate_text,
                build_db_reply_fn=build_db_reply, gui_defs_by_id=gui_defs_by_id,
            )
            if fallback_body:
                agent_state.set_fallback_body(fallback_body)
        return agent_state

    def _build_state(node: dict, is_initial: bool):
        data = node_data(node)
        if _is_reasoning_node(node, data):
            _build_reasoning_state(node.get("id"), data, is_initial)
        else:
            _build_agent_state(node.get("id"), data, is_initial)

    if initial_state_id:
        _build_state(nodes_by_id[initial_state_id], is_initial=True)
    for node in nodes:
        if _is_state_node(node) and node.get("id") != initial_state_id:
            _build_state(node, is_initial=False)

    # Intent names are unique case-insensitively in BUML, so accept both exact and
    # casefold matches (personalization variants may change the casing).
    intent_lookup = {intent.name: intent for intent in agent.intents}
    intent_lookup_casefold = {
        intent.name.casefold(): intent
        for intent in agent.intents
        if isinstance(intent.name, str)
    }

    transition_count = 0
    for edge in edges:
        edge_type = edge.get("type")
        if edge_type in COMMENT_LINK_TYPES:
            source_id = edge.get("source")
            target_id = edge.get("target")
            comment_id = None
            target = None
            if source_id in comment_nodes:
                comment_id, target = source_id, target_id
            elif target_id in comment_nodes:
                comment_id, target = target_id, source_id
            if comment_id and target:
                comment_links.setdefault(comment_id, []).append(target)
            continue
        if edge_type not in ("AgentStateTransition", "AgentStateTransitionInit"):
            continue

        source_id = edge.get("source")
        target_id = edge.get("target")
        if (nodes_by_id.get(source_id) or {}).get("type") == "StateInitialNode":
            continue
        source_state = states_by_id.get(source_id)
        target_state = states_by_id.get(target_id)
        if not source_state or not target_state:
            logger.warning(
                "Skipping agent transition: source '%s' or target '%s' state not found.",
                source_id, target_id,
            )
            continue

        edge_data = edge.get("data") or {}
        predefined_block = edge_data.get("predefined") or {}
        custom_block = edge_data.get("custom") or {}

        if edge_data.get("transitionType") == "custom":
            selected_event = custom_block.get("event")
            custom_conditions = custom_block.get("condition")
            if not isinstance(custom_conditions, list):
                custom_conditions = []
            normalized_event = "None"
            if isinstance(selected_event, str) and selected_event and selected_event != "None":
                normalized_event = selected_event
            condition_name = "custom_transition"
            transition_payload = {"event": normalized_event, "conditions": custom_conditions}
            if custom_block.get("guiEventGuiId"):
                transition_payload["guiEventGuiId"] = custom_block.get("guiEventGuiId")
        else:
            condition_name = predefined_block.get("predefinedType") or ""
            if condition_name == "when_intent_matched":
                transition_payload = predefined_block.get("intentName")
            elif condition_name == "when_file_received":
                transition_payload = predefined_block.get("fileType")
            elif condition_name == "when_form_submitted":
                transition_payload = predefined_block.get("formGuiId") or ""
            else:
                transition_payload = predefined_block.get("conditionValue")
            if transition_payload is None:
                transition_payload = ""

        if condition_name == "when_intent_matched":
            intent_to_match = intent_lookup.get(transition_payload)
            if intent_to_match is None and isinstance(transition_payload, str):
                intent_to_match = intent_lookup_casefold.get(transition_payload.casefold())
            if intent_to_match:
                source_state.when_intent_matched(intent_to_match).go_to(target_state)
                transition_count += 1
            elif isinstance(transition_payload, str) and transition_payload.strip():
                unresolved_intent = Intent(transition_payload.strip())
                TransitionBuilder(
                    source=source_state,
                    event=ReceiveTextEvent(),
                    conditions=[IntentMatcher(unresolved_intent)],
                ).go_to(target_state)
                transition_count += 1

        elif condition_name == "when_no_intent_matched":
            source_state.when_no_intent_matched().go_to(target_state)
            transition_count += 1

        elif condition_name == "when_variable_operation_matched":
            if isinstance(transition_payload, dict):
                variable_name = transition_payload.get("variable")
                operator_value = transition_payload.get("operator")
                target_value = transition_payload.get("targetValue")
                if not variable_name or not operator_value:
                    logger.warning(
                        "Incomplete variable operation condition (variable=%s, operator=%s) "
                        "for transition from '%s' to '%s'. Falling back to no_intent_matched.",
                        variable_name, operator_value, source_state.name, target_state.name,
                    )
                    source_state.when_no_intent_matched().go_to(target_state)
                    transition_count += 1
                else:
                    operator_map = {
                        "<": operator.lt,
                        "<=": operator.le,
                        "==": operator.eq,
                        ">=": operator.ge,
                        ">": operator.gt,
                        "!=": operator.ne,
                    }
                    op_func = operator_map.get(operator_value)
                    if op_func:
                        source_state.when_variable_matches_operation(
                            var_name=variable_name,
                            operation=op_func,
                            target=target_value,
                        ).go_to(target_state)
                        transition_count += 1
                    else:
                        logger.warning(
                            "Unknown operator '%s' for variable operation transition from '%s' to '%s'. Skipping.",
                            operator_value, source_state.name, target_state.name,
                        )
            else:
                logger.warning(
                    "Expected dict for when_variable_operation_matched condition but got %s. "
                    "Falling back to no_intent_matched for transition from '%s' to '%s'.",
                    type(transition_payload).__name__, source_state.name, target_state.name,
                )
                source_state.when_no_intent_matched().go_to(target_state)
                transition_count += 1

        elif condition_name == "when_form_submitted":
            form_id = transition_payload if isinstance(transition_payload, str) and transition_payload else None
            source_state.when_form_submitted(form_id=form_id).go_to(target_state)
            transition_count += 1

        elif condition_name == "when_file_received":
            # Comma-separated list of MIME types or short extensions (pdf, csv, ...).
            if isinstance(transition_payload, str) and transition_payload.strip():
                tokens = [t.strip() for t in transition_payload.split(",") if t.strip()]
                resolved = [
                    token if "/" in token else _FILE_MIME_MAP.get(token.lower(), token)
                    for token in tokens
                ]
                if len(resolved) == 1:
                    source_state.when_file_received(resolved[0]).go_to(target_state)
                else:
                    source_state.when_file_received(resolved).go_to(target_state)
            else:
                source_state.when_file_received().go_to(target_state)
            transition_count += 1

        elif condition_name == "auto":
            source_state.go_to(target_state)
            transition_count += 1

        elif condition_name == "custom_transition":
            event_instance = None
            custom_conditions = []
            if isinstance(transition_payload, dict):
                selected_event = transition_payload.get("event")
                if selected_event == "ReceiveTextEvent":
                    event_instance = ReceiveTextEvent()
                elif selected_event == "ReceiveMessageEvent":
                    event_instance = ReceiveMessageEvent("")
                elif selected_event == "ReceiveJSONEvent":
                    event_instance = ReceiveJSONEvent()
                elif selected_event == "ReceiveFileEvent":
                    event_instance = ReceiveFileEvent()
                elif selected_event == "DummyEvent":
                    event_instance = DummyEvent()
                elif selected_event == "WildcardEvent":
                    event_instance = WildcardEvent()
                elif selected_event == "GUIEvent":
                    message_id = transition_payload.get("guiEventGuiId") or None
                    event_instance = GUIEvent(message_id=message_id)
                raw_conditions = transition_payload.get("conditions") or []
                if isinstance(raw_conditions, list):
                    custom_conditions = [c for c in raw_conditions if isinstance(c, str) and c.strip()]

            condition_objects = []
            for condition_index, custom_condition_code in enumerate(custom_conditions, start=1):
                generated_name = f"condition_{transition_count + 1}_{condition_index}"
                custom_condition = Condition(name=generated_name, callable=None)
                custom_condition.code = custom_condition_code
                condition_objects.append(custom_condition)

            transition_builder = None
            if event_instance is not None:
                transition_builder = source_state.when_event(event_instance)
            if condition_objects:
                if transition_builder is None:
                    transition_builder = source_state.when_condition(condition_objects[0])
                    for extra_condition in condition_objects[1:]:
                        transition_builder.with_condition(extra_condition)
                else:
                    for custom_condition in condition_objects:
                        transition_builder.with_condition(custom_condition)
            if transition_builder is not None:
                transition_builder.go_to(target_state)
            else:
                source_state.go_to(target_state)
            transition_count += 1

        else:
            source_state.when_no_intent_matched().go_to(target_state)
            transition_count += 1

    for comment_id, comment_text in comment_nodes.items():
        if comment_id in comment_links:
            for linked_id in comment_links[comment_id]:
                if linked_id in states_by_id:
                    state = states_by_id[linked_id]
                    if state.metadata is None:
                        state.metadata = Metadata(description=comment_text)
                    else:
                        existing_desc = state.metadata.description or ""
                        state.metadata.description = (
                            f"{existing_desc}\n{comment_text}" if existing_desc else comment_text
                        )
        else:
            if agent.metadata is None:
                agent.metadata = Metadata(description=comment_text)
            else:
                existing_desc = agent.metadata.description or ""
                agent.metadata.description = f"{existing_desc}\n{comment_text}" if existing_desc else comment_text

    # Apply the default LLM from the diagram config block (if set); without an
    # explicit pointer the agent already auto-defaulted to the first one registered.
    default_llm_name_cfg = (config or {}).get("default_llm_name")
    if isinstance(default_llm_name_cfg, str) and default_llm_name_cfg.strip():
        if any(existing.name == default_llm_name_cfg for existing in agent.llms):
            agent.set_default_llm(default_llm_name_cfg)

    # Validate the agent model at build time so all callers get validation for free.
    agent.validate(raise_exception=True)
    return agent
