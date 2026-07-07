"""
Agent diagram processing for converting v4 JSON to BUML format.

Reads the v4 wire shape (``{nodes, edges}``) natively. The
``_normalise_agent_transitions`` helper collapses the 5 historical
``AgentStateTransition`` shapes (see ``docs/source/migrations/uml-v4-shape.md``
"Legacy AgentStateTransition shapes") to the canonical
``transitionType + predefined|custom`` form that the rest of the processor
expects. This is *not* a v3 conversion — it normalises legacy v4 transition
shapes that survived from earlier iterations of the editor.
"""

import logging
import operator
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)
import json as json_lib
from besser.BUML.metamodel.state_machine.state_machine import (
    Body,
    Condition,
    ConfigProperty,
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
)
from besser.BUML.metamodel.structural import Metadata
from besser.utilities.web_modeling_editor.backend.services.converters.parsers import sanitize_text
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml._node_helpers import (
    node_data,
)

# The React Flow frontend creates ``comment`` nodes tethered by
# ``CommentLink`` edges; ``Comments`` / ``Link`` are the legacy spellings.
COMMENT_NODE_TYPES = ("comment", "Comments")
COMMENT_LINK_TYPES = ("CommentLink", "Link")


def _collect_body_messages(body_rows, language, source_language, translate_text,
                           serialize_db_reply_payload=None):
    """Collect and classify body messages from a v4 ``data.bodies`` row list.

    ``body_rows`` is the list of dicts each carrying ``replyType``, ``name``,
    and the various optional fields (``ragDatabaseName``, ``dbSelectionType``,
    ...). v4 collapses ``AgentStateBody`` children onto the parent state's
    ``data.bodies``, so we walk that list directly.
    """
    messages = []
    for body in body_rows or []:
        if not isinstance(body, dict):
            continue
        reply_type = body.get("replyType")
        body_content = body.get("name", "")

        if reply_type == "text":
            msg = sanitize_text(body_content)
            if language:
                msg = translate_text(msg, language, source_language)
            messages.append(msg)
        elif reply_type == "llm":
            # The system prompt lives on ``body.name`` (the inspector's
            # "System prompt" textarea); ``llm_name`` is an optional
            # passthrough field. Serialize both so the body builder can
            # hand them to ``LLMReply(prompt=..., llm_name=...)``.
            llm_payload = {
                "prompt": sanitize_text(body_content),
                "llm_name": sanitize_text(body.get("llm_name", "") or ""),
            }
            messages.append(f"LLM:{json_lib.dumps(llm_payload)}")
        elif reply_type == "rag":
            rag_name = sanitize_text(body.get("ragDatabaseName", ""))
            if not rag_name:
                rag_name = sanitize_text(body_content)
            if rag_name:
                rag_prompt = sanitize_text(body.get("prompt", "") or "") or None
                messages.append(f"RAG:{json_lib.dumps({'rag_db_name': rag_name, 'prompt': rag_prompt})}")
        elif reply_type == "db_reply":
            if serialize_db_reply_payload:
                messages.append(serialize_db_reply_payload(body))
        elif reply_type == "code":
            # v4 body rows carry the actual source on ``body.code``;
            # ``body.name`` is the display label only. Prefer ``code``
            # so JSON->BUML preserves the user's typed code body.
            code_source = body.get("code")
            if not isinstance(code_source, str) or not code_source:
                code_source = body_content
            messages.append(f"CODE:{sanitize_text(code_source)}")
        elif reply_type == "llm_chat":
            # Mirror the ``llm`` convention: the system prompt rides on
            # ``body.name``; ``llm_name`` is an optional passthrough.
            llm_chat_payload = {
                "prompt": sanitize_text(body_content),
                "llm_name": sanitize_text(body.get("llm_name", "") or ""),
            }
            messages.append(f"LLMCHAT:{json_lib.dumps(llm_chat_payload)}")
        elif reply_type == "web_crawl_llm":
            initial_url = sanitize_text(body.get("initial_url", "") or "")
            if initial_url:
                max_depth_raw = body.get("max_depth", 2)
                max_pages_raw = body.get("max_pages", 20)
                try:
                    max_depth = int(max_depth_raw) if max_depth_raw is not None else 2
                except (TypeError, ValueError):
                    max_depth = 2
                try:
                    max_pages = int(max_pages_raw) if max_pages_raw is not None else 20
                except (TypeError, ValueError):
                    max_pages = 20
                run_crawl_raw = body.get("run_crawl", True)
                web_crawl_payload = {
                    "initial_url": initial_url,
                    "max_depth": max_depth,
                    "max_pages": max_pages,
                    "crawl_format": sanitize_text(body.get("crawl_format", "markdown") or "markdown") or "markdown",
                    "base_url_prefix": sanitize_text(body.get("base_url_prefix", "") or "") or None,
                    "run_crawl": bool(run_crawl_raw) if run_crawl_raw is not None else True,
                    "no_crawl_error_message": (
                        sanitize_text(body.get("no_crawl_error_message", "No web crawl data is available yet.")
                                      or "No web crawl data is available yet.")
                        or "No web crawl data is available yet."
                    ),
                    "system_message_prefix": sanitize_text(body.get("system_message_prefix", "") or "") or None,
                    "llm_name": sanitize_text(body.get("llm_name", "") or "") or None,
                }
                messages.append(f"WEBCRAWL:{json_lib.dumps(web_crawl_payload)}")
        elif reply_type == "ws_markdown":
            messages.append(f"WSMD:{json_lib.dumps({'message': sanitize_text(body.get('ws_message', '') or '')})}")
        elif reply_type == "ws_html":
            messages.append(f"WSHTML:{json_lib.dumps({'message': sanitize_text(body.get('ws_message', '') or '')})}")
        elif reply_type == "ws_speech":
            speed_raw = body.get("ws_audio_speed")
            try:
                audio_speed = float(speed_raw) if speed_raw not in (None, "") else None
            except (TypeError, ValueError):
                audio_speed = None
            messages.append(
                f"WSSPEECH:{json_lib.dumps({'message': sanitize_text(body.get('ws_message', '') or ''), 'audio_speed': audio_speed})}"
            )
        elif reply_type == "ws_options":
            opts_raw = body.get("ws_options", "") or ""
            if isinstance(opts_raw, list):
                options = [sanitize_text(str(o).strip()) for o in opts_raw if str(o).strip()]
            else:
                options = [sanitize_text(o.strip()) for o in opts_raw.split('\n') if o.strip()]
            messages.append(f"WSOPTIONS:{json_lib.dumps({'options': options})}")
        elif reply_type == "ws_location":
            try:
                lat = float(body.get("ws_latitude", 0.0))
            except (TypeError, ValueError):
                lat = 0.0
            try:
                lon = float(body.get("ws_longitude", 0.0))
            except (TypeError, ValueError):
                lon = 0.0
            messages.append(f"WSLOCATION:{json_lib.dumps({'latitude': lat, 'longitude': lon})}")
        elif reply_type == "ws_file":
            messages.append("WSFILE:")
        elif reply_type == "ws_image":
            messages.append("WSIMAGE:")
        elif reply_type == "ws_dataframe":
            messages.append("WSDATAFRAME:")
        elif reply_type == "ws_plotly":
            messages.append("WSPLOTLY:")

    return messages


def _build_body_from_messages(body_name, messages, build_db_reply_fn=None):
    """Build a Body object from classified messages."""
    if not messages:
        return None

    has_db = any(m.startswith("DB:") for m in messages)
    has_rag = any(m.startswith("RAG:") for m in messages)
    has_llm = any(m.startswith("LLM:") for m in messages)
    has_code = any(m.startswith("CODE:") for m in messages)
    has_llm_chat = any(m.startswith("LLMCHAT:") for m in messages)
    has_web_crawl = any(m.startswith("WEBCRAWL:") for m in messages)
    ws_prefixes = ("WSMD:", "WSHTML:", "WSSPEECH:", "WSOPTIONS:", "WSLOCATION:",
                   "WSFILE:", "WSIMAGE:", "WSDATAFRAME:", "WSPLOTLY:")
    has_ws = any(m.startswith(ws_prefixes) for m in messages)

    body = Body(body_name)

    if has_db and build_db_reply_fn:
        db_replies = [json_lib.loads(m.split(":", 1)[1]) for m in messages if m.startswith("DB:")]
        for db_reply in db_replies:
            body.add_action(build_db_reply_fn(db_reply))
    elif has_rag:
        for m in messages:
            if not m.startswith("RAG:"):
                continue
            raw = m.split(":", 1)[1]
            try:
                payload = json_lib.loads(raw)
            except (ValueError, TypeError):
                payload = None
            if isinstance(payload, dict):
                rag_db_name = payload.get("rag_db_name", "")
                rag_prompt = payload.get("prompt") or None
            else:
                # Backward compatibility: legacy ``RAG:<name>`` bare-name form.
                rag_db_name = raw
                rag_prompt = None
            body.add_action(RAGReply(rag_db_name=rag_db_name, prompt=rag_prompt))
    elif has_llm:
        for m in messages:
            if not m.startswith("LLM:"):
                continue
            payload_str = m.split(":", 1)[1]
            try:
                payload = json_lib.loads(payload_str)
            except (ValueError, TypeError):
                payload = {"prompt": payload_str, "llm_name": ""}
            prompt = payload.get("prompt") or None
            llm_name = payload.get("llm_name") or None
            body.add_action(LLMReply(prompt=prompt, llm_name=llm_name))
    elif has_code:
        code_contents = [m[5:] for m in messages if m.startswith("CODE:")]
        for code_content in code_contents:
            body.add_action(CustomCodeAction(source=code_content))
    elif has_llm_chat:
        for m in messages:
            if not m.startswith("LLMCHAT:"):
                continue
            payload_str = m.split(":", 1)[1]
            try:
                payload = json_lib.loads(payload_str)
            except (ValueError, TypeError):
                payload = {"prompt": payload_str, "llm_name": ""}
            prompt = payload.get("prompt") or None
            llm_name = payload.get("llm_name") or None
            body.add_action(LLMChatReply(prompt=prompt, llm_name=llm_name))
    elif has_web_crawl:
        for m in messages:
            if not m.startswith("WEBCRAWL:"):
                continue
            try:
                payload = json_lib.loads(m.split(":", 1)[1])
            except (ValueError, TypeError):
                continue
            body.add_action(WebCrawlLLMReply(
                initial_url=payload.get("initial_url", ""),
                max_depth=payload.get("max_depth", 2),
                max_pages=payload.get("max_pages", 20),
                crawl_format=payload.get("crawl_format", "markdown"),
                base_url_prefix=payload.get("base_url_prefix"),
                run_crawl=payload.get("run_crawl", True),
                no_crawl_error_message=payload.get(
                    "no_crawl_error_message", "No web crawl data is available yet."
                ),
                system_message_prefix=payload.get("system_message_prefix"),
                llm_name=payload.get("llm_name"),
            ))
    elif has_ws:
        for m in messages:
            prefix, _, payload_str = m.partition(":")
            prefix = prefix + ":"
            try:
                payload = json_lib.loads(payload_str) if payload_str else {}
            except (ValueError, TypeError):
                payload = {}
            if prefix == "WSMD:":
                body.add_action(WebSocketReplyMarkdown(message=payload.get("message", "")))
            elif prefix == "WSHTML:":
                body.add_action(WebSocketReplyHTML(message=payload.get("message", "")))
            elif prefix == "WSSPEECH:":
                body.add_action(WebSocketReplySpeech(
                    message=payload.get("message", ""),
                    audio_speed=payload.get("audio_speed"),
                ))
            elif prefix == "WSOPTIONS:":
                body.add_action(WebSocketReplyOptions(options=payload.get("options", [])))
            elif prefix == "WSLOCATION:":
                body.add_action(WebSocketReplyLocation(
                    latitude=payload.get("latitude", 0.0),
                    longitude=payload.get("longitude", 0.0),
                ))
            elif prefix == "WSFILE:":
                body.add_action(WebSocketReplyFile())
            elif prefix == "WSIMAGE:":
                body.add_action(WebSocketReplyImage())
            elif prefix == "WSDATAFRAME:":
                body.add_action(WebSocketReplyDataframe())
            elif prefix == "WSPLOTLY:":
                body.add_action(WebSocketReplyPlotly())
    else:
        for message in messages:
            body.add_action(AgentReply(message=message))

    return body


def _normalise_agent_transitions(edges: list[dict]) -> list[dict]:
    """Collapse legacy AgentStateTransition shapes to the canonical v4 form.

    The v4 canonical shape is on ``edge.data``:
        transitionType: 'predefined' | 'custom'
        predefined: { predefinedType, intentName?, fileType?, conditionValue? }
        custom: { event, condition: string[] }

    Five legacy shapes are accepted (see uml-v4-shape.md). Fallthrough
    order:
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
            ndata["transitionType"] = "custom"
            ndata["custom"] = {"event": event, "condition": cond}
            ndata.pop("predefined", None)
        else:
            predefined_type = (
                predefined.get("predefinedType")
                or ndata.get("predefinedType")
                or (ndata.get("condition") if isinstance(ndata.get("condition"), str) else None)
                or "when_intent_matched"
            )
            block: dict = {"predefinedType": predefined_type}
            intent_name = (
                predefined.get("intentName")
                or ndata.get("intentName")
            )
            if intent_name is not None:
                block["intentName"] = intent_name
            file_type = predefined.get("fileType") or ndata.get("fileType")
            if file_type is not None:
                block["fileType"] = file_type
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


def process_agent_diagram(json_data):
    """Process an Agent Diagram in the v4 wire shape and return an Agent."""
    config = json_data.get('config') or {}
    lang_value = config.get('language', '')
    language = lang_value.lower() if isinstance(lang_value, str) and lang_value else None
    source_language = config.get('source_language')

    def translate_text(text, lang, src_lang=None):
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
            translated = GoogleTranslator(source=src_code, target=target_lang).translate(text)
            return translated
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
        )

    def serialize_db_reply_payload(element: dict) -> str:
        payload = {
            "dbSelectionType": element.get("dbSelectionType", "default") or "default",
            "dbCustomName": element.get("dbCustomName", "") or "",
            "dbQueryMode": element.get("dbQueryMode", "llm_query") or "llm_query",
            "dbOperation": element.get("dbOperation", "any") or "any",
            "dbSqlQuery": element.get("dbSqlQuery", "") or "",
            "llm_name": element.get("llm_name", "") or "",
        }
        return f"DB:{json_lib.dumps(payload)}"

    title = json_data.get('title', 'Generated_Agent')
    if ' ' in title:
        title = title.replace(' ', '_')

    agent = Agent(title)

    agent.add_property(ConfigProperty('websocket_platform', 'websocket.host', '0.0.0.0'))
    agent.add_property(ConfigProperty('websocket_platform', 'websocket.port', 8765))
    agent.add_property(ConfigProperty('websocket_platform', 'streamlit.host', '0.0.0.0'))
    agent.add_property(ConfigProperty('websocket_platform', 'streamlit.port', 5000))
    agent.add_property(ConfigProperty('nlp', 'nlp.language', 'en'))
    agent.add_property(ConfigProperty('nlp', 'nlp.region', 'US'))
    agent.add_property(ConfigProperty('nlp', 'nlp.timezone', 'Europe/Madrid'))
    agent.add_property(ConfigProperty('nlp', 'nlp.pre_processing', True))
    agent.add_property(ConfigProperty('nlp', 'nlp.intent_threshold', 0.4))
    agent.add_property(ConfigProperty('nlp', 'nlp.openai.api_key', 'YOUR-API-KEY'))
    agent.add_property(ConfigProperty('nlp', 'nlp.hf.api_key', 'YOUR-API-KEY'))
    agent.add_property(ConfigProperty('nlp', 'nlp.replicate.api_key', 'YOUR-API-KEY'))

    model_data = json_data.get('model') or {}
    nodes = model_data.get('nodes') or []
    edges = model_data.get('edges') or []
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    edges = _normalise_agent_transitions(edges)

    nodes_by_id = {n.get("id"): n for n in nodes if n.get("id")}

    states_by_id = {}
    intents_by_id = {}
    rag_dbs_by_id = {}
    rag_dbs_by_name = {}

    comment_nodes = {}
    comment_links = {}

    intent_count = 0
    for node in nodes:
        node_type = node.get("type")
        node_id = node.get("id")
        data = node_data(node)
        if node_type in COMMENT_NODE_TYPES:
            comment_nodes[node_id] = data.get("name", "")
            continue
        if node_type == "AgentLLM":
            # Data-only LLM definition managed from the Agent
            # Customization panel (LLMs card). Registers on the agent's
            # ``llms`` list; the first registered LLM auto-becomes the
            # default unless ``config.default_llm_name`` overrides it
            # below.
            llm_name = sanitize_text((data.get("name") or "").strip())
            if not llm_name:
                continue
            if any(existing.name == llm_name for existing in agent.llms):
                continue
            provider = (data.get("provider") or "openai").lower()
            llm_parameters = data.get("parameters")
            if not isinstance(llm_parameters, dict):
                llm_parameters = {}
            num_prev = data.get("num_previous_messages")
            try:
                num_prev_int = int(num_prev) if num_prev is not None else 1
            except (TypeError, ValueError):
                num_prev_int = 1
            global_ctx = data.get("global_context") or None
            agent.new_llm(
                name=llm_name,
                provider=provider,
                parameters=llm_parameters,
                num_previous_messages=num_prev_int,
                global_context=global_ctx,
            )
            continue
        if node_type == "AgentTool":
            tool_name = sanitize_text((data.get("name") or "").strip())
            if not tool_name:
                continue
            if any(t.name == tool_name for t in agent.tools):
                continue
            agent.new_tool(
                name=tool_name,
                description=data.get("description", "") or "",
                code=data.get("code", "") or "",
            )
            continue
        if node_type == "AgentSkill":
            skill_name = sanitize_text((data.get("name") or "").strip())
            if not skill_name:
                continue
            if any(s.name == skill_name for s in agent.skills):
                continue
            agent.new_skill(
                name=skill_name,
                content=data.get("content", "") or "",
                description=data.get("description") or None,
            )
            continue
        if node_type == "AgentWorkspace":
            ws_name = sanitize_text((data.get("name") or "").strip())
            if not ws_name:
                continue
            if any(w.name == ws_name for w in agent.workspaces):
                continue
            writable = data.get("writable")
            if writable is None:
                writable = True
            max_read_bytes = data.get("max_read_bytes")
            if max_read_bytes is None:
                max_read_bytes = 200_000
            agent.new_workspace(
                name=ws_name,
                path=data.get("path", "") or "",
                description=data.get("description") or None,
                writable=bool(writable),
                max_read_bytes=int(max_read_bytes),
            )
            continue
        if node_type == "AgentReasoningState":
            # Reasoning states are created in the state-construction passes
            # below (alongside AgentState).
            continue
        if node_type == "AgentIntent":
            intent_name = data.get("name")
            training_sentences = []
            intent_description = data.get("intent_description", None)
            # The frontend stores training utterances on
            # ``data.training_phrases`` (``AgentIntentNodeProps``);
            # ``bodies`` is the legacy spelling kept as a fallback.
            phrase_rows = data.get("training_phrases")
            if not isinstance(phrase_rows, list) or not phrase_rows:
                phrase_rows = data.get("bodies") or []
            for body in phrase_rows:
                if not isinstance(body, dict):
                    continue
                training_sentence = sanitize_text(body.get("name", ""))
                if language:
                    training_sentence = translate_text(training_sentence, language, source_language)
                if training_sentence:
                    training_sentences.append(training_sentence)
            intent = Intent(intent_name, training_sentences, description=intent_description)
            agent.add_intent(intent)
            intents_by_id[node_id] = intent
            intent_count += 1
        elif node_type == "AgentRagElement":
            rag_name = sanitize_text((data.get("name") or "").strip())
            if not rag_name:
                continue
            if rag_name in rag_dbs_by_name:
                rag_dbs_by_id[node_id] = rag_dbs_by_name[rag_name]
                continue
            sanitized_slug = rag_name.lower().replace(' ', '_') or "default"
            vector_store = RAGVectorStore(
                embedding_provider="openai",
                embedding_parameters={"api_key_property": "nlp.OPENAI_API_KEY"},
                persist_directory=f"vector_store/{sanitized_slug}",
            )
            splitter = RAGTextSplitter(
                splitter_type="recursive_character",
                chunk_size=1000,
                chunk_overlap=100,
            )
            # The LLM is referenced by name; an empty value means "resolve
            # the agent default at codegen time" and passes the metamodel's
            # LLM-reference validation (a hard-coded model name would not).
            rag_llm_name = sanitize_text((data.get("llm_name") or "").strip()) or ""
            rag_llm_prompt = sanitize_text((data.get("llm_prompt") or data.get("llmPrompt") or "").strip()) or None
            rag_config = agent.new_rag(
                name=rag_name,
                vector_store=vector_store,
                splitter=splitter,
                llm_name=rag_llm_name,
                llm_prompt=rag_llm_prompt,
                k=4,
                num_previous_messages=0,
            )
            rag_dbs_by_id[node_id] = rag_config
            rag_dbs_by_name[rag_name] = rag_config

    # Find initial state (regular or reasoning).
    # Prefer the explicit ``data.initial`` property (v4 model: the initial
    # state is a boolean flag on the state itself). Fall back to the legacy
    # ``StateInitialNode`` marker + init edge for older diagrams that predate
    # the property so they keep converting unchanged.
    initial_state_id = None
    for node in nodes:
        if node.get("type") not in ("AgentState", "AgentReasoningState"):
            continue
        if node_data(node).get("initial") is True:
            initial_state_id = node.get("id")
            break
    if initial_state_id is None:
        for node in nodes:
            if node.get("type") not in ("AgentState", "AgentReasoningState"):
                continue
            for edge in edges:
                if edge.get("type") not in ("AgentStateTransition", "AgentStateTransitionInit"):
                    continue
                if edge.get("target") != node.get("id"):
                    continue
                source_node = nodes_by_id.get(edge.get("source"))
                if source_node and source_node.get("type") == "StateInitialNode":
                    initial_state_id = node.get("id")
                    break
            if initial_state_id:
                break

    def _is_reasoning_node(node: dict, data: dict) -> bool:
        """True if this node represents a reasoning agent state.

        The canonical v4 shape folds reasoning states into
        ``type: "AgentState"`` with ``data.stateType == "reasoning"``. The
        legacy ``type: "AgentReasoningState"`` node is still accepted on
        read for back-compat with diagrams saved before the fold.
        """
        if node.get("type") == "AgentReasoningState":
            return True
        return node.get("type") == "AgentState" and data.get("stateType") == "reasoning"

    def _build_reasoning_state(node_id: str, data: dict, is_initial: bool):
        """Create a ReasoningState from a reasoning agent-state node.

        Handles both the canonical v4 folded shape (``type: "AgentState"``
        with ``data.stateType == "reasoning"``) and the legacy
        ``type: "AgentReasoningState"`` shape — the field names are the
        same in both cases and mirror ``AgentReasoningStateNodeProps`` on
        the frontend (``llm_name``, ``max_steps``, ``enable_task_planning``,
        ``stream_steps``, ``system_prompt``, ``fallback_message``).
        """
        state_name = data.get("name", "") or ""
        llm_name = data.get("llm_name") or data.get("llm")
        llm_value = llm_name.strip() if isinstance(llm_name, str) and llm_name.strip() else None
        kwargs = {
            "name": state_name,
            "llm": llm_value,
            "initial": is_initial,
        }
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
        rs = agent.new_reasoning_state(**kwargs)
        # Tools, skills and workspaces are registered at the agent level and
        # shared by every reasoning state; the metamodel has no per-state
        # subset concept, so no per-state ref lists are parsed here.
        states_by_id[node_id] = rs
        return rs

    def _build_agent_state(node_id: str, data: dict, is_initial: bool):
        state_name = data.get("name", "") or ""
        agent_state = agent.new_state(name=state_name, initial=is_initial)
        states_by_id[node_id] = agent_state

        body_messages = _collect_body_messages(
            data.get("bodies"), language, source_language, translate_text,
            serialize_db_reply_payload=serialize_db_reply_payload,
        )
        body = _build_body_from_messages(f"{state_name}_body", body_messages, build_db_reply_fn=build_db_reply)
        if body:
            agent_state.set_body(body)
        fallback_messages = _collect_body_messages(
            data.get("fallbackBodies"), language, source_language, translate_text,
            serialize_db_reply_payload=serialize_db_reply_payload,
        )
        fallback_body = _build_body_from_messages(f"{state_name}_fallback_body", fallback_messages, build_db_reply_fn=build_db_reply)
        if fallback_body:
            agent_state.set_fallback_body(fallback_body)
        return agent_state

    # Process initial state first if found.
    if initial_state_id:
        node = nodes_by_id[initial_state_id]
        data = node_data(node)
        if _is_reasoning_node(node, data):
            _build_reasoning_state(initial_state_id, data, is_initial=True)
        else:
            _build_agent_state(initial_state_id, data, is_initial=True)

    # Process the rest of the states (including reasoning states).
    for node in nodes:
        node_id = node.get("id")
        if node_id == initial_state_id:
            continue
        if node.get("type") not in ("AgentState", "AgentReasoningState"):
            continue
        data = node_data(node)
        if _is_reasoning_node(node, data):
            _build_reasoning_state(node_id, data, is_initial=False)
        else:
            _build_agent_state(node_id, data, is_initial=False)

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
                comment_id = source_id
                target = target_id
            elif target_id in comment_nodes:
                comment_id = target_id
                target = source_id
            if comment_id and target:
                comment_links.setdefault(comment_id, []).append(target)
        elif edge_type in ("AgentStateTransition", "AgentStateTransitionInit"):
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
            transition_type = edge_data.get("transitionType")
            predefined_block = edge_data.get("predefined") or {}
            custom_block = edge_data.get("custom") or {}

            condition_name = ""
            transition_payload: object = ""

            is_custom_transition = transition_type == "custom"

            if is_custom_transition:
                selected_event = (
                    custom_block.get("event")
                    or edge_data.get("event")
                    or edge_data.get("customEvent")
                )
                custom_conditions = (
                    custom_block.get("condition")
                    if isinstance(custom_block.get("condition"), list)
                    else edge_data.get("conditions")
                )
                if not isinstance(custom_conditions, list):
                    custom_conditions = edge_data.get("customConditions")
                if not isinstance(custom_conditions, list):
                    custom_conditions = []

                normalized_event = "None"
                if isinstance(selected_event, str) and selected_event and selected_event != "None":
                    normalized_event = selected_event
                condition_name = "custom_transition"
                transition_payload = {
                    "event": normalized_event,
                    "conditions": custom_conditions if isinstance(custom_conditions, list) else [],
                }
            else:
                condition_name = (
                    predefined_block.get("predefinedType")
                    or edge_data.get("predefinedType")
                    or ""
                )
                if condition_name == "when_intent_matched":
                    transition_payload = (
                        predefined_block.get("intentName")
                        or edge_data.get("intentName")
                    )
                elif condition_name == "when_file_received":
                    transition_payload = (
                        predefined_block.get("fileType")
                        or edge_data.get("fileType")
                    )
                else:
                    transition_payload = predefined_block.get("conditionValue")

                if transition_payload is None:
                    transition_payload = edge_data.get("conditionValue", "")

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
                            variable_name, operator_value,
                            source_state.name, target_state.name,
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

            elif condition_name == "when_file_received":
                mime_types = {
                    "PDF": "application/pdf",
                    "TXT": "text/plain",
                    "JSON": "application/json",
                }
                if isinstance(transition_payload, str) and "/" in transition_payload:
                    file_type = transition_payload
                else:
                    file_type = mime_types.get(transition_payload)
                if file_type:
                    source_state.when_file_received(file_type).go_to(target_state)
                    transition_count += 1
                else:
                    logger.warning(
                        "Unknown file type '%s' for when_file_received transition from '%s' to '%s'. "
                        "Falling back to when_file_received() without type filter.",
                        transition_payload, source_state.name, target_state.name,
                    )
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
                    if not selected_event:
                        raw_events = transition_payload.get("events") or []
                        if isinstance(raw_events, list) and raw_events:
                            selected_event = raw_events[0]

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
                    elif selected_event == "None":
                        event_instance = None

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
                    transition_count += 1
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
                        state.metadata.description = f"{existing_desc}\n{comment_text}" if existing_desc else comment_text
        else:
            if agent.metadata is None:
                agent.metadata = Metadata(description=comment_text)
            else:
                existing_desc = agent.metadata.description or ""
                agent.metadata.description = f"{existing_desc}\n{comment_text}" if existing_desc else comment_text

    # Apply default LLM from the customization config block (if set).
    # The customization tab persists which registered LLM is the default;
    # without an explicit pointer the agent already auto-defaulted to the
    # first one registered.
    default_llm_name_cfg = (config or {}).get("default_llm_name")
    if isinstance(default_llm_name_cfg, str) and default_llm_name_cfg.strip():
        if any(existing.name == default_llm_name_cfg for existing in agent.llms):
            agent.set_default_llm(default_llm_name_cfg)

    agent.validate(raise_exception=True)
    return agent
