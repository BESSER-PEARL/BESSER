"""
Context compaction for LLM conversation history.

When the conversation grows too large for the context window, older messages
are summarized and replaced with a compact representation that preserves
the essential information (what tools were called, what files exist, and —
when the caller passes it — what work is still open).

The summary is DETERMINISTIC on purpose: no LLM call. The target model is
often a weak local Qwen3-30B-A3B, and asking it to summarize its own
transcript at the exact moment its context is failing is a reliability
liability, not a feature.
"""

import json
import logging
import os
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# Context compaction config.
#
# This is the LAST-RESORT context guard, not a routine budget: every
# compaction is a lossy summarisation, and anything it drops that the model
# still needs gets RE-READ, which costs more than it saved. A threshold set
# too low therefore does not "save context" - it produces a read/compact/
# re-read spiral that can run until the runtime cap.
COMPACT_TOKEN_THRESHOLD = max(
    8_000, int(os.environ.get("BESSER_LLM_COMPACT_THRESHOLD", "80000") or 80_000)
)
# Set explicitly, the same variable is also a hard cap on the adaptive
# threshold of every known-window model (see ``effective_threshold``): an
# operator lowering it to cut cost gets exactly that, not silence. Unset,
# the adaptive value stands, up to MAX_COMPACT_THRESHOLD.
_OPERATOR_CAP: int | None = (
    COMPACT_TOKEN_THRESHOLD if os.environ.get("BESSER_LLM_COMPACT_THRESHOLD") else None
)
# The preserved tail is sized in TOKENS, not messages: six write_file turns
# and six one-line turns differ by two orders of magnitude, so a fixed count
# either frees nothing or leaves the model without its working set.
# Expressed as a share of the LIVE threshold so a clamped small-window model
# gets a proportionally smaller tail instead of one that fills its window.
COMPACT_PRESERVE_TAIL_FRACTION = 0.3
# Floor for very small thresholds: under ~one whole-file read the tail cannot
# hold a working set at all.
COMPACT_MIN_PRESERVE_TOKENS = 4_000

# Headroom the model needs for its next response. The threshold is
# really "context window minus reserve" - the fixed 80k default silently
# overflows small-window models, which then truncate or hallucinate
# instead of compacting.
#
# NOTE this is a FLOOR, not the whole story: the from-scratch and modify
# paths raise ``client.max_tokens`` to FROM_SCRATCH_MAX_TOKENS (32_768),
# which is larger than this reserve. Callers that raise max_tokens should
# pass ``reserve=`` explicitly - see ``effective_threshold``.
COMPACT_RESERVE_TOKENS = 16_000

# Known small context windows by model-name substring.
#
# Add a row ONLY with evidence that the SERVED window is small - not the
# model's reputation, and not another deployment of the same family. A
# wrong row here is far more damaging than a missing one: an unknown model
# gets the frontier-sized default (fine, because a real overflow surfaces
# as a provider error), whereas an under-stated window compacts constantly
# and silently destroys the model's working context.
#
# Deliberately absent: a bare "qwen" row (hosted qwen models serve 262_144,
# and a 32k row would request more output than the window it claims), a bare
# "mistral" row (matches ``mistral-large-latest``, 256k), and "llama" /
# "deepseek" (no evidence of a small served window).
_SMALL_CONTEXT_WINDOWS: tuple = (
    # Self-hosted Ollama loads these with a 32k window (server config)
    # regardless of the model's native maximum; check via /api/ps. A restart
    # of ``ollama serve`` without the env var falls back to a much smaller
    # default, silently.
    ("devstral", 32_000),
    # Small self-hosted Mistral variants only. Deliberately NOT a bare
    # "mistral" marker, which would also match mistral-large (256k).
    ("mistral-small", 32_000),
    ("mistral-7b", 32_000),
    # Self-hosted Ollama (single GPU): OLLAMA_CONTEXT_LENGTH advertises
    # 131_072, but that is capability, not usable window. On qwen3-coder:30b a
    # 128k prompt comes back truncated to ~65_536 - from the FRONT, silently
    # dropping the system prompt. 60_000 keeps history plus the 32_768 output
    # reserve under that ceiling. Covers every self-hosted qwen3 tag.
    ("qwen3", 60_000),
)

# A threshold this close to the cost of a single tool result cannot hold a
# working set, so compaction fires between reads and the model re-reads
# what it just lost. Sized against tool_executor.MAX_FILE_READ (20_000
# chars, ~4_100 tokens at the measured 4.83 chars/token for code): fewer
# than ~4 whole-file reads of headroom is pathological, not tight.
_MIN_WORKABLE_THRESHOLD = 16_500


# Advertised context windows from the provider's model catalog.
#
# Only the Command Code endpoint (the keyless tier) returns ``context_length``
# from GET /models; OpenAI's and Anthropic's catalogs do not, so every other
# provider resolves through ``_KNOWN_CONTEXT_WINDOWS`` or the default. Read
# from the same server env the client uses (BESSER_FREE_LLM_BASE_URL /
# BESSER_FREE_LLM_TOKEN). Fetched at most once per process, failure included,
# and never allowed to fail a run: this runs inside the generation loop, so a
# dead endpoint costs one short timeout and then the fall-through.
_CATALOG_TIMEOUT_S = 3.0
_CATALOG: dict[str, int] | None = None
_CATALOG_LOADED = False

# Advertised context windows for hosted models, by family substring.
# Consulted after the measured table and the catalog, before the default.
#
# Rows are generous on purpose. Overflow against a hosted API (OpenAI,
# Anthropic, Command Code) surfaces as a clean provider error - loud and
# recoverable - whereas a self-hosted Ollama server truncates silently from
# the FRONT and eats the system prompt. An optimistic row is cheap here and
# dangerous in the measured table, which is why the measured table wins
# over everything below it. Family markers rather than exact ids so a point
# release does not fall back to the default; each marker must be specific
# enough not to catch a sibling family (see the bare "mistral" note above).
# Windows from the Command Code catalog and provider documentation.
_KNOWN_CONTEXT_WINDOWS: tuple = (
    ("claude-haiku", 200_000),
    ("claude-sonnet", 1_000_000),
    ("claude-opus", 1_000_000),
    ("claude-fable", 1_000_000),
    ("gpt-5.6", 1_050_000),
    ("gpt-5.5", 400_000),
    ("gpt-5.4", 400_000),
    ("gpt-5.3-codex", 400_000),
    ("gpt-4o", 128_000),
    ("gpt-4.1", 1_000_000),
    ("mistral-large", 256_000),
    # Nebius Token Factory endpoint properties: 262K context, FP8, tool
    # calling available.
    ("qwen3-30b-a3b", 262_144),
)

# Share of the USABLE advertised window (window - reserve) the history may
# fill. It must absorb the system prompt and tool schemas (outside
# ``messages``) and the tiktoken-vs-served tokenizer gap; one turn of growth
# is not on top, because the check runs before every request. 0.8 leaves
# ~15-19% of the window for those and keeps every hosted row within one
# file read of the flat default - gpt-4o (128k, the OpenAI default) is the
# binding case at ~76k. A whole-window 0.5 would put it at 31k, i.e. the
# re-read spiral.
_USABLE_WINDOW_FRACTION = 0.8

# Absolute ceiling on what an advertised window may hand back. A 1M window
# would otherwise send ~800k-token prompts every turn: metered credits on a
# paid plan, and prefill latency the runtime cap cannot absorb. 200k is 2.5x
# the flat default and bounds the per-turn prompt at ~threshold + reserve
# whatever the window says.
MAX_COMPACT_THRESHOLD = int(
    os.environ.get("BESSER_LLM_MAX_COMPACT_THRESHOLD", "200000") or 200_000
)


def _fetch_models(base_url: str, token: str) -> dict[str, int]:
    """``{model id: context_length}`` from one endpoint's GET /models."""
    request = urllib.request.Request(base_url.rstrip("/") + "/models")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=_CATALOG_TIMEOUT_S) as response:
        payload = json.load(response)
    entries = payload.get("data") if isinstance(payload, dict) else payload
    windows: dict[str, int] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        model_id, window = entry.get("id"), entry.get("context_length")
        if isinstance(model_id, str) and isinstance(window, int) and window > 0:
            windows[model_id] = window
    return windows


def _fetch_catalog() -> dict[str, int]:
    """Union of the free and sponsored tiers' catalogs (same server env the
    client uses). Empty when neither is configured; a shared endpoint is
    fetched once; raises only when every configured endpoint failed (the
    caller caches that as unavailable).
    """
    endpoints: list[tuple[str, str]] = []
    for prefix in ("BESSER_FREE_LLM", "BESSER_SPONSORED_LLM"):
        base_url = os.environ.get(f"{prefix}_BASE_URL", "").strip()
        if base_url and base_url not in (b for b, _ in endpoints):
            endpoints.append((base_url, os.environ.get(f"{prefix}_TOKEN", "").strip()))
    windows: dict[str, int] = {}
    failure: Exception | None = None
    for base_url, token in endpoints:
        try:
            windows.update(_fetch_models(base_url, token))
        except Exception as exc:
            failure = exc
    if failure is not None and not windows:
        raise failure
    return windows


def _advertised_window(model: str) -> int | None:
    """Catalog window for the exact model id, or None. One fetch per process."""
    global _CATALOG, _CATALOG_LOADED
    if not _CATALOG_LOADED:
        _CATALOG_LOADED = True
        try:
            _CATALOG = _fetch_catalog()
        except Exception as exc:
            logger.warning("Model catalog unavailable (%s); not retried this process.", exc)
            _CATALOG = {}
    return (_CATALOG or {}).get(model)


def effective_threshold(
    model: str | None,
    threshold: int = COMPACT_TOKEN_THRESHOLD,
    reserve: int = COMPACT_RESERVE_TOKENS,
) -> int:
    """Resolve the compaction threshold for ``model``.

    Most trusted source first:

    1. ``_SMALL_CONTEXT_WINDOWS`` - measured on the deployment we call.
       Always wins, even over a larger advertised figure: ``window -
       reserve``, never above ``threshold``.
    2. The keyless tier's catalog ``context_length`` for the exact model id.
    3. ``_KNOWN_CONTEXT_WINDOWS`` by family substring.
    4. The unchanged ``threshold`` for anything else (unknown names are
       assumed frontier-sized - wrongly clamping a big model would compact
       constantly).

    An advertised window (2, 3) yields ``(window - reserve) * fraction``,
    capped at ``MAX_COMPACT_THRESHOLD`` and, when BESSER_LLM_COMPACT_THRESHOLD
    is set explicitly, at that value too.

    ``reserve`` should be the live ``client.max_tokens`` when the caller
    has raised it above the default, so the headroom actually matches the
    response the model is allowed to produce.
    """
    if not model:
        return threshold
    low = model.lower()
    # Every measured row is for a SELF-HOSTED deployment, which serves bare
    # ``name:tag`` ids. A cloud provider
    # namespaces its ids by vendor ("Qwen/Qwen3-30B-A3B-Instruct-2507" on
    # Nebius, 262k native), and clamping one of those to a self-hosted server's
    # measured window compacts constantly for no reason. The "/" is what
    # tells them apart, so a namespaced id skips the measured table and falls
    # through to the advertised/known windows below.
    if "/" not in low:
        for marker, window in _SMALL_CONTEXT_WINDOWS:
            if marker in low:
                return _guard_floor(model, min(threshold, window - reserve), window, reserve)
    window = _advertised_window(model)
    if window is None:
        for marker, known in _KNOWN_CONTEXT_WINDOWS:
            if marker in low:
                window = known
                break
    if window:
        effective = min(
            MAX_COMPACT_THRESHOLD, int((window - reserve) * _USABLE_WINDOW_FRACTION)
        )
        if _OPERATOR_CAP is not None:
            effective = min(effective, _OPERATOR_CAP)
        return _guard_floor(model, effective, window, reserve)
    return threshold


def _guard_floor(model: str, effective: int, window: int, reserve: int) -> int:
    """Floor at 8k and warn when the result cannot hold a working set."""
    effective = max(8_000, effective)
    if effective < _MIN_WORKABLE_THRESHOLD:
        logger.warning(
            "Compaction threshold for %s is %d tokens (window=%d, "
            "reserve=%d). That is under ~4 whole-file reads, so "
            "compaction will fire between tool calls and the model "
            "will re-read what it loses. Verify the SERVED context "
            "window for this model.",
            model, effective, window, reserve,
        )
    return effective


def _is_tool_result_message(msg: dict) -> bool:
    """True when ``msg`` is the user-role message carrying tool results."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype == "tool_result":
            return True
    return False


# Real-tokenizer singleton. ``tiktoken`` (a declared dependency) is close
# enough to Anthropic's tokenizer for the ``is the context over threshold?``
# decision.
#
# If it is missing we fall back to chars/4. Measured on BESSER Python source
# the true ratio is 4.83 chars/token, so chars/4 reports ~121% of the real
# count and the threshold trips EARLY, not late.
_TOKENIZER: Any = None
_TOKENIZER_LOADED = False


def _get_tokenizer():
    global _TOKENIZER, _TOKENIZER_LOADED
    if _TOKENIZER_LOADED:
        return _TOKENIZER
    _TOKENIZER_LOADED = True
    try:
        import tiktoken
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TOKENIZER = None
    return _TOKENIZER


def _count_tokens(text: str) -> int:
    """Count tokens via tiktoken if available, else chars/4 heuristic."""
    if not text:
        return 0
    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, disallowed_special=()))
        except Exception:
            # Tokenizer can fail on weird control characters; fall back.
            pass
    return len(text) // 4


def _estimate_tokens(messages: list[dict]) -> int:
    """Estimate token count of a message list, preferring a real tokenizer."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _count_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    # ``input`` included so a dict-shaped tool_use block does
                    # not count as ZERO tokens: write_file arguments are the
                    # single largest thing the model emits, and any path that
                    # normalises blocks to dicts would otherwise hide them.
                    text = (
                        block.get("content", "")
                        or block.get("text", "")
                        or block.get("input", "")
                    )
                    total += _count_tokens(str(text))
                elif hasattr(block, "text"):
                    total += _count_tokens(block.text)
                elif hasattr(block, "input"):
                    total += _count_tokens(str(block.input))
    return total


def _tail_cut_index(messages: list[dict], budget: int) -> int:
    """Index of the first message to KEEP: the longest tail fitting ``budget``.

    Walks backwards over whole messages. The last message is kept even when it
    alone busts the budget — a tail of nothing leaves the model with a summary
    and no work in hand. Turns are never split: the walk stops on a message
    boundary and the caller's tool_result walk-back then pulls the paired
    ``tool_use`` back in, which may push the tail slightly over budget.
    """
    total = 0
    cut = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        cost = _estimate_tokens([messages[index]])
        if cut < len(messages) and total + cost > budget:
            break
        total += cost
        cut = index
    return cut


def maybe_compact(
    messages: list[dict],
    tool_calls_log: list[dict],
    output_dir: str,
    threshold: int = COMPACT_TOKEN_THRESHOLD,
    preserve_tokens: int | None = None,
    domain_model: Any | None = None,
    gui_model: Any | None = None,
    agent_model: Any | None = None,
    state_machines: list[Any] | None = None,
    object_model: Any | None = None,
    quantum_circuit: Any | None = None,
    bpmn_model: Any | None = None,
    nn_model: Any | None = None,
    primary_kind: str | None = None,
    model: str | None = None,
    reserve: int = COMPACT_RESERVE_TOKENS,
    work_state: Any | None = None,
) -> tuple[list[dict], bool]:
    """
    Compact conversation history if it exceeds the token threshold.

    Args:
        messages: The current conversation messages.
        tool_calls_log: Log of tool calls made so far.
        output_dir: Path to the output directory (for file listing).
        threshold: Token threshold above which compaction triggers.
        preserve_tokens: Token budget for the preserved tail. ``None``
            derives it from the resolved threshold.
        domain_model: Optional BUML DomainModel. When provided, a minimal
            recap (class names + association summary) is preserved in the
            summary so the LLM doesn't have to re-discover structure from
            file contents after compaction.
        model: The LLM model name, used to clamp ``threshold`` to the
            model's context window (small local models overflow the
            fixed default long before it trips).
        work_state: Optional open-work state; see :func:`_work_state_section`.
            Omitted, no work-state section is rendered.

    Returns:
        A tuple of (compacted_messages, did_compact).
    """
    threshold = effective_threshold(model, threshold, reserve)
    est_tokens = _estimate_tokens(messages)
    if est_tokens < threshold:
        return messages, False
    if preserve_tokens is None:
        preserve_tokens = max(
            COMPACT_MIN_PRESERVE_TOKENS,
            int(threshold * COMPACT_PRESERVE_TAIL_FRACTION),
        )

    cut = _tail_cut_index(messages, preserve_tokens)
    # The cut must not orphan a tool_use/tool_result pair: a preserved
    # tail that OPENS with tool results whose tool_use call was
    # summarized away is an invalid conversation for both providers.
    # Walk the cut back until the tail opens on a clean boundary.
    while cut > 0 and _is_tool_result_message(messages[cut]):
        cut -= 1
    if cut <= 0:
        return messages, False

    logger.info(
        "Compacting: ~%d tokens (threshold %d) -> summarizing %d messages, "
        "preserving the last %d (~%d token tail budget)",
        est_tokens, threshold, cut, len(messages) - cut, preserve_tokens,
    )

    to_summarize = messages[:cut]
    to_preserve = messages[cut:]
    summary = _summarize_messages(
        to_summarize, tool_calls_log, output_dir,
        domain_model=domain_model,
        gui_model=gui_model,
        agent_model=agent_model,
        state_machines=state_machines,
        object_model=object_model,
        quantum_circuit=quantum_circuit,
        bpmn_model=bpmn_model,
        nn_model=nn_model,
        primary_kind=primary_kind,
        work_state=work_state,
    )

    compacted = [
        {"role": "user", "content": f"[Earlier work summarized]\n\n{summary}\n\nContinue."},
    ]
    # Only insert the synthetic assistant turn when the preserved tail
    # opens with a user message — if the boundary walk above landed the
    # cut on an assistant tool_use message, adding another assistant
    # message would produce two consecutive assistant turns (invalid).
    if to_preserve and to_preserve[0].get("role") != "assistant":
        compacted.append(
            {"role": "assistant", "content": [{"type": "text", "text": "Continuing."}]}
        )
    compacted.extend(to_preserve)
    return compacted, True


def _summarize_messages(
    messages: list[dict],
    tool_calls_log: list[dict],
    output_dir: str,
    domain_model: Any | None = None,
    gui_model: Any | None = None,
    agent_model: Any | None = None,
    state_machines: list[Any] | None = None,
    object_model: Any | None = None,
    quantum_circuit: Any | None = None,
    bpmn_model: Any | None = None,
    nn_model: Any | None = None,
    primary_kind: str | None = None,
    work_state: Any | None = None,
) -> str:
    """Build a compact summary of earlier conversation messages."""
    lines = [f"Earlier: {len(messages)} messages"]

    if primary_kind:
        lines.append(f"Primary model: {primary_kind}")

    # Keep a minimal model recap so the LLM still knows what entities and
    # associations exist after the JSON was pruned from the context. A full
    # model JSON can be KB; this is ~hundreds of bytes.
    recap = _compact_model_recap(
        domain_model=domain_model,
        gui_model=gui_model,
        agent_model=agent_model,
        state_machines=state_machines,
        object_model=object_model,
        quantum_circuit=quantum_circuit,
        bpmn_model=bpmn_model,
        nn_model=nn_model,
    )
    if recap:
        lines.append(recap)

    if tool_calls_log:
        tools: dict[str, int] = {}
        for tc in tool_calls_log:
            tools[tc["tool"]] = tools.get(tc["tool"], 0) + 1
        lines.append(f"Tools: {', '.join(f'{k}({v}x)' for k, v in sorted(tools.items()))}")

    # File operations, cumulative across the WHOLE run (the log survives
    # every previous compaction, so nothing is ever forgotten). This is
    # the memory that stops a post-compaction model re-writing a file it
    # already customised or re-reading everything from scratch.
    written, read = _file_operations(tool_calls_log)
    if written:
        lines.append(
            "Files you already WROTE or MODIFIED (your edits are on disk — "
            "re-read before editing again, never rewrite blindly): "
            + ", ".join(written[:30])
            + (f" … +{len(written) - 30} more" if len(written) > 30 else "")
        )
    if read:
        lines.append(
            "Files you already read: "
            + ", ".join(read[:30])
            + (f" … +{len(read) - 30} more" if len(read) > 30 else "")
        )
    try:
        files = []
        for root, _, fnames in os.walk(output_dir):
            for f in fnames:
                if not f.startswith(".besser_"):
                    files.append(
                        os.path.relpath(os.path.join(root, f), output_dir).replace("\\", "/")
                    )
        if files:
            lines.append(f"Files: {', '.join(sorted(files)[:25])}")
    except Exception:
        pass

    # Last, next to the "Continue." the caller appends: this is the section
    # the model must act on, and the end-of-run gate keeps blocking on the
    # checklist whether or not the model can still see it.
    work = _work_state_section(work_state)
    if work:
        lines.append(work)
    return "\n".join(lines)


# Caps for the work-state section. This competes with the live tail for
# context, so every list is bounded and the remainder reported as a count.
_MAX_SUMMARY_OPEN_TASKS = 12
_MAX_SUMMARY_BLOCKED_TASKS = 5
_MAX_SUMMARY_DONE_IDS = 30
_MAX_SUMMARY_BLOCKERS = 8
_MAX_SUMMARY_RULES = 8
_MAX_SUMMARY_LINE_CHARS = 200


def _clip(text: Any, limit: int = _MAX_SUMMARY_LINE_CHARS) -> str:
    """One whitespace-collapsed line, truncated to ``limit`` characters."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _more(total: int, shown: int) -> list[str]:
    return [f"    … +{total - shown} more"] if total > shown else []


def _task_status(task: dict) -> str:
    """open / blocked / done / dropped, in the checklist's own precedence."""
    if task.get("dropped"):
        return "dropped"
    if task.get("blocked"):
        return "blocked"
    if task.get("done"):
        return "done"
    return "open"


def _checklist_lines(tasks: Any) -> list[str]:
    """Render the checklist: every open item, every blocker, done as ids."""
    if not isinstance(tasks, (list, tuple)):
        return []
    items = [t for t in tasks if isinstance(t, dict) and t.get("text")]
    if not items:
        return []
    by_status: dict[str, list[dict]] = {}
    for task in items:
        by_status.setdefault(_task_status(task), []).append(task)
    open_items = by_status.get("open", [])
    blocked = by_status.get("blocked", [])
    done = by_status.get("done", [])

    counts = [f"{len(open_items)} open", f"{len(blocked)} blocked"]
    if done:
        counts.append(f"{len(done)} done")
    if by_status.get("dropped"):
        counts.append(f"{len(by_status['dropped'])} dropped")
    lines = [
        f"  Checklist ({', '.join(counts)}) — the run cannot finish while an "
        "item is open; close each one with task_list (done / drop / blocked)."
    ]
    for task in open_items[:_MAX_SUMMARY_OPEN_TASKS]:
        flag = " [has verifier]" if _has_verifier(task) else ""
        lines.append(f"    OPEN {task.get('id')}{flag}: {_clip(task['text'])}")
    lines += _more(len(open_items), _MAX_SUMMARY_OPEN_TASKS)
    for task in blocked[:_MAX_SUMMARY_BLOCKED_TASKS]:
        reason = task.get("blocked_reason")
        lines.append(
            f"    BLOCKED {task.get('id')}: {_clip(task['text'])}"
            + (f" — {_clip(reason, 120)}" if reason else "")
        )
    lines += _more(len(blocked), _MAX_SUMMARY_BLOCKED_TASKS)
    if done:
        ids = [str(t.get("id")) for t in done[:_MAX_SUMMARY_DONE_IDS]]
        unverified = sum(
            1 for t in done if t.get("verification", "unverified") != "verified"
        )
        lines.append(
            f"    DONE (do not redo): {', '.join(ids)}"
            + (f" +{len(done) - len(ids)} more" if len(done) > len(ids) else "")
            + (f" — {unverified} of them unverified" if unverified else "")
        )
    return lines


def _has_verifier(task: dict) -> bool:
    """True when the item carries a verifier, so 'done' is machine-checked.

    ``verify`` is the live callable the executor holds; ``has_verifier`` is
    the flag a serialized snapshot can carry instead.
    """
    return bool(task.get("verify") or task.get("has_verifier"))


def _issue_text(entry: Any) -> str:
    """Blocker text from a string, a mapping, or a ValidationIssue-like."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return str(entry.get("message") or entry)
    return str(getattr(entry, "message", entry))


def _blocker_lines(blockers: Any) -> list[str]:
    if not isinstance(blockers, (list, tuple)) or not blockers:
        return []
    lines = [f"  Open blockers ({len(blockers)}) — these must be fixed:"]
    for entry in blockers[:_MAX_SUMMARY_BLOCKERS]:
        lines.append(f"    - {_clip(_issue_text(entry))}")
    return lines + _more(len(blockers), _MAX_SUMMARY_BLOCKERS)


def _contract_lines(rules: Any) -> list[str]:
    if isinstance(rules, str):
        rules = [line for line in rules.splitlines() if line.strip()]
    if not isinstance(rules, (list, tuple)) or not rules:
        return []
    lines = ["  Data contract — still NON-NEGOTIABLE:"]
    for rule in rules[:_MAX_SUMMARY_RULES]:
        lines.append(f"    - {_clip(rule)}")
    return lines + _more(len(rules), _MAX_SUMMARY_RULES)


def _work_state_section(work_state: Any) -> str:
    """Render the open work the summary must not drop.

    ``work_state`` is an optional mapping so this module stays a leaf — it
    must not import the orchestrator or the executor that own this state.
    Recognized keys, all optional:

    * ``tasks`` — checklist items as the executor stores them: ``id``,
      ``text``, ``done``, ``blocked``/``blocked_reason``, ``dropped``,
      ``verification``, and either the live ``verify`` callable or a
      ``has_verifier`` flag.
    * ``blockers`` — currently open blockers: strings, mappings with a
      ``message`` key, or objects with a ``.message`` attribute.
    * ``contract_rules`` — model-derived hard rules, as a list of short
      strings or one newline-separated block.

    Anything else (including ``None``) renders nothing, so an unwired caller
    gets no work-state section.
    """
    if not isinstance(work_state, dict):
        return ""
    lines = (
        _checklist_lines(work_state.get("tasks"))
        + _blocker_lines(work_state.get("blockers"))
        + _contract_lines(work_state.get("contract_rules"))
    )
    if not lines:
        return ""
    return "Work state (carried through compaction — still binding):\n" + "\n".join(lines)


def _file_operations(tool_calls_log: list[dict]) -> tuple[list, list]:
    """Split the tool log into (written_or_modified, read_only) paths.

    A path that was both read and written reports as written (the write
    is what the model must remember). Deleted paths drop out entirely.
    """
    written: set = set()
    read: set = set()
    for tc in tool_calls_log or []:
        args = tc.get("input") or {}
        path = args.get("path") if isinstance(args, dict) else None
        if not isinstance(path, str) or not path:
            continue
        tool = tc.get("tool")
        if tool in ("write_file", "modify_file", "replace_file_lines"):
            written.add(path)
        elif tool == "read_file":
            read.add(path)
        elif tool == "delete_file":
            written.discard(path)
            read.discard(path)
    return sorted(written), sorted(read - written)


def _compact_model_recap(
    domain_model: Any | None = None,
    gui_model: Any | None = None,
    agent_model: Any | None = None,
    state_machines: list[Any] | None = None,
    object_model: Any | None = None,
    quantum_circuit: Any | None = None,
    bpmn_model: Any | None = None,
    nn_model: Any | None = None,
) -> str:
    """One-line summary of the loaded BUML models for compaction context.

    Each model contributes a short recap if present. Nothing is serialized
    in full — the goal is to preserve just enough structure (names,
    counts) for the LLM to reason about entities after compaction drops
    the original JSON.
    """
    parts: list[str] = []

    if domain_model is not None:
        try:
            class_names = sorted(
                c.name for c in domain_model.get_classes() if getattr(c, "name", None)
            )
        except Exception:
            class_names = []
        try:
            enum_names = sorted(
                e.name for e in domain_model.get_enumerations() if getattr(e, "name", None)
            )
        except Exception:
            enum_names = []
        try:
            assoc_count = len(list(getattr(domain_model, "associations", []) or []))
        except Exception:
            assoc_count = 0
        if class_names:
            parts.append(f"Classes ({len(class_names)}): {', '.join(class_names)}")
        if enum_names:
            parts.append(f"Enums: {', '.join(enum_names)}")
        if assoc_count:
            parts.append(f"Associations: {assoc_count}")

    if state_machines:
        sm_names = [getattr(sm, "name", None) or "unnamed" for sm in state_machines]
        parts.append(f"State machines ({len(sm_names)}): {', '.join(sm_names)}")

    if agent_model is not None:
        parts.append("Agent present")

    if gui_model is not None:
        try:
            screen_count = sum(
                len(m.screens or []) for m in (gui_model.modules or [])
            )
        except Exception:
            screen_count = 0
        parts.append(f"GUI ({screen_count} screens)" if screen_count else "GUI present")

    if object_model is not None:
        parts.append("Object instances present")

    if quantum_circuit is not None:
        parts.append("Quantum circuit present")

    if bpmn_model is not None:
        process_count = len(getattr(bpmn_model, "processes", []) or [])
        label = getattr(bpmn_model, "name", None) or "BPMN"
        parts.append(f"BPMN {label} ({process_count} processes)")

    if nn_model is not None:
        module_count = len(getattr(nn_model, "modules", []) or [])
        label = getattr(nn_model, "name", None) or "NN"
        parts.append(f"Neural network {label} ({module_count} modules)")

    if not parts:
        return ""
    return "Model recap → " + " | ".join(parts)
