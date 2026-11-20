"""The lab agent's tool loop: one model call per turn, then that turn's tool calls.

The Spec-Driven Agent's Phase 2 loop needs a B-UML model and an application to
validate, so the lab has its own. It keeps that loop's provider handling (the
``max_tokens`` retry, compaction, key errors) and caps, and drops what is about
generated applications. It has no no-progress limit: reading for many turns is
normal on a repository the size of BESSER, and the turn and cost caps end a run
that loops.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from besser.spec_driven_agent.agent.compaction import COMPACT_RESERVE_TOKENS, maybe_compact
from besser.spec_driven_agent.errors import InvalidApiKeyError
from besser.spec_driven_agent.lab.contracts import LabCallbacks, LabLimits, LoopResult
from besser.spec_driven_agent.lab.tooling import READ_TOOLS, LabToolExecutor
from besser.spec_driven_agent.pipeline.constants import (
    _MAX_PARALLEL_WORKERS,
    _tool_call_detail,
    _tool_use_blocks,
)
from besser.spec_driven_agent.pipeline.orchestrator import _trace_safe_input

logger = logging.getLogger(__name__)

#: Output-limit retries per loop. One smaller turn usually fits; a model that is
#: cut off three times running is not going to fit the next one either.
MAX_TRUNCATION_RETRIES = 3

#: How ``maybe_compact`` opens its summary message.
_SUMMARY_PREFIX = "[Earlier work summarized]"

TRUNCATION_PROMPT = (
    "Your previous response was cut off at the output token limit, so it was discarded and "
    "nothing in it was run or written. Continue in a smaller step: at most one or two tool calls, "
    "one file per call, and a large file in several modify_file edits rather than one write."
)


def run_lab_loop(*, client, executor: LabToolExecutor, system: str, messages: list[dict],
                 limits: LabLimits, callbacks: LabCallbacks, finish_tool: str | None = None,
                 started_at: float | None = None) -> LoopResult:
    """Run the model with ``executor``'s tools until it stops or a cap is reached.

    ``messages`` is the conversation so far (at least the user's message) and is
    extended in place, so a caller can continue it with another message. The cost
    cap applies to ``client.usage.estimated_cost`` and the runtime cap counts from
    ``started_at`` (``time.monotonic()``, default now): loops that share a client
    and a start time share both budgets. ``LoopResult.cost_usd`` is this loop's
    own spend. With ``finish_tool`` the loop ends as ``completed`` right after a
    successful call of that tool; a turn calling it runs even past the cost cap
    (a refused call then stops the loop on the cap). ``InvalidApiKeyError`` propagates; any other
    provider error ends the loop as ``api_error``.
    """
    started = time.monotonic() if started_at is None else started_at
    cost_before = client.usage.estimated_cost
    result = LoopResult(stop_reason="max_turns")
    # A caller that leaves on_text at its no-op default does not want text.
    on_text = None if callbacks.on_text is LabCallbacks.on_text else callbacks.on_text
    truncations = 0

    def finish(stop_reason: str, detail: str = "") -> LoopResult:
        result.stop_reason, result.detail = stop_reason, detail
        result.cost_usd = client.usage.estimated_cost - cost_before
        if detail:
            logger.warning("Lab loop stopped (%s): %s", stop_reason, detail)
        return result

    for turn in range(1, limits.max_turns + 1):
        if not callbacks.should_continue():
            return finish("cancelled")
        if time.monotonic() - started >= limits.max_runtime_seconds:
            return finish("timeout", f"the run reached its {limits.max_runtime_seconds:.0f} s limit")
        if client.usage.estimated_cost >= limits.max_cost_usd:
            return finish("cost_cap", f"the run reached its ${limits.max_cost_usd:.2f} cost limit")
        _compact(messages, client, executor, result.tool_calls)
        try:
            response = _call_model(client, system, messages, executor.schemas, on_text)
        except InvalidApiKeyError:
            raise
        except Exception as exc:
            return finish("api_error", str(exc))
        result.turns = turn
        stop_reason = response.get("stop_reason")
        content = response.get("content") or []

        if stop_reason in ("max_tokens", "length"):
            if truncations >= MAX_TRUNCATION_RETRIES:
                return finish("api_error", f"the model hit its output token limit on {truncations + 1} turns")
            truncations += 1
            messages.append({"role": "user", "content": TRUNCATION_PROMPT})
            continue
        if stop_reason == "refusal":
            return finish("api_error", "the model declined this request")

        blocks = _tool_use_blocks(content)
        # A turn that calls finish_tool is run whatever it cost: it is paid for, and the loop ends after it.
        finishing = finish_tool is not None and any(getattr(b, "name", None) == finish_tool for b in blocks)
        if blocks and not finishing and client.usage.estimated_cost > limits.max_cost_usd:
            # The calls are not run, so the reply is not kept: a call without its result is invalid.
            return finish("cost_cap", f"the run reached its ${limits.max_cost_usd:.2f} cost limit")
        text = _text_of(content)
        if text:
            result.final_text = text
        if content:
            messages.append({"role": "assistant", "content": content})
        if not blocks:
            if stop_reason in ("end_turn", "stop_sequence"):
                return finish("completed")
            return finish("api_error", f"unexpected stop reason {stop_reason!r}")

        tool_results, finished = _run_tools(blocks, executor, callbacks, turn, result.tool_calls, finish_tool)
        messages.append({"role": "user", "content": tool_results})
        if finished:
            return finish("completed")
    return finish("max_turns", f"the run used all {limits.max_turns} turns")


def _call_model(client, system: str, messages: list[dict], tools: list[dict], on_text) -> dict:
    """One model call: streamed when the client streams and someone wants the text."""
    if on_text is None or not hasattr(client, "chat_stream"):
        return client.chat(system=system, messages=messages, tools=tools)
    for event in client.chat_stream(system=system, messages=messages, tools=tools):
        if event.get("type") == "text_delta":
            on_text(event.get("text", ""))
        elif event.get("type") == "message_done":
            return event
    raise RuntimeError("the model's stream ended without a final message")


def _run_tools(blocks: list, executor: LabToolExecutor, callbacks: LabCallbacks, turn: int,
               log: list[dict[str, Any]], finish_tool: str | None) -> tuple[list[dict], bool]:
    """The tool results of one turn, in the model's order, and whether ``finish_tool`` succeeded.

    Consecutive read calls run in parallel; every other call runs alone, in order,
    so a read the model placed after a write sees that write.
    """
    detail_count = len(blocks)
    outcomes: list[tuple[str, bool]] = []
    pending_reads: list = []

    def run(block) -> tuple[str, bool]:
        name = getattr(block, "name", None) or ""
        arguments = getattr(block, "input", None)
        callbacks.on_progress(turn, name, "executing", _tool_call_detail(name, arguments, detail_count))
        text, is_error = executor.execute(name, arguments)
        callbacks.on_progress(turn, name, "error" if is_error else "done", None)
        return text, is_error

    def flush_reads() -> None:
        if len(pending_reads) == 1:
            outcomes.append(run(pending_reads[0]))
        elif pending_reads:
            with ThreadPoolExecutor(max_workers=_MAX_PARALLEL_WORKERS) as pool:
                outcomes.extend(pool.map(run, pending_reads))
        pending_reads.clear()

    for block in blocks:
        if getattr(block, "name", None) in READ_TOOLS:
            pending_reads.append(block)
            continue
        flush_reads()
        outcomes.append(run(block))
    flush_reads()

    finished = False
    results = []
    for block, (text, is_error) in zip(blocks, outcomes):
        name = getattr(block, "name", None) or ""
        log.append({"turn": turn, "tool": name, "input": _trace_safe_input(name, getattr(block, "input", None)),
                    "success": not is_error})
        finished = finished or (name == finish_tool and not is_error)
        results.append({"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error})
    return results, finished


def _compact(messages: list[dict], client, executor: LabToolExecutor, tool_calls: list[dict]) -> None:
    """Summarize older turns in place once the history nears the model's window.

    The first message, the user's request, is kept word for word in front of the summary.
    """
    reserve = max(COMPACT_RESERVE_TOKENS, int(getattr(client, "max_tokens", 0) or 0))
    compacted, did_compact = maybe_compact(messages=messages, tool_calls_log=tool_calls,
                                           output_dir=executor.workspace,
                                           model=getattr(client, "model", None), reserve=reserve)
    if not did_compact:
        return
    # After an earlier compaction the first message also holds that summary; the new one replaces it.
    request = [block for block in _text_blocks(messages[0]["content"])
               if not str(_block_text(block)).startswith(_SUMMARY_PREFIX)]
    messages[:] = [{"role": "user", "content": request + _text_blocks(compacted[0]["content"])},
                   *compacted[1:]]
    # Earlier results are gone from the conversation: a re-read must show the text again.
    executor.forget_shown_results()
    logger.info("Lab loop compacted its history to %d messages", len(messages))


def _text_blocks(content: Any) -> list:
    """A message's content as a list of blocks."""
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content)


def _block_text(block: Any) -> str:
    """The text of a dict or SDK text block, or ""."""
    return block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")


def _text_of(content: list) -> str:
    """The text blocks of a reply, joined."""
    return "\n".join(getattr(block, "text", "") for block in content
                     if getattr(block, "type", None) == "text" and getattr(block, "text", "")).strip()
