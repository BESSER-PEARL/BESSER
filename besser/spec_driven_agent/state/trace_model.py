"""The run trace: what a Spec-Driven run did, as a B-UML traceability ``TraceModel``.

A run keeps three records of itself. They have three names, and only this one is a
:class:`~besser.BUML.metamodel.traceability.TraceModel`:

* the **run log** -- ``.besser_trace.jsonl`` (:mod:`.tracing`, ``TraceWriter``): JSON lines of
  phases, turns, tool calls, token counters, generator outputs and findings, for debugging and
  as the source of this module;
* the **Agent Trace record** -- ``.agent-trace/<run id>.json``, kept up to date in
  ``.besser_attribution.json`` (:mod:`.agent_trace`): who wrote each line of the delivered files,
  in the `agent-trace.dev <https://agent-trace.dev/>`_ format;
* the **run trace** -- this module: a fragment of the project's trace (``projectId`` None)
  that the editor merges into the project trace, as it merges the fragment of a traced
  generation.

:func:`build_run_trace_model` assembles the TraceModel at the end of a run from what the run
persisted in its workspace (the run log, the recipe, the attribution state, the delivered files),
so it needs nothing threaded through the orchestrator and works the same after ``resume()``: the
run log is append-only and every record carries the run id. How a run maps onto the metamodel:

* One :class:`CodeGeneration` per run (id ``spec-driven:<run id>``), performed by the
  Spec-Driven :class:`Agent` (and the :class:`Person` who started it, when known). It reads the
  diagrams (one root :class:`ModelArtifact` each: the diagram id as ``element_id``, the B-UML
  metaclass as ``element_type``), the instructions (:class:`PromptArtifact`) and the requirements
  the editor sent, and writes every version of every file the run produced.
* Its tool calls, in time order: one :class:`CodeGenerator` call per generator run (Phase 1, a
  generator tool the LLM called, a modify run's regeneration), with :class:`RuleEvidence` links
  from the models it read to its files and, where the generator records ``traced_files``, from
  elements to line ranges; one :class:`LLM` call per agent turn (Phases 2 and 3) with the tokens
  of that turn (the difference of the client's cumulative counters around it); one LLM call for
  the calls made outside the turns of a phase (generator selection, planning, the judge) when
  they used tokens; one LLM call holding the judge's verdicts. A turn's tool calls, validation
  findings and gate outcomes are :class:`LogRecord` messages (bounded and redacted) of the call
  that ran or preceded them.
* A file the LLM wrote is a :class:`CodeArtifact` linked from the prompt with
  :class:`AIEvidence`, at the line ranges the attribution gives to this run's LLM; a generated
  file the LLM then edited is a new version whose ``revision_of`` is the generator's version
  (which keeps the generator's links).
* Requirements: those the editor sent are referenced by their editor id; requirements the run
  extracted itself are written by a :class:`RequirementExtraction` (id
  ``spec-driven:<run id>:requirements``) with ids ``<run id>:R<n>``, linked from the prompt.
* The judge's verdicts: a requirement found implemented or partly implemented is linked to the
  cited code (the cited line when it is found, else the file) with :class:`AIEvidence` resting on
  the verdict's log record and a :class:`Confidence` (1 for implemented, 0.5 for partial) whose
  explanation is the verdict. A requirement found missing, or that no verdict settled, gets no
  link: a requirement without a link is one the code does not realize (as a model element's), and
  the verdict's log record says why.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import posixpath
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

from besser.BUML.metamodel.traceability import (
    AIEvidence, Agent, Artifact, CodeArtifact, CodeGeneration, CodeGenerator, Confidence, ExecutionStatus, LLM,
    LogRecord, LogType, ModelArtifact, Person, PromptArtifact, RequirementArtifact, RequirementExtraction,
    RequirementKind, RuleEvidence, Tool, ToolCall, TraceLink, TraceModel,
)
from besser.generators.registry import GENERATORS
from besser.spec_driven_agent.state import agent_trace
from besser.spec_driven_agent.state.tracing import (
    EVENT_ERROR, EVENT_GENERATOR_OUTPUT, EVENT_PHASE_ENTER, EVENT_PHASE_EXIT, EVENT_REQUIREMENTS, EVENT_ROLLBACK,
    EVENT_RUN_END, EVENT_TOOL_CALL, EVENT_TURN_START, EVENT_VALIDATION_ISSUE, TRACE_FILENAME,
)
from besser.utilities.generation_trace import (
    MAX_TRACED_REGIONS, ContentBudget, ElementArtifacts, ElementRef, FileTrace, InputModel, prefixed_traces,
)
from besser.utilities.secret_redaction import redact_trace_text
from besser.utilities.workspace_fs import open_plain

logger = logging.getLogger(__name__)

SPEC_DRIVEN_AGENT_NAME = agent_trace.TOOL_NAME
"""The Agent a run is performed by (the editor records runs under the same name)."""

TRACE_MODEL_FILENAME = ".besser_trace_model.json"
"""The run trace as trace-file JSON, delivered in the ZIP beside the Agent Trace record."""

MODEL_KINDS = ("domain", "gui", "agent", "object", "bpmn", "nn", "quantum", "state_machine")
"""The kinds of model a run reads; the first seven are the names generator tools require them by
(``AgentTool.requires`` in ``besser/generators/registry.py``)."""

# A log record is a message, not a transcript: long enough for a finding or a tool call's
# arguments, short enough that a run of hundreds of turns stays a small fragment.
MAX_LOG_CHARS = 500
# Per tool call; the rest are counted in one closing record.
MAX_LOGS_PER_CALL = 40
# A file the LLM rewrote line by line can split into many runs; above this the link is file level.
MAX_AI_RANGES_PER_FILE = 50
# The regions one generator call logs: the cap of a traced generation (a large web app has
# thousands); the rest stay file level.
MAX_REGIONS_PER_GENERATOR_CALL = MAX_TRACED_REGIONS

# Confidence of a judge's verdict link: it read the code and its citation was re-checked.
_VERDICT_CONFIDENCE = {"implemented": 1.0, "partial": 0.5}
_TOKEN_KEYS = ("api_calls", "input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens")
_WRITE_TOOLS = frozenset({"write_file", "modify_file", "replace_file_lines", "delete_file"})
_BODY_ARGUMENTS = frozenset({"content", "old_text", "new_text", "new_content", "lines"})
# Validation findings of the run's end (the recipe's), logged on the run's last call: blockers first.
MAX_FINDINGS = 40


def run_operation_id(run_id: str) -> str:
    """The id of a run's CodeGeneration (the editor records a run without a trace under it too)."""
    return f"spec-driven:{run_id}"


def requirement_trace_id(run_id: str, requirement: Mapping[str, Any]) -> str:
    """The RequirementArtifact id of a ledger item or verdict: the editor's id (``ref``) for a
    requirement the editor sent, ``<run id>:R<n>`` for one the run extracted."""
    ref = requirement.get("ref")
    return str(ref) if ref else f"{run_id}:R{requirement.get('id')}"


def traced_requirement_id(run_id: str, requirement: Mapping[str, Any]) -> Optional[str]:
    """:func:`requirement_trace_id` of a requirement the run's trace records, None for a ledger item
    that is not a requirement (an incomplete-extraction marker, an entity-coverage check)."""
    try:
        RequirementKind(requirement.get("kind"))
    except ValueError:
        return None
    return requirement_trace_id(run_id, requirement)


@dataclass(frozen=True)
class RunModel:
    """A model the run read: its kind (one of :data:`MODEL_KINDS`) and how a trace records it."""
    kind: str
    model: InputModel


# ---------------------------------------------------------------------------------------------
# Generator output, as the run log records it
# ---------------------------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snippet_hash(lines: list[str], start: int, end: int) -> str:
    return _sha256("\n".join(lines[start - 1:end]).encode("utf-8"))


def _read_bytes(workspace: str, rel: str) -> Optional[bytes]:
    try:
        with open_plain(os.path.join(workspace, rel), "rb", root=workspace) as handle:
            return handle.read()
    except OSError:
        return None


def _element_ref(ref: Mapping[str, Any]) -> ElementRef:
    """The run log's record of an element (``element_id``, ``element_type``, ``element_name``)."""
    return ElementRef(ref["element_id"], ref.get("element_type") or "Element", ref.get("element_name") or ref["element_id"])


def _ref_dict(element_id: str, element_type: str, element_name: str) -> dict:
    return {"element_id": element_id, "element_type": element_type, "element_name": element_name}


def generator_output_record(workspace: str, output_dir: str, written: Iterable[str],
                            traced_files: Mapping[str, FileTrace], tool: str, started_at: float) -> dict:
    """The run-log payload of one generator call (``EVENT_GENERATOR_OUTPUT``).

    Args:
        workspace: The run's workspace.
        output_dir: The folder the generator wrote into (its ``traced_files`` paths are relative to it).
        written: The files the call wrote, workspace-relative and ``/``-separated.
        traced_files: The generator's ``traced_files``.
        tool: The generator tool (``generate_web_app``) or generator key.
        started_at: When the call started (epoch seconds).

    Each file gets its SHA-256 and, from ``traced_files``, the regions (with the hash of their
    lines) and whole-file sources, so the trace keeps the generator's version even after the
    LLM edits the file.
    """
    spec = next((s for s in GENERATORS if s.agent_tool is not None and s.agent_tool.name == tool), None)
    if spec is None:
        spec = next((s for s in GENERATORS if s.id == tool), None)
    prefix = os.path.relpath(output_dir, workspace).replace(os.sep, "/")
    traces = prefixed_traces(traced_files or {}, "" if prefix == "." else prefix)
    budget = MAX_REGIONS_PER_GENERATOR_CALL
    files = []
    for rel in sorted(set(written)):
        data = _read_bytes(workspace, rel)
        if data is None:
            continue
        entry: dict[str, Any] = {"path": rel, "sha256": _sha256(data)}
        trace = traces.get(rel)
        if trace is not None:
            try:
                lines = data.decode("utf-8").split("\n")
            except UnicodeDecodeError:
                lines = None
            if lines is not None and (trace.regions or trace.sources):
                regions = []
                for region in trace.regions[:max(budget, 0)]:
                    owner = region.owner
                    regions.append({
                        **_ref_dict(region.element_id, region.element_type, region.element_name),
                        "rule": region.rule, "start_line": region.start_line, "end_line": region.end_line,
                        "sha256": _snippet_hash(lines, region.start_line, region.end_line),
                        "owner": _ref_dict(owner.element_id, owner.element_type, owner.element_name) if owner else None,
                    })
                if len(trace.regions) > budget:
                    logger.warning("Generator %s traced more than %d regions; %s keeps file-level links only",
                                   tool, MAX_REGIONS_PER_GENERATOR_CALL, rel)
                budget -= len(regions)
                entry["regions"] = regions
                entry["sources"] = [{**_ref_dict(s.element.element_id, s.element.element_type,
                                                 s.element.element_name), "rule": s.rule} for s in trace.sources]
        files.append(entry)
    return {
        "tool": tool,
        "generator": spec.id if spec else tool,
        "reads": sorted(spec.agent_tool.requires) if spec and spec.agent_tool else [],
        "started": started_at,
        "files": files,
    }


# ---------------------------------------------------------------------------------------------
# Reading back what the run persisted
# ---------------------------------------------------------------------------------------------


def read_run_log(workspace: str, run_id: str) -> list[dict]:
    """The run log's records of ``run_id``, in order. A seeded workspace also holds the records
    of the run it was seeded from; a resumed run keeps its id, so its records are all here."""
    path = os.path.join(workspace, TRACE_FILENAME)
    records = []
    try:
        with open_plain(path, "r", root=workspace, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line torn by a crash
                if isinstance(record, dict) and record.get("run_id") == run_id and isinstance(
                        record.get("payload"), dict):
                    records.append(record)
    except OSError:
        return []
    return records


def _read_recipe(workspace: str) -> dict:
    try:
        with open_plain(os.path.join(workspace, ".besser_recipe.json"), "r", root=workspace,
                        encoding="utf-8") as handle:
            recipe = json.load(handle)
        return recipe if isinstance(recipe, dict) else {}
    except (OSError, ValueError):
        return {}


def _bounded(text: Any) -> str:
    text = redact_trace_text(str(text))
    return text if len(text) <= MAX_LOG_CHARS else text[:MAX_LOG_CHARS - 1] + "…"


def _when(epoch: Any) -> datetime:
    try:
        return datetime.fromtimestamp(float(epoch), timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return datetime.now(timezone.utc)


def _counters(usage: Any) -> Optional[dict]:
    if not isinstance(usage, dict):
        return None
    return {key: int(usage.get(key) or 0) for key in _TOKEN_KEYS}


def _tokens_read(delta: Mapping[str, int]) -> int:
    """Every token the LLM read: the counters keep fresh input apart from cache reads and writes."""
    return delta["input_tokens"] + delta["cache_creation_tokens"] + delta["cache_read_tokens"]


# ---------------------------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------------------------


@dataclass
class _Call:
    """A tool call being assembled: the metamodel's ToolCall checks its fields as they are set."""
    tool: Tool
    started: datetime
    ended: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.SUCCEEDED
    logs: list = field(default_factory=list)
    delta: Optional[dict] = None
    dropped_logs: int = 0
    # None keeps every message: the judge's call, whose records the verdict links rest on.
    log_limit: Optional[int] = MAX_LOGS_PER_CALL

    def log(self, content: Any, log_type: LogType, when: datetime) -> Optional[LogRecord]:
        if self.log_limit is not None and len(self.logs) >= self.log_limit:
            self.dropped_logs += 1
            return None
        record = LogRecord(_bounded(content), len(self.logs), log_type, timestamp=when)
        self.logs.append(record)
        return record

    def add(self, delta: dict) -> None:
        self.delta = dict(delta) if self.delta is None else {k: self.delta[k] + delta[k] for k in _TOKEN_KEYS}

    @property
    def used_llm(self) -> bool:
        return self.delta is not None and self.delta["api_calls"] > 0


class _RunTraceBuilder:
    def __init__(self, workspace: str, run_id: str, models: Sequence[RunModel], instructions: Optional[str],
                 provider: Optional[str], agent_version: Optional[str], actor_name: Optional[str],
                 excluded: Iterable[str], max_stored_content_bytes: int, max_total_content_bytes: Optional[int]):
        self.workspace, self.run_id, self.provider = workspace, run_id, (provider or "unknown")
        self.version = agent_version
        self.excluded = list(excluded)
        self.contents = ContentBudget(max_stored_content_bytes, max_total_content_bytes)
        self.agent = Agent(SPEC_DRIVEN_AGENT_NAME, agent_version)
        self.actors = [self.agent] + ([Person(actor_name)] if actor_name and actor_name.strip() else [])
        self.tools: dict[tuple, Tool] = {}
        self.artifacts: list[Artifact] = []
        self.reads: list[Artifact] = []
        self.writes: list[Artifact] = []
        self.links: list[tuple] = []
        self.calls: list[_Call] = []
        self.roots = [(m.kind, ModelArtifact(m.model.ref.element_name, m.model.ref.element_id,
                                             m.model.ref.element_type), m.model) for m in models]
        for _, root, _ in self.roots:
            self._add(root, read=True)
        self.elements = ElementArtifacts([root for _, root, _ in self.roots], [model for _, _, model in self.roots],
                                         lambda artifact: self._add(artifact, read=True))
        self.prompt: Optional[PromptArtifact] = None
        if instructions and instructions.strip():
            self.prompt = PromptArtifact("Spec-Driven instructions", instructions,
                                         _sha256(instructions.encode("utf-8")))
            self._add(self.prompt, read=True)
        # path -> generator versions in order: (artifact, logged file entry)
        self.generated: dict[str, list[tuple[CodeArtifact, dict]]] = {}
        self.llm_touched: set[str] = set()
        self.requirements: Optional[dict] = None
        self.extraction: Optional[_Call] = None
        self.served_model: Optional[str] = None

    # -- helpers ------------------------------------------------------------------------------

    def _add(self, artifact: Artifact, read: bool = False, write: bool = False) -> Artifact:
        self.artifacts.append(artifact)
        if read:
            self.reads.append(artifact)
        if write:
            self.writes.append(artifact)
        return artifact

    def _llm(self, model: Optional[str]) -> LLM:
        name = model or self.served_model or "unknown"
        key = ("llm", name)
        if key not in self.tools:
            self.tools[key] = LLM(name, self.provider)
        return self.tools[key]

    def _generator(self, name: str) -> CodeGenerator:
        key = ("generator", name)
        if key not in self.tools:
            self.tools[key] = CodeGenerator(name, self.version)
        return self.tools[key]

    def _algorithm(self, model: Optional[str]) -> str:
        return agent_trace.normalize_model_id(self.provider, model or self.served_model) or "LLM"

    def _link(self, source, target, evidence, confidence: Optional[Confidence], when: datetime):
        self.links.append((source, target, evidence, confidence, when))

    def _element(self, ref: Mapping[str, Any], owner: Optional[Mapping[str, Any]] = None) -> Optional[ModelArtifact]:
        if not ref.get("element_id"):
            return None
        return self.elements.artifact(_element_ref(ref), _element_ref(owner) if owner and owner.get("element_id")
                                      else None)

    # -- run log --------------------------------------------------------------------------------

    def replay(self, records: list[dict]) -> None:
        baseline: Optional[dict] = None
        carved = {key: 0 for key in _TOKEN_KEYS}
        phase = "run"
        turn: Optional[_Call] = None
        turn_label = ""
        segment_start: Optional[datetime] = None
        pending: list[tuple] = []

        def last_call() -> Optional[_Call]:
            return turn or (self.calls[-1] if self.calls else None)

        def note(content, log_type, when):
            call = last_call()
            if call is None:
                pending.append((content, log_type, when))
            else:
                call.log(content, log_type, when)

        def close_turn(when):
            nonlocal turn
            if turn is None:
                return
            turn.ended = when
            if turn.used_llm or turn.logs:  # a turn cut off before its LLM call is not a call
                if turn.delta is not None and turn.used_llm:
                    turn.logs.insert(0, LogRecord(_bounded(f"{turn_label}: {self._usage_text(turn.delta)}"),
                                                  0, LogType.INFO, timestamp=turn.started))
                self.calls.append(turn)
            turn = None

        for record in records:
            event, payload, when = record.get("event"), record["payload"], _when(record.get("ts"))
            usage = _counters(payload.get("usage"))
            if usage is not None:
                if isinstance(payload["usage"].get("model"), str):
                    self.served_model = payload["usage"]["model"]
                delta = None
                if baseline is not None and all(usage[k] >= baseline[k] for k in _TOKEN_KEYS):
                    delta = {k: max(usage[k] - baseline[k] - carved[k], 0) for k in _TOKEN_KEYS}
                # A counter that went down is a new client: a resumed run starts from zero.
                if delta is not None:
                    if turn is not None:
                        turn.add(delta)
                        turn.tool = self._llm(self.served_model)  # the model that served it, after any switch
                    elif delta["api_calls"] > 0:
                        outside = _Call(self._llm(self.served_model), segment_start or when, when, delta=delta)
                        outside.log(f"LLM calls of {phase} outside the agent's turns (generator selection, "
                                    f"planning, judging): {self._usage_text(delta)}", LogType.INFO, when)
                        self.calls.append(outside)
                baseline, carved, segment_start = usage, {key: 0 for key in _TOKEN_KEYS}, when
                close_turn(when)
            if event == EVENT_PHASE_ENTER:
                phase = str(payload.get("phase") or phase)
            elif event == EVENT_TURN_START:
                close_turn(when)
                turn_phase = str(payload.get("phase") or phase)
                turn_label = f"Turn {payload.get('turn')} ({turn_phase})"
                turn = _Call(self._llm(self.served_model), when)
            elif event == EVENT_TOOL_CALL:
                if payload.get("success") and payload.get("tool") in _WRITE_TOOLS:
                    path = (payload.get("input") or {}).get("path") if isinstance(payload.get("input"), dict) else None
                    if isinstance(path, str):
                        self.llm_touched.add(_relpath(path))
                arguments = json.dumps(_without_bodies(payload.get("input")), default=str, sort_keys=True)
                note(f"{payload.get('tool')} -> {payload.get('status') or ('ok' if payload.get('success') else 'failed')}"
                     f": {arguments}", LogType.TOOL_CALL, when)
            elif event == EVENT_GENERATOR_OUTPUT:
                self._generator_call(payload, when)
            elif event == EVENT_REQUIREMENTS:
                extraction = payload.get("extraction_usage")
                if isinstance(extraction, dict):
                    delta = {k: int(extraction.get(k) or 0) for k in _TOKEN_KEYS}
                    carved = {k: carved[k] + delta[k] for k in _TOKEN_KEYS}
                    self.extraction = _Call(self._llm(payload.get("model")), when, when, delta=delta)
                    self.extraction.log(f"Requirements extracted from the instructions: {self._usage_text(delta)}",
                                        LogType.INFO, when)
                self.requirements = payload
            elif event == EVENT_VALIDATION_ISSUE:
                severity = payload.get("severity")
                log_type = LogType.INFO if severity in ("warning", "style") else LogType.ERROR
                note(f"Validation ({payload.get('phase')}): {payload.get('message')}", log_type, when)
            elif event == EVENT_ROLLBACK:
                note(f"Rollback ({payload.get('phase')}): {payload.get('reason') or ''}", LogType.INFO, when)
            elif event == EVENT_ERROR:
                if turn is not None:
                    turn.status = ExecutionStatus.FAILED
                note(f"Error ({payload.get('phase')}): {payload.get('reason')}", LogType.ERROR, when)
            elif event == EVENT_PHASE_EXIT:
                details = [f"{key}={payload[key]}" for key in ("stop_reason", "stop_detail", "exit_reason",
                                                                "unresolved_blockers", "not_confirmed",
                                                                "generator_used")
                           if payload.get(key) not in (None, "")]
                if details:
                    note(f"{payload.get('phase')} ended: {', '.join(map(str, details))}", LogType.INFO, when)
            elif event == EVENT_RUN_END:
                close_turn(when)
        close_turn(_when(records[-1].get("ts")) if records else datetime.now(timezone.utc))
        if pending and self.calls:
            first = min(self.calls, key=lambda c: c.started)
            for content, log_type, when in pending:
                first.log(content, log_type, when)

    @staticmethod
    def _usage_text(delta: Mapping[str, int]) -> str:
        return (f"{delta['api_calls']} call(s), {_tokens_read(delta)} input tokens (cache read "
                f"{delta['cache_read_tokens']}, cache write {delta['cache_creation_tokens']}), "
                f"{delta['output_tokens']} output tokens")

    def _generator_call(self, payload: dict, when: datetime) -> None:
        generator = str(payload.get("generator") or payload.get("tool") or "generator")
        started = _when(payload.get("started")) if payload.get("started") else when
        call = _Call(self._generator(generator), min(started, when), when)
        files = [f for f in payload.get("files") or [] if isinstance(f, dict) and f.get("path") and f.get("sha256")]
        call.log(f"{payload.get('tool') or generator} wrote {len(files)} file(s)", LogType.TOOL_RESULT, when)
        self.calls.append(call)
        kinds = set(payload.get("reads") or [])
        roots = [root for kind, root, _ in self.roots if kind in kinds] or [root for _, root, _ in self.roots]
        rule = f"generator:{generator}"
        for entry in files:
            path = entry["path"]
            versions = self.generated.setdefault(path, [])
            previous = versions[-1][0] if versions else None
            artifact = CodeArtifact(path, path, entry["sha256"], revision_of=previous, timestamp=when)
            self._add(artifact, write=True)
            versions.append((artifact, entry))
            for root in roots:
                self._link(root, artifact, RuleEvidence(rule, timestamp=when), Confidence(1.0, timestamp=when), when)
            for source in entry.get("sources") or []:
                element = self._element(source)
                if element is not None:
                    self._link(element, artifact, RuleEvidence(source.get("rule") or rule, timestamp=when),
                               Confidence(1.0, timestamp=when), when)
            for region in entry.get("regions") or []:
                element = self._element(region, region.get("owner"))
                if element is None:
                    continue
                fragment = CodeArtifact(f"{element.name} ({region['start_line']}-{region['end_line']})", path,
                                        region["sha256"], start_line=region["start_line"],
                                        end_line=region["end_line"], parent=artifact, timestamp=when)
                self._add(fragment, write=True)
                self._link(element, fragment, RuleEvidence(region.get("rule") or rule, timestamp=when),
                           Confidence(1.0, timestamp=when), when)

    # -- delivered files ------------------------------------------------------------------------

    def files(self, when: datetime) -> dict[str, CodeArtifact]:
        """The version of each delivered file this run produced, by path."""
        try:
            state = agent_trace.AttributionState.load(self.workspace)
        except agent_trace.AttributionStateError as exc:
            logger.warning("Run trace: the attribution state is unusable (%s); LLM edits are traced per file", exc)
            state = None
        this_run = set()
        if state is not None:
            this_run = {i for i, c in enumerate(state.contributors)
                        if c.get("kind") == agent_trace.KIND_AI and c.get("run_id") == self.run_id}
        final: dict[str, CodeArtifact] = {}
        for rel in agent_trace.iter_attributable_files(self.workspace, self.excluded):
            data = agent_trace.read_text_bytes(self.workspace, rel)
            if data is None:
                continue
            content = data.decode("utf-8")
            digest = _sha256(data)
            versions = self.generated.get(rel)
            generated = versions[-1][0] if versions else None
            if generated is not None and generated.content_hash == digest:
                generated.content = self.contents.keep(rel, content)
                final[rel] = generated
                continue
            lines = content.splitlines()
            ranges = None
            if state is not None:
                owners = state.owners_of(rel, lines)
                ranges = _runs(owners, this_run | {-1})
            if generated is None and rel not in self.llm_touched and not ranges:
                continue  # not written by this run (a seeded file left as it was)
            version = CodeArtifact(rel, rel, digest, content=self.contents.keep(rel, content),
                                   revision_of=generated, timestamp=when)
            self._add(version, write=True)
            final[rel] = version
            if self.prompt is None or (ranges is not None and not ranges):
                continue  # changed after the LLM, by redaction or a deterministic repair: no AI link
            targets: list[Artifact] = [version]
            if ranges and len(ranges) <= MAX_AI_RANGES_PER_FILE:
                targets = []
                for start, end in ranges:
                    fragment = CodeArtifact(f"{posixpath.basename(rel)} ({start}-{end})", rel,
                                            _snippet_hash(lines, start, end), start_line=start, end_line=end,
                                            parent=version, timestamp=when)
                    self._add(fragment, write=True)
                    targets.append(fragment)
            for target in targets:
                self._link(self.prompt, target, AIEvidence(self._algorithm(None), timestamp=when), None, when)
        return final

    # -- requirements -------------------------------------------------------------------------

    def requirement_artifacts(self, recipe: dict, when: datetime) -> tuple[dict[str, RequirementArtifact],
                                                                         Optional[RequirementExtraction]]:
        source = (self.requirements or {}).get("source") or recipe.get("requirements_source")
        items = (self.requirements or {}).get("items")
        if not isinstance(items, list):
            items = [v for v in recipe.get("requirements") or [] if isinstance(v, dict)]
        artifacts: dict[str, RequirementArtifact] = {}
        for item in items:
            if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                continue
            rid = traced_requirement_id(self.run_id, item)
            if rid is None or (source == "editor" and not item.get("ref")):
                continue
            text = str(item["text"])
            artifacts[rid] = RequirementArtifact(text, text, RequirementKind(item["kind"]), id=rid, timestamp=when)
        if not artifacts:
            return {}, None
        if source != "extracted":
            for artifact in artifacts.values():
                self._add(artifact, read=True)
            return artifacts, None
        for artifact in artifacts.values():
            self.artifacts.append(artifact)
        call = self.extraction or _Call(self._llm(None), when, when)
        extraction = RequirementExtraction(
            list(self.actors), id=f"{run_operation_id(self.run_id)}:requirements", started_at=call.started,
            ended_at=call.ended or call.started, tool_calls=[self._tool_call(call, 0)],
            reads=[self.prompt] if self.prompt else [], writes=list(artifacts.values()))
        if self.prompt is not None:
            for artifact in artifacts.values():
                self.links.append((self.prompt, artifact, AIEvidence(self._algorithm(None), timestamp=when), None,
                                   when, extraction))
        return artifacts, extraction

    def verdicts(self, recipe: dict, requirements: dict[str, RequirementArtifact],
                 files: dict[str, CodeArtifact], when: datetime) -> None:
        verdicts = [v for v in recipe.get("requirements") or [] if isinstance(v, dict)]
        if not verdicts:
            return
        judge = _Call(self._llm(None), when, when, log_limit=None)
        judge.log("The judge's verdicts on the requirements, read from the delivered code", LogType.INFO, when)
        self.calls.append(judge)
        for verdict in verdicts:
            status = str(verdict.get("status") or "unverified")
            evidence, note = str(verdict.get("evidence") or ""), str(verdict.get("note") or "")
            text = f"R{verdict.get('id')} {status}: {verdict.get('text')}"
            if evidence:
                text += f" | evidence: {evidence}"
            if note:
                text += f" | note: {note}"
            record = judge.log(text, LogType.OUTPUT_RESPONSE, when)
            requirement = requirements.get(requirement_trace_id(self.run_id, verdict))
            confidence = _VERDICT_CONFIDENCE.get(status)
            if requirement is None or confidence is None or record is None:
                continue  # unmet or unsettled: no link, the record says why
            target = self._cited(evidence, files, when)
            if target is None:
                judge.log(f"R{verdict.get('id')}: the cited code is not among this run's files", LogType.INFO, when)
                continue
            self._link(requirement, target, AIEvidence(self._algorithm(None), logs=[record], timestamp=when),
                       Confidence(confidence, _bounded(f"{status}: {note or evidence}"), timestamp=when), when)

    def _cited(self, evidence: str, files: dict[str, CodeArtifact], when: datetime) -> Optional[Artifact]:
        """The code a citation (``path: quoted line``) points at: the line in this run's version of the
        file when it is found there, else the file; None when no delivered file of this run matches."""
        path, _, quote = evidence.partition(":")
        path = _relpath(path.strip())
        if path in ("", "."):
            return None
        matches = [rel for rel in files if rel == path or rel.endswith("/" + path)]
        if len(matches) != 1:
            return None
        file = files[matches[0]]
        quote = " ".join(quote.split())
        if quote and file.content is not None:
            lines = file.content.splitlines()
            for number, line in enumerate(lines, start=1):
                if quote in " ".join(line.split()):
                    fragment = CodeArtifact(f"{posixpath.basename(file.path)} ({number}-{number})", file.path,
                                            _snippet_hash(lines, number, number), start_line=number,
                                            end_line=number, parent=file, timestamp=when)
                    self._add(fragment, write=True)
                    return fragment
        return file

    def findings(self, recipe: dict, when: datetime) -> None:
        """Log the run's final validation findings (the gates of Phase 3) on its last call."""
        issues = [i for i in recipe.get("validation_issues") or [] if isinstance(i, dict) and i.get("message")]
        if not issues or not self.calls:
            return
        last = max(self.calls, key=lambda c: c.started)
        issues.sort(key=lambda i: i.get("severity") != "blocker")
        if last.log_limit is not None:
            last.log_limit = len(last.logs) + MAX_FINDINGS + 1
        for issue in issues[:MAX_FINDINGS]:
            last.log(f"Validation ({issue.get('severity')}): {issue['message']}",
                     LogType.ERROR if issue.get("severity") == "blocker" else LogType.INFO, when)
        if len(issues) > MAX_FINDINGS:
            last.log(f"... {len(issues) - MAX_FINDINGS} more finding(s) in the run's recipe", LogType.INFO, when)

    # -- assembly -------------------------------------------------------------------------------

    def _tool_call(self, call: _Call, sequence: int) -> ToolCall:
        tokens = isinstance(call.tool, LLM) and call.used_llm
        logs = list(call.logs)
        if call.dropped_logs:
            logs.append(LogRecord(f"... {call.dropped_logs} more message(s) not kept", len(logs), LogType.INFO,
                                  timestamp=call.ended or call.started))
        for number, record in enumerate(logs):
            record.sequence_number = number
        ended = call.ended if call.ended is not None and call.ended >= call.started else call.started
        return ToolCall(call.tool, sequence, status=call.status, started_at=call.started, ended_at=ended, logs=logs,
                        input_tokens=_tokens_read(call.delta) if tokens else None,
                        output_tokens=call.delta["output_tokens"] if tokens else None)


def _without_bodies(arguments: Any) -> Any:
    """A tool call's arguments with file bodies replaced by their size: a log says what was
    done to which file, the file's version says what it holds."""
    if not isinstance(arguments, dict):
        return arguments
    return {key: f"<{len(value)} characters>" if key in _BODY_ARGUMENTS and isinstance(value, str) else value
            for key, value in arguments.items()}


def _relpath(path: str) -> str:
    """A workspace-relative path as the run's files are keyed: ``/``-separated, no leading ``./``."""
    path = posixpath.normpath(path.replace("\\", "/"))
    return path[2:] if path.startswith("./") else path


def _runs(owners: list[int], wanted: set[int]) -> list[tuple[int, int]]:
    """The 1-based inclusive line ranges whose owner is in ``wanted``."""
    ranges, start = [], None
    for number, owner in enumerate(owners, start=1):
        if owner in wanted and start is None:
            start = number
        elif owner not in wanted and start is not None:
            ranges.append((start, number - 1))
            start = None
    if start is not None:
        ranges.append((start, len(owners)))
    return ranges


def build_run_trace_model(
    workspace: str,
    *,
    run_id: str,
    models: Sequence[RunModel] = (),
    instructions: Optional[str] = None,
    provider: Optional[str] = None,
    agent_version: Optional[str] = None,
    actor_name: Optional[str] = None,
    status: ExecutionStatus = ExecutionStatus.SUCCEEDED,
    excluded: Iterable[str] = (),
    max_stored_content_bytes: int = 1_000_000,
    max_total_content_bytes: Optional[int] = None,
    ended_at: Optional[datetime] = None,
) -> Optional[TraceModel]:
    """The TraceModel fragment of run ``run_id``, from what it persisted in ``workspace``.

    Args:
        workspace: The run's workspace, after packaging reconciled the line attribution.
        run_id: The run (a resumed run keeps its id).
        models: The models the run read.
        instructions: The run's instructions (the prompt the code and requirements derive from).
        provider: The LLM provider (``openai``, ``anthropic``, ``free``, ...).
        agent_version: The BESSER version (the Agent's and the generators' version).
        actor_name: The person who started the run, when known.
        status: How the run ended.
        excluded: Directory names and file globs left out of the delivery (as packaging skips them).
        max_stored_content_bytes: A file's content is kept up to this size; above it only its hash is.
        max_total_content_bytes: The contents kept in all, in file order (None: no limit). A content
            kept has its credentials redacted (``generation_trace.ContentBudget``).
        ended_at: When the run ended (the last run-log record's time when None).

    Returns:
        The fragment (``project_id`` None), which passes ``TraceModel.validate()``; None when the run
        called neither a generator nor the LLM (nothing to trace: a run that failed before Phase 1).
    """
    records = read_run_log(workspace, run_id)
    recipe = _read_recipe(workspace)
    builder = _RunTraceBuilder(workspace, run_id, models, instructions, provider, agent_version, actor_name,
                               excluded, max_stored_content_bytes, max_total_content_bytes)
    builder.served_model = recipe.get("llm_model") if isinstance(recipe.get("llm_model"), str) else None
    builder.replay(records)
    end = ended_at or (_when(records[-1].get("ts")) if records else datetime.now(timezone.utc))
    start = _when(records[0].get("ts")) if records else end
    files = builder.files(end)
    builder.contents.report(f"The trace of run {run_id}")
    requirements, extraction = builder.requirement_artifacts(recipe, end)
    builder.findings(recipe, end)
    builder.verdicts(recipe, requirements, files, end)
    if not builder.calls:
        logger.info("Run %s called neither a generator nor the LLM: no trace", run_id)
        return None
    ordered = sorted(builder.calls, key=lambda c: c.started)
    operation = CodeGeneration(
        list(builder.actors), status=status, id=run_operation_id(run_id), started_at=min(start, ordered[0].started),
        ended_at=max(end, start), tool_calls=[builder._tool_call(c, n) for n, c in enumerate(ordered)],
        reads=builder.reads, writes=builder.writes)
    links = []
    for entry in builder.links:
        source, target, evidence, confidence, when = entry[:5]
        made_by = entry[5] if len(entry) > 5 else operation
        links.append(TraceLink(source, target, made_by, [evidence], confidence, timestamp=when))
    operations = [extraction, operation] if extraction is not None else [operation]
    called = {id(call.tool) for op in operations for call in op.tool_calls}
    trace = TraceModel(artifacts=builder.artifacts, operations=operations, trace_links=links, actors=builder.actors,
                       tools=[tool for tool in builder.tools.values() if id(tool) in called])
    trace.validate()
    return trace
