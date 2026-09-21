"""Per-turn checkpoint persistence for crash-recovery.

After each completed customization/repair turn, the orchestrator calls
:func:`save_checkpoint` to write a snapshot of its mutable conversation
state to ``.besser_checkpoint.json`` in the output directory. If the
backend process crashes (OOM, container restart, Ctrl-C), the caller
can re-issue the generation against the same output dir and the
orchestrator picks up from the saved turn via :func:`load_checkpoint`.

What we save
------------

* The original request (``instructions``, ``primary_kind``, ``run_id``).
* The message list that's been accumulating in Phase 2. This is the
  heaviest payload — full conversation history including tool results.
  Assistant content arrives as provider SDK block objects, which are
  converted to their JSON wire shape by :func:`_to_wire` before the
  dump. That conversion is load-bearing: ``default=str`` used to
  stringify a ``tool_use`` block into its ``repr()``, which destroyed
  the block's ``id`` while the ``tool_result`` in the next message kept
  referencing it. Every resume from a checkpoint saved mid-tool-turn
  then sent the provider a ``tool_result`` with no matching
  ``tool_use`` and was rejected with a 400.
* Running costs, turn counter, compaction counter, tool-call log,
  validation issues, selected generator, inventory.
* A project fingerprint (hash of primary-model names + instruction
  prefix). The resume endpoint refuses to load a checkpoint if the
  fingerprint doesn't match — it's cheap protection against
  accidentally resuming a run against a completely different project.

What we don't save
------------------

* The LLM client — you need a valid API key to resume, so the resume
  endpoint rebuilds the client from the new request.
* The domain / gui / agent models themselves — those come from the
  request's ``ProjectInput`` on resume. We save only a fingerprint and
  let the caller re-assemble.

The file is designed to be durable: we write to a ``.tmp`` sidecar and
then ``os.replace`` so partial writes don't corrupt the checkpoint.
"""

from __future__ import annotations

import hashlib
import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = ".besser_checkpoint.json"
# Version 2 distinguishes a state-only repair checkpoint from a Phase 2
# conversation. Older runners must reject it, not replay repairs as Phase 2.
CHECKPOINT_SCHEMA_VERSION = 2
# Shared by snapshot persistence and source scans that must exclude old output.
_SNAPSHOT_DIR = ".besser_snapshot"


@dataclass
class Checkpoint:
    """In-memory representation of a saved run."""

    schema_version: int
    run_id: str
    instructions: str
    primary_kind: str
    turn: int                          # last completed turn in the saved phase
    total_turns: int
    messages: list[dict]               # Phase 2 conversation only; empty for repair
    tool_calls_log: list[dict]
    validation_issues: list[dict]
    inventory: str
    generator_used: str | None
    estimated_cost_usd: float
    compaction_count: int
    project_fingerprint: str           # see ``compute_fingerprint``
    saved_at: float                    # unix ts
    # Serialized executor checklist. Added compatibly to schema v1 so older
    # checkpoints load with an empty list and newer resumes retain their gate.
    tasks: list[dict] = field(default_factory=list)
    # Scenario definitions survive crashes; reports are deliberately not trusted
    # on resume and are re-established against a fresh runtime copy.
    api_scenarios: list[dict] = field(default_factory=list)
    phase: str = "phase2"
    source_revision: str = ""
    phase2_stop_reason: str = "max_turns"
    phase2_exited_cleanly: bool = False
    # Scheduling/progress only, never cached validation or API success.
    repair_progress: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "instructions": self.instructions,
            "primary_kind": self.primary_kind,
            "turn": self.turn,
            "total_turns": self.total_turns,
            "messages": self.messages,
            "tool_calls_log": self.tool_calls_log,
            "validation_issues": self.validation_issues,
            "inventory": self.inventory,
            "generator_used": self.generator_used,
            "estimated_cost_usd": self.estimated_cost_usd,
            "compaction_count": self.compaction_count,
            "project_fingerprint": self.project_fingerprint,
            "saved_at": self.saved_at,
            "tasks": self.tasks,
            "api_scenarios": self.api_scenarios,
            "phase": self.phase,
            "source_revision": self.source_revision,
            "phase2_stop_reason": self.phase2_stop_reason,
            "phase2_exited_cleanly": self.phase2_exited_cleanly,
            "repair_progress": self.repair_progress,
        }


def api_scenario_snapshot(records, *, include_reports: bool = False) -> list[dict]:
    """Bounded, detached workflow definitions for checkpoints and recipes."""
    from besser.spec_driven_agent.validation.api_probe import _validate_requests

    records = list(records)
    if len(records) > 10:
        raise ValueError("At most ten retained API scenarios may be saved/restored")
    output, identifiers = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid saved API scenario record")
        identifier, scenario = record.get("scenario_id"), record.get("scenario")
        if (not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 80
                or identifier in identifiers or not isinstance(scenario, dict)):
            raise ValueError("Saved API scenarios need distinct nonempty IDs and declarative definitions")
        _validate_requests(scenario.get("requests"))
        backend = scenario.get("backend")
        if backend is not None and (
            not isinstance(backend, str) or backend.startswith(("/", "\\"))
            or ":" in backend or ".." in backend.replace("\\", "/").split("/")
        ):
            raise ValueError("Saved API scenario backend must be an in-workspace relative path")
        identifiers.add(identifier)
        saved = {"scenario_id": identifier,
                 "scenario": copy.deepcopy({"requests": scenario["requests"], "backend": backend})}
        history = record.get("correction_history", [])
        if not isinstance(history, list):
            raise ValueError("Saved API correction_history must be a list")
        history = list(history)
        correction = (record.get("report") or {}).get("correction")
        if correction and (not history or history[-1] != correction):
            history.append(correction)
        saved["correction_history"] = [
            {"reason": str(item.get("reason", ""))[:1000],
             "previous_status": str(item.get("previous_status", ""))[:40]}
            for item in history[-10:] if isinstance(item, dict)
        ]
        if include_reports:
            saved["report"] = copy.deepcopy(record.get("report") or {"status": "unverified"})
        output.append(saved)
    return output


def restore_api_scenarios(entries: list[dict]) -> dict[str, dict]:
    """Validate definitions, never restore a cached success as fresh evidence."""
    if not isinstance(entries, list):
        raise ValueError("Saved api_scenarios must be a list")
    restored = {}
    for saved in api_scenario_snapshot(entries):
        restored["named:" + saved["scenario_id"]] = {
            **saved, "revision": None,
            "report": {"status": "unverified", "boot": "not_checked",
                       "error": "Restored API workflow must be rerun against the current source."},
        }
    return restored


def compute_fingerprint(
    instructions: str,
    primary_kind: str,
    domain_model: Any | None = None,
    state_machines: list[Any] | None = None,
    gui_model: Any | None = None,
    agent_model: Any | None = None,
    object_model: Any | None = None,
    quantum_circuit: Any | None = None,
    bpmn_model: Any | None = None,
    nn_model: Any | None = None,
) -> str:
    """Build a stable fingerprint for checkpoint validation.

    The fingerprint covers the things that, if they changed between the
    original run and the resume, would make resumption semantically
    wrong: the instructions, the primary kind, and the set of model
    "identities" (class names, state machine names, etc.). It does NOT
    include full model content — a user who edits an attribute should
    still be able to resume; we're guarding against *whole project
    swaps*, not every possible model edit.
    """
    parts: list[str] = [
        # bounded: fingerprint; guards against whole-project swaps, not edits
        instructions.strip()[:500],
        primary_kind,
    ]

    if domain_model is not None:
        try:
            class_names = sorted(
                c.name for c in domain_model.get_classes() if getattr(c, "name", None)
            )
            parts.append("classes=" + ",".join(class_names))
        except Exception:
            parts.append("classes=<unreadable>")
    if state_machines:
        parts.append(
            "sms=" + ",".join(sorted(
                getattr(sm, "name", "?") or "?" for sm in state_machines
            ))
        )
    if gui_model is not None:
        parts.append("gui=present")
    if agent_model is not None:
        parts.append("agent=present")
    if object_model is not None:
        parts.append("object=present")
    if quantum_circuit is not None:
        parts.append("quantum=present")
    if bpmn_model is not None:
        parts.append(f"bpmn={getattr(bpmn_model, 'name', None) or 'present'}")
    if nn_model is not None:
        parts.append(f"nn={getattr(nn_model, 'name', None) or 'present'}")

    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:32]



def _to_wire(obj: Any) -> Any:
    """Convert provider SDK content blocks into their JSON wire shape.

    The orchestrator stores ``response["content"]`` verbatim, so a checkpoint
    holds whatever objects the provider SDK returned. Those are not
    JSON-serializable, and stringifying them loses the ``id`` that ties a
    ``tool_use`` to the ``tool_result`` that follows it.

    Falls back to ``str`` only for genuinely opaque values, which keeps
    :func:`save_checkpoint` from ever raising.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_wire(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_wire(v) for v in obj]

    # Anthropic/OpenAI SDK blocks are pydantic models.
    dump = getattr(obj, "model_dump", None) or getattr(obj, "dict", None)
    if callable(dump):
        try:
            return {k: _to_wire(v) for k, v in dump().items() if v is not None}
        except Exception:
            pass
    if callable(getattr(obj, "to_dict", None)):
        try:
            return {k: _to_wire(v) for k, v in obj.to_dict().items() if v is not None}
        except Exception:
            pass
    # Plain objects that still look like a block (our test doubles, and
    # any provider shim that isn't pydantic).
    attrs = getattr(obj, "__dict__", None)
    if attrs and "type" in attrs:
        return {k: _to_wire(v) for k, v in attrs.items()
                if not k.startswith("_") and v is not None}
    # Our provider-agnostic blocks use __slots__, not __dict__. Stringifying
    # them dropped every assistant tool call in run f6770633's checkpoint.
    block_fields = {
        "text": ("type", "text"),
        "tool_use": ("type", "id", "name", "input"),
    }.get(getattr(obj, "type", None))
    if block_fields and all(hasattr(obj, key) for key in block_fields):
        return {key: _to_wire(getattr(obj, key)) for key in block_fields}
    return str(obj)

def save_checkpoint(
    output_dir: str,
    checkpoint: Checkpoint,
) -> str | None:
    """Write ``checkpoint`` to ``output_dir`` atomically.

    Returns the absolute path on success, ``None`` on failure.
    Failures are logged at debug level and never raised — a checkpoint
    write failing shouldn't take down an otherwise-healthy run.
    """
    final_path = os.path.join(output_dir, CHECKPOINT_FILENAME)
    tmp_path = final_path + ".tmp"
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as fh:
            # default=str is only a last resort, so a write never raises.
            json.dump(_to_wire(checkpoint.to_dict()), fh, default=str, indent=2)
        os.replace(tmp_path, final_path)
        return final_path
    except Exception as exc:
        logger.debug("Failed to save checkpoint to %s: %s", final_path, exc)
        # Best-effort cleanup of the half-written sidecar.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None


def load_checkpoint(output_dir: str) -> Checkpoint | None:
    """Read and parse a checkpoint file. Returns ``None`` if missing
    or unreadable or from an incompatible schema version.
    """
    path = os.path.join(output_dir, CHECKPOINT_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        logger.warning("Failed to read checkpoint at %s: %s", path, exc)
        return None

    if data.get("schema_version") not in {1, CHECKPOINT_SCHEMA_VERSION}:
        logger.warning(
            "Checkpoint at %s has schema version %s, expected %s — "
            "refusing to load to avoid silent corruption",
            path, data.get("schema_version"), CHECKPOINT_SCHEMA_VERSION,
        )
        return None

    try:
        phase = data.get("phase", "phase2")
        if phase not in {"phase2", "phase3"}:
            raise ValueError("Unknown checkpoint phase")
        if data["schema_version"] == 1 and phase != "phase2":
            raise ValueError("Legacy checkpoints cannot contain repair-phase state")
        progress = data.get("repair_progress", {})
        revision = data.get("source_revision", "")
        if not isinstance(progress, dict) or not isinstance(revision, str):
            raise ValueError("Invalid repair progress or source revision")
        for key in ("attempts_run", "no_progress_streak"):
            if type(progress.get(key, 0)) is not int or progress.get(key, 0) < 0:
                raise ValueError("Invalid repair progress counter")
        states = progress.get("seen_states", [])
        if not isinstance(states, list) or any(
            not isinstance(item, list) or len(item) != 3
            or not isinstance(item[0], str) or not isinstance(item[1], str)
            or not isinstance(item[2], list)
            or any(not isinstance(message, str) for message in item[2])
            for item in states
        ):
            raise ValueError("Invalid repair progress states")
        return Checkpoint(
            schema_version=data["schema_version"],
            run_id=data.get("run_id", ""),
            instructions=data["instructions"],
            primary_kind=data.get("primary_kind", "class"),
            turn=int(data.get("turn", 0)),
            total_turns=int(data.get("total_turns", 0)),
            messages=data.get("messages") or [],
            tool_calls_log=data.get("tool_calls_log") or [],
            validation_issues=data.get("validation_issues") or [],
            inventory=data.get("inventory", ""),
            generator_used=data.get("generator_used"),
            estimated_cost_usd=float(data.get("estimated_cost_usd", 0.0)),
            compaction_count=int(data.get("compaction_count", 0)),
            project_fingerprint=data.get("project_fingerprint", ""),
            saved_at=float(data.get("saved_at", 0.0)),
            tasks=data.get("tasks") or [],
            api_scenarios=data.get("api_scenarios", []),
            phase=phase,
            source_revision=revision,
            phase2_stop_reason=str(data.get("phase2_stop_reason", "max_turns")),
            phase2_exited_cleanly=data.get("phase2_exited_cleanly") is True,
            repair_progress=progress,
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Checkpoint at %s has unexpected shape: %s", path, exc)
        return None


def delete_checkpoint(output_dir: str) -> None:
    """Remove the checkpoint file after a successful run.

    Called on the happy path so a future re-invocation against the same
    output dir doesn't accidentally resume a completed run. Never
    raises.
    """
    path = os.path.join(output_dir, CHECKPOINT_FILENAME)
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception as exc:
        logger.debug("Failed to delete checkpoint at %s: %s", path, exc)
