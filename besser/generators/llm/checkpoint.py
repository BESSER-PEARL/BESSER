"""Per-turn checkpoint persistence for crash-recovery.

After every successful Phase 2 turn, the orchestrator calls
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
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = ".besser_checkpoint.json"
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class Checkpoint:
    """In-memory representation of a saved run."""

    schema_version: int
    run_id: str
    instructions: str
    primary_kind: str
    turn: int                          # last completed Phase 2 turn
    total_turns: int
    messages: list[dict]               # Phase 2 conversation state
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
        }


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

    The orchestrator stores ``response["content"]`` verbatim in the
    message list, so a checkpoint holds whatever objects the provider
    SDK returned (``TextBlock`` / ``ToolUseBlock`` / thinking blocks —
    pydantic models on Anthropic, plain objects elsewhere). Those are
    not JSON-serializable, and stringifying them loses the ``id`` that
    ties a ``tool_use`` to the ``tool_result`` that follows it.

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
            # _to_wire turns SDK blocks into the dicts the provider APIs
            # accept, preserving tool_use ids so the tool_results that
            # follow them still resolve on resume. default=str stays only
            # as a last resort so a checkpoint write never raises.
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

    if data.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        logger.warning(
            "Checkpoint at %s has schema version %s, expected %s — "
            "refusing to load to avoid silent corruption",
            path, data.get("schema_version"), CHECKPOINT_SCHEMA_VERSION,
        )
        return None

    try:
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
