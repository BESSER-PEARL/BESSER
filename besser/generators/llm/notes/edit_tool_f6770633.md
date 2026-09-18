# Replay-safe file edits (EC2 run f6770633)

The run's turns 31–105 sent the same insertion 75 times. Its replacement
contained the entire search anchor. Every call therefore matched and reported
success, duplicating the price assignment and clearing the failure counters.

The fix excludes search anchors inside completed replacement regions. It is
content-based (works after restart), handles the existing indentation/numbered
quote ladder, and leaves pending locations editable. `replace_all` skips
completed sites. An already-applied edit is a non-writing error with an explicit
`already_applied` payload, so the existing repeat guard can force a task review,
freeze the file, then stop a stuck phase. A changed edit remains allowed.

Missing-anchor completion feedback additionally requires more than one non-blank
replacement line and a whole-line match. A short/common line found incidentally
must give the ordinary miss response, never tell the model its change is done.
This feedback gate is independent of retained-anchor replay protection. The
`TestUnseenFile` tests assert the status as well as the wording, covering `x`,
`pass`, and `import x`; checking only the error prefix missed this defect.

## Open-source reuse

Reviewed [OpenCode's edit tool](https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/tool/edit.ts)
and its read/write tools. The Python implementation adopts its resolved-path
transaction locking and uniqueness checking across matching tiers, plus CRLF
quote normalization. These are adaptations of the design, not a vendored
TypeScript runtime. Locks cover BESSER read/modify/write/delete operations within
the worker process and are shared between executor instances. They do not lock
external shell processes.

The existing conservative Aider-derived matcher remains. OpenCode's broader
similarity and whitespace-collapse fallbacks were not copied: guessing which
Python block the model intended would weaken the existing safety guarantees.
OpenCode's literal replacement itself does not detect retained-anchor replay;
the content-based protection above addresses that separately.

The handoff also exposed checkpoint corruption: provider shim objects use
`__slots__`, but serialization only handled dictionaries/Pydantic objects.
Checkpoints now preserve real text/tool-use fields and tool-call IDs. This does
not retroactively reconstruct already-corrupted checkpoints.

## Verification

Focused regressions use the exact booking edit from the handoff, replayed 75
times both with and without executor restart; pending `replace_all` sites;
normalized quotes; concurrent path aliases; checkpoint round-trip; and the
orchestrator's bounded retry escalation. No live model or paid generation is
required. This fixes the demonstrated editor defects, not a guarantee that any
model-generated application is functionally correct.
