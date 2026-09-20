Validation and auto-fix
=======================

Phase 3 attempts validation within the run's budget. Core checks need no network
and no Docker. Missing runtime prerequisites do not count as verified execution:

- Python syntax on every ``.py`` file.
- Python declaration contracts: invalid enum members, overwritten declarations,
  and SQLite-incompatible constant CHECK constraints.
- Isolated backend startup and create-request probes. An import or DDL failure
  is a blocker, not a skipped check. Schema changes are checked against router
  consumers as well as syntax.
- Retained ``test_api`` workflows replayed after source changes.

- Dockerfile coherence — referenced ``requirements.txt`` / ``package.json``
  must exist. Several common mistakes are repaired outright without spending
  an LLM turn (``npm ci`` with no lockfile anywhere in the project becomes
  ``npm install``; a ``COPY`` of a non-existent ``package-lock.json`` is
  dropped; ``passlib`` pins ``bcrypt==4.0.1``).
- Local-import resolution — an import naming a module the app does not ship
  where it is used. ``ruff`` is structurally blind to this (a star import
  excuses every name), and it is fatal at startup.
- Frontend contract — high-precision correctness defects that leave the UI
  visibly broken: a router with no home route (blank on load), or a form whose
  submit handler is a no-op (cannot save).
- Data contract — generated code that disagrees with the domain model's
  declared identifier types or server-owned fields, or that fakes success for
  an unimplemented method.
- Framework coherence — generated code importing a rival framework into the
  chosen scaffold (Flask into a FastAPI project).
- Generated model-action contracts — actual handler paths and functions, with
  deterministic checks for missing handlers and known unimplemented bodies.
  Absence of a known stub is structural evidence, not proof of business behavior.
- Frontend/backend endpoint coherence — literal ``fetch`` and Axios URLs that
  match no generated backend route.
- ``ruff`` lint.
- Optionally ``tsc --noEmit``, ``cargo check`` and ``kotlinc``, per project.

For discovered TypeScript projects and frontend applications, required checks
that are disabled, unavailable, timed out, or only partially run are recorded as
``validation unverified:`` warnings. These keep the delivered output incomplete
and retain its repair checkpoint, but do not spend LLM turns repairing an
environment restriction. Optional lint checks remain advisory.

When both toolchain validation and shell tools are explicitly enabled, npm is
available, and project dependencies are missing, ``verification setup:`` is
instead actionable. The agent may use its existing authorized dependency tool,
then request fresh validation. This is a verification-only obligation: Phase 3
does not force application edits to satisfy it. Missing system tools or disabled
permissions remain unverified; validation itself never installs anything.
Authorized TypeScript checks prefer the project's installed compiler over a
potentially incompatible global version.

Frontend production builds are distinct from typechecking. A discovered app's
configured ``npm run build`` is checked only when both toolchain validation and
shell tools are explicitly enabled and its dependencies are already installed.
Validation never installs packages or enables shell access. Configured checks
execute generated build scripts under that explicit permission; this is not an
operating-system sandbox. Source is checked before and after execution. Results
are reused only for unchanged source and dependency-install markers during the
same run; source changes during a build invalidate its result. With shell tools
disabled, a separate external build can verify the output, but raw generation
continues to state that frontend build verification is pending.

Generated create probes use distinct fixture values and native association-link
payloads. A legitimate 4xx rejection of guessed data is **unknown**, not proof
that every valid request fails. An unresolved create probe can be discharged by
a passing current-revision ``test_api`` scenario that creates through the exact
endpoint and reads back the identifiable record. Health responses, an isolated
200, stale results, and evidence from a different route/backend do not qualify.
Observed server crashes remain failures even if another scenario passes.

Severities
----------

Findings are classified into three severities:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Severity
     - What lands here
   * - ``blocker``
     - Syntax errors; dependency conflicts; a Dockerfile referencing a file
       that doesn't exist; unresolvable local imports; an ORM module that
       fails to import or to configure its SQLAlchemy mappers (``mapper
       config:`` — the generated ``sql_alchemy.py`` is imported in a
       subprocess and ``configure_mappers()`` is run, which is the only way
       to see a ``relationship()`` whose string arguments resolve to
       nothing); a requirement the user stated that the code does not
       implement (``requirement:`` — the verbatim request is turned into
       atomic requirements once and each is judged against the generated
       code, with every "implemented" citation re-checked by the harness);
       partial requirements and unverified evidence (distinct from proven missing
       behavior); unresolved checklist work and unimplemented action contracts;
       a method button that takes its row id from a table of another
       entity; frontend-contract and
       data-contract violations; ``ruff`` **F821** / **F822** / **F823**
       (undefined name — the classic "ships green, boots dead" bug) and
       **F811** (redefinition, e.g. an ORM model shadowed by a Pydantic model
       of the same name); per-project toolchain errors from ``tsc`` /
       ``cargo`` / ``kotlinc``. These drive the auto-fix loop.
   * - ``style``
     - Cosmetic ``ruff`` rules: ``F401`` / ``F841`` (unused import or
       variable), ``E501``, whitespace, blank lines, import order.
   * - ``warning``
     - Everything else, including endpoint-coherence findings (report-only for
       now) and the model-derived acceptance matrix.

Only ``blocker`` findings spend LLM turns. The ``done`` event reports
``blockerCount`` — completion-blocking defects and required verification gaps
still standing when the run ended. A non-zero value means the downloaded output
is not verified complete; it does not necessarily mean that the code is broken.

The bounded auto-fix loop
-------------------------

Blocker-level findings drive a repair/recheck loop within the remaining turn,
cost and runtime budgets. The loop stops when blockers reach zero, after two
consecutive unchanged/repeated source states, or after three consecutive rounds
that change the tree without improving it. A round is progress when the tree
scores better, the source changed, or a verification obligation was discharged
— writing no source is not by itself a stop, because a round spent closing
checklist items or correcting a scenario can resolve blockers without touching
a file. A larger error count does not trigger an automatic rollback: fixing one
import may expose several previously unreachable CRUD failures. Unresolved
output remains explicitly incomplete. Startup and data-entry failures are
repaired before spending tokens on business requirement judgment.

For concrete code defects, an attempt without a successful edit — the model explained
the fix instead of making it, or read files until its turn budget ran out — is
re-prompted once, with ``modify_file`` forced through ``tool_choice`` where the
provider honours it and a reminder in the message either way; if that still
produces no edit the attempt ends, and the log says ``ended with no successful
edit``. Phase 3 tool calls are recorded in the trace and the recipe exactly
like Phase 2's.

Evidence-only findings and generated API assertions do not force a source edit.
The agent can inspect saved scenarios, correct a mistaken expectation with a
specification-grounded reason, or cite an already-present implementation.
Accepted scenario/checklist corrections count as progress; changing a judge's
opinion alone does not. Cancellation is checked between repair and validation,
so stopping a run does not trigger another paid coverage judgment.

Repair prompts retain the original specification, current files/symbols, actual
action handlers and recent rejected operations. Excerpts use the file/line
formats emitted by the checks. Unchanged source is not counted as repair
progress merely because a judge's blocker count changed.

Requirement judgments are cached by source/configuration revision. A single
bounded citation-repair judgment may revisit only unverified entries without a
code change. Evidence must name a real application source file and quote an
executable line; traces, recipes and unrelated files cannot validate a claim.
Failure to extract requirements remains unknown and blocks verified completion.

Model conversion losses
-----------------------

An OCL rule that cannot be parsed or attached is not silently removed from the
spec-driven handoff. The converted model retains ``conversion_issues`` with
stable identifiers, diagram/element provenance, the original expression and
the conversion reason. Invalid rules stay out of executable model constraints.

The planner, customization prompt and repair prompt receive these diagnostics.
Each loss becomes a harness-owned recovery task and a requirement checked
against current application source. Closing or dropping a task cannot discharge
the corresponding requirement. If coverage verification is disabled or cannot
run, recovery remains explicitly unverified and blocks verified completion.
Successful evidence is tied to the current code revision; it does not rewrite
the user's model or prove end-to-end behavior. Diagnostics remain in the recipe
under ``model_conversion_issues`` for review even after code recovery succeeds.

The original natural-language request takes precedence when a rejected model
expression conflicts with it. Relationship names must be resolved from the
actual model and code, not guessed or automatically renamed.

Validation is attempted within the run's runtime budget. If that budget is
already exhausted, the run records an unverified-completion blocker instead
of silently reporting an empty validation result. The *repair* half is a flag:
``LLMOrchestrator(auto_fix_issues=...)`` defaults to ``False`` for library
users — report by default, fix on request, the usual static-analyser
contract. The web deployment turns it on
(``BESSER_LLM_ENABLE_AUTO_FIX``, default ``true``), because shipping a
broken artifact as a green success is worse than spending a few more turns.

.. note::
   The heavy per-project compilers (``tsc`` / ``cargo`` / ``kotlinc``) are the
   one part of Phase 3 that is **opt-in** for the hosted deployment
   (``BESSER_LLM_ENABLE_TOOLCHAIN_VALIDATION``) — they were the main driver of
   a duration and cost regression on non-Python stacks. Every other check
   above runs on every run.

.. warning::
   Validation includes isolated backend execution, not a deployment or full
   browser acceptance run. Phase 3 does not build a container, and passing
   create probes or submitted API scenarios does not prove every workflow.
   Independently exercise the generated user interface and specification.

Phase 3 is not the only place findings surface: the per-write diagnostics guard
described in :doc:`how_it_works` parses every file the moment the model writes
it, so the cheapest defects never reach Phase 3 at all.
