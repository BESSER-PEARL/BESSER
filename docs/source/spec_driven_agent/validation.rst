Validation and auto-fix
=======================

Phase 3 attempts validation within the run's budget. Core checks need no network
and no Docker. Missing runtime prerequisites do not count as verified execution:

- Python syntax on every ``.py`` file.
- Structural syntax on every ``.ts`` / ``.tsx`` / ``.js`` / ``.jsx`` file:
  delimiter balance across comments, strings, template literals, regexes and
  JSX, plus strict-JSON parsing of the ``attr={{...}}`` containers the
  generated table configuration is written as. No Node toolchain is required,
  so this holds when ``tsc`` is unavailable or disabled.
- Python declaration contracts: invalid enum members, overwritten declarations,
  and SQLite-incompatible constant CHECK constraints.
- Isolated backend startup and create-request probes. An import or DDL failure
  is a blocker, not a skipped check. Schema changes are checked against router
  consumers as well as syntax.
- Retained ``test_api`` workflows replayed after source changes.

- Dockerfile coherence — referenced ``requirements.txt`` / ``package.json``
  must exist. A ``COPY`` source is resolved against the build context, as
  Docker does: the compose ``build.context`` when a compose file names the
  Dockerfile, else the Dockerfile's own folder. ``COPY --from=<stage>`` copies
  from a build stage and is not checked. This check only reports; it never
  writes a file. A separate repair step before validation restores a
  ``requirements.txt`` that a Phase 2 edit deleted from a FastAPI scaffold.
  Several other common mistakes are repaired outright without spending
  an LLM turn (``npm ci`` with no lockfile anywhere in the project becomes
  ``npm install``; a ``COPY`` of a non-existent ``package-lock.json`` is
  dropped).
- Known-incompatible dependencies — a table of pairs that pip resolves
  without complaint but that break at runtime (``passlib`` needs
  ``bcrypt<4.1``: bcrypt 5 rejects the >72-byte secret passlib's self-test
  hashes, so every password hash fails). Each ``requirements.txt`` that
  declares the first package gets the second pinned, unless its declared range
  is already compatible. The same table is applied when scaffold repair writes
  a ``requirements.txt`` and by ``install_dependencies`` before it installs,
  so the delivered file is pinned even when Phase 3 is skipped.
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

Every import a generated frontend makes is resolved against the files it
actually ships, with no install, no bundler and no shell, so this runs in every
configuration. A relative import that resolves to no file, and a ``.jsx`` /
``.tsx`` file containing JSX in a project that imports no React and configures
no automatic JSX runtime, are blockers: both leave a blank page in the browser.
An undeclared package is reported as a warning, because a bundler can still
satisfy it. The JSX rule matters even where build verification is enabled:
``React is not defined`` is a runtime error in a bundle that builds cleanly, so
no build check can see it.

For discovered TypeScript projects and frontend applications, required checks
that are disabled, unavailable, timed out, or only partially run are recorded as
``validation unverified:`` warnings. Their effect on completion depends on the
check: some contribute to the completion gate, while others are advisory.
The editor follows the ``done`` event's ``incomplete`` flag; it may present an
application as ready with unverified checks. Read the findings to distinguish
checks that passed from checks that did not run. Optional lint checks remain
advisory.

Checks that execute generated code (the import and runtime probes and the
compiler and build checks) run in the bubblewrap sandbox described in
:ref:`spec-driven-shell-tools`, with no network. Two skip reasons come from
that. *The sandbox is unavailable* means the host could not start it; the check
is skipped rather than run unconfined. *Its dependencies could not be
fetched* means ``cargo check`` needed a crate the run had not already
downloaded through ``run_command``. Both are reported as checks that did not
run, not as defects in the generated code.

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
execute generated build scripts under that explicit permission, inside a fresh
bubblewrap sandbox with no network. Source is checked before and after execution. Results
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
       code, with every "implemented" citation re-checked by the harness;
       only behaviour the request states is listed, so a vague request
       yields few requirements or none);
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
       now), the model-derived acceptance matrix, and ``requirement out of
       scope:`` — a UI requirement on a run that neither has nor asked for a
       frontend, which a backend-only output has no files to satisfy.

Only ``blocker`` findings spend LLM turns. The ``done`` event reports
``blockerCount`` — completion-blocking defects and required verification gaps
still standing when the run ended. A non-zero value means the downloaded output
is not verified complete; it does not necessarily mean that the code is broken.

.. _spec-driven-design-fidelity:

Design fidelity
---------------

When the GUI model carries a stylesheet, the React generator writes it to
``src/design.css`` and renders every designed screen from it. To keep the agent from discarding that
design, Phase 2's system prompt gains a short section, built from the project's own ``design.css``: its ``--ds-*`` variables
and its classes grouped by role (layout, navigation, cards, forms, buttons,
tables, status, text), then the rules — never edit ``design.css`` (new rules go
in ``src/design-overrides.css``, imported after it); edit designed pages in
place with ``modify_file`` / ``replace_file_lines``, never ``write_file`` over
one; keep each page's nav, layout wrappers and generated components
(``TableBlock``, ``MethodButton``, ``CrudButton``, ``FormBlock``, charts,
metric cards); build new markup from the design classes rather than bare tags,
``style={{...}}`` objects or hex colours; plus brief design-quality guidance.
The general styling advice it would contradict is switched off for such runs:
the "loose hint" framing of the GUI model, Rule 2's ``write_file`` fallback for
pages, and the "define concrete CSS" / "one shared stylesheet" bullets. Without
``design.css`` the section falls back to the GUI model's stylesheet, embedded in
the prompt; with neither, the prompt is unchanged.

The bounded auto-fix loop
-------------------------

Blocker-level findings drive a repair/recheck loop within the remaining turn,
cost and runtime budgets. The loop stops when blockers reach zero, after two
consecutive unchanged/repeated source states, or after three consecutive rounds
that change the tree without improving it. A round is progress when the tree
scores better, the source changed, or a verification obligation was discharged
(a checklist or scenario change counts only if it changed the set of blockers)
— writing no source is not by itself a stop, because a round spent closing
checklist items or correcting a scenario can resolve blockers without touching
a file, and a round whose edits were all rejected feeds those rejections into
the next attempt's prompt. A round that wrote nothing and never called an edit
tool is the exception: it leaves the next prompt identical to its own, so it
ends the loop immediately rather than paying for the same attempt twice. A larger error count does not trigger an automatic rollback: fixing one
import may expose several previously unreachable CRUD failures. Unresolved
output remains explicitly incomplete. Startup and data-entry failures are
repaired before spending tokens on business requirement judgment.

A fix turn whose reply is cut off at the output-token limit is retried with
an instruction to emit less, up to four times per attempt, rather than
applying a half-written edit.

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
   one part of Phase 3 that is **opt-in** for the web backend
   (``BESSER_LLM_ENABLE_TOOLCHAIN_VALIDATION``) — they were the main driver of
   a duration and cost regression on non-Python stacks. Two further checks
   are on by default and can be switched off per deployment: executing the
   generated backend (the ORM import check, the isolated startup and create
   probes, and ``test_api`` replay) is gated by
   ``BESSER_LLM_ENABLE_IMPORT_SMOKE_CHECK``, and the requirement judgment by
   ``BESSER_LLM_ENABLE_REQUIREMENTS_LEDGER``. Turning either off makes the run
   report that check as ``unverified``, never as passed. Every other check
   above runs on every run. See :doc:`configuration`.

.. note::
   A check that could not run reports that it did not, rather than returning
   nothing. A collector that times out or cannot be launched emits a
   ``validation: <tool> did not run (...) - its checks were SKIPPED`` finding,
   because an empty result is otherwise identical to a clean one and the run
   would report "0 blockers" having verified nothing. These are ``warning``
   findings on purpose: not looking is not evidence of a defect.

   For the same reason the acceptance matrix's ``route`` cell is *unmeasured*,
   not *missing*, on stacks its route parser does not read. That parser
   understands FastAPI decorators in ``.py`` files, so an Express, Django,
   Spring or axum backend reports no route finding at all rather than one per
   entity. The ``page`` and ``create`` cells are stack-independent and still
   apply.

.. warning::
   Validation includes isolated backend execution, not a deployment or full
   browser acceptance run. Phase 3 does not build a container, and passing
   create probes or submitted API scenarios does not prove every workflow.
   Independently exercise the generated user interface and specification.

Phase 3 is not the only place findings surface: the per-write diagnostics guard
described in :doc:`how_it_works` parses every file the moment the model writes
it, so the cheapest defects never reach Phase 3 at all.
