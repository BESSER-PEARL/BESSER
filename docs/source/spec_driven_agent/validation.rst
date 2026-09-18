Validation and auto-fix
=======================

Phase 3 always runs. Its checks need no network and no Docker, and each
external tool skips silently when its binary isn't on ``PATH``:

- Python syntax on every ``.py`` file.
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
- Frontend/backend endpoint coherence — literal ``fetch`` and Axios URLs that
  match no generated backend route.
- ``ruff`` lint.
- Optionally ``tsc --noEmit``, ``cargo check`` and ``kotlinc``, per project.

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
       that doesn't exist; unresolvable local imports; frontend-contract and
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
``blockerCount`` — the blockers still standing when the run ended — so a
non-zero value means the run finished but could not repair everything it found.

The bounded auto-fix loop
-------------------------

Blocker-level findings drive a bounded repair loop: up to five rounds of (LLM
fix turns → re-validate), against a snapshot taken before Phase 3 started. The
loop stops early when blockers reach zero, gives up after two consecutive
rounds without progress, and rolls back to that snapshot if the fixes made
blockers *worse* — a failed repair can never cost you the Phase 2 work.

An attempt that ends without a single successful edit — the model explained
the fix instead of making it, or read files until its turn budget ran out — is
re-prompted once, with ``modify_file`` forced through ``tool_choice`` where the
provider honours it and a reminder in the message either way; if that still
produces no edit the attempt ends, and the log says ``ended with no successful
edit``. Phase 3 tool calls are recorded in the trace and the recipe exactly
like Phase 2's.

Validation itself always runs. The *repair* half is a flag:
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
   Validation is static, not a build. Phase 3 does not build a container or run
   the app, so blockers that only appear at runtime can still get through.

Phase 3 is not the only place findings surface: the per-write diagnostics guard
described in :doc:`how_it_works` parses every file the moment the model writes
it, so the cheapest defects never reach Phase 3 at all.
