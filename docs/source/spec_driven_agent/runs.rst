Runs, budgets and modify mode
=============================

Budgets and caps
----------------

Every run is bounded by three soft caps, each clamped to a server-side hard cap
so a client can never exceed them. All six values are overridable per
deployment through ``BESSER_LLM_*`` environment variables (see
:doc:`configuration`); the figures below are the shipped defaults, and the live
values are exposed by ``GET /besser_api/spec-driven/config``.

.. list-table::
   :header-rows: 1
   :widths: 34 22 22

   * - Cap
     - Default
     - Hard limit
   * - ``max_cost_usd``
     - 5.0
     - 5.0
   * - ``max_runtime_seconds``
     - 2400
     - 2400
   * - ``max_turns``
     - 120
     - 150

When a run hits a cap it still returns whatever it has produced so far — the
cap is surfaced as a non-terminal warning before the final result, not as a
hard failure.

The caps are checked at turn boundaries: cancellation first, then elapsed time,
then spend — and the cost check runs *before* the next request is sent, so a
run cannot overshoot its cap by one more billable call. A single warning is
logged at 80% of the cost cap.

Only the paid LLM calls count against the budget: the planning pass, the
customize loop, and any Phase 3 fix turns. The deterministic scaffold, the
validators, and the download are free.

.. _spec-driven-durable-runs:

Durable runs
------------

A run is owned by the server, not by the browser connection. A background task
consumes the run to completion; every SSE frame is assigned a monotonically
increasing sequence number and stored before any subscriber sees it. That makes
runs survive a dropped connection:

- ``GET /besser_api/spec-driven/runs/{run_id}`` returns the run's lifecycle
  metadata (never the request or the API key).
- ``GET /besser_api/spec-driven/runs/{run_id}/events?after=N`` replays every
  frame after sequence ``N`` and then follows the live producer. Native
  ``EventSource`` clients can use the ``Last-Event-ID`` header instead; the
  larger of the two cursors wins, so a stale query value can't replay
  duplicates.

.. code-block:: bash

   # What happened to this run?
   curl https://<host>/besser_api/spec-driven/runs/<runId>

   # Replay everything after sequence 42, then follow the live stream
   curl -N -H 'Accept: text/event-stream' \
     'https://<host>/besser_api/spec-driven/runs/<runId>/events?after=42'

Losing *every* subscriber starts a grace period (default 5 minutes) before the
producer is cooperatively stopped — which bounds unattended BYOK spend. A
server restart marks still-running records ``interrupted``;
``POST /besser_api/spec-driven/resume/{run_id}`` can then continue them from
their checkpoint when the client re-supplies the request body. Resume relies on
the checkpoint written after every tool-use turn (see :doc:`how_it_works`): it
refuses to resume against a project whose fingerprint has changed, and seeds
the already-spent cost so a crash-and-resume cycle cannot double the bill.

.. _spec-driven-modify-mode:

Modifying a previous run
------------------------

A run can also be **incremental**. With ``mode: "modify"`` and a
``base_run_id`` naming a previous, still-downloadable run, the new run's
workspace is seeded from that run's generated files and edited in place instead
of being rebuilt from scratch. Phase 2 is never skipped for a modify run — you
asked for a change, so a change is made. When the base has expired the runner
warns and falls back to normal from-scratch generation rather than failing the
request.

A repository that was pushed to GitHub by a previous run can be brought back
the same way: ``POST /besser_api/spec-driven/import-github-run`` re-imports it
as a run whose ``run_id`` can then seed a ``mode: "modify"`` run. See
:doc:`api` for both request shapes.

.. _spec-driven-agent-trace:

Who wrote which lines (Agent Trace)
-----------------------------------

Every run ships an `Agent Trace <https://agent-trace.dev/>`_ record (spec
``0.1.0``, MIME type ``application/vnd.agent-trace.record+json``) at
``.agent-trace/<run id>.json``, in the download ZIP and in the GitHub push. It
lists each text file of the delivered tree with line ranges and who wrote them:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Contributor
     - Lines
   * - ``ai`` with ``model_id``
     - Written or modified by the LLM. ``model_id`` is ``provider/model``, the
       model that served the run (after any fallback switch). The
       conversation's ``related`` entry ``urn:besser:run:<run id>`` names the
       run, so a range can be traced back to the run's events.
   * - ``unknown`` (generator)
     - Unchanged from a deterministic BESSER generator's output. ``related``
       names the generator and BESSER version
       (``urn:besser:generator:django@<version>``). Deterministic packaging
       fixes (the frontend build configuration, secret redaction) are
       attributed the same way.
   * - ``human``
     - On a run that continues from a previous run or a GitHub import: lines a
       person changed in between.
   * - ``unknown`` (no generator)
     - Code carried over from a repository pushed before attribution existed.

Ranges refer to the files as delivered and carry a ``content_hash``
(``sha256:``) so moved code can be found again. ``metadata["org.besser-pearl"]``
holds the run id, the base generator, the model id and line counts per kind.
The ``done`` event carries the AI-written share (``aiShare``): AI lines overall
and for this run, and per file.

Attribution compares files, it does not replay edits: whenever a generator
writes a file, its lines are recorded (as short hashes) in
``.besser_attribution.json``; at packaging time every line that differs from
that record goes to the LLM. A line keeps its contributor however often the
file was edited, as long as the line itself is unchanged. The record holds
hashes only, never content, and is pushed to GitHub beside the recipe, so a
continue-from-GitHub run can tell a person's edits from the AI's. Binary files
and files over 2 MB are not attributed. The record carries no ``vcs`` revision:
it is part of the commit it would have to name. Older traces in the repository
are kept, and their ranges refer to the revision they were written for.
