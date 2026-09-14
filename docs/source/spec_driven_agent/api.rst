REST API and SSE stream
=======================

The agent has its own REST surface on the
:doc:`web editor backend <../web_editor_backend>`. This page is the canonical
contract: the endpoints, the request body, and the Server-Sent Events the run
emits.

All paths below are relative to the ``/besser_api`` router prefix.

Endpoints
---------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Endpoint
     - Method
     - Purpose
   * - ``/spec-driven/generate``
     - POST
     - Start a run; streams SSE, returns a download URL on completion.
   * - ``/spec-driven/preview``
     - POST
     - Pre-flight plan (primary model + target generator). No API key, no
       LLM call.
   * - ``/spec-driven/config``
     - GET
     - Caps, feature flags, provider default models, and free-tier
       availability.
   * - ``/spec-driven/runs/{run_id}``
     - GET
     - Durable lifecycle metadata for a run (never the request or the key).
   * - ``/spec-driven/runs/{run_id}/events``
     - GET
     - Replay this run's events after ``?after=N``, then follow the live
       stream. Also honours the ``Last-Event-ID`` header.
   * - ``/spec-driven/resume/{run_id}``
     - POST
     - Resume an interrupted run from its checkpoint (streams SSE).
   * - ``/spec-driven/cancel/{run_id}``
     - POST
     - Cancel an in-flight run at its next turn boundary.
   * - ``/spec-driven/download/{run_id}``
     - GET
     - Download the generated ZIP/file (re-fetchable within a TTL).
   * - ``/spec-driven/push-to-github``
     - POST
     - Push a finished run's code — plus the re-importable model under
       ``buml/`` — to a GitHub repository. Requires an ``X-GitHub-Session``
       header.
   * - ``/spec-driven/import-github-run``
     - POST
     - Import a BESSER-created repo back as a run, so it can be used as the
       seed of a subsequent modify run. Requires an ``X-GitHub-Session``
       header.

Request body
------------

**Request body of** ``POST /spec-driven/generate``:

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Field
     - Type
     - Notes
   * - ``project``
     - ProjectInput
     - The full project payload (same shape as ``/generate-output-from-project``).
   * - ``instructions``
     - string
     - Natural-language description of what to build (1–8000 chars).
   * - ``api_key``
     - string (secret)
     - BYOK. Required for ``anthropic`` / ``openai`` / ``mistral``; omitted for
       the keyless tiers. Sent only in the body, never logged or persisted.
   * - ``provider``
     - ``anthropic`` | ``openai`` | ``mistral`` | ``free`` | ``sponsored``
     - Which provider serves the run. Default ``anthropic``; the editor
       selects ``free`` when the deployment configures a free tier. See
       :doc:`models`.
   * - ``llm_model``
     - string (optional)
     - Model override; falls back to the provider default. For ``free`` only
       the ids the server has configured are honoured (the primary, an entry in
       ``BESSER_FREE_LLM_ALT_MODELS``, or the fallback model); any other value
       is ignored and the run pins to the primary rather than erroring.
   * - ``max_cost_usd``
     - float
     - Soft spend cap, clamped to the server hard cap (default 1.0, max 5.0).
   * - ``max_runtime_seconds``
     - int
     - Soft runtime cap, clamped to the server hard cap (default 600, max 900).
   * - ``max_turns``
     - int
     - Soft cap on LLM turns, clamped to the server hard cap (default 80,
       max 120).
   * - ``mode``
     - ``generate`` | ``modify``
     - ``modify`` seeds the workspace from ``base_run_id`` and edits it in
       place instead of rebuilding. Default ``generate``. See
       :ref:`spec-driven-modify-mode`.
   * - ``base_run_id``
     - string (optional)
     - The 32-hex run id of the run to modify. Optional even for
       ``mode: "modify"`` on ``/generate``: when it is missing or the base has
       expired, the run degrades to a normal from-scratch generation rather
       than failing. ``/preview`` is stricter and rejects
       ``mode: "modify"`` without it.
   * - ``primary_kind_override``
     - string (optional)
     - Force the primary model kind (``class``, ``gui``, ``agent``,
       ``state_machine``, ``object``, ``bpmn``, ``nn``, ``quantum``) instead of
       auto-detecting it.
   * - ``target_generator_override``
     - string (optional)
     - Bind the Phase-1 deterministic generator (a registered generator-tool
       name from an approved preview plan), skipping the agent's own
       selection. Mutually exclusive with ``skip_deterministic_generator``.
   * - ``skip_deterministic_generator``
     - bool
     - Explicitly skip Phase 1 and let the LLM scaffold from scratch.
       Default ``false``.

.. note::
   The caps above are the shipped defaults. Every one of them is overridable
   per deployment via ``BESSER_LLM_*`` environment variables — read the live
   values from ``GET /besser_api/spec-driven/config`` rather than hardcoding
   them in a client. The full variable reference is in :doc:`configuration`.

SSE event types
---------------

**SSE event types** emitted by ``/spec-driven/generate``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Event
     - Meaning
   * - ``start``
     - Run accepted; carries ``runId``, provider, model, and caps.
   * - ``phase``
     - Pipeline advanced to a phase: ``select``, ``generate``, ``gap``,
       ``customize``, ``validate``.
   * - ``phase_update``
     - Extra detail for the current phase (e.g. the gap task list).
   * - ``text``
     - A streaming text delta from the LLM.
   * - ``tool_call``
     - The LLM invoked a :doc:`tool <tools>` (read/write/modify a file, query
       the model, update its checklist, run a generator). Carries the turn
       number, the tool name, a status, and a short non-content ``detail``.
   * - ``model_update``
     - The model serving this run changed mid-run — the fallback chain took
       over after the primary failed or hit its quota. Only the
       server-configured tiers (``free``, ``sponsored``) have one; a BYOK run
       never switches models behind your back. The switch is sticky for the
       rest of the run.
   * - ``cost``
     - Periodic cost / runtime / turn-count tick.
   * - ``done``
     - Success. Carries ``downloadUrl``, ``fileName``, the run recipe, a file
       count and top-level listing, ``tokensUsed``, ``blockerCount`` (unfixed
       blocker-severity findings), ``incomplete`` / ``incompleteReason``, and
       a three-way authorship split of the output tree.
   * - ``error``
     - ``INVALID_KEY`` / ``UPSTREAM_LLM`` / ``INTERNAL`` / ``BAD_REQUEST`` /
       ``CANCELLED`` are terminal; ``COST_CAP`` / ``TIMEOUT`` / ``INCOMPLETE``
       are non-terminal warnings emitted just before ``done``.

Example
-------

**Example** — start a run with ``curl`` and read the stream::

   curl -N -X POST https://<host>/besser_api/spec-driven/generate \
     -H 'Content-Type: application/json' \
     -H 'Accept: text/event-stream' \
     -d '{
       "project": { "...": "full ProjectInput payload" },
       "instructions": "FastAPI backend with JWT auth and a Dockerfile",
       "api_key": "sk-...",
       "provider": "openai",
       "llm_model": "gpt-4o",
       "max_cost_usd": 1.0,
       "max_runtime_seconds": 600
     }'

The response is a Server-Sent Events stream::

   event: start
   data: {"event":"start","runId":"4f3c9a1d8e2b47c6905af1e3d7b28c40","provider":"openai","llmModel":"gpt-4o","maxCost":1.0,"maxRuntime":600}

   event: phase
   data: {"event":"phase","phase":"generate","message":"running generate_fastapi_backend"}

   event: tool_call
   data: {"event":"tool_call","turn":1,"tool":"write_file","status":"executing","detail":"auth.py"}

   event: cost
   data: {"event":"cost","usd":0.07,"turns":3,"elapsedSeconds":24.1}

   event: phase
   data: {"event":"phase","phase":"validate","message":"1 blockers / 4 total"}

   event: done
   data: {"event":"done","runId":"4f3c9a1d8e2b47c6905af1e3d7b28c40","downloadUrl":"/besser_api/spec-driven/download/4f3c9a1d8e2b47c6905af1e3d7b28c40","fileName":"besser_smart_4f3c9a1d8e2b47c6905af1e3d7b28c40.zip","isZip":true,"blockerCount":0}

Then issue a ``GET`` to the ``downloadUrl`` to retrieve the ZIP — it is
re-fetchable within the run's TTL.

.. note::
   The frames above are shown without their sequence number for readability.
   As actually delivered, every frame additionally carries an SSE ``id:`` line
   and a ``sequence`` field in the JSON body — ``id: 3`` /
   ``data: {"event":"cost", …,"sequence":3}``. That number is the cursor for
   ``?after=`` and for ``Last-Event-ID``, so a client that wants to reconnect
   must record it. See :ref:`spec-driven-durable-runs` for the reconnect flow.
