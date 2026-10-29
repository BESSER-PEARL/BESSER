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
       LLM call. For ``mode: "modify"`` it also returns ``model_change``
       (see :ref:`spec-driven-preview-model-change`).
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
   * - ``/spec-driven/runs/{run_id}/events.json``
     - GET
     - Polling alternative to the stream: the run's lifecycle metadata plus
       up to ``limit`` events (default 250, max 1000) after ``?after=N``, as
       a plain JSON response. Each event carries its ``sequence``, ``event``
       and decoded ``data``; ``cursor`` is the last sequence returned and
       ``hasMore`` says to poll again immediately. Polling and streaming
       share one sequence, so a client can switch between them. Use it where
       a proxy buffers responses and holds back an SSE stream.
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
       ``buml/`` — to a GitHub repository, as the change the run made: only
       files that differ from the base commit are sent, modes are kept, and
       paths the run never held (build output, ``*.db``, symlinks, secret
       ``.env`` files) are left as they are. ``deploy_config.owner`` is the
       owner of an existing repo (an organization or another user; the
       signed-in user when omitted). ``deploy_config.base_commit_sha``
       is the commit the run's code came from: commits made on the branch
       since are merged in, never reverted, with the model merged as data.
       ``deploy_config.push_mode`` is
       ``"commit"`` (default) or ``"pull_request"``. The response's
       ``status`` is ``"pushed"``, ``"merged"`` (with ``base_run_id``, a run
       holding the merged repo for the next modify run), ``"conflict"``
       (left on ``conflict_branch``) or ``"pull_request"``
       (``pull_request_url``, on ``pull_request_branch``); ``commit_sha`` is
       the next push's base and ``files_uploaded`` the number of files the
       push changed (0 when nothing changed: no commit is made).
       When the run changed the model (a modify run that added classes),
       ``project_export`` is the model written to ``buml/``, which the editor
       loads; otherwise it is ``null``. Requires an ``X-GitHub-Session``
       header.
   * - ``/spec-driven/import-github-run``
     - POST
     - Import a GitHub repo as a run, so it can be used as the seed of a
       subsequent modify run. The repo is pinned at its branch head
       (``commit_sha`` in the response) and written to the shared run
       workspace, where the Spec-Driven worker finds it. A repo without a
       BESSER model (``has_model: false``) is read for SQLAlchemy, Django,
       Pydantic or dataclass models, with their enumerations (including
       Django ``TextChoices``) and inheritance: ``analysis`` says what was
       found and ``proposed_diagram`` is the resulting class diagram, or ``null``.
       When no such models are found, or only Pydantic and dataclass models
       (mostly API shapes), and the server offers the keyless model, that
       model reads a bounded digest of the code (schemas, migrations, types,
       entities, README; secrets left out) and proposes the diagram instead,
       with ``analysis.ai: true``; a failed or timed-out reading falls back to
       the Pydantic or dataclass models when there are any, else returns no
       proposal and a ``message``. A body carrying the valid demo secret
       (``demo_token``, from a ``?demo=`` link) is read by the sponsored
       model instead; a wrong secret falls back to the keyless one.
       Requires an ``X-GitHub-Session`` header.

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
     - Natural-language description of what to build (1–64,000 characters).
       This limit includes any modeling-agent summary and appended original
       request. Longer input is rejected before generation, not truncated.
       Accepted instructions reach planning and requirement extraction in full;
       inventories and model summaries have separate context budgets.
   * - ``api_key``
     - string (secret)
     - BYOK. Required for ``anthropic`` / ``openai`` / ``mistral`` / ``nebius``;
       omitted for the keyless tiers. Sent only in the body, never logged or
       persisted.
   * - ``provider``
     - ``anthropic`` | ``openai`` | ``mistral`` | ``nebius`` | ``free`` |
       ``sponsored``
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
     - Soft spend cap, clamped to the server hard cap (default 5.0, max 5.0).
   * - ``max_runtime_seconds``
     - int
     - Soft runtime cap, clamped to the server hard cap (default 2400, max 2400).
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

.. _spec-driven-preview-model-change:

Model change in the preview
---------------------------

With ``mode: "modify"`` and a
``base_run_id`` whose workspace is still available, the ``/preview`` response
carries ``model_change``: how the project's class diagram relates to the one
the base run was built from. It is ``null`` for ``mode: "generate"``, an
unknown or expired base run, or a base run without a recorded class model.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``verdict``
     - ``small``, ``large`` or ``unrelated``. ``large`` means ``gone`` is
       above 0.15. ``unrelated`` means
       more than half of it is gone, most of the new model is new elements,
       and either most base classes or three quarters of the base model are
       gone. Deleting classes, even all of them, is ``large``.
   * - ``basis``
     - ``attributes``, or ``classes`` when the base run predates the recorded
       model (then only class names are compared and ``changes`` is empty).
   * - ``gone``
     - Share of the base model's classes and attributes (0 to 1) that are
       removed, or are attributes now under another name. A renamed attribute
       counts because the code still changes everywhere it is used; a renamed
       class, and a change of letter case only, do not.
   * - ``removed`` / ``added``
     - ``Class`` and ``Class.attribute`` names, for display. Renamed elements
       are not listed here.
   * - ``renamed``
     - ``[{"from": "Room", "to": "Suite"}, {"from": "Room.floorType", "to":
       "Room.floor"}]``: classes and attributes kept under a new name. Empty
       when none.
   * - ``changes``
     - One line per change, the list a modify run receives (see
       :ref:`spec-driven-modify-mode`).

The verdict, ``removed``/``added``/``renamed`` and ``changes`` come from one
matching of old and new elements, so they always agree. Two elements are the
same element when:

1. both carry an editor id and the ids are equal (two different ids are two
   different elements, whatever their names);
2. otherwise, they have the same name, compared exactly first and then
   ignoring case; attributes are matched only within their matched class;
3. for classes only, when neither is matched yet: they are the only class on
   each side with the same non-empty set of attribute names (a class whose
   name alone changed).

No element is matched twice.

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
       blocker-severity findings), ``incomplete`` / ``incompleteReason``,
       a three-way authorship split of the output tree, and ``aiShare``, the
       AI-written share of the lines from the run's Agent Trace record (see
       :ref:`spec-driven-agent-trace`). ``mode`` is ``create`` or ``modify``;
       ``changedFiles`` (capped) and ``changedFileCount`` are the files the
       agent wrote, edited or deleted, and ``modelChanges`` how the model
       differs from the one the base app was built from (modify runs).
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
