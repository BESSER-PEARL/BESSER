Web Editor Backend API
======================

The backend for the :doc:`Web Modeling Editor <web_editor>` is a FastAPI service that
handles code generation, model conversion, validation, and deployment. It lives under
``besser/utilities/web_modeling_editor/backend``.

Architecture
------------

The backend uses a **modular router architecture**. The application factory
(``backend.py``) registers middleware, includes routers, and starts background
services. Endpoints are organized by concern:

- **``routers/generation_router.py``** -- Code generation (single-diagram and project-based) and agent-configuration recommendation
- **``routers/conversion_router.py``** -- BUML import/export, CSV reverse engineering, image-to-model, SVG rendering
- **``routers/validation_router.py``** -- Diagram validation (metamodel + OCL constraints)
- **``routers/deployment_router.py``** -- Docker deployment and feedback
- **``routers/spec_driven_router.py``** -- The :doc:`Spec-Driven Agent <spec_driven_agent/index>`: SSE generation runs, preview, config, durable run status and event replay, resume, cancel, download, and GitHub push/import
- **``routers/telemetry_router.py``** -- Opt-in run telemetry collection and reporting
- **``routers/agent_simulator_router.py``** -- Live agent simulation (see `Agent Simulation`_)
- **``routers/error_handler.py``** -- Centralized ``@handle_endpoint_errors`` decorator
- **``routers/auth.py``** -- Shared GitHub-session gate (``require_github_session``, HTTP 401)

GitHub OAuth and GitHub deployment are registered from
``services/deployment/`` (``github_oauth.py`` and ``github_deploy_api.py``)
rather than from ``routers/``.

Additional infrastructure:

- **``middleware/request_logging.py``** -- Structured request logging with unique IDs and performance timing
- **``services/cleanup.py``** -- Background temp-file cleanup (removes stale directories every hour)
- **``services/exceptions.py``** -- Custom exception hierarchy: ``BesserError`` (base) with ``ConversionError``, ``ValidationError``, ``GenerationError`` and ``ConfigurationError``.
  ``CodeValidationError`` (invalid custom Python code in an agent) is a ``ValidationError`` and maps to HTTP 400.
- **``services/spec_driven/``** -- The machinery behind the Spec-Driven Agent: ``runner.py`` (drives a run and emits SSE), ``run_manager.py`` (durable ownership, SQLite-backed event store, replay), ``model_assembly.py`` (project payload to B-UML models), ``preview.py`` (pre-flight plan, no LLM call), ``sse_events.py`` (the typed event schema), ``secret_redaction.py``, ``telemetry.py``, ``incidents.py``
- **``constants/constants.py``** -- API version, temp prefixes, generator defaults, CORS origins, and the ``BESSER_LLM_*`` caps and feature flags
- **``models/responses.py``** -- Standardized Pydantic response models
- **``models/spec_driven.py``** -- Request/response models for the spec-driven endpoints (the API key is a ``SecretStr``; the caps are clamped by field validators)

Multi-Diagram Projects
^^^^^^^^^^^^^^^^^^^^^^

Projects support **multiple diagrams per type** via
``diagrams: Dict[str, List[DiagramInput]]``. Each diagram can reference other
diagrams by ID through the ``references`` field, and the active diagram per type
is tracked via ``currentDiagramIndices``. Old single-diagram projects are
auto-converted by a Pydantic model validator for backward compatibility.

Supported diagram types: ``ClassDiagram``, ``ObjectDiagram``,
``StateMachineDiagram``, ``AgentDiagram``, ``GUINoCodeDiagram``,
``QuantumCircuitDiagram``, ``UserDiagram``, ``NNDiagram``, ``BPMN``.


Neural Network Diagrams
^^^^^^^^^^^^^^^^^^^^^^^

The backend treats ``NNDiagram`` as a self-contained diagram type (no
cross-diagram references are required for code generation). The editor emits
an NN diagram JSON whose top-level ``type`` is ``"NNDiagram"`` and whose
``elements``/``relationships`` describe layers, containers, sub-network
references, tensor operations, configuration, and training/test datasets.

**Generators.** The registered NN generators are ``pytorch`` and
``tensorflow`` (see :doc:`generators/pytorch` and :doc:`generators/tensorflow`).
Both accept an optional ``config`` payload with:

- ``generation_type``: ``"subclassing"`` or ``"sequential"`` (default:
  ``"subclassing"``) — selects the target architectural style.
- ``channel_last`` (PyTorch only): ``true`` or ``false`` (default: ``false``) —
  when ``true``, input tensors are interpreted as NHWC instead of NCHW.

The response filename embeds the generation type, e.g.
``pytorch_nn_subclassing.py`` or ``tf_nn_sequential.py``.

**JSON ↔ BUML.** The ``/export-buml`` endpoint converts an NN diagram JSON
into a BUML Python file (``nn_model.py``) that reproduces the model when
executed. The converse path through ``/get-json-model`` auto-detects NN BUML
content by the presence of ``.add_layer(``, ``.add_tensor_op(``,
``.add_sub_nn(``, ``.add_configuration(``, ``.add_train_data(``, or
``.add_test_data(``.

**Validation.** ``/validate-diagram`` for ``NNDiagram`` runs the full
processor and surfaces ``ValueError`` as per-line validation errors rather
than a 500 response. Only ``ValueError`` is caught: any other exception
escaping the processor is mapped to a generic HTTP 500 by
``@handle_endpoint_errors``. The processor verifies whitelists for
``pooling_type``, ``return_type``, ``task_type``, ``input_format``,
``optimizer``, ``loss_function``, and ``metrics``, as well as conv layer
``kernel_dim`` / ``stride_dim`` lengths, and detects transitive
``NNReference`` cycles among sub-networks.

**Determinism.** ``nn_model_to_json`` produces byte-identical output for
identical BUML NN models across runs — element IDs are derived from a
thread-local counter via ``uuid.uuid5`` under a fixed namespace.


BPMN Diagrams
^^^^^^^^^^^^^

The backend handles ``BPMN`` as a self-contained diagram type backed by the
:doc:`BPMN metamodel <buml_language/model_types/bpmn>`. ``/export-buml``
converts a BPMN diagram JSON to an executable Python BUML file;
``/get-json-model`` reads it back; ``/validate-diagram`` runs the metamodel
``validate()``.

Middleware and Request Security
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Four middlewares wrap every request, outermost first:

- **Request logging** (``middleware/request_logging.py``) -- a UUID request ID,
  timing, and a warning for any request slower than one second.
- **CORS** -- origins come from ``CORS_ORIGINS`` (comma-separated) or the
  built-in ``DEFAULT_CORS_ORIGINS``. Credentials are allowed; methods are
  ``GET``/``POST``/``PUT``/``DELETE``/``OPTIONS``; the allowed request headers
  are ``Content-Type``, ``X-GitHub-Session``, ``Content-Disposition`` and
  ``Authorization``, and ``Content-Disposition`` plus ``X-BESSER-Run-Id`` are
  exposed to the browser.
- **Request size limit** -- bodies over **50 MB** are rejected with ``413``,
  checked both from ``Content-Length`` and from the read body so the header
  cannot be spoofed. This is a global ceiling on top of the per-file limits
  below.
- **Security headers** -- ``X-Content-Type-Options: nosniff``,
  ``X-Frame-Options: DENY``, ``Referrer-Policy``, ``Permissions-Policy``,
  HSTS, and a Content-Security-Policy on every response.

There is no per-client rate limiting. The only throughput control is
``BESSER_LLM_MAX_CONCURRENT_RUNS`` (default 10) on spec-driven runs, which
answers ``429`` when the slots are full; starting or resuming a run that is
already in flight answers ``409``. See
:doc:`spec_driven_agent/configuration`.

API Endpoints
-------------

Service
^^^^^^^

These two are declared on the application itself rather than through a router,
so the paths below are complete as written:

- ``GET /health`` -- liveness probe for load balancers and monitoring. Note it
  has **no** ``/besser_api`` prefix.
- ``GET /besser_api/`` -- API root: version, the list of supported generator
  keys, and a map of the main endpoint paths.

Code Generation
^^^^^^^^^^^^^^^

- ``POST /generate-output`` -- Single diagram to code generation
- ``POST /generate-output-from-project`` -- Multi-diagram project generation (e.g., WebApp needs ClassDiagram + GUINoCodeDiagram)

Spec-Driven Agent
^^^^^^^^^^^^^^^^^

Ten endpoints under ``/spec-driven/*`` back the
:doc:`Spec-Driven Agent <spec_driven_agent/index>` -- the hybrid pipeline that
runs a deterministic generator, lets an LLM customise its output, then validates
and repairs the result: ``generate``, ``preview``, ``config``,
``runs/{run_id}``, ``runs/{run_id}/events``, ``resume/{run_id}``,
``cancel/{run_id}``, ``download/{run_id}``, ``push-to-github`` and
``import-github-run``.

They are documented once, with the agent, rather than twice: see
:doc:`spec_driven_agent/api` for the endpoint table, the request contract, the
SSE event types and a worked ``curl`` example.

Backend-side notes that do not belong in the contract:

- The ``api_key`` field on ``/spec-driven/generate`` is a ``SecretStr``: never
  logged, never echoed in an event, never persisted. ``/spec-driven/preview``
  has no ``api_key`` field at all, and makes no LLM call -- it is pure local
  computation.
- ``/spec-driven/push-to-github`` pushes the run's *stored* code, not a fresh
  deterministic regeneration, which would discard the LLM's customizations.
- ``/spec-driven/push-to-github`` and ``/spec-driven/import-github-run`` require
  the ``X-GitHub-Session`` header, like the other GitHub endpoints below.

Telemetry
^^^^^^^^^

- ``POST /telemetry/event`` -- Accept one opt-in study telemetry event. Always
  returns 204 for well-formed input whether or not it was stored, so the
  endpoint cannot be used to probe server configuration.
- ``GET  /telemetry/report`` -- Aggregated report (Markdown by default, CSV via
  ``?format=csv``). Requires an ``X-Telemetry-Token`` header matching
  ``BESSER_TELEMETRY_ADMIN_TOKEN``.

Agent Personalization
^^^^^^^^^^^^^^^^^^^^^

These endpoints back the :doc:`agent personalization <generators/agent_personalization>`
workflow. They consume a serialized :doc:`UserDiagram <buml_language/model_types/user_diagram>`
and return a structured agent configuration.

.. note::
   The three ``*-agent-config-*`` endpoints below require a signed-in GitHub
   session: they read the ``X-GitHub-Session`` header and answer **401** when
   it is missing or expired. ``/personalize-gui-page`` and
   ``/transform-agent-model-json`` are open.

- ``POST /recommend-agent-config-llm`` -- LLM-based recommendation. Body:
  ``{userProfileModel, userProfileName?, currentConfig?, model?}``. Requires an
  OpenAI API key (passed in the request body, top-level
  ``openai_api_key``/``openaiApiKey``/``apiKey``, or under
  ``system.openaiApiKey``, or via ``OPENAI_API_KEY`` env var). Returns
  ``{config, source: "openai", model, generatedAt}``.
- ``POST /recommend-agent-config-mapping`` -- Deterministic rule-based
  recommendation. Same request body shape (no OpenAI key needed). Returns
  ``{config, matchedRules, signals, source: "manual_mapping", generatedAt}``.
- ``GET  /agent-config-manual-mapping`` -- The full rule table used by the
  deterministic recommender. Returns ``{mapping, source, generatedAt}``, where
  ``mapping`` is ``{version, sourceDocument, notes, allowedValues, rules}`` and
  each rule carries ``id``, ``label``, ``summary``, ``priority``, ``evidence``,
  ``matchMode``, ``conditions`` and ``recommendation``. Useful for UIs that
  want to show "why this recommendation".
- ``POST /transform-agent-model-json`` -- Apply an agent configuration to an
  agent diagram and return the personalized agent model JSON (used by the
  editor's "apply personalization" action).
- ``POST /personalize-gui-page`` -- Adapt a GrapesJS page snapshot
  (``{components, css}``) in style and content for a user profile, via an LLM.
  Returns ``{guiPage: {components, css}, source, model, generatedAt}``. Needs
  an OpenAI key (request body or ``OPENAI_API_KEY``).

Conversion
^^^^^^^^^^

- ``POST /export-buml`` -- Diagram JSON to BUML Python code
- ``POST /export-project-as-buml`` -- Full project to BUML Python code
- ``POST /get-json-model`` -- BUML Python file to JSON (auto-detects diagram type)
- ``POST /get-project-json-model`` -- BUML project file to JSON
- ``POST /get-json-model-from-image`` -- Image to ClassDiagram JSON (requires OpenAI API key)
- ``POST /get-json-model-from-kg`` -- Knowledge graph (TTL/RDF/JSON) to ClassDiagram JSON
- ``POST /csv-to-domain-model`` -- CSV **or XLSX** files to domain model JSON (both extensions are accepted; XLSX needs ``openpyxl``)
- ``POST /transform-agent-model-json`` -- Agent model transformation with personalization
- ``POST /get-svg`` -- B-UML class-diagram file to an auto-laid-out SVG. Rendering is delegated to the WME Node server (``WME_NODE_SERVER_URL``, default ``http://localhost:8080``), which runs ELK auto-layout headlessly.

Validation
^^^^^^^^^^

- ``POST /validate-diagram`` -- Unified diagram validation (metamodel + OCL constraints)
- ``POST /check-ocl`` -- **Deprecated.** Kept for backwards compatibility; delegates to ``/validate-diagram``.

Deployment
^^^^^^^^^^

- ``POST /deploy-app`` -- Docker Compose deployment for Django projects
- ``POST /feedback`` -- User feedback submission

Standalone Chatbot Deployment
"""""""""""""""""""""""""""""

The GitHub deploy endpoint supports a ``target: "agent"`` flag in
``deploy_config`` that switches the output from a full web-app to a standalone
chatbot (Streamlit frontend, Python backend, single-service Render blueprint).
This is the path used by the editor's "Publish to Render" action and reuses the
personalization flow end-to-end:

- Only an AgentDiagram is required (ClassDiagram / GUI are ignored).
- If the agent config carries a ``personalizationMapping``, it is normalized
  in-place before generation so the BAF generator sees profile *documents*
  rather than raw UML payloads.
- The generated ``render.yaml`` declares ``OPENAI_API_KEY`` as a secret env
  var the user must fill in on Render.

GitHub Integration
^^^^^^^^^^^^^^^^^^

- ``GET /github/auth/login`` -- Initiate GitHub OAuth flow
- ``GET /github/auth/callback`` -- OAuth callback handler
- ``GET /github/auth/status`` -- Check authentication status
- ``POST /github/auth/logout`` -- End session
- ``GET /github/star/status``, ``PUT /github/star``, ``DELETE /github/star`` -- Read / set / clear the signed-in user's star on the BESSER repository
- ``POST /github/deploy-webapp`` -- Deploy generated app to GitHub repository
- ``GET /github/repos``, ``GET /github/branches``, ``GET /github/commits``, ``GET /github/contents``, ``GET /github/file/exists`` -- Repository browsing for the editor's save/load flow
- ``POST /github/project/save``, ``GET /github/project/load``, ``GET /github/project/load-commit``, ``POST /github/project/create-repo`` -- Store and retrieve a BESSER project in a repository (including loading it back from a specific commit)
- ``POST /github/gist/create`` -- Share a model as a GitHub gist

Agent Simulation
^^^^^^^^^^^^^^^^

These endpoints let the editor run an AgentDiagram live. The backend generates
the BAF agent code and hands it to the separate **agent simulator** service
(``besser-wme-agent-simulator``), which runs it in a sandboxed subprocess and
exposes its WebSocket. All paths are under ``/besser_api/simulation``. The
simulator service itself (sandbox, security model, configuration) is described
in :doc:`utilities/agent_simulator`.

- ``GET /limits`` -- Resource limits and quota settings shown in the editor
  (``memoryMb``, ``cpuCores``, ``diskMb``, ``sessionLifetimeSeconds``,
  ``editorQuotaEnabled``). The numeric values are ``null`` while their environment
  variables are unset; in particular ``sessionLifetimeSeconds`` is ``null`` unless
  ``AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS`` is set on the backend.
- ``POST /validate`` -- Generate the agent code without starting a session.
  Body: ``{title, model, config?, configYaml?}``. Returns
  ``{valid, agentCode, eventList, errors}``; diagram errors come back as
  ``valid: false`` with messages.
- ``POST /sessions`` -- Generate the agent code and start a session. Body as
  ``/validate`` plus optional ``credentials`` (``openAiApiKey``,
  ``huggingFaceToken``, ``replicateApiKey``), forwarded to the agent as
  environment variables and in its ``config.yaml``. Returns ``{sessionId, eventList}``.
- ``GET /sessions/{sessionId}/files`` -- Files the running agent wrote to its
  workspace: ``{files: [{path, content}], directories}``.
- ``DELETE /sessions/{sessionId}`` -- Stop the session. Returns ``{ok: true}``.
- ``WS /{sessionId}/ws`` -- Relay between the editor and the running agent.
  The first frame the client sends must be
  ``{"type": "auth", "githubSession": "<session or empty>"}``, within 10 seconds.
  The backend applies the same checks as the HTTP endpoints, answers
  ``{"type": "auth_ok"}`` and starts relaying. Otherwise it sends
  ``{"type": "error", "message": ...}`` and closes with ``4401`` (not
  authenticated), ``4400`` (invalid session id), ``4404`` (not your session)
  or ``4429`` (rate limited). If the backend then cannot open the relay to the
  simulator (service unreachable, or ``AGENT_SIMULATOR_API_TOKEN`` not set), it
  sends an error frame and closes with ``1011``. The session is stopped when
  the socket closes.

**Access rules.**

- When ``AGENT_SIMULATOR_REQUIRE_AUTH`` is on (the default), every endpoint
  needs a valid ``X-GitHub-Session`` header (HTTP 401 otherwise).
- Each session belongs to the actor that created it: the GitHub session when
  sign-in is required, the client IP otherwise. Another actor gets 404 on
  ``files``, ``DELETE`` and the WebSocket.
- One actor may hold ``AGENT_SIMULATOR_MAX_SESSIONS_PER_ACTOR`` sessions at a
  time (default 1). Past that, ``POST /sessions`` returns 429.
- ``/validate``, ``/sessions`` and the WebSocket share a per-actor rate limit
  (default 12 requests per 60 seconds). Over the limit they return 429 with
  ``Retry-After``.
- When ``AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE`` is on (the default), agents that
  contain custom Python code actions are refused (``POST /sessions`` returns
  403, ``/validate`` returns ``valid: false``). With the restriction off,
  custom code and tool code go through a lint that rejects risky imports and
  built-ins. The lint only gives early feedback and is **not** a security
  boundary; isolation is the simulator sandbox's job.
- Invalid custom code returns 400 with the lint message. If the simulator
  service cannot be reached, or ``AGENT_SIMULATOR_API_TOKEN`` is not set, the
  endpoint returns 503. If the simulator is at capacity (all of its
  ``AGENT_SIMULATOR_MAX_SESSIONS`` slots are in use), ``POST /sessions``
  returns 429. If the simulator rejects the request otherwise, it returns 502.

.. note::
   The rate limiter, the per-actor session cap and the session-ownership map
   are kept in the backend process's memory. This is correct because the
   backend runs as a **single uvicorn process** (``backend.py`` and the
   Dockerfile ``CMD``). If you run several workers or replicas, each one
   keeps its own limits and only knows about the sessions it created.

.. tip::
   When the backend is running, the auto-generated Swagger UI is available at
   ``http://localhost:9000/docs`` (ReDoc at ``/redoc``, the raw schema at
   ``/openapi.json``) with interactive request/response examples. Note that
   ``/besser_api`` is a *router* prefix, so it does not apply to these — nor to
   ``GET /health``, the load-balancer probe, which is also mounted at the root.


File Upload Limits
------------------

- CSV files: 5 MB max
- Images: 10 MB max
- BUML Python files: 2 MB max


Environment Variables
---------------------

**Required for GitHub integration:**

- ``GITHUB_CLIENT_ID`` -- GitHub OAuth app ID
- ``GITHUB_CLIENT_SECRET`` -- GitHub OAuth app secret

**Optional:**

- ``OPENAI_API_KEY`` -- OpenAI key consumed by several features:

  - image-to-model and knowledge-graph-to-model conversion,
  - the :doc:`LLM-based agent recommendation <generators/agent_personalization>`
    endpoint (``/recommend-agent-config-llm``),
  - the BAF generator's personalization pipeline when ``agentLanguage`` /
    ``agentStyle`` / ``languageComplexity`` / ``sentenceLength`` /
    ``useAbbreviations`` differ from ``original`` (message re-writing and
    translation),
  - deployments that ship the generated agent to GitHub + Render
    (the generated ``render.yaml`` declares it as a required secret).

  The key can also be supplied per-request in the JSON body under
  ``system.openaiApiKey`` (or ``openai_api_key``). If both are set, the
  request-scoped key wins.

**Spec-Driven Agent (all optional):**

The agent's ``BESSER_LLM_*``, ``BESSER_FREE_LLM_*``,
``BESSER_SPONSORED_LLM_*`` and ``BESSER_DEMO_TOKEN`` variables -- the keyless
tiers, the caps and their hard limits, the feature flags, the context budgets,
and the durable-run storage paths -- are documented in one place, with the
agent: :doc:`spec_driven_agent/configuration`.

**Telemetry (all optional):**

- ``BESSER_TELEMETRY_ENABLED`` -- Master switch for telemetry collection.
- ``BESSER_TELEMETRY_ADMIN_TOKEN`` -- Required by ``GET /telemetry/report``;
  without it the endpoint answers ``404`` by design.
- ``BESSER_TELEMETRY_DIR`` (``/app/telemetry``) -- Where telemetry records are
  written. It is also the fallback location for provider-incident logs when
  ``BESSER_INCIDENT_LOG_DIR`` is unset.

**Other optional variables:**

- ``CORS_ORIGINS`` -- Comma-separated allowed origins; overrides the built-in
  default list
- ``WME_NODE_SERVER_URL`` -- WME Node server used by ``/get-svg`` for headless
  SVG rendering (default ``http://localhost:8080``)
- ``FEEDBACK_EMAIL`` -- Email recipients for feedback (comma-separated)
- ``SMTP_HOST`` -- SMTP server (default: ``smtp.gmail.com``)
- ``SMTP_PORT`` -- SMTP port (default: ``587``)
- ``SMTP_USERNAME`` -- SMTP authentication user (defaults to the first
  ``FEEDBACK_EMAIL`` recipient)
- ``SMTP_PASSWORD`` -- SMTP authentication password
- ``OAUTH_SESSIONS_PATH``, ``USER_TOKENS_PATH``, ``SESSION_STORE_PATH`` --
  On-disk locations for GitHub OAuth session and token state; set them to a
  persistent volume so sign-ins survive a restart
- ``GITHUB_REDIRECT_URI`` -- OAuth redirect URL (default: ``http://localhost:9000/besser_api/github/auth/callback``)
- ``DEPLOYMENT_URL`` -- Frontend URL for OAuth redirects (default: ``http://localhost:8080``)

**Agent simulation** (read once at startup, except the token):

- ``AGENT_SIMULATOR_API_TOKEN`` -- Shared secret sent as the
  ``X-Agent-Simulator-Token`` header on every HTTP request and WebSocket
  handshake to the simulator. Set the same value on the simulator container.
  Without it, the simulation endpoints return 503.
- ``AGENT_SIMULATOR_URL`` -- Simulator base URL (default:
  ``http://besser-wme-agent-simulator:8001``).
- ``AGENT_SIMULATOR_REQUIRE_AUTH`` -- Require GitHub sign-in (default: ``true``).
- ``AGENT_SIMULATOR_RESTRICT_CUSTOM_CODE`` -- Refuse agents with custom Python
  code actions (default: ``true``).
- ``AGENT_SIMULATOR_MAX_SESSIONS_PER_ACTOR`` -- Concurrent sessions per actor
  (default: ``1``).
- ``AGENT_SIMULATOR_RATE_LIMIT_MAX_REQUESTS`` / ``AGENT_SIMULATOR_RATE_LIMIT_WINDOW_SECONDS``
  -- Per-actor rate limit (default: ``12`` requests per ``60`` seconds).
- ``AGENT_SIMULATOR_RATE_LIMIT_MAX_KEYS`` -- Maximum number of actors the rate
  limiter tracks at once (default: ``10000``). The least recently seen actors
  are dropped first.
- ``AGENT_SIMULATOR_SESSION_LIFETIME_SECONDS`` -- How long a session lives.
  It is shown in ``/limits`` (``null`` when unset), and the backend forgets a
  session's ownership after this long (``900`` seconds, the simulator's
  default, when unset).
- ``AGENT_SIMULATOR_MEMORY_MB``, ``AGENT_SIMULATOR_CPU_CORES``,
  ``AGENT_SIMULATOR_DISK_MB``, ``AGENT_SIMULATOR_QUOTA_ENABLED`` -- Values
  reported by ``/limits``.


Generator Configuration
-----------------------

Each generator can receive configuration options via the ``config`` field in the
request body:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Generator
     - Configuration Options
   * - **Django**
     - ``project_name``, ``app_name``, ``containerization`` (bool)
   * - **SQL**
     - ``dialect`` (sqlite, postgresql, mysql, mssql, mariadb, oracle)
   * - **SQLAlchemy**
     - ``dbms`` (sqlite, postgresql, mysql, mssql, mariadb, oracle)
   * - **JSON Schema**
     - ``mode`` (regular, smart_data)
   * - **Qiskit**
     - ``backend`` (aer_simulator, fake_backend, ibm_quantum), ``shots``
   * - **Agent**
     - ``openai_api_key``, ``languages``, ``variations``, ``configurations``,
       ``personalizationMapping`` — see
       :doc:`generators/agent_personalization` for the variant mechanisms and
       configuration schema


Running the Backend
-------------------

Start the backend from the BESSER repository root:

.. code-block:: bash

   python besser/utilities/web_modeling_editor/backend/backend.py

The backend listens on ``http://localhost:9000/besser_api`` by default.

For the full-stack experience with Docker:

.. code-block:: bash

   docker-compose up --build
