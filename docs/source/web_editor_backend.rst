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

- **``routers/generation_router.py``** -- Code generation (single-diagram and project-based)
- **``routers/conversion_router.py``** -- BUML import/export, CSV reverse engineering, image-to-model
- **``routers/validation_router.py``** -- Diagram validation (metamodel + OCL constraints)
- **``routers/deployment_router.py``** -- GitHub deployment and Docker integration
- **``routers/agent_simulator_router.py``** -- Live agent simulation (see `Agent Simulation`_)
- **``routers/error_handler.py``** -- Centralized ``@handle_endpoint_errors`` decorator
- **``routers/auth.py``** -- Shared GitHub-session gate (``require_github_session``, HTTP 401)

Additional infrastructure:

- **``middleware/request_logging.py``** -- Structured request logging with unique IDs and performance timing
- **``services/cleanup.py``** -- Background temp-file cleanup (removes stale directories every hour)
- **``services/exceptions.py``** -- Custom exception hierarchy (``ConversionError``, ``ValidationError``, ``GenerationError``, ``DeploymentError``).
  ``CodeValidationError`` (invalid custom Python code in an agent) is a ``ValidationError`` and maps to HTTP 400.
- **``constants/constants.py``** -- API version, temp prefixes, generator defaults, CORS origins
- **``models/responses.py``** -- Standardized Pydantic response models

Multi-Diagram Projects
^^^^^^^^^^^^^^^^^^^^^^

Projects support **multiple diagrams per type** via
``diagrams: Dict[str, List[DiagramInput]]``. Each diagram can reference other
diagrams by ID through the ``references`` field, and the active diagram per type
is tracked via ``currentDiagramIndices``. Old single-diagram projects are
auto-converted by a Pydantic model validator for backward compatibility.

Supported diagram types: ``ClassDiagram``, ``ObjectDiagram``,
``StateMachineDiagram``, ``AgentDiagram``, ``GUINoCodeDiagram``,
``QuantumCircuitDiagram``, ``NNDiagram``, ``BPMN``.


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
processor and surfaces ``ValueError`` (plus ``KeyError``/``TypeError``/
``AttributeError`` on malformed payloads) as per-line validation errors
rather than 500 responses. The processor verifies whitelists for
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

API Endpoints
-------------

Code Generation
^^^^^^^^^^^^^^^

- ``POST /generate-output`` -- Single diagram to code generation
- ``POST /generate-output-from-project`` -- Multi-diagram project generation (e.g., WebApp needs ClassDiagram + GUINoCodeDiagram)

Agent Personalization
^^^^^^^^^^^^^^^^^^^^^

These endpoints back the :doc:`agent personalization <generators/agent_personalization>`
workflow. They consume a serialized :doc:`UserDiagram <buml_language/model_types/user_diagram>`
and return a structured agent configuration.

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
  deterministic recommender (every rule, evidence, priority, and payload).
  Useful for UIs that want to show "why this recommendation".
- ``POST /transform-agent-model-json`` -- Apply an agent configuration to an
  agent diagram and return the personalized agent model JSON (used by the
  editor's "apply personalization" action).

Conversion
^^^^^^^^^^

- ``POST /export-buml`` -- Diagram JSON to BUML Python code
- ``POST /export-project-as-buml`` -- Full project to BUML Python code
- ``POST /get-json-model`` -- BUML Python file to JSON (auto-detects diagram type)
- ``POST /get-project-json-model`` -- BUML project file to JSON
- ``POST /get-json-model-from-image`` -- Image to ClassDiagram JSON (requires OpenAI API key)
- ``POST /get-json-model-from-kg`` -- Knowledge graph (TTL/RDF/JSON) to ClassDiagram JSON
- ``POST /csv-to-domain-model`` -- CSV files to domain model JSON
- ``POST /transform-agent-model-json`` -- Agent model transformation with personalization

Validation
^^^^^^^^^^

- ``POST /validate-diagram`` -- Unified diagram validation (metamodel + OCL constraints)
- ``POST /semantic-consistency-check`` -- SAT-based semantic diagram validation (class diagram + OCL constraints)
- ``POST /generate-object-diagram`` -- SAT-based generation of a witness of semantic consistency (class diagram + OCL constraints) 
 
Deployment
^^^^^^^^^^

- ``POST /deploy-app`` -- Docker Compose deployment for Django projects
- ``POST /feedback`` -- User feedback submission

Standalone Chatbot Deployment
"""""""""""""""""""""""""""""

The GitHub deploy endpoint supports a ``target: "agent"`` flag in
``deploy_config`` that switches the output from a full web-app to a standalone
chatbot (Streamlit frontend, Python backend, single-service Render blueprint).
This is the path used by the editor's "Deploy chatbot" action and reuses the
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
- ``POST /github/deploy-webapp`` -- Deploy generated app to GitHub repository

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
   ``http://localhost:9000/besser_api/docs`` with interactive request/response examples.


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
- ``FEEDBACK_EMAIL`` -- Email recipients for feedback (comma-separated)
- ``SMTP_HOST`` -- SMTP server (default: ``smtp.gmail.com``)
- ``SMTP_PORT`` -- SMTP port (default: ``587``)
- ``SMTP_PASSWORD`` -- SMTP authentication password
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
