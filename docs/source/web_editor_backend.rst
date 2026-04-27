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
- **``routers/error_handler.py``** -- Centralized ``@handle_endpoint_errors`` decorator

Additional infrastructure:

- **``middleware/request_logging.py``** -- Structured request logging with unique IDs and performance timing
- **``services/cleanup.py``** -- Background temp-file cleanup (removes stale directories every hour)
- **``services/exceptions.py``** -- Custom exception hierarchy (``ConversionError``, ``ValidationError``, ``GenerationError``, ``DeploymentError``)
- **``constants/constants.py``** -- API version, temp prefixes, generator defaults, CORS origins
- **``models/responses.py``** -- Standardized Pydantic response models

Multi-Diagram Projects
^^^^^^^^^^^^^^^^^^^^^^

Projects support **multiple diagrams per type** via
``diagrams: Dict[str, List[DiagramInput]]``. Each diagram can reference other
diagrams by ID through the ``references`` field, and the active diagram per type
is tracked via ``currentDiagramIndices``. Old single-diagram projects are
auto-converted by a Pydantic model validator for backward compatibility.


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
