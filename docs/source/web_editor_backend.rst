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

Supported diagram types: ``ClassDiagram``, ``ObjectDiagram``,
``StateMachineDiagram``, ``AgentDiagram``, ``GUINoCodeDiagram``,
``QuantumCircuitDiagram``, ``NNDiagram``.


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


API Endpoints
-------------

Code Generation
^^^^^^^^^^^^^^^

- ``POST /generate-output`` -- Single diagram to code generation
- ``POST /generate-output-from-project`` -- Multi-diagram project generation (e.g., WebApp needs ClassDiagram + GUINoCodeDiagram)

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

- ``OPENAI_API_KEY`` -- Required for image-to-model and knowledge-graph-to-model conversion
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
     - ``openai_api_key``, ``languages``, ``variations``, ``configurations``, ``personalizationMapping``


Running the Backend
-------------------

Start the backend from the BESSER repository root:

.. code-block:: bash

   python besser/utilities/web_modeling_editor/backend/backend.py

The backend listens on ``http://localhost:9000/besser_api`` by default.

For the full-stack experience with Docker:

.. code-block:: bash

   docker-compose up --build
