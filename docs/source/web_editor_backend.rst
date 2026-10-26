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

Supported diagram types (the project keys): ``ClassDiagram``, ``ObjectDiagram``,
``StateMachineDiagram``, ``AgentDiagram``, ``GUINoCodeDiagram``,
``QuantumCircuitDiagram``, ``UserDiagram``, ``NNDiagram``, ``BPMN``,
``ComponentDiagram``, and ``DeploymentDiagram``. A single BPMN diagram payload
has type ``BPMNDiagram``.


Class Diagrams: Element Timestamps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every structural B-UML element carries a ``timestamp`` (its creation time),
which generators use to order members. The class-diagram converters keep it:
``class_buml_to_json`` writes an optional ISO 8601 ``timestamp`` string on the
diagram model, on each class, enumeration, attribute, enumeration literal,
method and OCL constraint element, on each association and inheritance
relationship, and on each association end (``source`` / ``target``). Only
explicit timestamps are written (``Element.has_explicit_timestamp``: given to
the constructor, set later, or read from JSON); one that defaulted to the
creation time is left out, so converting the same model twice gives the same
JSON, and exporting a timestamp-less diagram to B-UML twice gives the same file.
``process_class_diagram`` reads the field back. A missing or malformed value
falls back to the creation time, and a value with a UTC offset is converted to
UTC and stored naive, like the metamodel's own timestamps (UTC rather than the
server's local time, so the same JSON gives the same B-UML on every host).
Method parameters live
in the method signature string, so they are recreated with a fresh timestamp.

The editor itself does not keep the field yet: its element model serializes a
fixed set of properties, so a diagram opened and saved in the editor loses its
timestamps. They survive conversions that do not pass through the editor, such
as importing a B-UML file and exporting it back.


Class Diagrams: Element Ids
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Structural elements have an optional ``id`` (``Element.id``, ``None`` by
default). ``process_class_diagram`` stores the editor's element id on each
class, enumeration, enumeration literal, attribute, method, association,
association class and generalization, and ``class_buml_to_json`` writes it back
instead of minting a new one. An element keeps its id when it is renamed or the
diagram is re-converted (for example when a Spec-Driven Agent modify run adds a
class), so references from other diagrams keep resolving. An element without
an id gets a fresh uuid, as before. So does an element whose id is already
taken in the same diagram, so two elements never collapse into one, and one
whose id is not 1-128 characters of ``A-Za-z0-9_:-`` (or is ``__proto__``):
the editor uses ids as object keys and the GUI parser reads ``.`` as a path
separator. Each replacement is logged as a warning. Association ends, method
parameters and OCL constraints have no element id in the editor JSON and are
not covered. The id takes no part in equality or hashing.

Other diagrams refer to class-diagram elements by these ids:

- Object diagrams: an object's ``classId``, a slot's ``attributeId`` and a
  link's ``associationId``. ``process_object_diagram`` resolves the class and
  the attribute by id first, so an object diagram still converts after its
  class or attribute was renamed; it falls back to the names as before.
  Exporting an object model to JSON resolves the references against the class
  diagram by name, including links of compositions and aggregations.
- GUI diagrams: ``data-source``, ``data-field`` and ``label-field`` of charts
  and forms, a button's ``data-method`` and ``data-method-class``, and a
  table's lookup ``columns`` (``lookupEntity``). The GUI export writes a domain
  element's id when it has one, otherwise its name. A lookup column whose
  ``field`` does not name an association end (the editor writes the target
  class name for an end without a role) is resolved through ``lookupEntity``
  when exactly one end of the table's class leads to that class.
- Component diagrams: ``Component.realizes``.

A Spec-Driven Agent modify run re-serialises the class diagram and leaves the
other diagrams untouched, so these references keep resolving in the updated
project and in the pushed ``buml/diagrams.json``.

When a model is rebuilt on top of an existing one, such as an image imported
into a class diagram that already has elements, ``adopt_element_ids``
(``services/converters/element_ids.py``) carries the ids over with one rule:

- an element that already carries an id is matched by it;
- otherwise it is matched by name, ignoring case, within the same parent
  (types by kind and name, attributes, methods and literals within their class
  or enumeration, generalizations within their specific class);
- an id is never given to two elements, and a name shared by two previous
  elements matches neither;
- renames are not guessed: an element whose name changed gets a new id.

B-UML files: reading ``id="..."`` is always supported. The editor backend
writes ids into exported B-UML (``.py`` export, project export, Spec-Driven
Agent and GitHub pushes) only when the environment variable
``BESSER_BUML_EMIT_IDS=1`` is set; it is off by default because BESSER 8.2.1
and older cannot load a file that passes ``id=``. Without it, a project
exported to B-UML and imported again comes back with new class-diagram ids,
as before, and references from other diagrams no longer resolve. Ids of
objects, slots and links, and of diagrams themselves, are never written to
B-UML and are new after an import.


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

A third NN generator, ``forecaster``, wraps the network in a time-series
pipeline (see :doc:`generators/forecaster`). Its ``config`` holds the
forecaster settings (``window``, ``horizon``, ``season``, ``difference``,
``log``, ``calendar``, ``standardize``, ``seed`` or ``seeds``, ``description``)
plus ``channel_last``; the response is ``forecaster_<model name>.py`` (a name
outside ASCII is sent as ``filename*=UTF-8''...``). Invalid settings, and a
network whose first layer does not read the window or whose last layer does not
return the horizon, are a 400 naming every problem.

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

Projects store BPMN entries under ``BPMN``; an individual diagram payload
has type ``BPMNDiagram``. The backend converts it to ``BPMNModel``, backed by
the :doc:`BPMN metamodel <buml_language/model_types/bpmn>`.
``/export-buml`` writes executable BUML Python, ``/get-json-model`` reads
it back, ``/validate-diagram`` runs the metamodel's ``validate()``, and
the BPMN generator emits vendor-neutral BPMN 2.0 XML.


The Agentic extension is represented by ``AgenticTask``, ``AgenticGateway``,
and ``AgenticLane`` subclasses. The converter consumes the WME ``isAgentic``
flag and preserves ``reflectionMode``, ``reflectionReviewerLaneId``,
``gatewayRole``, ``trustScore``, lane ``role``, lane ``multiplicity`` (stored as
``AgenticLane.swarm_size``), ``agentDiagramRef``, and ``governanceDsl``. Flows
stay standard BPMN flows and carry no agentic fields.

In project-level deployment generation, BPMN provides the process context for
multi-agent system runtime behavior: lane ``agentDiagramRef`` values resolve gateway
owners to Agent diagrams, ``governanceDsl`` on merging gateways is attached to
the owner agent, and BPMN sequence-flow ids are used to route A2A messages into
the correct governed merge (``services/governance/swarm_bindings.py``).

An invalid Governance DSL on a merging gateway raises
``GovernanceDslValidationError``, which the endpoints report as **HTTP 422**
with the gateway name and the parser message. A missing ``governancedsl``
package raises ``ConfigurationError`` (**HTTP 500**, naming the package to
install).

Component and Deployment Diagrams
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The backend handles ``ComponentDiagram`` and ``DeploymentDiagram`` as the
whole-swarm structural and runtime-allocation views backing the
:doc:`UML Component <buml_language/model_types/uml_component>` and
:doc:`UML Deployment <buml_language/model_types/uml_deployment>` metamodels.
They are standalone diagrams for normal import/export, and they also
participate in project-level multi-agent system generation.

**Agentic profile.** The Component diagram's agentic vocabulary is carried in
the WME ``stereotype`` string. The converter promotes a component to
``AgenticComponent`` / ``Skill`` / ``Tool`` (and a dependency to
``AgenticEdge``) by matching stereotype tokens; permission scopes ride in a
``{permission: ...}`` suffix on agentic-edge stereotypes.

**JSON ↔ BUML.** ``/export-buml`` converts a Component, Deployment, or BPMN
diagram JSON into a BUML Python file; ``/get-json-model`` reads it back,
autodetecting the diagram type. ``/validate-diagram`` runs the metamodel
``validate()`` for supported diagram types.

**Cross-diagram references.** Within a project, component ``realizes`` and
deployment artifact ``manifests`` values resolve against peer diagrams;
``/export-buml`` of a project rejects dangling references. The Deployment
diagram's ``DeploymentComponent`` element is imported as a
``DeploymentComponent`` and manifests nothing itself.
Agentic deployment artifacts can also carry ``agentModelRef``: the Agent
diagram id that should be baked into that artifact's Docker Compose build
context.

**Project generation.** ``POST /generate-output-from-project`` gives deployment
generators access to the full project. For ``docker_compose``, the backend
builds an ``AgentDiagram`` id -> BUML Agent map, annotates each Agent with A2A
tags from the raw Agent diagram JSON, attaches Governance DSL from the BPMN
merge gateways, and passes the map to the Docker Compose generator so LOCAL
artifacts with resolvable ``agentModelRef`` values receive runnable BAF build
contexts.

Middleware and Request Security
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Four middlewares wrap every request, outermost first:

- **Request logging** (``middleware/request_logging.py``) -- a UUID request ID,
  timing, and a warning for any request slower than one second.
- **CORS** -- origins come from ``CORS_ORIGINS`` (comma-separated) or the
  built-in ``DEFAULT_CORS_ORIGINS``. Credentials are allowed; methods are
  ``GET``/``POST``/``PUT``/``DELETE``/``OPTIONS``; the allowed request headers
  are ``Content-Type``, ``X-GitHub-Session``, ``Content-Disposition``,
  ``Authorization`` and ``Idempotency-Key``, and ``Content-Disposition`` plus ``X-BESSER-Run-Id`` are
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
- ``GET /besser_api/`` -- API root: API version, the running BESSER version
  (``besser_version``) and, when the Docker image was built with ``GIT_SHA``,
  its commit (``besser_build``), the list of supported generator keys, and a
  map of the main endpoint paths. The editor reads ``besser_version`` to stamp
  its JSON exports: ``besserVersion`` is that value, omitted when the backend
  is unreachable, and ``editorVersion`` is the web editor's own package
  version, built into the frontend.

Code Generation
^^^^^^^^^^^^^^^

- ``POST /generate-output`` -- Single diagram to code generation
- ``POST /generate-output-from-project`` -- Multi-diagram project generation (e.g., WebApp needs ClassDiagram + GUINoCodeDiagram)

Every ZIP these endpoints return (and every GitHub deploy) has a
``BESSER_GENERATION.md`` at its root: the BESSER version, the build commit when
set, the generator and the user-facing options that shaped the output (never
credentials), plus a link back to the BESSER repository. It carries no timestamp, so the same model still yields the same
files. A single-file download starts with a ``Generated by BESSER <version>``
comment when it is a ``.py`` or ``.sql`` file; JSON, BPMN XML and Turtle files
have no safe place for one and are returned unchanged.

Spec-Driven Agent
^^^^^^^^^^^^^^^^^

Eleven endpoints under ``/spec-driven/*`` back the
:doc:`Spec-Driven Agent <spec_driven_agent/index>` -- the hybrid pipeline that
runs a deterministic generator, lets an LLM customise its output, then validates
and repairs the result: ``generate``, ``preview``, ``config``,
``runs/{run_id}``, ``runs/{run_id}/events``, ``runs/{run_id}/events.json``,
``resume/{run_id}``, ``cancel/{run_id}``, ``download/{run_id}``,
``push-to-github`` and ``import-github-run``.

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
- ``/spec-driven/import-github-run`` answers ``400`` for an owner or repository
  name that does not match ``^[A-Za-z0-9_.-]+$`` or consists only of dots
  (both are interpolated into GitHub API paths), and for a repository over the
  import limits: a 100 MB archive, 500 MB unpacked, or 50,000 files. The
  response detail says which limit was hit.

Telemetry (research study mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``POST /telemetry/event`` -- Accept one opt-in study telemetry event. Always
  returns 204 for well-formed input whether or not it was stored, so the
  endpoint cannot be used to probe server configuration.
- ``GET  /telemetry/report`` -- Aggregated report (Markdown by default, CSV via
  ``?format=csv``). Requires an ``X-Telemetry-Token`` header matching
  ``BESSER_TELEMETRY_ADMIN_TOKEN``; a missing or wrong token, including one
  with non-ASCII characters, answers ``403``.

Study mode records how participants of a facilitated research study use the
editor. It is off by default and needs two opt-ins at once:

- **The participant's link.** The editor tab must be opened with
  ``?study=<label>`` (``?pilot=<label>`` is accepted as an alias). The label
  is a short token such as ``P3``, matching ``^[A-Za-z0-9_-]{1,16}$``, never a name or email. It lives in the
  tab's ``sessionStorage``, the editor shows a notice for as long as it is
  set, and closing the tab ends it. Every event carries this label; one
  without it is dropped.
- **The server's switch.** ``BESSER_TELEMETRY_ENABLED=1`` (or ``true``).
  Without it nothing is written, even for labelled events.

What is recorded, per tab session id and participant label:

- ``prompt`` (from the modeling agent) -- the message text (at most 2000
  characters), what the assistant did with it, and the diagram type.
- ``run_summary`` (recorded in-process by the Spec-Driven runner, never
  accepted over HTTP) -- outcome, duration, turns, tokens, cost, model,
  validation blockers, and how many output files the generator wrote versus
  the LLM. For these runs the run trace (which includes the first 500
  characters of the instructions) is also copied to
  ``<BESSER_TELEMETRY_DIR>/traces/``.
- ``delivery`` (from the editor) -- a download, a push to GitHub, or a
  continue-from-repository import, with the run id.

Each session is one append-only JSON-lines file,
``<BESSER_TELEMETRY_DIR>/<session>.jsonl``; the collector caps each payload at
8 KB. Nothing expires automatically: the operator deletes the files when the
study's data is no longer needed. Runs of sessions without a study label are
never written to this directory.

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
- ``POST /sdmx-to-domain-model`` -- The structure of an SDMX dataflow to domain model JSON, same response as ``/csv-to-domain-model``.
  Body: ``{"provider": "ecb", "dataflow": "EXR"}`` or ``{"baseUrl": "https://...", "dataflow": "AGENCY,ID,VERSION"}``
  (``agency``, ``version`` and ``modelName`` are optional; ``baseUrl`` only when ``BESSER_SDMX_ALLOW_CUSTOM_URL`` is on).
  Fetched in safe mode: https on port 443 to public addresses only, each connection pinned to the address that was
  checked, within the ``BESSER_SDMX_*`` limits below. 400 for invalid input, 404 for an unknown dataflow, 413 when
  too large, 502 when the provider fails, 503 when too many imports are running, 504 when the import takes too long.
  See :doc:`buml_language/model_building/sdmx_to_buml`.
- ``GET /sdmx-providers`` -- The preset SDMX providers (``key``, ``name``, ``baseUrl``, ``agency``) and ``customUrlAllowed``
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
- ``GET /github/repos``, ``GET /github/branches``, ``GET /github/commits``, ``GET /github/contents``, ``GET /github/file/exists`` -- Repository browsing for the editor's save/load flow (``/repos`` includes each repository's topics; ``/commits`` marks each commit's ``origin``)
- ``POST /github/project/save``, ``GET /github/project/load``, ``GET /github/project/load-commit``, ``POST /github/project/create-repo`` -- Store and retrieve a BESSER project in a repository (including loading it back from a specific commit)
- ``GET /github/repo-summary`` -- What a repository holds: the model's name, diagrams per type, classes, and the last commit
- ``GET /github/sync-status`` -- Commits on a branch since ``base_sha``, the commit the editor last synced
- ``GET /github/model-history`` -- Recent commits that changed the model, each with its class-diagram changes
- ``POST /github/pull-request`` -- Open a pull request, e.g. for a push GitHub could not merge
- ``GET /github/app/capabilities``, ``POST /github/app/go-live`` -- "Go live": commit Render deploy files into the project's own repository, for its real folder layout (see below). Off unless ``BESSER_DEPLOY_FROM_REPO=true``.

Every repository BESSER writes has one layout, defined in
``services/deployment/repo_model.py``: the model in ``buml/diagrams.json`` (a V2
project export; the legacy ``besser-project.json`` is still read), the B-UML
sources in ``buml/*.py``, the owning project in ``.besser/project.json``, and the
``besser`` topic on the repository. Commits BESSER makes end with a
``BESSER-Origin: spec-driven-agent`` or ``BESSER-Origin: editor`` trailer.

Writes never revert newer commits. ``/github/project/save`` and
``/spec-driven/push-to-github`` take the commit the editor last synced
(``base_commit_sha``). When the branch has moved past it, the new commit is made
on top of that base and merged into the branch with GitHub's merge API, so both
sides' changes are kept. A merge GitHub cannot resolve is left on a
``besser/sync-<sha>`` branch and reported as ``status: "conflict"``, and nothing
is overwritten.

The base of an agent push is the commit the run itself started from, recorded
in the run's workspace (``.besser_github_source.json``, written by the import
and updated by each push, and copied into every modify run seeded from that
run). The ``base_commit_sha`` the editor sends is used only for runs with no
recorded start, so pulling while a run is open cannot make its push revert the
pulled commits.

**Go live** (``POST /github/app/go-live``, behind ``BESSER_DEPLOY_FROM_REPO``)
deploys the code that is in the project's repository, which is the code the
Spec-Driven Agent wrote. ``services/deployment/render_blueprint.py`` reads the
repository's layout: a FastAPI backend (``main_api.py`` beside
``requirements.txt``) and an optional Vite frontend, at the root or nested (the
agent puts web apps under ``web_app/``), with the frontend's real build
folder. It writes ``render.yaml`` and the frontend's ``.env.production``, and
commits only those files. A layout it cannot deploy (Django, Flutter, several
backends, built-in agents) is refused with ``422`` and the reason; an existing
``render.yaml`` BESSER did not write is never overwritten (``409``). The
``render.yaml`` starts with a marker line; for a repository whose
``render.yaml`` carries it, every agent push regenerates both files for the
pushed tree and keeps the service names, so the URLs stay the same. Render's
build filter skips rebuilds for saves that only touch ``buml/`` or
``.besser/``. The classic ``/github/deploy-webapp`` is unchanged and shares the
same service writer.
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

**Telemetry, research study mode (all optional):**

- ``BESSER_TELEMETRY_ENABLED`` -- Master switch for study telemetry; off
  unless ``1`` or ``true``.
- ``BESSER_TELEMETRY_ADMIN_TOKEN`` -- Required by ``GET /telemetry/report``;
  without it the endpoint answers ``404`` by design.
- ``BESSER_TELEMETRY_DIR`` (``/app/telemetry``) -- Where study records are
  written, and where the traces of study runs are copied when
  ``BESSER_INCIDENT_LOG_DIR`` is unset.
- ``BESSER_PILOT_LLM_MODEL`` -- Keyless model a study session starts on (it
  must be one of the free-tier models the server offers). Unset, study
  sessions get the ordinary default.

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

**SDMX import** (``POST /sdmx-to-domain-model``, read once at startup):

- ``BESSER_SDMX_ALLOW_CUSTOM_URL`` (off) -- Accept an SDMX API that is not one of
  the preset providers. Off by default, like ``BESSER_LLM_ALLOW_CUSTOM_BASE_URL``:
  even with its public-address checks, a hosted server should not fetch any URL
  for anyone. Local and on-premises installs can turn it on.
- ``BESSER_SDMX_TIMEOUT`` (``60``) -- Seconds for one whole import: every request,
  redirect and retry, enforced while the answer is read.
- ``BESSER_SDMX_MAX_MB`` (``32``) -- Largest single structure message. The largest
  seen, Eurostat Comext's ``DS-045409``, is 23.6 MB; parsing takes about 15 times a
  message's size in memory.
- ``BESSER_SDMX_MAX_TOTAL_MB`` (``64``) -- All the messages of one import together.
- ``BESSER_SDMX_MAX_CONCURRENT_IMPORTS`` (``2``) -- Imports running at once; more
  are answered with 503, so imports cannot fill the thread pool the conversion and
  generation endpoints share.

The import does not use the environment's HTTP proxies (a proxy would resolve the
host itself, which the pinning prevents); it does use the CA bundle of
``REQUESTS_CA_BUNDLE`` or ``CURL_CA_BUNDLE``, for servers behind a TLS-inspecting
gateway.

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

The backend needs packages that the core ``besser`` install does not include
(FastAPI, openpyxl, the LLM SDKs, ...). Install them once, then start the
backend with the ``besser-wme-backend`` command, which ``pip install besser``
(or ``pip install -e .`` in a checkout) puts on your ``PATH``:

.. code-block:: bash

   pip install besser
   pip install -r besser/utilities/web_modeling_editor/backend/requirements.txt
   besser-wme-backend

The requirements file ships inside the installed package too. If a backend
dependency is missing, ``besser-wme-backend`` says which one and prints the
``pip install -r`` command with the file's full path, instead of an import
traceback.

The backend listens on ``http://localhost:9000/besser_api`` by default. The
command accepts:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Option
     - Default
     - Meaning
   * - ``--host``
     - ``0.0.0.0``
     - Interface to bind. The default accepts connections on every interface;
       pass ``127.0.0.1`` to accept local connections only.
   * - ``--port``
     - ``9000``
     - Port to listen on. The frontend's development server expects ``9000``.
   * - ``--log-level``
     - ``info``
     - uvicorn log level (``critical``, ``error``, ``warning``, ``info``,
       ``debug``, ``trace``).

From a checkout, ``python -m besser.utilities.web_modeling_editor.backend.backend``
(what the Docker image runs) and ``python besser/utilities/web_modeling_editor/backend/backend.py``
start the same server and take the same options. The backend always runs as a
single process (see the note under `Agent Simulation`_).

Startup and shutdown work (the temp-file cleanup, the spec-driven download
registry restore and sweepers) runs in the FastAPI ``lifespan`` handler
declared in ``backend.py``; the agent simulator service declares its own
``lifespan`` in ``agent_simulator_api.py``.

The Spec-Driven Agent's PIA provider is LIST's private AI gateway, an
OpenAI-compatible endpoint backed by AWS Bedrock and reachable only from the
LIST network. To use it, or the Local (e.g. Ollama) provider, from this local
backend, start it with ``BESSER_LLM_ALLOW_CUSTOM_BASE_URL=true`` (off by
default; the local ``docker-compose.yml`` sets it). See
:doc:`spec_driven_agent/configuration`.

For the full-stack experience with Docker:

.. code-block:: bash

   docker-compose up --build
