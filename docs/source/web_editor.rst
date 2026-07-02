Web Modeling Editor
===================

One of the practical ways to use BESSER is through the Web Modeling Editor, where you can rapidly 
design :doc:`B-UML <../buml_language>` models and use the :doc:`BESSER code generators <../generators>`.

.. note::
   The BESSER Web Modeling Editor is now live and available at
   `editor.besser-pearl.org <https://editor.besser-pearl.org>`_.
   You can access and use it directly in your browser without installing anything locally.

The full Web Modeling Editor documentation is published as a separate subproject:
`BESSER Web Modeling Editor documentation <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/>`_.
For contributor workflows like adding a new diagram type, see
`Adding a New Diagram Type <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/contributing/new-diagram-guide/index.html>`_.

.. image:: ./img/besser_new.gif
   :width: 900
   :alt: BESSER Web Modeling Editor interface
   :align: center

The editor's source code is available in the
`BESSER-WEB-MODELING-EDITOR GitHub repository <https://github.com/BESSER-PEARL/BESSER-WEB-MODELING-EDITOR>`_.
The frontend is vendored into this repository as a git submodule at
``besser/utilities/web_modeling_editor/frontend``, while the backend services live here under
``besser/utilities/web_modeling_editor/backend``.

Class Diagram Notation
----------------------

Class diagrams can be rendered in two equivalent notations, chosen from
**Project Settings → Display → Class Diagram Notation**. The underlying B-UML
model is identical in both cases — the notation only affects how the diagram is
drawn, so switching is lossless.

**UML** (default) — standard UML class notation with visibility prefixes,
``{id}`` markers on identifier attributes, and ``min..max`` multiplicities.

.. image:: ./img/class_diagram_uml.png
   :width: 800
   :alt: Library model rendered in UML notation
   :align: center

**ER** (Chen-style) — entity/relationship flavor aimed at users with a database
modeling background. Identifier attributes (``is_id``) are underlined, the
methods compartment is hidden, associations are drawn as named diamonds, and
multiplicities are shown as ``(min,max)`` cardinality pairs (with ``*`` rendered
as ``N``). Inheritance relationships keep their UML rendering since there is no
direct ER equivalent.

.. image:: ./img/class_diagram_er.png
   :width: 800
   :alt: Library model rendered in Chen-style ER notation
   :align: center

Agent Personalization
---------------------

The editor exposes the full :doc:`agent personalization <generators/agent_personalization>`
workflow under the *Agent* diagram type:

- **User diagrams** (see :doc:`buml_language/model_types/user_diagram`) describe
  an end-user — age, languages, skills, education, disabilities. They can be
  created alongside the agent diagram in the same project.
- The **Agent Configuration** panel lets you edit the structured configuration
  (language, style, readability, modality, platform, LLM…) directly, or ask the
  backend for a recommendation based on one of the saved user profiles.

  - *Deterministic mapping* produces a configuration from a literature-backed
    rule table — no OpenAI key needed.
  - *LLM recommendation* calls the configured OpenAI model. An API key must be
    set either in the panel or as the ``OPENAI_API_KEY`` environment variable
    on the backend.
- When generating or deploying, three variant mechanisms can run in parallel:
  multi-language output, configuration variants, and per-profile
  **personalization mappings** that bundle one agent variant per mapped user.

The *Deploy chatbot* action reuses the same pipeline to push a standalone,
Streamlit-based agent to a GitHub repository with a ready-to-use Render
blueprint. See :doc:`web_editor_backend` for the underlying endpoints.

AI Assistant & Vibe-Driven Generation
-------------------------------------

The editor ships with an AI assistant (a floating widget and a workspace
drawer) backed by the :doc:`modeling agent <generators/baf>`. Through it you
can create and modify diagrams in natural language, ask questions about your
model, and trigger code generation — including the
:doc:`Vibe-Driven (LLM-Augmented) Generator <generators/vibe_driven>`.

When you ask for a customised codebase ("a FastAPI backend for this model with
JWT auth and Docker", "build this in Rust"), the assistant routes the request
to the Vibe-Driven Generator. Because that generator calls a commercial LLM
with **your own API key** (BYOK), the assistant always asks for explicit
confirmation before starting — a run never spends your key silently.

The run streams over `Server-Sent Events
<https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events>`_ so the
assistant can show the phase timeline, the LLM's tool calls, and a live
cost/runtime meter as it works.

**Endpoints** (all under the ``/besser_api`` prefix):

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Endpoint
     - Method
     - Purpose
   * - ``/smart-generate``
     - POST
     - Start a run; streams SSE, returns a download URL on completion.
   * - ``/smart-preview``
     - POST
     - Pre-flight plan (primary model + target generator). No API key, no
       LLM call.
   * - ``/smart-gen/config``
     - GET
     - Feature flags and provider default models.
   * - ``/resume-smart-gen/{run_id}``
     - POST
     - Resume an interrupted run from its checkpoint (streams SSE).
   * - ``/cancel-smart-gen/{run_id}``
     - POST
     - Cancel an in-flight run.
   * - ``/download-smart/{run_id}``
     - GET
     - Download the generated ZIP/file (re-fetchable within a TTL).

**Request body of** ``POST /smart-generate``:

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
     - BYOK. Sent only in the body, never logged or persisted.
   * - ``provider``
     - ``anthropic`` | ``openai``
     - Which provider the key is for. Default ``anthropic``.
   * - ``llm_model``
     - string (optional)
     - Model override; falls back to the provider default.
   * - ``max_cost_usd``
     - float
     - Soft spend cap, clamped to the server hard cap (default 1.0, max 2.0).
   * - ``max_runtime_seconds``
     - int
     - Soft runtime cap, clamped to the server hard cap (default 600, max 900).

**SSE event types** emitted by ``/smart-generate``:

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
     - The LLM invoked a tool (read/write/modify a file, run a generator).
   * - ``cost``
     - Periodic cost / runtime / turn-count tick.
   * - ``done``
     - Success. Carries ``downloadUrl``, ``fileName``, and the run recipe.
   * - ``error``
     - ``INVALID_KEY`` / ``UPSTREAM_LLM`` / ``INTERNAL`` / ``BAD_REQUEST`` /
       ``CANCELLED`` are terminal; ``COST_CAP`` / ``TIMEOUT`` are
       non-terminal warnings emitted just before ``done``.

**Example** — start a run with ``curl`` and read the stream::

   curl -N -X POST https://<host>/besser_api/smart-generate \
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
   data: {"event":"start","runId":"a1b2c3","provider":"openai","llmModel":"gpt-4o","maxCost":1.0,"maxRuntime":600}

   event: phase
   data: {"event":"phase","phase":"generate","message":"Running the FastAPI generator…"}

   event: tool_call
   data: {"event":"tool_call","turn":1,"tool":"write_file","status":"done","summary":"auth.py"}

   event: cost
   data: {"event":"cost","usd":0.07,"turns":3,"elapsedSeconds":24.1}

   event: done
   data: {"event":"done","runId":"a1b2c3","downloadUrl":"/besser_api/download-smart/a1b2c3","fileName":"besser-smartgen.zip","isZip":true}

Then issue a ``GET`` to the ``downloadUrl`` to retrieve the ZIP — it is
re-fetchable within the run's TTL.

Neural Network Diagram
----------------------

The editor supports neural network architecture modeling through the *NN*
diagram type. The underlying B-UML model captures layers, tensor operations,
and training metadata in a form that the code generators use to produce
runnable training code.

- **Layers** cover the standard catalog (Conv1D/2D/3D, Pooling,
  SimpleRNN/LSTM/GRU, Linear, Flatten, Embedding, Dropout, LayerNorm,
  BatchNorm). Each layer carries the parameters needed to define it, such
  as ``kernel_dim``, ``hidden_size`` or ``return_type``.
- **Tensor operations** (``concatenate``, ``multiply``, ``matmultiply``,
  ``reshape``, ``transpose``, ``permute``) compose layer outputs and are
  placed inline with layers in the same container.
- **NNContainer** holds the modules of a neural network. A diagram has one
  top-level container and may include additional containers used as
  sub-networks, linked into the main one via **NNReference** elements.
- **NNNext** relationships order modules within a container, defining the
  flow of data through the network.
- **Training Dataset** and **Test Dataset** elements describe the data
  feeding into the model: name, path, task type
  (``binary``/``multi_class``/``regression``), and input format
  (``csv``/``images``). When the input format is ``images``, an **Image**
  element is attached to the dataset, holding the shape and an optional
  normalization flag.
- A **Configuration** element captures training hyperparameters: batch size,
  epochs, learning rate, optimizer, loss function, metrics, plus optional
  weight decay and momentum.

The *Generate* menu offers four output variants: **PyTorch** or
**TensorFlow**, each in **Subclassing** or **Sequential** form. Diagrams are
checked against a set of metamodel rules (cross-reference integrity,
identifier safety, numerical bounds, dataset consistency) and the
**Validate** action surfaces any violations before code is generated. See
:doc:`buml_language/model_types/nn` for the metamodel reference and
:doc:`generators/pytorch` / :doc:`generators/tensorflow` for the generator
details.

BPMN Diagram
------------

The editor supports BPMN 2.0 process modelling through the *BPMN* diagram
type. The underlying B-UML model (see :doc:`buml_language/model_types/bpmn`)
covers the WME palette one-to-one and follows the OMG BPMN 2.0.2 abstract
syntax.

- **Flow nodes** cover the standard catalog: ``BPMNTask`` (with
  ``taskType`` user / service / send / receive / manual / business-rule /
  script / default and an optional ``marker`` for loops or multi-instance),
  ``BPMNSubprocess``, ``BPMNTransaction``, ``BPMNCallActivity``,
  ``BPMNStartEvent`` / ``BPMNIntermediateEvent`` / ``BPMNEndEvent`` (with
  a flat ``eventType`` enum the backend splits into the spec's orthogonal
  direction × event-definition pair), and ``BPMNGateway``
  (``exclusive`` / ``inclusive`` / ``parallel`` / ``complex`` /
  ``event-based``).
- **Data and artifacts** are ``BPMNDataObject``, ``BPMNDataStore``,
  ``BPMNAnnotation``, ``BPMNGroup``.
- **Containment** is expressed via ``BPMNPool`` (a participant) holding
  ``BPMNSwimlane``\ s and flow nodes; sub-processes can nest flow nodes.
  Pool-less diagrams (one bare process) are valid.
- **Flows** all use the single ``BPMNFlow`` relationship type; the
  ``flowType`` field (``sequence`` / ``message`` / ``association`` /
  ``data association``) and ``isDefault`` flag select the four metamodel
  edge classes on the backend side.

The editor round-trips ``.bpmn`` files entirely in the browser (BPMN 2.0
XML import / export). The backend converters
(``process_bpmn_diagram`` / ``bpmn_object_to_json``) handle the
JSON ↔ B-UML side, and ``bpmn_model_to_code`` / ``bpmn_buml_to_json`` close the
round-trip through executable BUML ``.py`` files.

.. note::
   The frontend BPMN editor is being integrated into the
   `BESSER-WEB-MODELING-EDITOR <https://github.com/BESSER-PEARL/BESSER-WEB-MODELING-EDITOR>`_
   repository; check there for the latest availability.

Backend API Reference
---------------------

The backend services that power code generation, validation, and deployment are
documented separately. If you are integrating with the backend API or extending it,
see:

.. toctree::
   :maxdepth: 1

   web_editor_backend

.. note::
   The BESSER Web Modeling Editor is based on a fork of the
   `Apollon project <https://apollon-library.readthedocs.io/en/latest/>`_, a UML modeling editor.
