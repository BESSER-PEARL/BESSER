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
