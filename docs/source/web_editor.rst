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

The editor also supports BESSER's :ref:`bpmn-agentic-extension`. Agentic BPMN
is the process view within a multi-agent system: lanes identify participating
agent roles, tasks describe the work inside the process, and merging gateways
can carry Governance DSL for governed merge points.

Agentic BPMN JSON uses the same BPMN diagram shape with additional fields on
agentic elements:

- Tasks carry ``reflectionMode``, ``trustScore``, and optional
  ``agentDiagramRef`` links to Agent diagrams.
- Gateways carry ``gatewayRole``, ``trustScore``, and optional
  ``governanceDsl``. Agentic gateways are limited to the gateway kinds the
  backend metamodel accepts.
- Lanes carry ``role``, ``trustScore``, optional ``agentDiagramRef``, and
  ``multiplicity`` for the number of identical agent instances represented by
  the lane.

The backend converters construct the corresponding ``AgenticTask`` /
``AgenticGateway`` / ``AgenticLane`` subclasses on import, re-emit the WME JSON
shape on export, and preserve the agentic information in BPMN XML
``<extensionElements>``. Agent-to-agent runtime wiring is represented by A2A
tags in the derived Agent diagrams and consumed by project-level deployment
generation; it is not encoded as special BPMN message-flow attributes.

Component and Deployment Diagrams
---------------------------------

The editor supports UML 2.5 **Component** and **Deployment** diagrams as two
distinct diagram types. Within a multi-agent system (MAS) workflow, these diagrams 
provide system-wide views: Component diagrams describe the agents, capabilities, and
agentic dependencies; Deployment diagrams allocate those agents to runtime
nodes and carry the instance counts used by deployment generators.

- A **Component diagram** holds components, subsystems, and interfaces wired
  by provided/required-interface and dependency relationships. The agentic
  profile is carried through the editor's ``stereotype`` field: a component
  stereotyped ``solution`` / ``supervision`` / ``consensus`` /
  ``collaboration`` becomes an agent; ``skill`` / ``tool`` promote a
  component to that capability subtype; a dependency stereotyped
  ``delegates`` / ``has`` / ``uses`` / ... becomes a typed agentic edge. A
  dependency can also carry a permission suffix, e.g.
  ``delegates {permission: repo:merge:approve}``.
- A **Deployment diagram** holds nodes and artifacts. An artifact's
  multiplicity on a node is written as a ``[N]`` / ``[N..M]`` suffix on the
  artifact name. Agentic artifacts can also carry ``agentModelRef`` so the
  backend can resolve the deployed artifact back to the Agent diagram that
  defines its runtime behaviour.
- Project-level derivation connects the views: Agentic BPMN lanes can derive
  Component agents, Components can derive Deployment artifacts, and linked
  Deployment artifacts can be baked into runnable BAF agent build contexts by
  the Docker Compose generator.

Both diagram types round-trip losslessly: ``/export-buml`` produces BUML
Python and ``/get-json-model`` reads it back. See
:doc:`web_editor_backend` for the endpoints.

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
