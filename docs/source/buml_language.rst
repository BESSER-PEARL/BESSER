B-UML Language
==============

B-UML is BESSER's model language. A model describes the structure or behaviour
of a system; a generator turns that model into code for a chosen technology.
Its Python metamodel is inspired by UML and extends it for screens, agents,
processes, neural networks, and other domains.

Choose a model type
-------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - You want to describe
     - Model type
   * - Data entities and relationships
     - :doc:`Class diagram <buml_language/model_types/structural>`
   * - Instances of those entities
     - :doc:`Object diagram <buml_language/model_types/object>`
   * - Screens, forms, and navigation
     - :doc:`GUI <buml_language/model_types/gui>`
   * - Data rules and constraints
     - :doc:`OCL <buml_language/model_types/ocl>`
   * - States and transitions
     - :doc:`State machine <buml_language/model_types/state_machine>`
   * - Conversational behaviour
     - :doc:`Agent <buml_language/model_types/agent>`
   * - Processes and workflows
     - :doc:`BPMN <buml_language/model_types/bpmn>`
   * - Users and personalisation
     - :doc:`User diagram <buml_language/model_types/user_diagram>`
   * - Infrastructure
     - :doc:`Deployment <buml_language/model_types/deployment>`
   * - Product variants
     - :doc:`Feature model <buml_language/model_types/feature_model>`
   * - Training architectures
     - :doc:`Neural network <buml_language/model_types/nn>`
   * - Quantum circuits
     - :doc:`Quantum <buml_language/model_types/quantum>`

Create or import a model
------------------------

Start with :doc:`the browser editor <start/browser>` or
:doc:`the Python tutorial <start/first-model>`. More specialised paths include
:doc:`text grammars <buml_language/model_building/grammars>`,
:doc:`draw.io import <buml_language/model_building/drawio_structural>`,
:doc:`diagram images <buml_language/model_building/image_to_buml>`,
:doc:`knowledge graphs <buml_language/model_building/kg_to_buml>`, and
:doc:`UI mockups <buml_language/model_building/mockup_to_buml>`.
Supported model types and prerequisites are listed in each guide.

Language reference
------------------

.. toctree::
   :maxdepth: 1

   buml_language/model_types
   buml_language/model_building
   besser_action_language
