Web Modeling Editor
===================

The editor lets you draw B-UML models, describe changes to an assistant, and
download generated code. Open `editor.besser-pearl.org
<https://editor.besser-pearl.org>`_ to use the hosted version.

Start using the editor
----------------------

* :doc:`start/browser`: create a project and download your first code.
* `Editor user guide <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/>`_:
  diagram tools, project backups, display settings, and AI features.
* :doc:`guides/choose-generator`: choose an output for your model.
* :doc:`guides/build-with-ai`: review a model and generate an application.

The editor guide is the reference for interface instructions and screenshots.
This site covers the underlying model language, generators, and services.

Models and services
-------------------

Use class diagrams for data, GUI diagrams for screens, state machines and
agent diagrams for behaviour, BPMN for processes, and neural-network diagrams
for training architectures. See :doc:`buml_language` for the model catalogue.
Available generation actions depend on the selected diagram and deployment.

The assistant service creates and edits diagrams. The :doc:`Spec-Driven Agent
<spec_driven_agent/index>` handles application files. Agent chat simulation
uses a separate :doc:`simulator service <utilities/agent_simulator>`.

Run or extend the editor
------------------------

The frontend is a Git submodule at
``besser/utilities/web_modeling_editor/frontend``. Backend services live at
``besser/utilities/web_modeling_editor/backend``.

* `Run locally <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/user-guide/deploy_locally.html>`_.
* `Add a diagram type <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/contributing/new-diagram-guide/index.html>`_.
* `Frontend source <https://github.com/BESSER-PEARL/BESSER-Web-Modeling-Editor>`_.

.. toctree::
   :maxdepth: 1

   web_editor_backend

.. note::
   The BESSER Web Modeling Editor is built on `React Flow <https://reactflow.dev/>`_
   and `Zustand <https://github.com/pmndrs/zustand>`_, packaged as the
   ``@besser/wme`` editor library. The diagram engine lives in
   ``packages/library`` and the React SPA that hosts it lives in ``packages/webapp``.
