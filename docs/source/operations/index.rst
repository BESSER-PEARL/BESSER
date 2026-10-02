Install and deploy
==================

Use the hosted editor to try BESSER without installation. Use a local setup
when you need your own services, providers, or source changes.

.. toctree::
   :maxdepth: 1

   ../installation
   ../web_editor

* `Run the editor locally <https://besser.readthedocs.io/projects/besser-web-modeling-editor/en/latest/user-guide/deploy_locally.html>`_:
  clone the repository, configure the environment, and start Docker Compose.
* :doc:`../spec_driven_agent/production_deployment`: configure the isolated
  generation worker, reverse proxy, and verification checks for a hosted editor.
* :doc:`../utilities/agent_simulator`: operate the agent simulation service.
* `Modeling Agent deployment <https://modeling-agent.readthedocs.io/en/latest/deployment.html>`_:
  deploy the editor's conversational backend.

When upgrading to v8, read :doc:`../releases/v8/migration` before replacing
production images. The generation worker and its proxy routes must be in
place first.
