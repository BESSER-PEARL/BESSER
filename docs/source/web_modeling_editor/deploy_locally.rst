Deploying Locally
=================

The BESSER web editor can be launched as Docker containers using `Docker Compose <https://docs.docker.com/compose/>`_.

Prerequisites
--------------
* Install Docker Compose. The recommended way is via `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_

Clone and Launch the Project
-----------------------------
Follow these steps to set up the editor:

   .. code-block:: bash

      git clone https://github.com/BESSER-PEARL/BESSER.git
      cd BESSER
      git submodule init
      git submodule update
      docker-compose up

Once the setup is complete, open your browser and navigate to ``http://localhost:8080``.
