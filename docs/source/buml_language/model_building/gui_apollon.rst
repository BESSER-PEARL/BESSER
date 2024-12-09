Integration with Apollon
========================

BESSER provides integration with `Apollon <https://apollon-library.readthedocs.io/en/latest/>`_ for visual UML modeling and code generation. 
Apollon is an open source, modern UML editor supporting various diagram types.

.. image:: ../../img/library_apollon.png
  :width: 550
  :alt: Library model in Apollon
  :align: center

Prerequisites
--------------

Before using Apollon with BESSER, ensure you have:

* **Docker** installed
* Basic familiarity with command-line interfaces (CLI)

Key Features
-------------

Visual Modeling
^^^^^^^^^^^^^^^^^
* Interactive UML diagram creation
* Support BUML for:
    * Class diagrams (structural models)
    * State machine diagrams
    * More diagram types coming soon


Getting Started
---------------

Setting Up with Docker
^^^^^^^^^^^^^^^^^^^^^^^^

1. **Install Docker**
    First, install Docker and Docker Compose on your system.
    
    For installation guides, visit the `Docker Documentation <https://docs.docker.com/get-docker/>`_.

2. **Clone BESSER**
    Get the BESSER source code:

    .. code-block:: bash

        git clone https://github.com/BESSER-PEARL/BESSER.git
        cd BESSER

3. **Initialize Components**
    Set up required submodules:

    .. code-block:: bash

        git submodule init
        git submodule update

4. **Launch Services**
    Start the Apollon environment:

    .. code-block:: bash

        docker-compose up

Using Apollon
--------------

Creating Diagrams
^^^^^^^^^^^^^^^^^
1. Open your browser and navigate to ``http://localhost:8888``
2. Select the diagram type (Class or State Machine)
3. Use the toolbar to add elements and relationships
4. Models are automatically saved and synchronized

Generating Code
^^^^^^^^^^^^^^^^^
1. Create your UML diagram in Apollon
2. Select your BESSER Generator (e.g., Python classes, Backend)
3. Click "Generate/Download" in the toolbar
4. Download the generated code

Advanced Usage
---------------

Running Components Separately
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For development or debugging, you can run components independently:

**Apollon Frontend**:

.. code-block:: bash

    cd Apollon
    npm install
    npm start

**BESSER Backend**:

.. code-block:: bash

    cd besser.utilities.besser_backend
    python main.py

References
-----------

* `BESSER Documentation <https://besser.readthedocs.io/>`_
* `Apollon Documentation <https://apollon-library.readthedocs.io/en/latest/>`_
* `Apollon Fork Repository <https://github.com/BESSER-PEARL/Apollon>`_
