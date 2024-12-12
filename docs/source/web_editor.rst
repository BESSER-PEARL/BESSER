Web Modeling Editor
===================

One of the practical ways to use BESSER is through the Web Modeling Editor, where you can rapidly 
design :doc:`B-UML <../buml_language>` models and try the :doc:`BESSER code generators <../generators>`.

.. note::
   The BESSER Web Editor is based on a fork of the `Apollon project <https://apollon-library.readthedocs.io/en/latest/>`_, a UML modeling editor.

The BESSER web editor provides a graphical dashboard to create two types of B-UML models:

- Class diagram or :doc:`structural model <./buml_language/model_types/structural>`: 
- :doc:`State Machine diagram <./buml_language/model_types/state_machine>`: 

.. image:: ./img/GUI_WEB_SHOW.gif
   :width: 900
   :alt: BESSER Web Editor interface
   :align: center


Launching the Project
---------------------
The BESSER web editor can be launched as Docker containers using `Docker Compose <https://docs.docker.com/compose/>`_.

Prerequisites
^^^^^^^^^^^^^
* Install Docker Compose. The recommended way is via `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_

Clone and Launch the Project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Follow these steps to set up the editor:

   .. code-block:: bash

      git clone https://github.com/BESSER-PEARL/BESSER.git
      cd BESSER
      git submodule init
      git submodule update
      docker-compose up

Once the setup is complete, open your browser and navigate to ``http://localhost:8888``.


Using the BESSER Web Editor
---------------------------

Creating Diagrams
^^^^^^^^^^^^^^^^^
1. Open Apollon in your browser (``http://localhost:8888``).
2. Select the diagram type (Class or State Machine).
3. Use the toolbar to add elements and relationships.
4. Models are automatically saved and synchronized.
5. Export/Import the diagram as a B-UML or JSON file.

Generating Code
^^^^^^^^^^^^^^^^
1. Create your UML diagram in Apollon.
2. Select your BESSER Generator (e.g., Python classes, Backend).
3. Click "Generate/Download" in the toolbar.
4. Download the generated code.

.. note::
   The Web Editor will be available online.
