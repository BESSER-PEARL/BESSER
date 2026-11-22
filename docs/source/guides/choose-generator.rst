Choose a generator
==================

A deterministic generator turns a model into code for a specific technology.
It does not call an LLM. Choose it when its output matches the application
you want to build.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - I need
     - Use
     - Model input
   * - Python classes
     - :doc:`../generators/python`
     - Class diagram
   * - Database tables
     - :doc:`../generators/sql` or :doc:`../generators/alchemy`
     - Class diagram
   * - A REST backend with persistence
     - :doc:`../generators/backend`
     - Class diagram
   * - A Django project
     - :doc:`../generators/django`
     - Class diagram
   * - A React frontend and FastAPI backend
     - :doc:`../generators/full_web_app`
     - Class diagram and linked GUI model
   * - A different stack or extra application features
     - :doc:`build-with-ai`
     - Model and written requirements

See :doc:`the full generator catalogue <../generators>` for Java, JSON Schema,
agents, neural networks, quantum circuits, and other targets.

What to check before generation
-------------------------------

1. Validate the model and resolve errors.
2. Make sure the chosen generator accepts that model type.
3. For a web application, link the GUI's data-bound controls to the class model.
4. Choose a new output directory or export your existing work before
   regenerating. Generation may replace generated files.

After generation, read the generated project's run instructions and install
its dependencies in its own environment. A valid model does not guarantee
that every deployment environment is configured correctly.
