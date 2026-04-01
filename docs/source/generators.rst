Code Generators
===============

BESSER offers a suite of code generators designed for diverse technologies and purposes. These generators play
a pivotal role in translating your model, created using the :doc:`buml_language`, into executable code suitable for
various applications.

.. note::
   Most generators consume :doc:`structural models <buml_language/model_types/structural>` (class diagrams).
   Some generators require additional model types (GUI, agent, quantum, deployment) as noted below.

Choosing a Generator
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 45

   * - Generator
     - Input Model
     - Output
     - Use When
   * - **Full Web App**
     - Structural + GUI
     - ZIP (React + FastAPI)
     - You want a complete web application with frontend, backend, and database
   * - **Django**
     - Structural
     - ZIP (Django project)
     - You want a Django admin panel with ORM models
   * - **Backend (FastAPI)**
     - Structural
     - ZIP (FastAPI + SQLAlchemy)
     - You need only a REST API backend without frontend
   * - **REST API**
     - Structural
     - Python files
     - You need API endpoints and Pydantic models without database setup
   * - **Python**
     - Structural
     - .py
     - You need plain Python classes from your model
   * - **Pydantic**
     - Structural
     - .py
     - You need Pydantic validation models with OCL constraints
   * - **Java**
     - Structural
     - .java files
     - You need Java class files
   * - **SQL**
     - Structural
     - .sql
     - You need DDL statements for any SQL dialect
   * - **SQLAlchemy**
     - Structural
     - .py
     - You need SQLAlchemy ORM models
   * - **JSON Schema**
     - Structural
     - .json
     - You need JSON Schema or Smart Data Models
   * - **RDF**
     - Structural
     - .ttl
     - You need an RDF vocabulary in Turtle format
   * - **React**
     - Structural + GUI
     - ZIP
     - You need only the React frontend (no backend)
   * - **Flutter**
     - Structural + GUI
     - ZIP
     - You need a Flutter mobile application
   * - **Terraform**
     - Deployment
     - ZIP
     - You need infrastructure-as-code for AWS or GCP
   * - **PyTorch**
     - Neural Network
     - .py
     - You need a PyTorch neural network
   * - **TensorFlow**
     - Neural Network
     - .py
     - You need a TensorFlow neural network
   * - **Qiskit**
     - Quantum Circuit
     - .py
     - You need Qiskit quantum circuit code
   * - **BAF Agent**
     - Agent
     - ZIP
     - You need a BESSER Agentic Framework conversational agent


Web Application
---------------

Generate complete web applications with frontend, backend, and database:

.. toctree::
   :maxdepth: 2

   generators/full_web_app

Frameworks & Languages
----------------------

Generate code for various frameworks and programming languages:

.. toctree::
   :maxdepth: 1

   generators/django
   generators/backend
   generators/rest_api
   generators/python
   generators/pydantic
   generators/java
   generators/flutter
   generators/react

Data & API
----------

Generate database schemas, APIs, and data formats:

.. toctree::
   :maxdepth: 1

   generators/sql
   generators/alchemy
   generators/json_schema
   generators/rdf
   generators/terraform

Machine Learning
----------------

Generate machine learning model code:

.. toctree::
   :maxdepth: 1

   generators/pytorch
   generators/tensorflow

Quantum Computing
-----------------

Generate quantum circuit code:

.. toctree::
   :maxdepth: 1

   generators/qiskit

Agents
------

Generate conversational agents:

.. toctree::
   :maxdepth: 1

   generators/baf

Build Your Own
--------------

Create custom code generators:

.. toctree::
   :maxdepth: 1

   generators/build_generator


