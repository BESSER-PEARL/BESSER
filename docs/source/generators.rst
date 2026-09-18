Code Generators
===============

BESSER offers a suite of code generators designed for diverse technologies and purposes. These generators play
a pivotal role in translating your model, created using the :doc:`buml_language`, into executable code suitable for
various applications.

.. note::
   Most generators consume :doc:`structural models <buml_language/model_types/structural>` (class diagrams).
   Some generators require additional model types (GUI, agent, quantum, deployment) as noted below.

The generators on this page are **template-based**: one pass over your model,
same input in, same output out, with no natural-language step. (The one
qualification is :doc:`agent personalization <generators/agent_personalization>`,
whose variant mechanisms can optionally call an LLM to re-write an agent's
messages.)

.. tip::
   Need something the templates don't cover — authentication, Docker, tests —
   or a stack with no BESSER generator at all? The
   :doc:`Spec-Driven Agent <spec_driven_agent/index>` is the agentic layer on
   top of these generators: it runs one of them to get a model-faithful
   scaffold, lets an LLM customise it to satisfy a natural-language request,
   then validates the result and repairs blocker-level issues before handing it
   back. Most of the generators below are reachable as one of the agent's
   :doc:`tools <spec_driven_agent/tools>`.

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
   * - **Test Cases**
     - Structural
     - .py (pytest + Hypothesis)
     - You want an auto-generated pytest + Hypothesis test suite for your model
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
   * - **Supabase**
     - Structural
     - .sql
     - You need a Supabase migration: Postgres DDL plus ``auth.users``
       mirroring, grants and Row Level Security policies
   * - **JSON Schema**
     - Structural
     - .json
     - You need JSON Schema or Smart Data Models
   * - **JSON Object**
     - Object
     - .json
     - You need your *instances* as JSON — fixtures, seed data, a worked
       example of a system state
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
   * - **BAF Agent (personalized)**
     - Agent + UserDiagram
     - ZIP
     - You want a BAF agent adapted to an end-user profile (language, style,
       accessibility, modality); see :doc:`generators/agent_personalization`
   * - **BPMN**
     - BPMN
     - .bpmn (XML)
     - You need vendor-neutral BPMN 2.0 XML readable by every BPMN-aware tool

None of the above a fit? If you need a customised codebase — extra features
(auth, JWT, Docker, tests) or a stack with no built-in generator (Rails, Rust,
Kotlin, Next.js) — use the :doc:`Spec-Driven Agent <spec_driven_agent/index>`
instead. It runs keyless on the free tier, or with your own API key.

Web Application
---------------

Generate complete web applications with frontend, backend, and database:

.. toctree::
   :maxdepth: 2

   generators/full_web_app
   generators/maps

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
   generators/test_case
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
   generators/supabase
   generators/json_schema
   generators/json_object
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
   generators/agent_personalization

Business Process
----------------

Generate BPMN 2.0 XML for any BPMN-aware engine or modeller:

.. toctree::
   :maxdepth: 1

   generators/bpmn

Build Your Own
--------------

Create custom code generators:

.. toctree::
   :maxdepth: 1

   generators/build_generator


