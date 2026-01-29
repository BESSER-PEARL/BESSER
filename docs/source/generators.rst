Code Generators
===============

BESSER offers a suite of code generators designed for diverse technologies and purposes. These generators play 
a pivotal role in translating your model, created using the :doc:`buml_language`, into executable code suitable for 
various applications.

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
   generators/python
   generators/java
   generators/flutter

Data & API
----------

Generate database schemas, APIs, and data formats:

.. toctree::
   :maxdepth: 1

   generators/sql
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


.. warning::
   
   Right now, most of our available code generators can only handle :doc:`structural models <../buml_language/model_types/structural>`.
   But here's the cool part: BESSER offers an interface that makes it easy to :doc:`develop your own code generator </generators/build_generator>` 
   capable of handling any type of B-UML model.
