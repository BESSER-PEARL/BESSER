Code Generators
===============

BESSER offers a suite of code generators designed for diverse technologies and purposes. These generators play 
a pivotal role in translating your model, created using the :doc:`buml_language`, into executable code suitable for 
various applications, such as Django models or database structures compatible with SQLAlchemy.

Currently, BESSER covers the following range of predefined code generators (although you could also design your 
own generator).

.. toctree::
   :maxdepth: 1

   generators/full_web_app
   generators/django
   generators/python
   generators/java
   generators/pydantic
   generators/alchemy
   generators/sql
   generators/rest_api
   generators/backend
   generators/flutter
   generators/terraform
   generators/rdf
   generators/json_schema
   generators/build_generator
   generators/pytorch
   generators/tensorflow
   generators/baf


.. warning::
   
   Right now, most of our available code generators can only handle :doc:`structural models <../buml_language/model_types/structural>`.
   But here's the cool part: BESSER offers an interface that makes it easy to :doc:`develop your own code generator </generators/build_generator>` 
   capable of handling any type of B-UML model.
