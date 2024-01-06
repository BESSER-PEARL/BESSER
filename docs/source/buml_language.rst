B-UML Language
==============

B-UML is the base language of the BESSER low-code platform. This language (inspired by UML) is implemented 
in Python. The metamodel is summarised in the following figure (concepts shaded in blue inherit from *NamedElement*)

.. note::
  This metamodel contains only the main classes, attributes, and methods of the B-UML language. For a detailed 
  description please refer to the :doc:`API documentation </api>`.

.. image:: img/metamodel.jpg
  :width: 800
  :alt: B-UML metamodel
  :align: center

BESSER currently enables three ways to specify your model using B-UML: (1) build the model coding in Python 
(based on the B-UML metamodel), (2) import a `PlantUML <https://plantuml.com/>`_ model to transform it into 
a B-UML model, and (3) transform an image into a BUML model (e.g., a photo of a hand-drawn model on a board).

.. toctree::

  buml_language/buml_core
  buml_language/plantuml
  buml_language/image_to_buml
