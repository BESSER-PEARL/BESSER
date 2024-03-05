B-UML Language
==============

B-UML (short for BESSER's Universal Modeling Language) is the base language of the BESSER low-code platform. This language is heavily inspired by UML but doesn't aim to be fully compliant with it. It doesn't aim to be "unified" but "universal". Meaning that it is (will be) composed by a set of sublanguages modelers can opt to "activate" for a given project depending on their modeling needs. 

B-UML is implemented in Python. The metamodel of the core of B-UML is summarised in the following figure (concepts shaded in blue inherit from *NamedElement*)

.. note::
  This metamodel contains only the main classes, attributes, and methods of the B-UML language. For a detailed 
  description please refer to the :doc:`API documentation </api>`.

.. image:: img/structural_mm.png
  :width: 800
  :alt: B-UML metamodel
  :align: center

BESSER currently enables three ways to specify your model using B-UML: (1) build the model coding in Python 
(based on the B-UML metamodel), (2) import a `PlantUML <https://plantuml.com/>`_ model to transform it into 
a B-UML model, and (3) transform an image into a BUML model (e.g., a photo of a hand-drawn model on a board).

.. toctree::
  :maxdepth: 1

  buml_language/model_types
  buml_language/model_building
