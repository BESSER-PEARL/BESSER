B-UML Language
==============

B-UML (short for BESSER's Universal Modeling Language) is the base language of the BESSER low-code platform. This language is heavily inspired by UML but doesn't
aim to be fully compliant with it. It doesn't aim to be "unified" but "universal". Meaning that it is (will be) composed by a set of sublanguages that modelers
can opt to "activate" for a given project depending on their modeling needs.

With B-UML, you can design various types of models using its diverse sublanguages, including:

* :doc:`../buml_language/model_types/structural`
* :doc:`../buml_language/model_types/object`
* :doc:`../buml_language/model_types/gui`
* :doc:`../buml_language/model_types/ocl`
* :doc:`../buml_language/model_types/deployment`
* :doc:`../buml_language/model_types/state_machine`
* :doc:`../buml_language/model_types/agent`

BESSER currently offers five ways to specify your models with B-UML. However, not all model types support all three forms.

* Build the model coding in Python Using the B-UML python library
* Define the model via textual notation using one of our grammars
* Transform an image into a B-UML model (e.g., a photo of a hand-drawn model on a board)
* Use the BESSER Web Modeling Editor to design models with a graphical notation
* Transform mockup(s) into a B-UML model (e.g., a screenshot or a hand-drawn UI sketch)

The following table shows the different notations supported by each type of model

+----------------------------------------------------------------------------+------------+------------+------------+------------+------------+---------------+-------+
| Notation                                                                   | Structural |   Object   |    GUI     |    OCL     | Deployment | State Machine | Agent |
+============================================================================+============+============+============+============+============+===============+=======+
| :doc:`B-UML python library <buml_language/model_building/buml_core>`       |     X      |     X      |     X      |     X      |     X      |      X        |   X   |
+----------------------------------------------------------------------------+------------+------------+------------+------------+------------+---------------+-------+
| :doc:`B-UML Grammars <buml_language/model_building/grammars>`              |     X      |     X      |            |     X      |     X      |               |       |
+----------------------------------------------------------------------------+------------+------------+------------+------------+------------+---------------+-------+
| :doc:`Image transformation <buml_language/model_building/image_to_buml>`   |     X      |            |            |            |            |               |       |
+----------------------------------------------------------------------------+------------+------------+------------+------------+------------+---------------+-------+
| :doc:`Web Modeling Editor <./web_editor>`                                  |     X      |     X      |     X      |            |            |      X        |   X   |
+----------------------------------------------------------------------------+------------+------------+------------+------------+------------+---------------+-------+
| :doc:`Mock-up transformation <buml_language/model_building/mockup_to_buml>`|     X      |            |     X      |            |            |               |       |
+----------------------------------------------------------------------------+------------+------------+------------+------------+------------+---------------+-------+


Contents
--------
.. toctree::
   :maxdepth: 2

   buml_language/model_types
   buml_language/model_building
