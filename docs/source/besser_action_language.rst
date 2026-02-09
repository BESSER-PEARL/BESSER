BESSER Action Language (BAL)
==============

BAL (short for BESSER Action Language) is a language to specify the behavioral aspects of B-UML models.
This language is heavily inspired ALF, the Action Language for Foundational UML, but does not aim to be fully compliant with it.

BAL provides a textual notation to implement B-UML methods and generate the appropriate implementation for the target platform.
In its current version (v1.0.0), BAL can only be used in a :doc:`../buml_language/model_types/structural` to implement methods.
These methods are then generated with the model taking into account the implementation language and target platform specificities.
BAL is currently supported in the following generators:

* :doc:`../generators/rest_api`
* :doc:`../generators/backend`
* :doc:`../generators/full_web_app`


Contents
--------
.. toctree::
   :maxdepth: 2

   besser_action_language/overview
   besser_action_language/standard_library
