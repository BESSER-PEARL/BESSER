KG to B-UML
==============

Two independent paths turn a knowledge graph into B-UML, sharing no code:

* ``besser.utilities.kg_llm_to_buml`` — the **LLM-assisted** path. It asks a model to
  read an arbitrary graph and propose a class diagram. Needs an API key, and its
  output is a best-effort interpretation.
* ``besser.BUML.notations.kg_to_buml`` — the **deterministic** path. It applies
  the OWL 2 / SHACL transformation rules to a proper ontology, offline and with
  no API key, and produces the same model every time for the same input.

See :doc:`../../../../buml_language/model_building/kg_to_buml` for the narrative
version of both.

LLM-assisted conversion
-----------------------

.. automodule:: besser.utilities.kg_llm_to_buml
   :members:
   :undoc-members:
   :show-inheritance:

.. note::
   ``besser.utilities.kg_to_buml``, the module's former name, is a deprecated
   alias that warns on import; import from ``besser.utilities.kg_llm_to_buml``.

Deterministic conversion
------------------------

.. automodule:: besser.BUML.notations.kg_to_buml

.. automodule:: besser.BUML.notations.kg_to_buml.kg_to_class_diagram
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: besser.BUML.notations.kg_to_buml.to_buml
   :members:
   :undoc-members:
   :show-inheritance:

Preflight, resolutions and consistency
--------------------------------------

.. automodule:: besser.BUML.notations.kg_to_buml.preflight
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: besser.BUML.notations.kg_to_buml.resolutions
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: besser.BUML.notations.kg_to_buml.consistency
   :members:
   :undoc-members:
   :show-inheritance:

OWL / RDF import, export and projection
---------------------------------------

.. automodule:: besser.BUML.notations.kg_rdf
   :members:
   :show-inheritance:

.. automodule:: besser.BUML.notations.kg_rdf.importer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: besser.BUML.notations.kg_rdf.exporter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: besser.BUML.notations.kg_rdf.projection
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: besser.BUML.notations.kg_rdf._emit
   :members:
   :undoc-members:
   :show-inheritance:
