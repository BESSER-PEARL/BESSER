BESSER Agentic Framework (BAF) Generator
========================================

The BAF generator produces a BAF Agent based on a given B-UML agent model.
Let's generate the agent for Greetings Agent defined in :doc:`../buml_language/model_types/agent`. You should create a ``BAFGenerator`` object, provide the agent model, and use the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.agents.baf_generator import BAFGenerator

    generator: BAFGenerator = BAFGenerator(model=agent)
    generator.generate()

The corresponding ``agent.py`` file and its config file titled ``config.ini`` will be generated in the ``<<current_directory>>/output``
folder.

Check out the BAF documentation for more details on how to use the generated agent: `BESSER Agentic Framework Documentation <https://besser-agentic-framework.readthedocs.io/latest/>`_.


Missing BAF Features
--------------------

Currently, some features available in BAF are stil missing in the B-UML agent model and the BAF Generator. Most notably:

- **LLM Configuration**
- **Platform Configuration**
- **RAG**
- **Entities**
- **Processors**