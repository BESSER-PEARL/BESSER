BESSER Agentic Framework (BAF) Generator
========================================

The BAF generator produces a BAF Agent based on a given B-UML agent model.
Let's generate the agent for Greetings Agent defined in :doc:`../buml_language/model_types/agent`. You should create a ``BAFGenerator`` object, provide the agent model, and use the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.agents.baf_generator import BAFGenerator

    generator: BAFGenerator = BAFGenerator(model=agent)
    generator.generate()

Optional constructor parameters:

- ``config_path``: Path to a YAML configuration file for the agent.
- ``config``: Configuration dictionary (alternative to ``config_path``).
- ``openai_api_key``: OpenAI API key for LLM-powered agent features.

The corresponding ``agent.py`` file and its config file titled ``config.yaml`` will be generated in the ``<<current_directory>>/output``
folder.

Check out the BAF documentation for more details on how to use the generated agent: `BESSER Agentic Framework Documentation <https://besser-agentic-framework.readthedocs.io/latest/>`_.


Generation Modes
----------------

The BAF generator supports three generation modes via the ``generation_mode`` parameter:

.. code-block:: python

    from besser.generators.agents.baf_generator import BAFGenerator, GenerationMode

    # Default: full pipeline (personalization + templated code)
    generator = BAFGenerator(model=agent, generation_mode=GenerationMode.FULL)

    # Skip personalization, render templates immediately
    generator = BAFGenerator(model=agent, generation_mode=GenerationMode.CODE_ONLY)

    # Run personalization JSON/model export only (no code templates)
    generator = BAFGenerator(model=agent, generation_mode=GenerationMode.PERSONALIZED_ONLY)

- **FULL** (default): Runs personalization (if configured) followed by templated code generation.
- **CODE_ONLY**: Skips personalization helpers and renders templates immediately. Use this when you
  do not need personalization assets.
- **PERSONALIZED_ONLY**: Runs only the personalization JSON/model export. Use this to produce
  personalization artifacts without generating the agent code.


Personalization
---------------

The BAF generator can adapt the generated agent to an end-user's profile —
language, style, readability, modality, platform, LLM, and more. The
personalization flow is opt-in: with no ``config`` passed the generator behaves
identically to the classic pipeline.

See :doc:`agent_personalization` for the structured configuration schema, the
two recommendation backends (rule-based and LLM-based), the variant mechanisms
(languages, variations, configuration variants, personalization mapping), and
the ``OPENAI_API_KEY`` lookup order.


RAG Support
-----------

If the agent model includes RAG elements (see :doc:`../buml_language/model_types/agent`),
the generator produces the vector store setup (Chroma), text splitter configuration,
and ``session.run_rag()`` calls. A data folder is created for each RAG element
where you should place your PDF documents before running the agent. The folder
name is derived from the RAG element name (e.g. ``"Knowledge Base"`` becomes
``knowledge_base/``).


Missing BAF Features
--------------------

Currently, some features available in BAF are stil missing in the B-UML agent model and the BAF Generator. Most notably:

- **LLM Configuration**
- **Platform Configuration**
- **Entities**
- **Processors**