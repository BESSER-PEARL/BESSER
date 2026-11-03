LLM access
==========

``besser.llm`` holds what every LLM call in BESSER shares: the OpenAI chat
transport, the one retry policy, the SDK clients, reading JSON out of a reply,
the upstream error, the model catalog and the per-model token prices. It imports none of its callers and works from a base
install; the ``openai`` and ``anthropic`` SDKs (the ``agents`` extra) are
imported only when an SDK client is built. ``besser.utilities.llm_utils``
re-exports the names it used to define.

The model catalog (:mod:`besser.llm.catalog`, ``BESSER_MODEL_CATALOG`` merged
over ``besser/llm/data/model_catalog.json``) and the prices
(:mod:`besser.llm.pricing`, from the catalog and the vendored public list in
``besser/llm/data/model_prices.json``) live here; the old import paths in
``besser.spec_driven_agent.providers`` still work. The Spec-Driven Agent's
provider clients themselves are still in
``besser.spec_driven_agent.providers.llm_client`` (they use ``besser.llm.retry``
and the prices); they move here with its split.

.. automodule:: besser.llm.openai_chat
   :members:

.. automodule:: besser.llm.retry
   :members:

.. automodule:: besser.llm.sdk
   :members:

.. automodule:: besser.llm.replies
   :members:

.. automodule:: besser.llm.errors
   :members:

.. automodule:: besser.llm.catalog
   :members: ModelCatalog, ModelCatalogError, load_model_catalog

.. automodule:: besser.llm.pricing

.. automodule:: besser.llm.settings
   :members:
