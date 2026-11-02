LLM access
==========

``besser.llm`` holds what every LLM call in BESSER shares: the OpenAI chat
transport, the one retry policy, the SDK clients, reading JSON out of a reply
and the upstream error. It imports none of its callers and works from a base
install; the ``openai`` and ``anthropic`` SDKs (the ``agents`` extra) are
imported only when an SDK client is built. ``besser.utilities.llm_utils``
re-exports the names it used to define.

The Spec-Driven Agent's provider clients, their pricing and the model catalog
are still in ``besser.spec_driven_agent.providers`` (they use
``besser.llm.retry``); they move here with the split of its ``llm_client``.

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
