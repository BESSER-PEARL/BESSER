LLM access
==========

``besser.llm`` is the layer through which BESSER reaches a large language
model. It imports none of its callers and works from a base install; the
``openai`` and ``anthropic`` SDKs (the ``agents`` extra) are imported only when
an SDK client is built. ``besser.utilities.llm_utils`` re-exports the names it
used to define.

.. automodule:: besser.llm.openai_chat
   :members:

.. automodule:: besser.llm.sdk
   :members:

.. automodule:: besser.llm.replies
   :members:

.. automodule:: besser.llm.errors
   :members:
