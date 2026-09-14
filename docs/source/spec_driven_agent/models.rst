Providers and models
====================

A run needs an LLM. Which one serves it is decided per run, from the
``provider`` and ``llm_model`` fields of the request (see :doc:`api`).

Free tier, and Bring Your Own Key
---------------------------------

**Free tier (default, keyless).** When the deployment configures a free tier,
it is the **default**: the assistant runs on it with **no API-key prompt**, and
the user supplies **no key at all**. The server injects a hosted, open-weight
model on the request's behalf. The free tier is gated by
``free_tier_available()`` / ``free_tier_model()`` and configured through the
``BESSER_FREE_LLM_BASE_URL``, ``BESSER_FREE_LLM_MODEL``, and
``BESSER_FREE_LLM_TOKEN`` environment variables (see :doc:`configuration`); the
backend config endpoint reports it as ``free_tier: {available, model, models}``.

**Bring Your Own Key (BYOK, optional).** To target a commercial provider —
``anthropic``, ``openai``, or ``mistral`` — for higher-fidelity results, the
user supplies their **own API key**. The key:

- is sent only in the request body, never in the URL;
- is held as a Pydantic ``SecretStr`` so it never appears in logs or
  ``repr`` output;
- is read exactly once to construct the LLM client and is **never stored or
  persisted** server-side.

A request for the keyless tiers must not carry a key; one is silently discarded
rather than used. A request for a commercial provider without a key is rejected
with a provider-aware message.

The fallback chain
------------------

The free tier has a **fallback chain**, so an upstream outage or an exhausted
daily quota doesn't end the run. It is ordered cloud-first:

#. any extra model on the *primary* endpoint, listed in
   ``BESSER_FREE_LLM_ALT_MODELS`` (these share the primary's base URL and
   token, and are metered separately by the upstream aggregator);
#. the self-hosted fallback endpoint (``BESSER_FREE_LLM_FALLBACK_BASE_URL`` /
   ``_MODEL`` / ``_TOKEN``) — a single shared box that serves one request at a
   time, so it is a genuine last resort rather than a peer.

The switch is sticky for the rest of the run and is surfaced to the client as a
``model_update`` SSE event, so the run card shows the model actually serving it
rather than the one the run started with. Only the server-configured tiers
(``free``, ``sponsored``) have a fallback chain; a BYOK run never switches
models behind your back.

Default models per provider
---------------------------

The model is chosen per run. If the request doesn't specify one, the
provider default is used:

.. list-table::
   :header-rows: 1
   :widths: 18 30 26 26

   * - Provider
     - Default model
     - Planning model
     - Override field
   * - ``anthropic``
     - ``claude-sonnet-4-6``
     - ``claude-haiku-4-5``
     - ``llm_model``
   * - ``openai``
     - ``gpt-4o``
     - ``gpt-4o-mini``
     - ``llm_model``
   * - ``mistral``
     - ``mistral-large-latest``
     - ``mistral-small-latest``
     - ``llm_model``
   * - ``free``
     - server-configured (``BESSER_FREE_LLM_MODEL``)
     - the main model
     - ``llm_model``, restricted to the server's allowlist
   * - ``sponsored``
     - server-configured (``BESSER_SPONSORED_LLM_MODEL``)
     - ``gpt-4o-mini``, unless the model reads as open-weight
     - ``llm_model``

For the free tier, ``llm_model`` may only name a model the server has
configured — the primary, one of ``BESSER_FREE_LLM_ALT_MODELS``, or the
fallback. Any other value pins the run to the primary.

The planning model
------------------

The cheap **planning** call (the *gap* phase, see :doc:`how_it_works`) uses the
small sibling model above rather than the main one, so planning never costs
more than it needs to. Cheap routing is skipped when it would not help or would
not work: when the main model is already on the cheap tier (a ``haiku`` model
on Anthropic, a ``mini`` / ``nano`` model on OpenAI), and — on the
OpenAI-compatible providers — when the model reads as self-hosted or
open-weight, whose endpoint has no such sibling. That last test is name-based
(a ``name:size`` tag, or a known open-weight family), which is what makes the
``free`` tier route planning to its own model.

.. important::
   The sponsored tier is served through an OpenAI-compatible aggregator, so a
   *paid* sponsored model id does **not** match the open-weight test and its
   planning call is routed to ``gpt-4o-mini`` — which that aggregator may not
   serve. On such a deploy, set ``BESSER_LLM_PLANNING_MODEL`` explicitly: to
   another model id the gateway does serve, or to ``primary`` to disable cheap
   routing entirely and plan on the main model.
