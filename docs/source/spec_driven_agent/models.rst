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

The user pays nothing for a free-tier run, and the run card says "No cost".
Internally, a free-tier model that draws on the deployment's provider credits
(for example ``moonshotai/Kimi-K3``) is still priced at its list rate, so the
per-run cost cap bounds it; a run that reaches the cap reports "Free tier
per-run usage limit reached". Models with an explicit free marker
(``:free`` / ``-free``) and a self-hosted fallback are priced at $0.

**Bring Your Own Key (BYOK, optional).** To target a commercial provider —
``anthropic``, ``openai``, ``mistral``, or ``nebius`` — for higher-fidelity
results, the
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
     - ``claude-sonnet-5``
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
   * - ``nebius``
     - ``Qwen/Qwen3-30B-A3B-Instruct-2507``
     - the main model
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

``nebius`` is `Nebius Token Factory <https://tokenfactory.nebius.com/>`_
(formerly Nebius AI Studio), reached over its OpenAI-compatible Chat
Completions API. Its endpoint is **fixed server-side**
(``https://api.tokenfactory.nebius.com/v1/``, overridable only by the operator
through ``NEBIUS_BASE_URL``), so a Nebius run carries no ``base_url`` and is
not subject to the custom-endpoint gate that ``local`` / ``PIA`` runs are.
Because the whole customization loop is tool-driven, any ``llm_model`` chosen
here must support OpenAI-style function calling.

Model-specific request handling
-------------------------------

Any tool-capable model id can be passed as ``llm_model``. The models below need
a request shaped for them, which the client does automatically:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Handling
   * - ``gpt-6-sol``, ``gpt-6-luna``
     - ``max_completion_tokens`` (``max_tokens`` is rejected), and
       ``reasoning_effort="none"`` whenever tools are sent: with reasoning on,
       ``/v1/chat/completions`` rejects function tools. Sent only to
       ``api.openai.com``; gateways get no ``reasoning_effort``.
   * - ``gpt-5.6-sol``, ``gpt-5.6-terra``, ``gpt-5.6-luna``
     - Same as GPT-6 sol / luna.
   * - ``gpt-6-astra``
     - **Not usable.** It accepts no ``reasoning_effort="none"`` and rejects
       tools with reasoning on, so it cannot run the tool loop on
       ``/v1/chat/completions``. A run on it fails with a message asking for a
       tool-capable model.
   * - ``claude-fable-5-1``, ``claude-opus-5-5``
     - A forced tool call (planning, gap analysis, requirements ledger,
       recovery steps) is sent as ``tool_choice: auto`` with an instruction
       naming the tool, plus ``strict: true`` when it is the only tool, and is
       asked again once if the reply skips the tool. ``thinking`` is never
       sent.
   * - ``claude-fable-5``
     - Forced tool calls keep ``tool_choice``, but without
       ``thinking: disabled``, which the model rejects.
   * - ``claude-opus-5``, ``claude-sonnet-5``, ``claude-haiku-4-5``, and the
       4.x models
     - Forced tool calls use ``tool_choice`` with thinking disabled.

No sampling parameters (``temperature``, ``top_p``, ``top_k``) are sent to any
Anthropic or OpenAI model. A reply that a provider safety classifier stops (Anthropic's
``stop_reason: "refusal"``, OpenAI's ``finish_reason: "content_filter"``) ends
the run with that reason and its category rather than an opaque provider
error.

Every model above has a price, so ``max_cost_usd`` counts its real spend.
Prices come first from a vendored copy of litellm's published price table.
A route to a paid vendor API (Anthropic, the official OpenAI endpoint,
Mistral, Nebius) is always billed: the name-based open-weight test that prices
self-hosted models at zero is skipped there. A paid model id found in no table
is billed at the ``gpt-4o`` rate, a deliberately middle-tier fallback. After a
fallback-chain switch, tokens from then on are billed at the fallback model's
rate; spend so far is kept.

A provider timeout is not retried at length: when the tier has a fallback
chain the run switches to the next model at once, and without one it is
retried twice before the run fails.

The planning model
------------------

The cheap **planning** call (the *gap* phase, see :doc:`how_it_works`) uses the
small sibling model above rather than the main one, so planning never costs
more than it needs to. Cheap routing is skipped when it would not help or would
not work: when the main model is already on the cheap tier (a ``haiku`` model
on Anthropic, a ``mini`` / ``nano`` model on OpenAI), and — on the
OpenAI-compatible providers — when the model reads as self-hosted or
open-weight, whose endpoint has no such sibling. That last test is name-based
(a ``name:size`` tag, or a known open-weight family). The ``free`` tier always
runs its planning calls on its own model. ``nebius`` opts out explicitly:
its default is already a small-activation MoE, and the OpenAI cheap sibling
does not exist on that endpoint.

.. important::
   The sponsored tier is served through an OpenAI-compatible aggregator, so a
   *paid* sponsored model id does **not** match the open-weight test and its
   planning call is routed to ``gpt-4o-mini`` — which that aggregator may not
   serve. On such a deploy, set ``BESSER_LLM_PLANNING_MODEL`` explicitly: to
   another model id the gateway does serve, or to ``primary`` to disable cheap
   routing entirely and plan on the main model.
