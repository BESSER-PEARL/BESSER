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
(for example ``moonshotai/Kimi-K3``) is priced at the gateway's published
rate, including its cached-input rate, so the per-run cost cap bounds it; a run that reaches the cap reports "Free tier
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

The model catalog
-----------------

Which providers and models the editor's key dialog offers, the endpoint of a
gateway provider, and the rates a gateway bills are configuration, held in one
JSON file. BESSER ships a public default,
``besser/spec_driven_agent/providers/data/model_catalog.json``. A deployment
points ``BESSER_MODEL_CATALOG`` at its own file, which is merged over the
default (see :doc:`configuration`). Adding, removing or repricing a model is an
edit to that file and a restart of the backend, the Spec-Driven worker and the
modeling agent; no code change or release.

A deployment's file names only what differs from the default:

.. code-block:: json

   {
     "providers": {
       "pia": {
         "label": "Company gateway",
         "hint": "Your personal gateway key (starts with sk-).",
         "key_prefix": "sk-",
         "base_url": "https://gateway.example.internal/v1",
         "models": [
           {"id": "claude-sonnet-5-5", "label": "Claude Sonnet 5.5", "default": true,
            "price": {"input": 2.2, "output": 11.0}},
           {"id": "gpt-6-luna", "label": "GPT-6 Luna — fast & cheap",
            "price": {"input": 0.1, "output": 0.5}}
         ]
       },
       "mistral": null
     },
     "route_prices": {
       "llm-proxy.example.org": {
         "moonshotai/kimi-k3": {"input": 3.0, "output": 15.0, "cache_read": 0.3}
       }
     }
   }

- ``providers.<id>`` replaces that provider's entry, ``null`` removes it, and a
  new id is added after the default ones. ``"replace": true`` at the top level
  drops the default catalog altogether. Keys starting with ``_`` are notes.
- A provider has a ``label``, optional ``hint`` and ``key_prefix`` for the key
  field, and its ``models``: an ``id``, a ``label``, an optional ``default``
  and an optional ``price``. The model marked ``"default": true`` is what a
  run uses when the request names no model ("Recommended default" in the key
  dialog, which preselects it for ``pia`` and ``local``); without one, the
  built-in default in the table below applies.
- ``base_url`` makes the provider a **gateway**: its runs go to that
  OpenAI-compatible endpoint, which the server resolves, so the browser never
  holds the host. A request for a gateway carries no ``base_url`` and is not
  subject to the custom-endpoint gate below. The editor knows one gateway
  provider, ``pia``; the public default has none, so the editor offers it only
  where a deployment's catalog defines it.
- A model's optional ``request`` object sets how BESSER shapes requests to
  it, over the built-in rules in the table under `Model-specific request
  handling`_: ``forced_tool_choice`` (``false`` when ``tool_choice`` of type
  ``tool`` / ``any`` is rejected; the call is then sent as ``auto`` plus an
  instruction), ``thinking_can_be_disabled`` (``false`` when
  ``thinking: {"type": "disabled"}`` is rejected) and
  ``reasoning_effort_for_tools`` (the ``reasoning_effort`` an OpenAI model
  needs to accept function tools, e.g. ``"none"``). A new model with one of
  these quirks is then a catalog edit, not a release:

  .. code-block:: json

     {"id": "claude-sonnet-6", "label": "Claude Sonnet 6",
      "request": {"forced_tool_choice": false, "thinking_can_be_disabled": false}}

- Prices are USD per 1M tokens (``input``, ``output``, optional
  ``cache_read`` and ``cache_write``; a missing cache rate bills at the input
  rate). A price on a gateway model applies on that gateway; ``route_prices``
  prices a gateway that is not a picker provider, such as the keyless tiers'
  endpoint. A price on a vendor model overrides the published list price.

``GET /besser_api/spec-driven/config`` returns the catalog's providers and
models as ``providers`` (``id``, ``label``, ``hint``, ``key_prefix``,
``models``), without endpoints or rates. A broken catalog file stops the
backend at startup with a message naming the file and the field. Validation
is strict: a key not listed above (a typo such as ``base_ur1``), a value of the
wrong type (``"default": "true"``) or a ``base_url`` without a host is an
error, not ignored.

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
provider default is used: the model catalog's ``"default": true`` model when
it marks one, otherwise the built-in default below:

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
     - ``gpt-6-luna``
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
     - ``gpt-6-luna``, unless the model reads as open-weight
     - ``llm_model``

For the free tier, ``llm_model`` may only name a model the server has
configured — the primary, one of ``BESSER_FREE_LLM_ALT_MODELS``, or the
fallback. Any other value pins the run to the primary.

``nebius`` is `Nebius Token Factory <https://tokenfactory.nebius.com/>`_
(formerly Nebius AI Studio), reached over its OpenAI-compatible Chat
Completions API. Its endpoint is **fixed server-side**
(``https://api.tokenfactory.nebius.com/v1/``, overridable only by the operator
through ``NEBIUS_BASE_URL``), so a Nebius run carries no ``base_url`` and is
not subject to the custom-endpoint gate that ``local`` runs are.
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
   * - ``claude-fable-5-1``, ``claude-opus-5-5``, ``claude-sonnet-5-5``
     - A forced tool call (planning, gap analysis, requirements ledger,
       recovery steps) is sent as ``tool_choice: auto`` with an instruction
       naming the tool, plus ``strict: true`` when it is the only tool, and is
       asked again once if the reply skips the tool. ``thinking`` is never
       sent.
   * - ``claude-fable-5``
     - Forced tool calls keep ``tool_choice``, but without
       ``thinking: disabled``, which the model rejects.
   * - ``claude-opus-5``, ``claude-sonnet-5``, ``claude-haiku-5-5``, ``claude-haiku-4-5``, and the
       4.x models
     - Forced tool calls use ``tool_choice`` with thinking disabled.

No sampling parameters (``temperature``, ``top_p``, ``top_k``) are sent to any
Anthropic or OpenAI model. A reply that a provider safety classifier stops (Anthropic's
``stop_reason: "refusal"``, OpenAI's ``finish_reason: "content_filter"``) ends
the run with that reason and its category rather than an opaque provider
error.

Every model above has a price, so ``max_cost_usd`` counts its real spend.
Prices come first from the gateway rates in the model catalog (the default
holds Command Code's), then from a rate the catalog sets on a vendor model,
then from a vendored copy of litellm's published price table.
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
on Anthropic, a ``mini`` / ``nano`` / ``luna`` model on OpenAI), and — on the
OpenAI-compatible providers — when the model reads as self-hosted or
open-weight, whose endpoint has no such sibling. That last test is name-based
(a ``name:size`` tag, or a known open-weight family). The ``free`` tier always
runs its planning calls on its own model. ``nebius`` opts out explicitly:
its default is already a small-activation MoE, and the OpenAI cheap sibling
does not exist on that endpoint.

.. important::
   The sponsored tier is served through an OpenAI-compatible aggregator, so a
   *paid* sponsored model id does **not** match the open-weight test and its
   planning call is routed to ``gpt-6-luna`` — which that aggregator may not
   serve. On such a deploy, set ``BESSER_LLM_PLANNING_MODEL`` explicitly: to
   another model id the gateway does serve, or to ``primary`` to disable cheap
   routing entirely and plan on the main model.
