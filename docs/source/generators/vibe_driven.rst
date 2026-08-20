Vibe-Driven (LLM-Augmented) Generator
=====================================

.. note::
   **Experimental.** The Vibe-Driven Generator is under active development.
   Its pipeline, request shape, and defaults may change between releases.

The Vibe-Driven Generator (also called the *smart generator* or
*LLM-augmented generator*) bridges the gap between BESSER's deterministic,
template-based generators and free-form natural-language code generation.

Where a template generator (Django, FastAPI, SQLAlchemy, …) emits a fixed
scaffold from your :doc:`structural model <../buml_language/model_types/structural>`,
the Vibe-Driven Generator **starts from that deterministic scaffold and then
lets an LLM customise it** to satisfy a natural-language request — adding
features the templates don't cover (authentication, JWT, Docker, custom
middleware, migrations, tests) or targeting a stack BESSER has no built-in
generator for (Rails, Rust/Axum, Kotlin/Spring, Next.js, …).

The key idea: the model remains the source of truth. The LLM never starts
from a blank page — it edits a correct, model-faithful baseline, which keeps
the output anchored to your diagram instead of drifting into invention.

How it works
------------

A run advances through a sequence of phases (surfaced live as the run
streams):

#. **select** — pick the deterministic Phase-1 generator that best matches
   the request (or honour an explicitly approved target). Only generators
   whose required models are present are offered.
#. **generate** — run that deterministic generator to produce a correct
   baseline scaffold (ORM models, schemas, CRUD, pages).
#. **gap** — a cheap *planning* LLM call analyses the gap between the
   scaffold and the request and produces a task list (or determines the
   scaffold already suffices and Phase 2 is skipped).
#. **customize** — the main LLM loop. The LLM reads and edits files through
   a constrained tool surface (read / write / modify files, run further
   generators), making surgical, scoped changes on top of the scaffold.
#. **validate** — an optional toolchain/compile pass. *Disabled by default
   in production* for cost and latency; the benchmark harness enables it.

The whole run streams progress over Server-Sent Events, so the editor shows
the phase timeline, the LLM's tool calls, a live cost/runtime meter, and the
streamed text as it happens.

Providers, Free tier, and BYOK
------------------------------

The Vibe-Driven Generator supports **four** providers: ``free``, ``anthropic``,
``openai``, and ``mistral``.

**Free tier (default, keyless).** When the deployment configures a free tier,
it is the **default**: the assistant runs on it with **no API-key prompt**, and
the user supplies **no key at all**. The server injects a hosted, open-weight
model on the request's behalf. The free tier is gated by
``free_tier_available()`` / ``free_tier_model()`` and configured through the
``BESSER_FREE_LLM_BASE_URL``, ``BESSER_FREE_LLM_MODEL``, and
``BESSER_FREE_LLM_TOKEN`` environment variables; the backend config endpoint
reports it as ``free_tier: {available, model}``.

**Bring Your Own Key (BYOK, optional).** To target a commercial provider —
``anthropic``, ``openai``, or ``mistral`` — for higher-fidelity results, the
user supplies their **own API key**. The key:

- is sent only in the request body, never in the URL;
- is held as a Pydantic ``SecretStr`` so it never appears in logs or
  ``repr`` output;
- is read exactly once to construct the LLM client and is **never stored or
  persisted** server-side.

Models
------

The model is chosen per run. If the request doesn't specify one, the
provider default is used:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Provider
     - Default model
     - Override field
   * - ``anthropic``
     - ``claude-sonnet-4-6``
     - ``llm_model``
   * - ``openai``
     - ``gpt-4o``
     - ``llm_model``
   * - ``mistral``
     - ``mistral-large-latest``
     - ``llm_model``
   * - ``free``
     - server-configured (``BESSER_FREE_LLM_MODEL``)
     - n/a (server-injected)

The cheap **planning** call (the *gap* phase) uses a small model —
``gpt-4o-mini`` for OpenAI, ``mistral-small-latest`` for Mistral — regardless of
the main model, so planning never costs more than it needs to. It can be
overridden with the ``BESSER_LLM_PLANNING_MODEL`` environment variable.

Budgets and caps
----------------

Every run is bounded by two soft caps, each clamped to a server-side hard
cap so a client can never exceed them:

.. list-table::
   :header-rows: 1
   :widths: 30 25 25

   * - Cap
     - Default
     - Hard limit
   * - ``max_cost_usd``
     - 1.0
     - 2.0
   * - ``max_runtime_seconds``
     - 600
     - 900

When a run hits a cap it still returns whatever it has produced so far — the
cap is surfaced as a non-terminal warning before the final result, not as a
hard failure.

Using it
--------

**From the editor.** Open the AI assistant, describe what you want
("a FastAPI backend for this model with JWT auth and Docker"), and confirm the
run. When a free tier is configured it runs by default with no API-key prompt;
to use a commercial provider for higher-fidelity results, supply your own key in
the optional BYOK dialog first. See :doc:`../web_editor` for the assistant
workflow.

**From the API.** ``POST /besser_api/smart-generate`` streams the run as
Server-Sent Events and returns a download URL on completion. The full
endpoint contract — request fields, the SSE event types, preview, resume,
cancel, and download endpoints — is documented in :doc:`../web_editor`.

Worked example
--------------

Suppose your project contains a class diagram with ``Book``, ``Author``, and
``Member`` (and their associations), and you ask:

   *"Generate a FastAPI backend for this model with JWT authentication and a
   Dockerfile."*

The run unfolds like this:

#. **select / generate** — the FastAPI (Backend) generator runs and produces a
   model-faithful baseline: SQLAlchemy models for ``Book`` / ``Author`` /
   ``Member``, Pydantic schemas, and CRUD routes.
#. **gap** — the planner notes the request needs two things the template
   doesn't provide: JWT auth and a Dockerfile.
#. **customize** — the LLM adds them on top of the scaffold, e.g.::

      app/
      ├── models.py            # Book, Author, Member — from your diagram
      ├── schemas.py
      ├── routers/
      │   ├── books.py         # CRUD, now behind auth
      │   ├── authors.py
      │   └── members.py
      ├── auth.py              # NEW — JWT issue/verify, password hashing
      ├── deps.py              # NEW — get_current_user dependency
      └── main.py              # wires the auth router in
      Dockerfile               # NEW
      requirements.txt         # + python-jose, + passlib
      pyproject.toml

The domain classes come from your diagram — the model stays the source of
truth; only the auth and container scaffolding are LLM-authored.

Limitations
-----------

- **Round-trip preservation.** Each run regenerates from the model and
  instructions; hand-written edits that aren't re-derivable from the model
  are not yet carried across re-generations. Closing this gap is the most
  significant planned improvement.
- **No build gate in production.** The *validate* phase is disabled by
  default, so generated code is not compiled before it is returned. Treat
  the output as a strong starting point, not guaranteed-buildable.
- **Quality depends on the model.** Output fidelity tracks the chosen LLM;
  a weaker or cheaper model produces a thinner result.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Symptom
     - What it means / what to do
   * - ``INVALID_KEY`` error
     - Only applies when using BYOK (never the keyless free tier). The provider
       rejected the API key. Check the key, and that it matches the selected
       provider (``anthropic`` / ``openai`` / ``mistral``).
   * - ``COST_CAP`` warning
     - The run reached ``max_cost_usd`` and stopped early. Whatever it
       produced so far is still returned (a ``done`` event follows). Raise the
       cap — up to the server hard cap — for a fuller result.
   * - ``TIMEOUT`` warning
     - The same, for ``max_runtime_seconds``: partial output is still
       delivered.
   * - ``UPSTREAM_LLM`` error
     - The provider returned an error (rate limit, overload, content filter).
       Usually transient — retry.
   * - The run seems stuck
     - Watch the ``cost`` ticks: while turns / elapsed keep advancing, the LLM
       is working. A genuinely hung run is cancelled after the runtime cap.
   * - "What counts against my budget?"
     - Only the paid LLM calls (the planning pass + the customize loop). The
       deterministic scaffold and the download are free.
