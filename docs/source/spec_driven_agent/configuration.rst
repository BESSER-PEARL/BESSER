Configuration reference
=======================

Everything the Spec-Driven Agent reads from the environment. All of it is
optional — the agent runs without any of it, minus the keyless tiers. These
variables are read by the :doc:`web editor backend <../web_editor_backend>`
and by ``besser/generators/llm/``; the live values of the caps and feature
flags are reported by ``GET /besser_api/spec-driven/config``, which is what a
client should read rather than hardcoding the figures below.

Keyless tiers
-------------

The free tier is what makes the agent usable without an API key. It is off
unless configured, in which case the editor offers it as the default and the
config endpoint reports ``free_tier.available``. See :doc:`models`.

- ``BESSER_FREE_LLM_BASE_URL`` / ``BESSER_FREE_LLM_MODEL`` / ``BESSER_FREE_LLM_TOKEN``
  -- The keyless free tier's primary endpoint. Both the URL and the model must
  be set for the tier to be offered.
- ``BESSER_FREE_LLM_ALT_MODELS`` -- Comma-separated extra model ids served by
  that *same* endpoint and token. They are offered as alternative free choices
  and used as the first step of the fallback chain.
- ``BESSER_FREE_LLM_FALLBACK_BASE_URL`` / ``_MODEL`` / ``_TOKEN`` -- A second,
  independently credentialed endpoint used as the last resort in the fallback
  chain.
- ``BESSER_SPONSORED_LLM_BASE_URL`` / ``_MODEL`` / ``_TOKEN`` -- An additional
  server-funded keyless tier.
- ``BESSER_DEMO_TOKEN`` -- Shared secret authorising the ``sponsored`` tier.
  A request for that tier must carry the same value in ``demo_token`` or the
  server answers 403; the web editor sends it for tabs opened through a
  ``?demo=<token>`` link. The check **fails closed**: leaving this unset
  refuses every sponsored run, so the tier cannot be left open by a
  half-finished configuration. Because the tier spends the deployment's own
  credits, pair it with a hard spend cap on the provider account — that limit
  holds even if the link is shared further than intended.
- ``BESSER_LLM_PLANNING_MODEL`` -- Override the small model used for the gap
  analysis call. Set it to ``primary`` to plan on the main model instead —
  necessary behind a gateway that does not serve the cheap sibling.
- ``BESSER_LLM_ALLOW_CUSTOM_BASE_URL`` (**off**) -- Permit a request to carry
  its own OpenAI-compatible ``base_url``. Having the server open a
  user-supplied URL is an SSRF surface, so this is meant for local or
  single-tenant deployments.

Caps and limits
---------------

See :doc:`runs` for how the caps are enforced.

- ``BESSER_LLM_MAX_COST_USD_HARD_CAP`` (5.0), ``BESSER_LLM_MAX_RUNTIME_SECONDS_HARD_CAP``
  (2400), ``BESSER_LLM_MAX_TURNS_HARD_CAP`` (150) -- Server-side ceilings a
  client request can never exceed.
- ``BESSER_LLM_DEFAULT_MAX_COST_USD`` (5.0), ``BESSER_LLM_DEFAULT_MAX_RUNTIME_SECONDS``
  (1200), ``BESSER_LLM_DEFAULT_MAX_TURNS`` (120) -- Defaults when the request
  omits a cap. Each is clamped to its hard cap.
- ``BESSER_LLM_MAX_CONCURRENT_RUNS`` (10) -- Runs in flight before new requests
  get a ``429``. Starting or resuming a run that is already in flight answers
  ``409``.
- ``BESSER_LLM_CALL_TIMEOUT_SECONDS`` (300) -- Per-request SDK timeout, so a
  stalled provider call cannot hang a turn indefinitely.
- ``BESSER_LLM_WATCHDOG_GRACE_SECONDS`` (120) -- Extra time allowed on top of
  the request's ``max_runtime_seconds`` before the runner-level watchdog
  force-cancels a run. It covers Phase 3 and packaging, which happen after the
  Phase 2 loop has already hit its own runtime check.

Feature flags
-------------

- ``BESSER_LLM_ENABLE_SHELL_TOOLS`` (**off**) -- Give the LLM
  ``run_command`` / ``install_dependencies``. Arbitrary shell on a shared BYOK
  host is an RCE and secret-exfiltration surface, so this is opt-in and
  intended for trusted local or CLI runs only. It also gates Phase 3's
  ``pip install --dry-run`` dependency check. See :doc:`tools`.
- ``BESSER_LLM_ENABLE_TOOLCHAIN_VALIDATION`` (**off**) -- Run ``tsc`` /
  ``cargo`` / ``kotlinc`` in Phase 3. Costly on non-Python stacks. The cheap
  in-process checks (syntax, Dockerfile references, contracts, ``ruff``) run
  regardless; the ``pip install --dry-run`` dependency check is gated on
  ``BESSER_LLM_ENABLE_SHELL_TOOLS`` instead. See :doc:`validation`.
- ``BESSER_LLM_ENABLE_AUTO_FIX`` (on) -- Let Phase 3 spend LLM turns repairing
  blocker-severity findings.
- ``BESSER_LLM_ENABLE_TRACING`` (on) -- Append the run's phases, turns, tool
  calls and findings to ``.besser_trace.jsonl`` in the workspace.
- ``BESSER_LLM_ENABLE_CHECKPOINTING`` (on, required for resume) -- Write
  ``.besser_checkpoint.json`` after every tool-use turn.
- ``BESSER_LLM_PER_WRITE_DIAGNOSTICS`` (on) -- Parse every file the LLM writes
  immediately and feed findings back in the same tool result.
- ``BESSER_LLM_INLINE_SCAFFOLD`` (on; set to ``0`` / ``false`` to disable) --
  Inline the small Phase-1 scaffold files into the Phase-2 prompt so the LLM
  does not burn its first turns on ``read_file`` calls.
- ``BESSER_LLM_ROLLING_CACHE`` (**off**; set to ``1`` to enable) -- Add a
  rolling prompt-cache breakpoint on the growing conversation, so the prior
  prefix is served from cache instead of re-billed each turn. Anthropic path
  only — the OpenAI-compatible path caches by prefix automatically. Off by
  default because it changes the request shape on the paid path.

Context and token budgets
-------------------------

- ``BESSER_LLM_COMPACT_THRESHOLD`` (``80000``) -- Estimated-token threshold at
  which the Phase 2 conversation is compacted, for models whose context
  window is unknown. Models with a known window (from the model catalogs of
  the free and sponsored tiers, or the built-in table of hosted frontier
  models) get an
  adaptive threshold derived from that window instead. Setting this variable
  also caps the adaptive value, so lowering it to cut cost applies to every
  model, not only unknown ones. Floored at 8,000.
- ``BESSER_LLM_MAX_COMPACT_THRESHOLD`` (``200000``) -- Ceiling on the adaptive
  threshold for models with a known window, so a million-token model does
  not send prompts of that size every turn. Models measured to serve a
  smaller window than they advertise are clamped below both values
  regardless.
- ``BESSER_LLM_HISTORY_EVICTION`` (**off**) -- Opt into the lossless
  alternative to compaction: large file bodies in older messages are replaced
  by stubs pointing back to disk, instead of whole messages being summarised.
- ``BESSER_LLM_FROM_SCRATCH_MAX_TOKENS`` (``32768``) -- Output-token ceiling
  for from-scratch generation (no Phase-1 scaffold to build on). Raising it
  costs nothing unless the extra output is actually produced; the truncation
  guard in :doc:`how_it_works` is what keeps a cut-off turn from landing
  half-written.
- ``BESSER_LLM_MODIFY_MAX_TOKENS`` (defaults to the from-scratch value) --
  The same ceiling for modify and fix runs, which routinely rewrite a whole
  existing file in one turn.

Storage and lifecycle
---------------------

- ``BESSER_LLM_RUN_STORE_PATH`` -- Path of the SQLite durable-run store
  (default: ``besser_spec_driven_runs.sqlite3`` in the system temp directory).
  Point it at a persistent volume so runs survive a container restart.
- ``BESSER_LLM_RUN_WORKSPACE_ROOT`` -- Persistent parent directory for run
  workspaces and checkpoints (default: the OS temp directory).
- ``BESSER_LLM_DOWNLOAD_TTL_SECONDS`` (1800) -- How long an output stays
  downloadable.
- ``BESSER_LLM_CANCEL_ABANDONED_RUNS`` (on) and
  ``BESSER_LLM_DISCONNECTED_GRACE_SECONDS`` (``300``) -- How long a run with no
  subscribers keeps going before it is cooperatively cancelled through the
  checkpoint-aware path.
- ``BESSER_LLM_COST_EMITTER_INTERVAL_SECONDS`` (``2.0``) -- Cadence of the
  ``cost`` SSE tick.
- ``BESSER_INCIDENT_LOG_DIR`` (``/app/incidents``) -- Where provider-incident
  records (a failed endpoint, a fallback-chain switch) are written. When it is
  unset the incident writer falls back to ``BESSER_TELEMETRY_DIR``; both are
  described alongside the telemetry endpoints in
  :doc:`../web_editor_backend`.

.. note::
   A run driven from Python rather than from the backend does not read most of
   these: ``LLMOrchestrator`` takes the flags as constructor arguments, and its
   own defaults differ from the hosted deployment's. See
   :doc:`usage`.
