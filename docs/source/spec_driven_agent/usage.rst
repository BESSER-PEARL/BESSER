Using the agent
===============

There are three ways to drive a run: the web editor's AI assistant, the REST
API, and the Python library.

From the web editor
-------------------

Open the AI assistant, describe what you want ("a FastAPI backend for this
model with JWT auth and Docker"), and confirm the run. When a free tier is
configured it runs by default with no API-key prompt; to use a commercial
provider for higher-fidelity results, supply your own key in the optional BYOK
dialog first. Whenever a run would spend your own key, the assistant asks for
explicit confirmation — a run never spends your key silently.

The assistant shows the phase timeline, the LLM's tool calls, and a live
cost/runtime meter as the run streams. Closing the tab does not kill the run:
it is owned by the server, and you can reattach — see :doc:`runs`.

See :doc:`../web_editor` for the rest of the assistant workflow, including how
it builds the model and the screens before you ask it for code.

From the REST API
-----------------

``POST /besser_api/spec-driven/generate`` streams the run as Server-Sent Events
and returns a download URL on completion. The full endpoint contract — request
fields, the SSE event types, preview, resume, cancel, and download endpoints —
is in :doc:`api`.

From Python
-----------

The agent is also a normal BESSER generator: ``LLMGenerator`` implements
``GeneratorInterface``, so it is constructed and run like any generator in
:doc:`../generators`.

.. code-block:: python

   from besser.spec_driven_agent import LLMGenerator

   gen = LLMGenerator(
       model=library_model,
       instructions="Build a FastAPI backend with JWT auth and PostgreSQL",
       api_key="sk-ant-...",              # or the provider's env var
       provider="anthropic",              # "anthropic" | "openai" | "mistral" | "nebius"
       llm_model="claude-sonnet-4-6",     # optional; provider default otherwise
       output_dir="./my_app",
   )
   output_path = gen.generate()

A domain model is not required — a run can also be driven from a ``gui_model``,
``agent_model``, ``object_model``, ``quantum_circuit``, or a non-empty
``state_machines`` list, passed as the corresponding keyword argument. At least
one model must be supplied.

After the run, ``fix_error()`` feeds a runtime error from the generated app
back to the same orchestrator, which either repairs the code or explains what
to do:

.. code-block:: python

   response = gen.fix_error("ImportError: No module named 'auth'")

.. note::
   The library defaults are **not** the hosted-editor defaults. ``LLMGenerator``
   ships ``max_cost_usd=5.0`` and ``max_runtime_seconds=1200``, against the
   editor's 1.0 / 600; and the underlying ``LLMOrchestrator`` defaults
   ``auto_fix_issues=False``, so a library run **reports** Phase 3 findings
   rather than spending turns repairing them. Pass ``auto_fix_issues=True`` to
   ``LLMOrchestrator`` directly if you want the repair loop. See
   :doc:`validation`.

.. note::
   ``allow_shell_tools`` and ``enable_toolchain_validation`` both default to
   ``False``, on every path — library, CLI and hosted alike. A library run does
   **not** get ``run_command`` / ``install_dependencies`` unless you ask for
   them, and the refusal is enforced when the tool is called, not merely by
   leaving it out of the advertised list. Opt in deliberately, and only where
   running model-authored commands on that machine is acceptable::

       LLMGenerator(model=model, instructions=...,
                    allow_shell_tools=True, enable_toolchain_validation=True)

   The hosted backend keeps both off via ``BESSER_LLM_ENABLE_SHELL_TOOLS``.
   See :doc:`tools`.

Troubleshooting a run
---------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Symptom
     - What it means / what to do
   * - ``INVALID_KEY`` error
     - Only applies when using BYOK (never the keyless free tier). The provider
       rejected the API key. Check the key, and that it matches the selected
       provider (``anthropic`` / ``openai`` / ``mistral`` / ``nebius``).
   * - ``COST_CAP`` warning
     - The run reached ``max_cost_usd`` and stopped early. Whatever it
       produced so far is still returned (a ``done`` event follows). Raise the
       cap — up to the server hard cap — for a fuller result.
   * - ``TIMEOUT`` warning
     - The same, for ``max_runtime_seconds``: partial output is still
       delivered.
   * - ``INCOMPLETE`` warning
     - The customization loop was cut short (a provider rate-limit, or the
       turn cap). Output is still returned; the ``done`` event sets
       ``incomplete`` so the client can say so.
   * - ``UPSTREAM_LLM`` error
     - The provider returned an error (rate limit, overload, content filter).
       Usually transient — retry.
   * - The run seems stuck
     - Watch the ``cost`` ticks: while turns / elapsed keep advancing, the LLM
       is working. A genuinely hung run is cancelled after the runtime cap.
   * - The browser tab closed mid-run
     - The run keeps going server-side. Reattach with
       ``GET /besser_api/spec-driven/runs/{id}/events?after=N`` to replay what you missed
       and follow the rest live.
   * - The model changed mid-run
     - A ``model_update`` event means the free tier's fallback chain took over
       after the primary endpoint failed or ran out of quota. The switch is
       sticky for the rest of the run.
   * - "It finished, but the app is broken"
     - Check ``blockerCount`` on the ``done`` event: a non-zero value means the
       run completed but Phase 3 could not repair everything it found.
   * - "What counts against my budget?"
     - Only the paid LLM calls (the planning pass, the customize loop, and any
       Phase 3 fix turns). The deterministic scaffold, the validators, and the
       download are free.
