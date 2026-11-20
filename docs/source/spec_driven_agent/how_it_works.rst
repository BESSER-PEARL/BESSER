How it works
============

A run is a three-phase pipeline (``LLMOrchestrator.run`` in
``besser/spec_driven_agent/pipeline/orchestrator.py``).

.. container:: besser-diagram

   .. raw:: html
      :file: ../_static/diagrams/create-run.svg

   A new run. Teal boxes are LLM calls, and everything else is deterministic
   code. The agent is one conversation with a tool list. Around Phase 1, two
   deterministic checks run: Phase 0 turns the model's design defects (a
   mandatory creation cycle, an invalid name, two tables with one name, …) into
   blockers before anything is generated, and Phase 1.5 validates the scaffold.
   See :ref:`spec-driven-model-contract`.

The three phases
----------------

**Phase 1 — deterministic generation.**
   A deterministic BESSER generator is selected and run against your B-UML
   model to produce a correct scaffold (ORM models, schemas, CRUD routes,
   pages). This is exactly the machinery documented under
   :doc:`../generators` — the agent drives those generators rather than
   replacing them. Only generators whose required models are present are
   offered, and a generator approved on the preview screen is honoured as-is,
   with no selection call.

   The same selection call decides what the run builds, its *target*: a
   ``web_app`` (anything used in a browser, the default for a bare "app"), a
   ``backend`` (API-only or headless), or ``other`` (a native mobile or desktop
   app, a CLI, a game, an extension, a library, ...). A generator is picked
   only when its output *is* that target, so an ``other`` target is written
   from scratch in the stack the user named, or the standard one for that
   target. An app that uses an API is that app, not a backend, and an
   architecture pattern such as MVC describes the app's structure, not its
   platform. Every web-specific rule below applies to a ``web_app`` target only.
   The call also names a stack the user asked for that no BESSER generator
   builds (``unsupported_stack``); words of the domain ("express delivery",
   "Swift Logistics") are left to the model, not to a word list. Such a stack
   is written from scratch ("a REST API in Go"), even when the answer also
   names a generator. A NONE for a web app or a backend with no unsupported
   stack contradicts the rules, so it gets the generator the loaded models
   call for (the FastAPI scaffold for a class diagram); for an ``other``
   target, NONE stands. A gateway that ignores the forced tool call, or a
   client without one, is asked for the same answer as a JSON object in plain
   text. When no selection call runs (an approved generator, a run with no
   class diagram), or the selection call fails, one small call decides the
   target alone (a failed selection is logged as a warning). An approved
   ``generate_web_app`` settles it as a web app and costs no call. Only when no
   answer comes back at all (the provider is failing) do keywords pick the
   generator (``planning/generator_keywords.py``). The ``/preview`` endpoint
   makes no LLM call and needs no key, so its generator is a keyword estimate:
   the same keywords, plus a short list of stacks no generator builds.

   The target is recorded in the trace, the checkpoint and
   ``.besser_recipe.json``, so a resume or a later modify run of the same app
   keeps it. Every modify run makes the same call once, on the primary model
   rather than the cheap planning one (its answer can rebuild the whole app),
   with the recorded target and the model's class names, and it answers three
   more questions:
   whether the request changes the target (``changes_target``: "turn it into a
   Flutter app", "I only need the REST API now"; new features and a reworded
   description of the same target do not), whether the newest message reports
   a failure of the existing app (``is_failure_report``), and which entity it
   is about (``reported_entity``). A changed target is built next to the
   existing project; a failure report makes the run a fix run, gated on that
   failure: a blocker that names the reported entity, from a check of the app
   (a runtime probe, the data contract, the acceptance matrix), leaves it
   unconfirmed, and so does one on this run's own requirements or fix tasks.
   A finding on a requirement of an earlier run's request never does: it names
   the app's entities whatever the failure. A call with no usable answer, even as JSON text, keeps the
   recorded target and makes no fix run, with a warning. This call is one
   primary-model call more per modify run.

   The request is free text. When the user asks for something that is not a
   web app, the BESSER modeling agent adds a ``Target:`` line of its own to the
   instructions it sends (under "Assistant's notes" when the original request
   is attached):

   .. code-block:: text

      Target: Windows 11 desktop app (not a web app).

   The selection call reads it as part of the request. The line is optional:
   the modeling agent omits it for a web app or a backend and when the
   instructions already name the target, and an oversized specification falls
   back to the original request without it.

   Two smaller steps bracket it. When no deterministic generator fits the
   target stack (Next.js, Rust, Kotlin/Spring), a *stack-metadata* step writes
   the minimal build-config files that stack needs instead. Those templates are
   server projects, so this step does not run for an ``other`` target. Afterwards, the
   Phase-1 output is validated so any pre-existing issue is handed to Phase 2
   as context rather than discovered later.

**Phase 2 — LLM customization loop.**
   The main agent loop. The LLM works through a constrained
   :doc:`tool surface <tools>` — reading and writing files, querying the domain
   model, invoking further generators, and tracking its own work through a
   ``task_list`` checklist — making surgical, scoped changes on top of the
   scaffold. Before the loop starts, a cheap *planning* call analyses the gap
   between the scaffold and the request and produces the task list; when it
   judges the scaffold already sufficient, Phase 2 is skipped entirely (a
   create run only; a modify run always runs it), and the run record says the
   run finished, not that a limit stopped it. The planner labels each task
   as a change, or as removing the scaffold or switching its framework; the
   latter two are dropped, since the scaffold's stack is fixed. Only the
   planner's own empty list skips Phase 2: a list emptied by dropped tasks
   runs Phase 2 from the request. The planner is sent the user's verbatim
   request as the request, and the modeling agent's "Assistant's notes" in a
   separate block labelled as context that is not the user's words and adds no
   task the request does not name (the requirements ledger reads the request
   the same way). It is also told what is being built, from the routing
   answer: for a web app, that people use it through screens and that the
   harness already adds the item that builds a missing frontend. An HTTP path the request names (``GET
   /stats``, "a /health/live endpoint") that the FastAPI scaffold does not
   serve becomes a checklist item. Routes are read from the source: method
   decorators and ``add_api_route``, under their router prefixes, the
   ``include_router`` chain and a mounted sub-application, plus FastAPI's own
   ``/docs``, ``/redoc`` and ``/openapi.json`` unless the app turns them off.
   A scaffold serving
   ``/statistics`` never counts as covering ``/stats``; the item closes only
   once the route exists.

   A ``web_app`` target (an app, web app, UI, website, dashboard or portal
   with no other platform named) gets a frontend even without a GUI model:
   the FastAPI scaffold, plus a React frontend the LLM writes in Phase 2. An
   explicitly headless request (API or backend only, a REST API, no UI) stays
   backend-only, and in a modify run a bare "app" names the existing project
   rather than asking for a UI. Such a request also puts "build the frontend"
   on the checklist, and Phase 3 reports a blocker if the output ends with no
   frontend files. With no GUI model, the item names the classes that have
   backend routes, says that a narrower UI scope in the request wins, and
   closes only once a routed app with at least one entity screen exists. A new project with both a class model and an agent model
   gets a checklist item to connect them (with a GUI model, only when a
   screen uses the agent or the run targets the agent; the generated web app
   includes the agent only when a screen uses it, as in the editor's own web
   app generation, and otherwise reports ``agent left out:`` as a warning): the agent's intents about the
   model's records must read and write them, not answer with canned text.
   The item also keeps Phase 2 from being skipped, and the agent may drop it
   with a reason when it has nothing to do with the data. Phase 3 checks the
   same thing as a judged requirement (see :doc:`validation`), in any language
   and however the agent reaches the data. The
   system prompt adapts to the run: the rule that asks for a complete CRUD
   frontend applies only when the run has or needs a frontend, and when shell
   tools are on and a FastAPI backend is present it gains a runtime
   verification runbook (see :ref:`spec-driven-shell-tools`). For a scaffold
   from the FastAPI backend generator, the endpoint manifest in the system
   prompt states what create, update and get-by-id return. The relationship
   mutation coverage block is left out only when the requirements ledger is
   known to be empty and the model has no constraint. The shell is
   one session for the whole run, like a terminal: ``cd`` and exported
   variables carry over between commands, and on Linux, where the run has a
   sandbox, a server started in one command answers requests in the next,
   until the run ends.

   On Anthropic the Phase 2 system prompt is sent as two cached blocks, split
   at ``## Variable context`` (``llm_client.SYSTEM_CACHE_SPLIT``). The first
   holds the role, the plan and the rules and depends only on the run
   configuration, so web-app runs of one configuration read it from each
   other's cache (a non-web target is named in the rules, so that block is
   shared only between runs for the same target); everything per project or per request (models, data contract,
   design system, idioms, inventory, routes, requirements, the request) comes
   after the marker. A Phase 2 request then uses all four cache breakpoints
   the API allows (tools, the stable block, the end of the system prompt, the
   rolling message marker), so no other one can be added. Other providers get
   the prompt as one string.

**Phase 3 — validation and bounded auto-fix.**
   The workspace is snapshotted, then swept by a set of static validators.
   Findings are classified by the kind of evidence: a **blocker** is an observed
   failure or a requirement the judge found missing or partial; a check that
   reached no conclusion (a citation not located, a guessed request the app
   refused, a check that did not run) is *not confirmed*, listed for the user and
   never a blocker. Blockers drive a
   bounded repair loop of (LLM fix turns → re-validate) rounds: at least five,
   and as many as the turns Phase 2 left over, each round up to ten LLM turns.
   With auto-fix on, Phase 2 leaves this loop part of ``max_turns`` (see
   :doc:`runs`), so a model that never ends its turn still has its output
   checked and repaired. Each round is steered to the files its blockers name
   and their neighbours (a blocker that names no file leaves the round open),
   and no tool is forced. A round that edits is kept only if it retires a
   blocker it faced, adds no observed failure except ones a failure it fixed had
   hidden, leaves no more blockers by the same measure, does not stop an app
   that started from starting, and leaves no more entities or actions failing
   at runtime; otherwise the tree from before it is restored and the loop ends.
   A round that edits nothing continues the loop only if the evidence or
   dependencies it changed pass the same check, or once after every write it
   tried was refused. Every kept tree is re-snapshotted as the next round's rollback
   target; the pre-Phase-3 tree is the first snapshot, so a failed repair can
   never cost you the Phase 2 work.
   Non-blocker findings are recorded but never burn fix turns.
   See :doc:`validation` for the full list of checks, the severity model, and
   how to turn the repair half on or off.

The whole run streams progress over Server-Sent Events, so the editor shows the
phase timeline, the LLM's tool calls, a live cost/runtime meter, and the
streamed text as it happens. The stream reports five coarse phases:
``select``, ``generate``, ``gap``, ``customize``, and ``validate``. Within
``validate``, which can run for many minutes, each step is named as it starts
(the check round, starting the backend, judging the requirements and how many,
each fix attempt and how many blockers it faces), and the card shows how long
the current step has been running. The event schema is documented in :doc:`api`.

A run, end to end
-----------------

Suppose your project contains a class diagram with ``Book``, ``Author``, and
``Member`` (and their associations), and you ask:

   *"Generate a FastAPI backend for this model with JWT authentication and a
   Dockerfile."*

The run unfolds like this:

#. **select / generate** — the :doc:`FastAPI (Backend) generator
   <../generators/backend>` runs and produces a model-faithful baseline:
   SQLAlchemy models for ``Book`` / ``Author`` / ``Member``, Pydantic schemas,
   and a router per resource.
#. **gap** — the planner notes the request needs two things the template
   doesn't provide: JWT auth and a Dockerfile.
#. **customize** — the LLM adds them on top of the scaffold, e.g.::

      main_api.py             # slim app — now wires the auth router in
      database.py             # engine / session / get_db
      sql_alchemy.py          # Book, Author, Member — from your diagram
      pydantic_classes.py
      bal_stdlib.py
      routers/
      ├── book.py             # CRUD, now behind auth
      ├── author.py
      └── member.py
      auth.py                 # NEW — JWT issue/verify, password hashing
      deps.py                 # NEW — get_current_user dependency
      Dockerfile              # NEW
      requirements.txt        # + python-jose, + passlib

#. **validate** — Phase 3 sweeps the result. Say the LLM referenced
   ``SECRET_KEY`` in ``deps.py`` without importing it: ``ruff`` reports
   ``F821``, which is a blocker, and the fix loop repairs it before the ZIP is
   handed back.

The domain classes come from your diagram — the model stays the source of
truth; only the auth and container scaffolding are LLM-authored. The ``done``
event carries a three-way authorship split over the final tree (untouched
generator output / generator output the LLM modified / LLM-authored), so you
can see exactly how much of the app each side carried.

How a run stays on the rails
----------------------------

An agent loop fails in characteristic ways — it fills its context, it edits the
same file forever, it declares victory with work outstanding, it gets cut off
mid-write. Phase 2 carries a specific guard for each. They matter when you are
reading a trace and wondering why a run did what it did.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Guard
     - Behaviour
   * - Batched tool calls
     - Up to four tool calls from one turn execute in parallel, *except* that
       calls writing to the same path are grouped and run sequentially in the
       order the model emitted them — otherwise two edits to one file in one
       turn would both read the pre-turn content and the second would clobber
       the first.
   * - Context compaction
     - Above ``BESSER_LLM_COMPACT_THRESHOLD`` estimated tokens (default
       80,000) everything but the most recent turns (about 30% of the
       threshold, in tokens) is replaced by a summary: a model recap, a
       tool-call histogram, the lists of files written and read, the project's
       files (dependency and build directories left out) and the open
       checklist items and blockers. The cut never lands between a tool call
       and its result, and the kept turns lose their thinking blocks (see
       :doc:`models`). A compaction that would not shrink the history (the
       kept turns alone fill the budget) is skipped with a warning. For models
       with a known small context window the threshold is lowered to that
       window minus a reserve for the reply (the run's output-token ceiling,
       at least 16,000 tokens), so a small model compacts instead of
       overflowing; a model with a known large window gets 80% of that
       difference, capped at ``BESSER_LLM_MAX_COMPACT_THRESHOLD``.
   * - Truncation recovery
     - When a response is cut off at the output-token limit, the partial turn
       is discarded rather than executed — nothing lands half-written on disk —
       and the model is told to emit a smaller turn. At most four such
       recoveries per run; after that the run stops with a resumable
       ``api_error`` rather than burning the cost cap. The budget is a
       per-run total that does NOT reset after a clean turn, so it is sized
       against the whole turn budget, not against a single bad patch.
   * - Per-file modify loop
     - Three consecutive ``modify_file`` or ``replace_file_lines`` calls on
       the same path that were all refused inject a reminder naming the
       refusal (a text that did not match, or an edit the syntax guard
       rejected) and how to recover, never to rewrite the file from memory.
       Successful edits do not count; it fires once per path. A
       ``read_file`` of that same file does not break the streak; any other
       tool, or a read of another file, does.
   * - Checklist gate
     - The run does not finish while ``task_list`` items are open: an
       ``end_turn`` with open items is sent back with the list, twice — four
       times when an open item carries a machine check. Some items are verified
       before they are accepted (the "build the frontend" item checks that
       frontend files exist). Ending the turn also runs ``validate_app``; a
       blocker from its static frontend checks is sent back to the model once
       before Phase 2 ends. The harness closes an item itself when its check
       passes, and refuses ``done`` while the check fails; the model closes
       an item without a check with ``task_list(action='done')``, without
       quoting code. An edit to a file the item names never closes it. Work
       the model cannot finish is recorded with ``action='blocked'`` and
       stops holding the gate. An item the user did not ask for is closed
       honestly with ``task_list(action='drop', id=N, reason=...)`` rather
       than marked done. In Phase 3 only an item whose check fails is a
       blocker; an item nothing closed is listed in the run report as not
       confirmed.
   * - Per-write diagnostics
     - Every file the model writes is parsed immediately — ``ast`` plus
       pyflakes' undefined-name checks for Python, a structural scanner for
       TS/TSX/JS/JSX, and the respective parser for JSON, YAML and TOML — and
       any finding comes back in the same tool result, while the file is still
       in context. An edit that would turn a file the harness can parse into
       one it cannot is refused outright and never reaches disk, in those same
       languages. Data-contract violations ride along the same way. A
       cross-file finding (a router reading a field its create schema lacks, a
       form field the request schema refuses) comes back on a write to a file
       it names, and otherwise once per content of those files, with the other
       files under ``in_other_files``; it is not repeated on unrelated writes.
       Toggle with ``BESSER_LLM_PER_WRITE_DIAGNOSTICS`` (default on).
   * - Checkpointing
     - A checkpoint is written to ``.besser_checkpoint.json`` in the workspace
       after every tool-use turn and again when Phase 3 starts and as its
       repair progresses. It is deleted only when Phase 2 ended cleanly,
       Phase 3 was not interrupted and no blocker or not-confirmed finding
       remains. A run cut short by an error, a cap, or
       a cancellation, or one that ends with open findings, therefore leaves
       one behind for :ref:`resume <spec-driven-durable-runs>`, which refuses to
       resume against a project whose fingerprint has changed and seeds the
       already-spent cost so a crash-and-resume cycle cannot double the bill.
       Because the file's *absence* is what marks a clean finish, a write that
       fails would otherwise be indistinguishable from one: such a run is not
       resumable, and an ``error`` record naming the reason is appended to the
       trace so a post-mortem can tell the two apart.
       Toggle with ``BESSER_LLM_ENABLE_CHECKPOINTING`` (default on).
   * - Tracing
     - Every run start and end, phase transition, turn, tool call, cost
       update, checkpoint write (or the ``error`` of a failed one), target
       decision, snapshot, rollback, regeneration and validation finding is
       appended to ``.besser_trace.jsonl`` in the workspace (events
       ``run_start``, ``phase_enter`` / ``phase_exit``, ``turn_start``,
       ``tool_call``, ``cost_update``, ``llm_call``, ``compaction``, ``eviction``,
       ``checkpoint``, ``error``, ``target``, ``snapshot``, ``rollback``, ``regeneration``,
       ``validation_issue``, ``requirement_cited``, ``run_end``). ``run_start`` holds the start of
       the user's own request, not the modeling agent's notes. Every Phase 2
       turn, the closing reply included, and every Phase 3 fix turn
       (``phase: "phase3_fix"``) writes a ``turn_start`` and a ``cost_update``
       with the cumulative token counters. Each paid call outside those loops
       writes one ``llm_call`` with its ``purpose`` (``routing``,
       ``routing_target``, ``model_deltas`` (a modify run's model sync; a
       modify run writes it and its ``routing`` record before its
       ``run_start``, with the run's id), ``gap_analysis``, ``requirements_extraction``,
       ``requirements_judge``, ``requirements_rejudge``), ``outcome``, the
       calls, tokens and ``cost_usd`` it added, and for a judge the verdict
       count per status, how many "implemented" citations the check verified
       (``citations``) and why each other failed (``failures``, by requirement
       id). Each also names ``model`` (and ``planning_model``, which gap analysis
       and the judge try first). A ``compaction`` record (the history's
       estimated ``tokens_before`` and ``tokens_after``) or an ``eviction``
       record (how many stale file bodies were ``stubbed``) names the ``turn``
       whose request carried the rewritten history, so the cache miss that
       turn pays is expected; compactions are also counted in the checkpoint
       and the recipe (``compactions``). A ``tool_call`` record holds its
       ``batch_index``, its place among the calls the model sent in that turn
       (parallel calls are logged as each finishes), the input and a
       ``result`` excerpt (its length and the first 500
       characters), with credential-shaped values redacted before the cut;
       each input value is cut at 500 characters, and the untruncated input
       of every write tool goes to ``.besser_tool_inputs.jsonl``. A
       ``generator_output`` record holds each generator call's files (SHA-256,
       traced line ranges), a ``requirements`` record the requirements ledger,
       and the run, phase and turn records carry the client's cumulative token
       counters (``usage``). The run's trace (:ref:`spec-driven-run-trace`) is
       built from this log, so with tracing off it has no turns. It
       is best-effort — a failed
       trace write never fails a run. Toggle with
       ``BESSER_LLM_ENABLE_TRACING`` (default on).
   * - Secret redaction
     - Credential-shaped values are stripped at both boundaries where output
       leaves the process: every SSE frame is redacted before it is sent, and
       the workspace is swept before the artifact is packaged — a populated
       ``.env`` is deleted outright, a template ``.env.example`` is kept. The
       run's own API key is also matched literally, so it is redacted from
       every frame and from the packaged workspace whatever its format. The
       count of findings is recorded in the run's recipe.

       In source files only the secret itself is replaced, inside its quotes,
       so the file still parses. Provider tokens are recognised by the
       prefix-anchored rules of `gitleaks <https://github.com/gitleaks/gitleaks>`_
       (MIT), vendored as ``besser/utilities/data/gitleaks_rules.toml``
       (``ghp_``/``github_pat_``, ``AKIA``/``ASIA``, ``sk_live_``, ``AIza``,
       Slack webhooks, ``glpat-``, ``hf_``, ``npm_``, ``SG.``, PEM private
       keys, JWTs and more). Rules that match on a variable name or on
       entropy alone are left out: a false hit would rewrite working code.
       A private key is redacted even when it is cut before its ``END``
       line. A database URL's password is redacted unless the host is local
       (a single label such as ``db`` or ``localhost``, ``*.localhost``,
       ``*.local``, ``host.docker.internal``, or a loopback or private
       address), since the generated ``docker-compose.yml`` sets the same
       password on the database. Code that builds the URL
       (``{settings.db_password}``, ``%s``, ``$DB_PASSWORD``) is left alone.
       Symbolic links, FIFOs and other special files in the workspace are
       skipped, never opened.

       The rules live in one module, ``besser.utilities.secret_redaction``,
       which the trace (above), the shared OpenAI chat client's error logs
       (``besser.llm.openai_chat``) and the editor
       backend's generator-config logs use as well. A log line masks a local
       host's database password too: only the generated app needs it.
   * - Artifact exclusion
     - The download ZIP leaves out the same runtime and build artifacts as the
       GitHub push: ``*.log``, ``*.db``, ``*.db-journal``, ``*.sqlite`` /
       ``*.sqlite3``, ``*.zip`` and ``*.pyc`` files, plus dependency and build
       directories such as ``node_modules`` and ``target``, and the run's own
       ``.besser_*`` bookkeeping except the files a later run carries forward
       (the recipe, the attribution and generator-output records).
   * - Link-free workspace access
     - The workspace is written by model-authored code, so a symbolic link in
       it could point at a file outside the run. The download ZIP, the seed
       copied for a ``modify`` run and the tree pushed to GitHub keep only
       regular files and real directories; symbolic links, devices, sockets
       and FIFOs are dropped, never read through. The harness itself never
       follows a link or special file in the workspace either: not when it
       walks or copies the tree (prompt sections, the Phase 3 snapshot, the
       probes' scratch copies), and not when it opens or writes a named file
       (the recipe, checkpoint and trace, the ``BESSER_GENERATION.md``
       provenance file, or a generated source file).

Between Phase 2 and Phase 3, the Python files the LLM wrote whole in a create
run are passed through ``ruff format --isolated``, which changes layout only,
so Phase 3 validates the layout that ships; a file it only edited, generator
files and the files of a modify run keep their own layout. A formatter that
could not run is reported under *Not confirmed*. The download's
``BESSER_GENERATION.md`` gains a *This run* section: what was built, the base
generator and the verification counts (confirmed by running the app's API, found
in the code but not run, not confirmed, checked and not met, and test workflows that ran and
failed), a *Not verified* line when a whole
class of checks left no evidence, with up to eight lines each for what was
checked and not met (a rule missing in the code, cited code that does not do it), what ran and
failed (a test workflow the agent wrote) or not confirmed (a check that could not run, a
requirement whose citation was not located), and how many more there are. A list
of more than five lines puts findings of one kind on one line ("5 create
operations could not be tested automatically, because each needs an existing
record first: ..."). These items, the run card's verification report and the
first issue named in an incomplete notice are one plain sentence per finding
(``validation/user_wording.py``): a validator's message is written for the
agent, with the request it sent, the file and line to edit and what to call
next, so it stays in the agent's prompt, the trace and the recipe's
``validation_issues``. A finding with no sentence of its own reads as a generic
one, never as the agent's text. A request that states no business rule adds a *Not
confirmed* line saying so, and an implemented requirement that was only found in
the source is counted as found in the code, never as confirmed; a confirmed one
names the call that exercised it. No check opens
the screens, so only a call to the running app's API confirms anything. It leaves out the request itself, because the file
also goes into GitHub pushes, and shows model-written values as code, never as
links or markup.

Every output says how to run it. When the output has no README at its root,
packaging writes one from the files delivered: for a FastAPI backend beside a
Vite frontend, the Dockerfiles, compose file and README of the scaffold repair;
for an app started by one compose file, ``docker compose up --build`` in its
folder and the ports it publishes; for a FastAPI backend on its own, the virtual
environment, ``pip install -r requirements.txt``, the start command of its entry
file and the address of its API documentation. Nothing is written for other
shapes, such as a backend beside a frontend of another framework, or an entry
file Python cannot import by name. The README BESSER writes ends with an HTML
comment line saying so: a later run rewrites that README to match its output (a
modify run that adds a frontend gets the two-service README) and removes it when
no shape fits any more. A README without that line is the user's and is never
touched. The *This run* section then points to the README.

Every ``BESSER_LLM_*`` switch named above is collected in
:doc:`configuration`.
