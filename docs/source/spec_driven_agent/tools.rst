The agent's tools
=================

Phase 2 is an agent loop, and everything the LLM can do to a project it does
through a declared tool. The tool list is scoped per run: a generator whose
required model is absent is never offered, so the LLM cannot pick a tool that
would immediately fail.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Group
     - Tools
   * - Files
     - ``list_files``, ``read_file``, ``write_file``, ``modify_file``, ``replace_file_lines``,
       ``search_in_files``, ``delete_file``
   * - Validation / bookkeeping
     - ``validate_app``, ``test_api`` (while the workspace holds a generated
       FastAPI backend), ``task_list``
   * - Generators
     - The 19 generator tools listed below.
   * - Shell
     - ``run_command``

A tool is offered only when it can return something the context lacks. The
system prompt holds the whole serialized domain model (inherited members, OCL
constraints and rejected expressions included), so there is no tool to query
it; the write tools check the syntax of what they write, and ``validate_app``
runs the application checks.

File edits and the checklist
----------------------------

``modify_file`` refuses ambiguous anchors, identical no-ops and duplicate-looking
insertions. ``already_applied`` requires a receipt for that exact successful edit
and an unchanged post-edit file hash. After restart, matching replacement-shaped
content may instead produce ``possible_replay``: no write is made, but no success
is claimed. Missing-file responses suggest existing paths without substituting
them for the requested path.

``write_file`` over an existing file requires the whole file to have been read
(or written) this run; a rewrite from memory would drop scaffold code, and a
landed ``modify_file`` does not count as a read. Like the other
editors, it refuses an edit that would make the ORM or Pydantic module
(``sql_alchemy.py``, ``pydantic_classes.py``) fail to import. Only the
``N| `` line numbering that ``read_file`` displays (``7| code``, or the padded
``   7| code`` of the post-edit echo) is stripped from edit text, and only
when it is uniform; a single copied numbered line is refused
rather than guessed at. ``write_file`` strips it only when the content is
exactly that display of a whole file (every line numbered from 1), so numbered
data such as a pipe-delimited id column is written as sent; any stripping is
reported in the result's ``note``. ``list_files`` and ``search_in_files`` skip links that
resolve outside the workspace, and the harness never reads or writes a
workspace file through a symbolic link or special file. A call with a missing required argument, or
arguments that are not valid JSON, returns a tool error naming the problem
instead of running. Each argument is also checked against the type its tool's
schema declares: a null optional argument counts as omitted, ``"12"`` or ``12.0``
is accepted for an integer, ``"true"``/``"false"`` for a boolean and a single value
for an array; any other mismatch (a boolean or ``1.5`` for an integer, ``inf``, an
object for a string, a value outside an ``enum``) is refused with an error naming
the argument, the expected type and the type received. An argument the tool
does not declare (``old_text`` on a range edit, a ``path`` scope on
``search_in_files``) is not used, and the result says so under
``ignored_arguments``, with the arguments the tool takes.

A refused edit carries one next step (``edit_recovery``), the same in
customization and validation repair. When a ``modify_file`` miss brackets one
region, or a range edit's ``read_id`` is stale, the result holds that span as a
``located_range`` with a fresh ``read_id`` for ``replace_file_lines``. From the
second refused edit on a file (with no landed edit between), the step is to read
the whole file and ``write_file`` it back. The orchestrator's reminders repeat
that step. No tool call is forced: the step is in the result, and the model
chooses its next call, in Phase 2 as in a repair attempt.

A rewrite of a file this run did not create must keep its code when the file
is a generated one over 200 lines, or, in a modify run, any file over 15
lines. Keeping its code means: at least 60% of its non-blank lines; every
top-level declaration (Python ``def``, ``class`` and assigned names, and each
Python class's own methods, fields and assigned names; TS/JS column-0
declarations outside comments); no fewer Python decorators and
module-level calls (``app.include_router(...)``), and no fewer TS/JS ``export
default`` statements, ``<Route>`` elements and re-exports (``export { A } from
"./a"``); no new elision such as ``{/* ... unchanged ... */}``; and no new
comment-only line that reads as a placeholder (``{/* remaining fields
unchanged */}``) in place of three or more of the file's lines that the
rewrite has fewer copies of (a repeated section counts), or added while a
block of three or more such lines is deleted elsewhere, however much the
rewrite adds. A rewrite that misses one is refused saying which
(``rejection_kind`` ``rewrite_loss``), whatever edits came before it. The
same rule holds a ``modify_file`` or ``replace_file_lines`` edit that writes
new code over 90% or more of the file's non-blank lines (with ``replace_all``,
counting every occurrence), so a whole-range edit cannot apply a rewrite the
rule refused. An edit over less of the file is a targeted edit and is not held
to it, however much shorter it makes the file (simplifying a function, folding
calls into a loop). An empty replacement is a removal and
lands however much it spans: for a removal the model means to make, the
refusal says to replace only that code's own lines with nothing (naming the
``read_id`` of its whole-file read), and for a rename to edit only the line
that names it. A file
``write_file`` created this run, and a smaller file, is rewritten freely. A
change these checks cannot see (one entry dropped from a large object literal)
is a targeted edit's job, and is not caught here; a file with no declarations
to compare (CSS, JSON, YAML, HTML, ``requirements.txt``) is held only to the
60% share and the elision checks. Repeated failures do not
permanently freeze the file; an agent that ignores recovery is still bounded
by the loop and run budgets.

The rule guards against losing code by accident; it is not a wall against an
agent that removes code in several steps. Each step is judged against the file
as it is then, so these sequences still land on a large generated file:
``delete_file`` (or ``rm`` through ``run_command``)
and then ``write_file``, since the file is then new to this run; emptying the
file with a range edit and then ``write_file``, since an empty file is below
the size limit; and several range edits that each replace less than 90% of
the file.

``delete_file`` removes one regular file, for every model alike: whether a
delete runs never depends on the model. When other files still import a
top-level module or package the delete removed, the result lists those imports
under ``still_imported`` (the same unresolved-import findings validation
reports, new since the delete), so the model can change or remove them. An
import of a deleted module inside a package that still exists (``from services
import billing``) is not listed; the build and startup checks catch it, and any
import the model leaves.

``read_file`` returns a ``read_id`` tied to the resolved path, whole-file content
hash, and actually displayed lines. ``replace_file_lines`` accepts that ID,
1-based inclusive ``start_line``/``end_line``, and complete ``new_text``. It does
not require reproducing the old text. A line longer than one read's cap is
shown clipped and never authorized by a ``read_id``. Unread/truncated ranges, stale IDs,
no-ops, new elisions, and changes that break previously valid Python syntax are
refused without writing. After a successful edit, read again before another
same-file range edit. IDs are session-local and expire on resume. The tool shares
path containment, per-file locking, diagnostics, and checklist write tracking
with the other editors. An accepted edit is not proof of business correctness.

A ``read_file`` repeating an earlier call (same path, ``offset`` and
``limit``) on unchanged content, while that earlier result is still in the
conversation, returns a short ``unchanged`` stub naming the still-valid
``read_id`` and line range instead of the text. Repeating the call right after
a stub returns the text. Compaction, history eviction, a resume and each
Phase 3 fix round (a fresh conversation) reset this, so the next read is full.

Every tool result reaches the model as JSON with non-ASCII text kept literal
(``é``, not ``\u00e9``). ``list_files`` returns one ``<path> <size>`` line per
file, skipping installed dependencies, build output and any directory holding
a ``pyvenv.cfg`` (a virtualenv, whatever its name; it is named under
``not_listed``). Past 300 files it lists source files first and ends with a
``... N more files not listed`` line counting the rest per directory.
``search_in_files`` returns up to 50 matches; it skips lock files
(``package-lock.json``, ``yarn.lock``, ``pnpm-lock.yaml``, ``poetry.lock``,
``Cargo.lock``), minified bundles, source maps and binaries unless
``file_glob`` names the file exactly, and says so under ``not_searched``.
``file_glob`` matches file names only; one containing a path separator is
refused with an error rather than returning no matches.

``task_list`` takes one shape with four optional fields, any combination in
one call: ``done`` (item ids), ``add`` (item texts), ``drop`` and ``blocked``
(lists of ``{id, reason}``). A call with none lists every item with its status,
and every result ends with ``open_items``, the id and a short label of each open
item. The earlier ``action=...`` shape (``list``, ``done``, ``add``, ``drop``,
``blocked``, ``mixed``) is still accepted. ``done`` runs an attached verifier
and refuses the item while it fails; a verifier exception is a failed check,
not success. An item without a verifier closes on the model's word, with no
code to quote.

After every successful edit the harness re-runs the verifiers and closes the
items whose check now passes. Nothing else closes an item: an edit to a file
an item names is recorded with the item (as ``written``) and quoted in its
"not confirmed" finding, but it does not show the item is done. An item names
a file only by a path written in its text (``backend/routers/order.py``),
matched exactly against the edited path; the planner's paths are first
repaired against the workspace (a suffix matching exactly one file becomes
that file's path). Only an edit that changed the file after the item was
created is recorded: a refused or failed call, an identical rewrite and an
edit a Phase 3 rollback discarded are not. A named path that is not a
workspace file is reported as absent; the item for a modeled rule the
generators could not enforce names ``routers/<class>.py``, relative to the
backend folder, so its finding says no workspace file is there.

``blocked`` requires a reason and records required work the model
could not finish; the item stops holding the end-turn gate. A later passing
check can clear the block. ``drop`` is only for work outside the user's
request. Task outcomes are retained in the recipe.

Application checks during editing
---------------------------------

``validate_app`` runs the deterministic checks that decide the run, the same
function Phase 3 runs after every fix attempt and at the end: source and
schema/router contracts, startup and create requests, retained ``test_api``
workflows, frontend, data and model checks, ruff, the dependency check, and the
type check, frontend build and screen check. Phase 3 alone adds the requirements
judgment (``validate_app`` lists its last verdicts), the checklist's findings
(open items are the end-turn gate's) and the authorization check. It lists at most 40 findings, blockers
first, and counts the rest (``remaining_issue_count``); a finding longer than 1,200
characters is cut as in the Phase 3 fix prompt, keeping its head and tail. Validation runs after writes in the same tool batch.
A warning is marked ``advisory, not a blocker``. The acceptance-matrix findings
wait for frontend files and are left out with a GUI model. tsc and the frontend
build reuse their last verdict while no file in the folders holding a
``package.json`` or ``tsconfig.json`` changes and the installed dependencies are
the same, and the dependency check while its ``requirements.txt`` is unchanged.
tsc does not run on a project whose dependencies are not installed, and the
screen check needs a browser; a check that did not run is listed as not run,
never as a blocker, and ``verified`` is true only when none is listed and no
blocker remains. When the run has a checklist, a passing result
also lists its open items (``open_items``: id and a short label, empty when
none is open), the ids ``task_list`` needs to close them. Ending a turn runs
the same checks, so the tool's description asks for a call after a code change
only, not a second one before finishing.

``test_api`` executes up to 20 declarative requests against a disposable copy of
one generated FastAPI backend and a fresh SQLite database. It is offered, and the
prompts name it, only while the workspace holds such a backend (``main_api.py``
next to ``sql_alchemy.py``), decided when Phase 2 and Phase 3 start, so the tool
list does not change within a phase. Requests share state
within a scenario; later requests may reference response JSON using
``{{0.room.id}}``. When the path is missing from a response that wraps the record
under exactly one key (``{"todolist": {"id": 1}, "items_ids": []}``), ``{{0.id}}``
resolves through that key and the response carries a note such as ``{{0.id}}``
resolved as ``{{0.todolist.id}}``; ``expected_fields`` paths are matched exactly.
Assertions specify expected statuses and dotted JSON fields;
an expected field value accepts the same references, so a check can name an id
an earlier request returned. Braces that resolve to no earlier value are text,
compared and reported as written.

A request body is ``json``, or ``form``: urlencoded fields (an OAuth2 password
login, a form post), whose values accept the same references. With ``files`` the
body is multipart, for an upload: each file field maps to ``{"filename",
"content", "content_type"}`` with text content (a CSV, a small text document).
``json`` and ``form``/``files`` in one request are refused.

A request may carry ``headers``, whose values accept the same references, so a
scenario can register, log in and send the returned token on later requests:
``{"Authorization": "Bearer {{1.access_token}}"}``. Allowed names are
``Authorization``, ``Accept``, ``Content-Type``, ``Accept-Language``,
``If-Match``, ``If-None-Match``, ``Idempotency-Key`` and ``X-*`` API headers;
``Host``, ``Cookie``, framing and hop-by-hop headers, ``X-Forwarded-*`` and
method-override headers are refused. Cookies a response sets are kept for the
rest of the scenario. The values of credential-bearing headers (``Authorization``
and names containing words such as *key*, *token*, *secret* or *auth*) are
redacted from the report, from ``action="get"`` and from the trace, as is the
value wherever the app echoes it. Response fields named like a credential
(``access_token``, ``api_key``, ``client_secret``, ``password``,
``password_hash``, ``passwordNew``, ``secret_answer``, ``reset_token``, a
non-numeric ``session_id``, ``reset_code``, nested ``tokens.access``; not a
page cursor such as ``next_page_token`` nor a field whose next word only
describes or identifies one, such as ``password_changed_at``, ``token_count``,
``api_key_name`` or ``secret_id``), the input a
validation error echoes for such a field, and any JWT-shaped string are
redacted too, even when no later request sends them, including inside
assertion failure messages and in the path, error and failures of a later
request that sends such a field back by reference; other fields that happen to
hold the same value are shown. References such as ``{{1.access_token}}`` still
receive the real value.
Named scenarios are retained during the run and replayed after source changes;
failures block completion. Correcting a mistaken test requires an explicit
``correction_reason``. A correction that removes a request which passed in the
scenario's last run (same method and route no longer present) is refused: such
checks stay in the corrected scenario or go under their own ``scenario_id``.
Model-authored scenarios are supplemental checks, not an
independent proof of specification coverage.

Use ``test_api(action="list")`` to discover retained workflows and
``test_api(action="get", scenario_id="...")`` to inspect their exact inputs,
expectations and last report without executing them. ``action="run"`` with
only a scenario ID replays its saved definition. ``action="retire"`` with a
scenario ID and ``correction_reason`` drops a workflow another one replaced: it
is no longer replayed and no longer blocks the run, and the run report lists it
under *Could not verify* as retired, with the reason; a workflow that had failed
when it was retired is listed as *failed, retired*: retiring it settled nothing. The original specification
remains authoritative: a generated assertion can be wrong and must not force
correct application behavior to regress.

In the run report a failed workflow is listed under *Could not verify*, as a
test the agent wrote that failed, not under *shipped unenforced*: it is not a
rule the user stated, and either the test or the app may be wrong. The rules the
report counts as shipped unenforced are the user's requirements, the model's
OCL constraints and its modeled operations.

Both runtime probes honor ``enable_import_smoke_check``. The API tool accepts no
shell commands or external URLs. The generated app runs in the same bubblewrap
sandbox as ``run_command``, with no network; see
:ref:`spec-driven-shell-tools` for when the sandbox applies.

Generator tools
---------------

Each of these calls a BESSER :doc:`code generator <../generators>` directly —
the same deterministic generator you would run yourself from Python. They are
what makes the agent an orchestrator rather than a code writer: for anything
BESSER already templates, the LLM asks for the template output and edits it
instead of writing it from scratch. They are declared with the generators
themselves, in ``besser/generators/registry.py``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Tool
     - Generator
   * - ``generate_fastapi_backend``
     - :doc:`Backend (FastAPI + SQLAlchemy + Pydantic) <../generators/backend>`
   * - ``generate_django``
     - :doc:`Django <../generators/django>`
   * - ``generate_rest_api``
     - :doc:`REST API <../generators/rest_api>`
   * - ``generate_web_app``
     - :doc:`Full Web App <../generators/full_web_app>`
   * - ``generate_react``
     - :doc:`React <../generators/react>`
   * - ``generate_python_classes``
     - :doc:`Python <../generators/python>`
   * - ``generate_java_classes``
     - :doc:`Java <../generators/java>`
   * - ``generate_pydantic``
     - :doc:`Pydantic <../generators/pydantic>`
   * - ``generate_sqlalchemy``
     - :doc:`SQLAlchemy <../generators/alchemy>`
   * - ``generate_sql``
     - :doc:`SQL <../generators/sql>`
   * - ``generate_supabase``
     - :doc:`Supabase <../generators/supabase>`
   * - ``generate_json_schema``
     - :doc:`JSON Schema <../generators/json_schema>`
   * - ``generate_json_object``
     - :doc:`JSON Object <../generators/json_object>`
   * - ``generate_rdf``
     - :doc:`RDF <../generators/rdf>`
   * - ``generate_baf``
     - :doc:`BAF agent <../generators/baf>`
   * - ``generate_bpmn``
     - :doc:`BPMN <../generators/bpmn>`
   * - ``generate_qiskit``
     - :doc:`Qiskit <../generators/qiskit>`
   * - ``generate_pytorch``
     - :doc:`PyTorch <../generators/pytorch>`
   * - ``generate_tensorflow``
     - :doc:`TensorFlow <../generators/tensorflow>`

The whole-application generators (``generate_web_app``, ``generate_django``,
``generate_fastapi_backend``, ``generate_rest_api``) run
at most once per application. Once one of them has built the app (in Phase 1,
or in the run a modify run continues), all of them are removed from the
toolset and the agent's calls to them are refused: running one again would
overwrite every edit made since. In a modify run on any other seed, only the
generators whose output folder the seed already has (``backend/``,
``web_app/``, ``django/``, ``rest_api/``) are refused. A resumed
run keeps the same lock. To change the stack, start a fresh run. The lock covers
the agent's calls only: when the model changed, a modify run itself regenerates
the generated files and merges the change into edited ones (see
:ref:`spec-driven-regeneration`).

After Phase 1, Phase 2 drops the generators whose output the chosen generator
already bundles (Pydantic, SQLAlchemy and the like next to a backend). The
single-artifact generators that remain (``generate_python_classes``,
``generate_java_classes``, ``generate_rdf``, ``generate_supabase``) stay
offered whether or not the request asks for that artifact: nothing structured
in a run says which extra artifacts the user wants, and the tool list is never
chosen by matching the request's wording.

.. _spec-driven-shell-tools:

Shell tools
-----------

``run_command`` runs arbitrary commands, installs included. It is
what lets the agent close its own loop — run ``pytest``, run ``npm run build``,
read the failures and fix them — and they are the single largest capability
difference between a run with it and a run without. It is also arbitrary
code execution on the host that serves generation, so who is allowed to switch it on is a
deployment decision, never a per-request one.

A command cannot change the run's own records: ``.besser_checkpoint.json`` and
``.besser_recipe.json`` are restored after each command if it rewrote, deleted
or planted them, and the result says so under ``run_records_restored``.

When a write tool writes a ``requirements.txt``, the harness pins the
known-incompatible dependency pairs described in :doc:`validation` into it and
lists each change under ``pinned`` in the write's result, with a note that
packages installed from the file before no longer match it.

The policy
~~~~~~~~~~

**Off by default.** The code default is off, and so is every configuration
that does not set it: the local ``docker-compose.yml``, the backend service in
``docker-compose.prod.yml``, and a backend started by hand. With the tools off,
every static tool (``read_file``, ``write_file``, ``modify_file``,
``validate_app``, the generators, the runtime probes) stays available, so the
agent still produces and validates a full application; it just cannot shell
out. The ``pip install --dry-run`` dependency check in
:doc:`Phase 3 <validation>` is behind the same flag, since resolving an sdist
can execute its build backend.

**On for the hosted editor's isolated worker only.** In production, generation
runs in its own container, ``besser-wme-smartgen``, and that service alone sets
``BESSER_LLM_ENABLE_SHELL_TOOLS=true`` in its own ``environment:`` block. The
worker has no ``env_file`` and receives only the LLM credentials, sits on its
own network, and confines every command in the bubblewrap sandbox described
below. The backend, which holds the SMTP, GitHub OAuth and telemetry secrets,
keeps the tools off; never enable them through the shared ``.env``, which the
backend loads. See :doc:`production_deployment`.

**On, deliberately, for a local or on-prem install.** One machine, one tenant,
the operator's own data: the multi-tenant objection does not apply, and
withholding the tools is pure loss. The local ``docker-compose.yml`` already
carries the sandbox's ``seccomp`` and ``apparmor`` entries, so setting the
variable in ``./.env`` is enough there on Docker Desktop; a Linux host such as
Amazon Linux also needs ``systempaths=unconfined`` (see
:doc:`production_deployment`).

The decision is read from the process environment at start-up:

.. code-block:: bash

   BESSER_LLM_ENABLE_SHELL_TOOLS=1
   # worth pairing with, so Phase 3 can actually compile what it just built:
   BESSER_LLM_ENABLE_TOOLCHAIN_VALIDATION=1

``GET /besser_api/spec-driven/config`` reports the resulting value as
``features.shell_tools_enabled`` — check it after a deploy rather than trusting
that the variable reached the container. A :doc:`Python library run <usage>`
opts in per call instead::

   LLMGenerator(model=model, instructions=...,
                allow_shell_tools=True, enable_toolchain_validation=True)

There is no request field, header or query parameter that turns shell tools
on. A request body carrying ``allow_shell_tools`` is ignored, so a hosted
deployment cannot be talked into granting the capability by a client.

What the local path does and does not guarantee
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What it does:

- **Refusal, not concealment.** The gate is enforced when the tool is called,
  not by leaving it out of the advertised list — a model that names a tool it
  was never offered is refused too.
- **One shell per run.** Every command starts in the workspace root, or in
  its ``working_dir``: a ``cd`` lasts for its own command only, since models
  write each command as if from the root. The variables a command exports
  (``export``, ``source .venv/bin/activate``) carry over to the next command,
  and on Linux a process started in the background (``python app.py >
  app.log 2>&1 &``) keeps running after its command returns, so a server
  started in one command answers ``curl`` in the next. What a background process prints after its command returned is
  discarded.
- **Working directory inside the run workspace.** ``working_dir`` is resolved
  against the workspace and a path that escapes it is rejected.
- **A per-command timeout**: 120 seconds by default. A command may set
  ``timeout`` up to 600 seconds for a slow build or a first install; no
  command runs past the run's remaining runtime. A command still
  running then is killed together with everything it started (its process
  group); the session and the processes earlier commands left running are
  untouched.
- **A stripped environment.** The child process gets an allowlist (``PATH``,
  ``HOME``, locale, temp dirs, a few Python/Node variables) with anything
  name-matching a secret removed, so provider API keys, OAuth secrets and SMTP
  credentials are not visible to a command or to a build backend it triggers.
- **A denylist** for the obvious catastrophes: ``sudo``, ``rm -rf /``,
  curl-pipe-shell, fork bombs, reads of ``~/.ssh`` and ``~/.aws``.
- **A bounded amount of output** in the model's context: 15,000 characters
  per command, half for each stream, with the full log
  still reachable (below). Each stream is also capped at 8 MB while it is
  captured: past that the command's process tree is killed, and the head and
  tail are kept with a marker for the dropped middle.

- **A bubblewrap sandbox on Linux, one per run.** A run's shell is a single
  sandbox, started at its first ``run_command``,
  with its own user, PID and mount namespaces; no other run's processes or
  files are visible in it. The container filesystem is visible read-only, other
  runs' workspaces, the telemetry folder and the incident log folder
  (``BESSER_INCIDENT_LOG_DIR``) are hidden, and only the run
  workspace and a per-run ``$HOME`` (``<run dir>.sandbox-home``) are writable.
  ``/usr/local`` and ``/root`` stay read-only, so an install
  (``pip install``, ``npm install``) goes into that per-run ``$HOME``
  rather than into the worker or into later runs. Where the worker has
  ``pasta`` (passt) and ``/dev/net/tun``, the session also gets a network
  namespace of its own (``BESSER_LLM_SHELL_NETWORK``): its ``localhost``
  belongs to that run, so any port is free, the worker's own API and other
  runs' servers cannot be reached, and outbound traffic (package installs)
  goes out through pasta, which forwards no port in either direction. Without
  them, in the default ``auto`` mode, runs on one worker share ``localhost``
  and a warning is logged; ``private`` refuses every command instead. The per-run ``$HOME`` sits
  outside the workspace, so it is never packaged or pushed; it is removed
  with the run, and by the 24-hour temp cleanup. If the sandbox cannot start, every command is
  refused; ``BESSER_LLM_SHELL_SANDBOX=off`` lifts that on a single-tenant
  Linux host whose kernel forbids unprivileged user namespaces. The model,
  and any validation finding, sees only "the shell sandbox is unavailable on
  this server"; the cause and the override are written to the server log
  only.
- **The session ends with the run.** Every process in it is killed when the
  run finishes, fails, is cancelled or times out, when its run folder is
  removed, and when the worker process exits or restarts (the session's
  control channel is a pipe to the worker, and bubblewrap's init takes the
  whole PID namespace with it). A session with no command for
  ``BESSER_LLM_SHELL_SESSION_IDLE_SECONDS`` (900; ``0`` never) is closed too,
  and the next command starts a fresh one; the environment
  survives that, since the worker keeps it, and the result's ``notes`` tell
  the model its background processes are gone. At most
  ``BESSER_LLM_SHELL_SESSION_MAX`` (10) sessions live per worker; past that a
  command runs in a one-off sandbox with the same confinement, whose
  processes stop when it returns (``0`` gives every command its own sandbox).
- **Checks do not run in the session.** The validators and probes that
  execute generated code (import checks, the startup and API probes,
  ``tsc``, ``cargo check``, the frontend build, the ``pip`` dry-run) each get
  a fresh sandbox of their own, so nothing a command left behind can steer a
  check. Before each Phase 3 validation sweep the session's processes are
  stopped as well, so a dev server or file watcher the agent left running
  cannot write to the workspace while it is checked; the environment
  carries over to the next command, whose notes say so.

What it does not:

- **No sandbox on Windows or macOS.** There is no namespace sandbox on those
  platforms, so commands run unconfined, as the backend user, and a warning
  is logged. The same applies on Linux with ``BESSER_LLM_SHELL_SANDBOX=off``.
  Each command is then its own process: the environment still
  carries over, but the worker does not track what a command left running, so
  ending the run does not stop it.
- **It does not cut the network for commands.** ``run_command`` keeps
  outbound network access so that installs work, as do the Phase 1
  ``npm install`` and the ``pip install --dry-run`` check, which are
  sandboxed the same way. Only the validators that execute generated code run
  without it.
- **The denylist stops mistakes**, not a determined prompt-injection payload.
  The sandbox is what contains a command, not the denylist.
- **It does not make generated code safe to execute.** Download and run the
  output with the care you would give any unreviewed code.

When shell tools are on and the workspace holds a generated FastAPI backend,
Phase 2's system prompt gains a *runtime verification* runbook, and a probe
script, ``.besser_probe.py``, is written to the workspace root. Its
subcommands (``up``, ``routes``, ``req``, ``log``, ``down``) boot the server
detached, list its routes and send requests, so the agent can run the app
rather than only read it. The runbook has the agent create every entity and
call every action through ``test_api`` scenarios, which are retained and
replayed as evidence; ``up``, ``log`` and ``req`` are for debugging a failure
a scenario reported. ``routes`` prints, per route, the JSON body it reads
as its top-level keys (``<- {"params": {...}}`` for a method route, whose
values go inside that key) and what a 2xx answer is (``-> list`` or
``-> object {...}`` with its keys): the response schema the app's OpenAPI
declares, else what the handler's ``return`` statements send, read from its
source as the ``api call:`` check reads it (see :doc:`validation`). The server ``up`` starts keeps running in the run's
shell session, so ``routes`` and ``req`` reach that same server; when the
session was restarted in between, they start their own, and the SQLite file
keeps records either way. A new server takes the port of the previous one
again while nothing else holds it, so a command or script written with that
port keeps working after a restart. ``req`` takes a JSON body as one argument
in single quotes (unquoted, the shell strips its double quotes), from a file
as ``@body.json``, or from standard input as ``-``, for instance a quoted
here-document; a body whose quotes the shell removed, an empty file or empty
standard input, or an option ``req`` does not have, is refused with a message
saying which. ``req`` sends
extra headers with ``-H "Name: value"`` (e.g. a bearer token from a login) and
prints only their names. It sends form fields and files as
``multipart/form-data`` with curl's syntax, ``-F "name=value"`` and
``-F "file=@path"`` (``;type=<mime>`` and ``;filename=<name>`` set a file's
type and name), for an upload endpoint; a field value keeps its own quotes. ``down`` stops a process only when its command line
identifies it as the probe's own server, since a recorded PID from a previous
session can name an unrelated process. Several ``run_command`` calls in one
turn run one after another, in order.
In a turn that also writes files, a command starts only after every write the
model emitted before it has finished, and a write emitted after it starts only
once it has returned; reads still run in parallel.
A line that joins two commands with ``;`` or
a new line, exits 0 and reports a failure in stderr gets a ``note`` that
``exit_code`` is the last command's alone.
``BESSER_LLM_SHELL_RUNBOOK=0`` leaves the runbook out.

The probe runs the app inside the workspace, so what the app writes on startup
or while answering a request (its database, a signing key it generates, an
upload) lands beside the source. The probe records the files that appear under
the backend during ``up`` and ``req`` in ``.besser_probe.runtime_files.json``,
and the run deletes them before it writes the recipe, unless a generator or a
write tool wrote them; the recipe lists them under ``runtime_files_removed``.
They are therefore in neither the download nor a GitHub push.

The worker also lists the backend's files before and after every
``run_command`` call, and records a file the call
left as written by the running app (a request sent with ``curl``, a test
client, a script importing the app) only when all of these hold: it is new
under the backend folder, outside dot-folders; it is a data file (``.json``,
``.pdf``, an image), not code, configuration or docs (``.py``, ``.ts``,
``.toml``, ``.txt``, ``.yaml``, ``.sql``, ``.html`` ...); its name holds a
generated id, 32 hex digits or a UUID, as an app names a stored upload; and the
command names neither its path nor its name, nor does another file the same
command created or changed (a script writing an image under a generated name
and the manifest naming it; UTF-8 text files up to 1 MB, at most 50, never a
database, log or other file the output leaves out by name, so the app's database
storing an upload's path does not keep the upload). Those
are removed by the same rules as the probe's record. A file the agent creates on
purpose through the shell is kept: one the command names (``cat >
data/seed.json``, ``cp a b``), code such as a migration ``alembic`` names with a
hex id, data a script writes under a plain name (``fixtures.json``), and a
generated name another file of that command names. A file of the app's data that
the agent's own script writes under a generated name that nothing names is still
taken for the app's and removed. The listing skips virtualenvs (a folder holding
``pyvenv.cfg``, under any name) and stops at 5,000 files, in which case nothing is
recorded for that command.

Only the name-based exclusions (databases, logs, caches) apply, and a generated
key or upload is delivered with the app, when the file is:

- written while the app was started some other way than the probe and not
  matching the rule above: a key under a plain name (``signing.key``), an
  upload kept under the name it was sent with;
- written by the running server between two shell commands, for example by a
  background task;
- written anywhere when the backend is the workspace root: the probe then
  records nothing and nothing is removed;
- in a dot-folder or outside the backend folder, an existing file the app
  rewrote, or a recorded file changed afterwards.

At the same point the run looks for files written this run whose content names
the run's temporary workspace folder (``besser_llm_...``), which is removed
after the run. Only a shell script (``.sh``, ``.ps1``, ``.bat`` or ``.cmd``)
at the workspace root that a write tool of this run wrote and no other file
names, such as a check script that ``cd``s into that folder, is deleted; the
recipe lists those under ``scratch_files_removed``. Every other such file (any
Python file, a script in a folder, a README, a test) is kept and listed under
``temporary_path_files``. ``BESSER_GENERATION.md`` names both. A workspace
folder the caller chose (a :doc:`Python library run <usage>` with its own
``output_dir``) is left alone, since a path to it stays valid.

Where no sandbox applies, treat "enable shell tools" as "I am willing to run
model-authored commands on this machine, as this user". That is acceptable on
a developer laptop or a single-tenant on-prem box, and not on a shared one.

Command output that overruns the cap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each stream is truncated before it reaches the model. For stderr the truncation
keeps the head and the tail, and a failing ``tsc`` or ``npm run build`` puts its
banner in the head and ``Found N errors.`` in the tail — the diagnostics
themselves are in the discarded middle, which is the part :doc:`Phase 3
<validation>` has to act on.

So when either stream is cut, the complete output is written to
``.besser_command_output/`` inside the run workspace and the tool result gains
``full_output_path`` and ``full_output_note``. The agent can ``search_in_files``
that path for the errors and ``read_file`` the surrounding lines. The directory
is run-internal: it is excluded from the download archive, from a GitHub push,
from the scaffold inventory and from the recipe's file manifest. Each spilled
stream is itself capped at 2,000,000 characters — far above anything a build log needs, and
only there so a runaway command cannot fill the disk.

.. note::
   Adding a tool takes three edits: its declaration in
   ``besser/spec_driven_agent/agent/tools.py``, an entry in
   ``_TOOL_MODEL_REQUIREMENTS`` in the same file (so the tool is only offered
   when the models it needs are present), and a handler in
   ``ToolExecutor._handlers`` (``agent/tool_executor.py``). A generator tool is
   the ``agent_tool`` of the generator's entry in
   ``besser/generators/registry.py`` (declaration and requirements in one), its
   id in ``AGENT_TOOL_ORDER``, plus the handler. See :doc:`../contributor_guide`.
