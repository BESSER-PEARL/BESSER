.. _besser-lab:

BESSER Lab
==========

.. note::
   **Experimental, off by default.** BESSER Lab is meant for a private
   deployment used by BESSER's own maintainers. Every lab route answers 404
   unless ``BESSER_LAB_ENABLED`` is on.

BESSER Lab is a deployment of the web editor whose chat changes BESSER's own
source code: the B-UML metamodel, the generators, the editor's backend, the web
editor and the modeling agent. The *lab agent* plans the change, makes it on
``lab/`` branches in a working copy on the server, runs the repositories' own
CI checks, reports what changed, and opens draft pull requests when the user
asks for them. Nothing leaves the server before that.

The lab agent is the Spec-Driven Agent's shape with BESSER as the target: a
plan, a tool loop, then checks with a bounded number of fix rounds. It does not
reuse the Spec-Driven Agent's Phase 2 loop, which needs a B-UML model and a
generated application to validate. It reuses its provider clients, its file
and shell tools, its context compaction and its shell sandbox. The engine lives
in ``besser/spec_driven_agent/lab/``; the editor backend's part lives in
``besser/utilities/web_modeling_editor/backend/services/lab/``.

The working copy
----------------

The lab works in one folder on the server, ``BESSER_LAB_WORKSPACE_ROOT``:

- ``besser/`` -- a checkout of BESSER (base branch ``development``), with the
  web editor submodule checked out at
  ``besser/utilities/web_modeling_editor/frontend`` (base branch ``develop``);
- ``modeling-agent/`` -- a checkout of the modeling agent (base branch
  ``develop``);
- ``.lab/`` -- unless ``BESSER_LAB_STATE_DIR`` names a folder outside the
  working copy, the lab's state: ``changes/<id>/`` holds the state of each
  change (``state.json``), its notes (``notes.md``: what was asked and what was
  done, which later messages of the same change read back) and the checks of
  its last run (``last_run.json``, kept across a restart). ``pending.json``
  holds a plan exchange that ended without a plan, and ``preview-logs/`` the
  preview's logs. On a server, keep it outside the working copy (see
  `Security`_).

``LabWorkspace`` (``lab/workspace.py``) runs every git command for the lab, as
an argument list with the scrubbed child environment, ``core.autocrlf=false``
and ``GIT_TERMINAL_PROMPT=0``. Git's error messages pass through the secret
redaction before they are shown.

A **change** is one unit of work, with one branch per repository and its
state. Its branch is ``lab/<user>/<slug>``: the GitHub login and the title,
both reduced to ``a-z``, ``0-9`` and ``-``, with ``-2``, ``-3``, ... added when
the name is taken locally, on origin, or by a branch the lab pushed earlier
(``refs/remotes/github/<branch>``, kept after a drop). The branch is created in every repository when the change
starts, from the commit each checkout is on; a repository the change never
modifies keeps a branch identical to its start and is left out of the report
and of the pull requests. The rules:

- At most one change is active (checked out) at a time.
- A change starts, or a parked one resumes, only when no working copy has
  uncommitted work outside a change, so unrelated edits never leak into it.
- **Park** commits the work in progress (``wip: <title>``) and checks out the
  base branches. **Resume** checks the branches out again.
- **Drop** discards the uncommitted work of an active change
  (``git reset --hard`` and ``git clean -ffd``, which also removes a
  repository made inside a checkout; ignored files such as ``node_modules``
  stay), checks out the base branches, deletes the lab branches and removes
  the change's state. It refuses, before discarding anything, when a
  repository holding the change's branch has another branch checked out.
- Park and drop delete the untracked files that only the change's own ignore
  rules hid, so they never show as uncommitted work on the base branch.
- ``commit_all``, and so park and the pull requests, refuses a folder that is
  a git repository of its own (made by ``git init`` or a scaffolder).
- The changed files of a repository are its committed and uncommitted changes
  against the merge-base with its base branch, new files and renames included.
  Editor files are reported under the editor, and BESSER's list never shows the
  submodule pointer. The tools' run records (``.besser_*``) and the shell
  sandbox's ``*.sandbox-home`` folders are never listed and never committed.
- Commits are authored as the user (``<login>`` and
  ``<login>@users.noreply.github.com``) through ``git -c``; the lab never
  writes git configuration.

How a message is handled
------------------------

A user message goes through at most three steps.

1. **Plan.** When there is no active change, or the message asks for a plan,
   the lab agent works with read-only tools. It reads the repositories'
   ``CLAUDE.md`` files and the code the change touches, then calls
   ``submit_plan`` with the kind of change, a title, a summary, the steps, the
   repositories it touches and the questions only the user can answer. A
   metamodel or diagram-type plan also carries a class diagram of the
   extension (``model_spec``), which the editor opens as a new class diagram
   for the user to review. The plan creates the change and is stored in its
   notes. A plan that touches no repository (the request is already met)
   creates no change and no branch. A reply to a plan that is not approved yet
   revises it, starting from the class diagram as the user corrected it in the
   editor. A plan step that ends without a change, with a question or with a
   plan that touches nothing, keeps the exchange in ``.lab/pending.json``, and
   the same user's next message continues it.
2. **Build.** When the user builds the plan (the plan card's Build button
   sends action ``build``), or sends a message to a change whose plan is
   already approved, the lab agent works in the working copy with the write
   and shell tools, following the guidance of the change's kind. It runs the
   checks itself with ``run_checks``, which cuts each check to the time the
   run has left. A stop ends a running check with every process it started,
   skips the tool calls the model had not started yet, and the stopped build
   reports the checks that ran. When its
   loop ends, the lab runs the checks of the changed files once more; while a
   check fails and fix rounds remain, the failures go back to the model in the
   same conversation. A fix round that ends without a tool call keeps the
   results as they are. The checks after the build loop are cut to the time
   the run has left; a check that cannot start is reported as "Not run: the
   run reached its N s limit".
3. **Report.** The changed files per repository, the check results, the
   checklist of the change's kind and a short summary of what changed.

``submit_plan`` (``lab/prompts.py``) refuses an incomplete plan with every
problem listed, and the model is asked again: a kind that is not one of the
kinds below, a blank title or summary, a title over 72 characters (it names the
branch and the pull request), no step, an unknown repository, or a metamodel or
diagram-type plan without ``model_spec``. A ``model_spec`` is checked like a
class diagram: names are Python identifiers and unique, attribute types are
B-UML primitives (``str``, ``int``, ``float``, ``bool``, ``date``,
``datetime``, ``time``, ``timedelta``, ``any``) or classes and enumerations of
the same diagram, association ends and generalizations name its classes,
multiplicities read ``1``, ``0..1``, ``0..*``, ``1..*`` or ``*`` with an upper
bound above 0, association names are unique, and no class generalizes itself,
directly or through a cycle. ``composition: true`` makes the target the whole and the
source the part, as in the editor.

Change kinds and checklists
---------------------------

The kind of a change selects the guidance the build step reads and the
checklist of the report. Guidance and rules come from the repositories' own
guides: BESSER's ``CLAUDE.md`` and ``docs/source/contributing/``, and the
``CLAUDE.md`` of the web editor and of the modeling agent (``lab/kinds.py``).

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Kind
     - Adds
     - Checklist items besides the common ones
   * - ``generator``
     - A new code generator
     - The package with ``__init__.py`` and a ``GeneratorInterface`` module;
       its ``GeneratorSpec`` in ``registry.py``; its id in the pinned lists
       and snapshot of the registry test; regeneration coverage (a builder or
       an entry in ``NOT_REGENERABLE``); tests under
       ``tests/generators/<name>/``; its docs page, its toctree entry and its
       row in the "Choosing a Generator" table; with an ``agent_tool``, the
       tool's handler and documentation.
   * - ``metamodel``
     - New or changed B-UML metaclasses
     - The metamodel changed; tests under ``tests/BUML/metamodel/``; docs
       under ``docs/source/buml_language``; the editor converters changed in
       both directions or in neither.
   * - ``diagram_type``
     - A new diagram type
     - The backend converters both ways and both project converters; the
       editor's diagram type, its elements in ``components.ts``,
       ``uml-elements.ts`` and ``popups.ts``, its palette preview
       (``*-preview.ts``) and its ``create-pane.tsx`` case, and
       ``shared/types/project.ts``; the modeling agent's handler, schema,
       ``SUPPORTED_DIAGRAM_TYPES`` and routing.
   * - ``editor_element``
     - A new element on an existing diagram
     - ``components.ts``, ``uml-elements.ts`` and ``popups.ts`` under
       ``packages/editor/src/main/packages/``, and its palette entry in its
       diagram package's ``*-preview.ts``.
   * - ``backend``, ``agent``, ``bugfix``, ``other``
     - An endpoint or service; a modeling agent change; a fix; anything else
     - None.

Every kind also gets three common items: tests changed in every repository
whose source changed; no file under ``.github/workflows/`` changed (the
sign-in token cannot push those); and every editor string used in a changed
file present in ``packages/i18n/en/``. Each item is done, missing, or *not
judged* when its repository is not in the working copy or a file it reads
does not parse; the report says which and why.

The tools
---------

The lab agent's tools are the Spec-Driven Agent's file and shell tools,
offered per step by ``LabToolExecutor`` (``lab/tooling.py``), plus three of
the lab's own. A call to a tool that is not offered gets an error result, not
an exception.

.. list-table::
   :header-rows: 1
   :widths: 20 50 15 15

   * - Group
     - Tools
     - Plan
     - Build
   * - Read
     - ``list_files``, ``read_file``, ``search_in_files``
     - yes
     - yes
   * - State
     - ``task_list``, ``check_syntax``
     - yes
     - yes
   * - Write
     - ``write_file``, ``modify_file``, ``replace_file_lines``, ``delete_file``
     - no
     - yes
   * - Shell
     - ``run_command`` (sandboxed, see `Security`_)
     - no
     - yes
   * - Lab
     - ``submit_plan`` -- ends the plan step with the plan
     - yes
     - no
   * - Lab
     - ``run_checks`` -- the CI checks of the files changed so far, failures
       first with the end of their output
     - no
     - yes
   * - Lab
     - ``scaffold_generator`` -- writes what a new deterministic generator
       needs, for a generator change
     - no
     - yes

``scaffold_generator`` (``lab/scaffolds/``) follows the steps of
:doc:`../contributing/create_generator`: a package with a
``GeneratorInterface`` subclass and one Jinja template that lists each class
with its attributes, the ``GeneratorSpec`` in ``registry.py``, a test, the
docs page with its toctree and table entries, the pinned lists and snapshot of
``tests/generators/test_generator_registry.py``, and a regeneration builder.
The generated generator runs and its tests pass as written; the model then
turns it into the real one. It refuses a name that exists or is not
lower-case snake case, and computes every edit before it writes, so a refusal
leaves the checkout unchanged.

The loop (``lab/loop.py``) makes one model call per turn and runs that turn's
tool calls: consecutive read calls in parallel, every other call alone and in
order. It stops on the turn cap (150), the cost cap and the runtime cap, all
checked against the whole message's spend, or when the user cancels. It has no
no-progress limit: reading for many turns is normal on a repository of
BESSER's size. A reply cut off at the output-token limit is discarded and the
model is asked for a smaller step, at most three times. The context is kept in
bounds with the Spec-Driven Agent's compaction, which keeps the user's first
message word for word. A tool result over 80,000 characters is cut, with a
note on how to read the rest. A rejected API key stops the run; any other
provider error ends the loop as ``api_error``.

Edits to existing files are refused when they break the file's syntax, as in
the Spec-Driven Agent. The workspace-wide checks the Spec-Driven Agent runs
after each write look for generated-application mismatches and are off in the
lab; ``run_checks`` covers what they would.

The checks
----------

``lab/checks.py`` runs the checks the repositories' CI workflows run
(``.github/workflows/ci.yml`` of BESSER and of the editor, ``tests.yml`` of the
modeling agent), narrowed to the changed files:

.. list-table::
   :header-rows: 1
   :widths: 18 37 45

   * - Repository
     - Check
     - When
   * - BESSER
     - ``pytest`` on the tests of the touched areas
     - A changed file maps to tests (below), or a test file changed.
   * - BESSER
     - ``ruff check`` on the changed files
     - A Python file under ``besser/`` changed.
   * - BESSER
     - ``lint-imports``
     - A Python file under ``besser/`` changed.
   * - BESSER
     - ``python docs/check_docs.py --offline``
     - Something under ``docs/`` changed, or a module the docs render with
       autodoc.
   * - BESSER
     - ``python .github/scripts/check_openapi.py``
     - A Python file under ``besser/`` changed (any module the app imports can
       define a schema component; CI runs it on every pull request), or the
       editor's ``besser-openapi.json``.
   * - Editor
     - ``npx vitest run`` on the tests of the touched areas
     - A webapp file maps to tests, or a test file changed.
   * - Editor
     - ``npm run lint``
     - A script file of the webapp or the server changed.
   * - Editor
     - ``npm run typecheck`` (fails above its own error baseline, as in CI)
     - A TypeScript file of the webapp or the editor package changed.
   * - Editor
     - ``npm test --workspace=server``
     - A file under ``packages/server/`` changed.
   * - Editor
     - ``node scripts/i18n-check.mjs``
     - A file under ``packages/i18n/`` changed.
   * - Editor
     - The OpenAPI types (``besser-openapi.d.ts``) match the committed schema,
       as CI's ``npm run api:types`` step
     - ``besser-openapi.json`` or ``besser-openapi.d.ts`` changed.
   * - Modeling agent
     - ``ruff check src patches modeling_agent.py``
     - A Python file changed.
   * - Modeling agent
     - ``lint-imports``
     - A Python file under ``src/`` changed.
   * - Modeling agent
     - ``pytest tests --ignore=tests/live``
     - Anything but ``docs/`` and Markdown changed (its tests are flat, so the
       whole suite runs, as in its CI).

How a changed file maps to tests:

- **BESSER** -- the test files directly in the mirror of the file's Python
  package under ``tests/``, with the generators' renames (``java_classes`` to
  ``java``, ``pydantic_classes`` to ``pydantic``, ``python_classes`` to
  ``python``, ``sql_alchemy`` to ``sqlalchemy``, ``json`` to
  ``json_object`` and ``json_schema``, ``nn/forecaster`` to ``forecaster``).
  A package without such tests falls back to its nearest ancestor that has
  some, but never to a whole top-level area such as
  ``tests/spec_driven_agent``. A module that maps to nothing that way gets the
  test files of its top-level area that import it (``tests/spec_driven_agent``
  is flat). A module the change adds maps only to ``test_<name>.py`` in its
  package's mirror and to the tests of its area that import it. A change to
  ``besser/generators/registry.py`` also runs the registry, regeneration and
  generator-tool tests.
- **Editor** -- a file under ``src/main/features/<x>/`` or
  ``src/main/shared/<y>/`` maps to every test under that folder; any other
  webapp file to the tests next to it and in its ``__tests__`` folder.
- A changed file that maps to no tests gets a skipped "No tests found for ..."
  result rather than nothing. A file of the editor package
  (``packages/editor/src``) gets a skipped result saying that the package has no
  test runner and that its tests go in ``packages/webapp``.

The Python checks of a repository run with the interpreter, ``ruff`` and
``lint-imports`` of its checkout's ``venv/`` when it has one, so they import the
working copy with its own dependencies; otherwise with the lab's own Python
(or, for the two tools, whatever ``PATH`` finds).

A check whose tool is not installed (exit code 127, or a missing ``pytest``,
``sphinx``, ``sphinx_immaterial``, ``ruff`` or ``importlinter`` module) is reported as *skipped* with
the reason and logged as a warning; it does not start a fix round. Commands
run with the scrubbed child environment and their own timeout (30 minutes for
tests); a timeout kills the whole process tree. ``checks_for(..., full=True)``
gives each touched repository's whole CI suite instead.

Pull requests
-------------

``services/lab/pull_requests.py`` opens the pull requests of the active change
with the signed-in user's GitHub token:

1. It refuses, before anything is committed or pushed, a change with no
   modifications, a branch that does not start with ``lab/``, a change to a
   file under ``.github/workflows/`` (a file moved out of it counts too), and
   an added or modified ``.env*`` file holding a credential (the detection of
   the GitHub push boundary, ``scrub_secret_files``). It also refuses a change
   other than the one the editor's card shows (``changeId``, see `HTTP API`_).
2. It commits the work in progress, with the pull request's title as the
   message.
3. It pushes each branch to ``https://github.com/<owner>/<name>.git``, the
   editor first. When the editor changed and BESSER has changes of its own,
   BESSER's branch first gets one commit that moves the submodule pointer to
   the pushed editor commit. The token reaches git only through the
   environment (``GIT_CONFIG_COUNT`` with an ``Authorization`` header for
   github.com and an empty ``credential.helper``), never through the command
   line, a remote URL or a git config file. Every command that carries the
   token (``ls-remote``, ``fetch``, ``push``) runs in a scratch bare
   repository that has no config of its own and borrows the working copy's
   objects. Commits added to the branch on GitHub are merged first; a
   conflict, or a same-named branch that shares no commit with the change, is
   refused with a message. The push is never forced.
4. It opens a **draft** pull request against each base branch, titled
   ``feat: <title>`` (``fix:`` for a bug fix) unless the user gave a title.
   When several were opened, each description lists the others in merge
   order, the editor first.

Every error message is passed through the secret redaction with the token
before it is shown. A failure the user can act on is told in plain words, with
git's or GitHub's own message only in the server log: a refused sign-in (the
route answers 401, and the editor offers to sign in again), no write access, a
branch protection rule, or no connection to GitHub. A push failure opens no pull request; a GitHub failure
part way names the pull requests already opened. Asking again returns the
pull request already open for a branch; its description is kept; only the
related pull requests section at its end is replaced.

The preview
-----------

With ``BESSER_LAB_PREVIEW_ENABLED``, ``services/lab/preview.py`` runs the
working copy as a live preview, so a change can be tried before its pull
request: the backend (``python -m
besser.utilities.web_modeling_editor.backend.backend`` on port 9100), the
editor's Vite dev server (``npx vite`` on port 8180) and the modeling agent
(``modeling_agent.py``, on the port of its own ``config.yaml``, 8865 by
convention), each bound to ``127.0.0.1``. Each child runs in its own process
group, writes to its own log file, and is restarted when one of its watched
``.py`` files changes, even after it exited, so fixing the error that stopped
it brings it back; the editor reloads by itself.
``stop()`` sends ``SIGTERM`` to the group and ``SIGKILL`` ten seconds later.
The logs are in ``preview-logs/`` of the lab's state folder
(``BESSER_LAB_STATE_DIR``, by default ``.lab/`` of the workspace root).

The backend and the modeling agent run with the Python of their checkout's
``venv/`` (as each repository's ``CLAUDE.md`` sets it up); without one they run
with the lab's own interpreter and a warning is logged. They get
``PYTHONPATH`` pointing at the working copy, so the working copy's code runs
rather than an installed copy. The backend also gets ``CORS_ORIGINS`` for the
editor's origin and ``BESSER_LAB_ENABLED=false``, so it is a plain editor
backend, not a second lab. The preview's public address is ``DEPLOYMENT_URL``:
the editor gets ``VITE_DEV_BACKEND_URL`` (``<DEPLOYMENT_URL>/besser_api``,
which a dev build of the editor uses as its backend), ``UML_BOT_WS_URL``
(``wss://<host>/agent``) and the host name Vite's dev server has to accept.
Without ``DEPLOYMENT_URL`` each service is addressed on its own localhost
port.

The preview runs separately from a lab instance's own backend, so a proxy in
front of both decides which of them a route reaches. A route sent to the lab
backend (the lab routes, GitHub sign-in, and whatever else needs its sessions)
runs the lab's code: a change to it cannot be tried in the preview.

Configuration
-------------

The lab's variables are fields of ``BackendSettings``
(``besser/utilities/web_modeling_editor/backend/settings.py``), read once per
process. As for every backend setting, a flag is on only for ``1``, ``true``,
``yes`` or ``on``, and a number that does not parse or is not positive falls
back to its default.

.. list-table::
   :header-rows: 1
   :widths: 32 14 54

   * - Variable
     - Default
     - Meaning
   * - ``BESSER_LAB_ENABLED``
     - off
     - This instance is a BESSER Lab: the lab routes answer and GitHub sign-in
       is limited to ``BESSER_LAB_ALLOWED_LOGINS``. Off, every lab route
       answers 404 and sign-in is unchanged.
   * - ``BESSER_LAB_ALLOWED_LOGINS``
     - empty
     - Comma-separated GitHub logins allowed to use the lab, ignoring case.
       Empty allows no one.
   * - ``BESSER_LAB_WORKSPACE_ROOT``
     - unset
     - Absolute path of the folder holding the working copies (see `The
       working copy`_).
   * - ``BESSER_LAB_STATE_DIR``
     - unset
     - Absolute path of the lab's state: the changes' state, notes and last
       checks, the pending plan exchange and the preview's logs. Unset, it is
       ``.lab/`` of ``BESSER_LAB_WORKSPACE_ROOT``, which the lab agent's shell
       commands can write; on a server, set it to a folder outside the working
       copy (see `Security`_).
   * - ``BESSER_LAB_LLM_PROVIDER``
     - ``anthropic``
     - LLM provider of the lab agent. A blank value gives the default.
   * - ``BESSER_LAB_LLM_MODEL``
     - unset
     - Model of the lab agent; unset or blank, the provider's default.
   * - ``BESSER_LAB_LLM_API_KEY``
     - unset
     - The server's API key for that provider. Users never send a key of
       their own.
   * - ``BESSER_LAB_MAX_COST_USD``
     - ``5.0``
     - Cost cap of one lab run, in US dollars.
   * - ``BESSER_LAB_MAX_RUNTIME_SECONDS``
     - ``2400``
     - Runtime cap of one lab run, in seconds.
   * - ``BESSER_LAB_PREVIEW_ENABLED``
     - off
     - Run the working copy's backend, editor and modeling agent as a live
       preview (see `The preview`_).

The GitHub OAuth app is configured as for any instance (``GITHUB_CLIENT_ID``,
``GITHUB_CLIENT_SECRET``, ``GITHUB_REDIRECT_URI``, ``DEPLOYMENT_URL``, see
:ref:`the backend's environment variables <web-editor-backend-env>`).

HTTP API
--------

All lab routes are under ``/besser_api`` and need the ``X-GitHub-Session``
header of a signed-in user on the list (``LabUserDep`` in
``routers/auth.py``). They are mounted on every instance, so ``/openapi.json``
is the same whether ``BESSER_LAB_ENABLED`` is set or not; with the lab off the
gate answers 404 before the request body is validated (a body that is not JSON
at all still gets the usual 422). They answer, in this order of checks:

- **404** when ``BESSER_LAB_ENABLED`` is off;
- **401** when the session is missing or expired (the deprecated
  ``session_id`` query parameter is not accepted);
- **403** when the login is not in ``BESSER_LAB_ALLOWED_LOGINS``. The list is
  read when the backend starts, so removing a login takes effect only after a
  restart, which also cuts off a message in progress.

On a lab instance the OAuth callback also refuses a login that is not on the
list: it stores no session, revokes the GitHub token, logs a warning with the
login and redirects to ``DEPLOYMENT_URL`` with ``?error=not_allowed``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Route
     - What it does
   * - ``POST /lab/message``
     - Body ``{"message": str, "action": "auto" | "plan" | "build", "project":
       <ProjectInput> | null}``. Runs one message (see `How a message is
       handled`_) and answers with a Server-Sent Events stream. ``auto`` plans
       when there is no active change or its plan is not approved yet (a reply
       revises the plan), and builds otherwise; ``plan`` forces a plan,
       ``build`` builds and approves the plan. The editor sends ``plan`` for a
       reply typed after a plan card, ``build`` from the card's "Build this
       plan" button, and ``auto`` otherwise. ``project`` holds only the class
       diagram the lab proposed for the change being continued (pending, or
       kept by the user), never the user's other diagrams, and is null when
       there is none. A plan revision starts from that diagram and the build
       step follows it: the one the editor
       marks as the change's (``config.labProposalChangeId``, or
       ``labChangeId`` once kept), not the active class diagram. One run at a
       time per instance (409 otherwise).
   * - ``GET /lab/state``
     - ``{"enabled": true, "user": <login>, "change": <the lab object of done,
       without plan, modelProposal and finalText> | null, "parked":
       [{"changeId", "title", "kind", "createdAt"}], "preview": {"enabled":
       bool, "services": [{"name", "running", "detail"}]}, "run": {"runId",
       "startedAt"} | null}``. ``run`` is the run in progress, which
       ``/spec-driven/runs/{runId}/events`` follows and
       ``/spec-driven/cancel/{runId}`` stops. The change's checks are those of
       its last run, kept in ``.lab/changes/<id>/last_run.json``, so they
       survive a restart.
   * - ``POST /lab/pull-request``
     - Body ``{"changeId": str | null, "title": str | null, "body": str |
       null}``. Opens the draft pull requests (see `Pull requests`_) and
       returns ``{"pullRequests": [{"repo", "number", "url", "branch",
       "base"}], "checksNote": str}``. 409 when ``changeId`` is not the active
       change, before anything is pushed; 401 when GitHub refuses the user's
       sign-in. The lab runs only the checks of the
       touched areas before pushing, never the full suites; ``checksNote`` says
       so with the last build's counts.
   * - ``POST /lab/change/park``
     - Parks the active change; returns the state.
   * - ``POST /lab/change/resume``
     - Body ``{"changeId": str}``. Resumes a parked change; returns the state.
   * - ``POST /lab/change/drop``
     - Drops the active change; returns the state.
   * - ``POST /lab/preview/reset``
     - Checks out the base branches when no change is active (409 otherwise);
       returns the state. It parks nothing.

The stream of ``/lab/message`` uses the Spec-Driven Agent's event types
(``start``, ``phase``, ``tool_call``, ``text``, ``cost``, ``error``, ``done``;
see :doc:`api`), so the editor's run card shows a lab run unchanged. ``start``
carries the 32-hex ``runId``, which ``POST /spec-driven/cancel/{run_id}``
stops. On a lab instance the run routes (``GET /spec-driven/runs/{run_id}``,
its ``/events`` and ``/events.json``, and ``POST /spec-driven/cancel/{run_id}``)
need the ``X-GitHub-Session`` of an allowed user, as the lab routes do. A
stopped run ends with ``done`` like any other, with ``cancelled: true``,
``lab.stopReason: "cancelled"`` and the checks that ran. The phases are
``plan``, ``build`` and ``check``. ``done`` has
``mode: "modify"``, empty ``downloadUrl`` and ``fileName``, ``changedFiles``
(each path prefixed with its repository) and one extra field, ``lab``:

.. code-block:: json

   {
     "step": "build",
     "changeId": "3f9c2a1b7d4e",
     "title": "Add a note element",
     "kind": "editor_element",
     "plan": null,
     "modelProposal": null,
     "repoChanges": [{"repo": "frontend", "branch": "lab/jordi/add-a-note-element", "base": "develop",
                      "files": [{"path": "packages/editor/src/main/packages/components.ts",
                                 "status": "modified"}]}],
     "checks": [{"id": "frontend:tests", "repo": "frontend", "title": "Tests of the touched areas",
                 "passed": true, "skipped": false, "summary": "12 passed", "durationSeconds": 12.4}],
     "checklist": [{"id": "editor_element.registries", "text": "...", "done": true}],
     "summary": "What changed, per repository.",
     "stopReason": "completed",
     "finalText": "The model's own closing account of the build."
   }

A plan step carries ``plan`` (``kind``, ``title``, ``summary``, ``steps``,
``repos``, ``questions``) and, for a metamodel or diagram-type change,
``modelProposal`` (``title``, ``diagramType: "ClassDiagram"`` and the editor's
class diagram JSON). A checklist item's ``done`` is ``null`` when it could not
be judged. ``finalText`` is the model's own closing account of a build (empty
for a plan), which ``summary`` also ends with after its deterministic lines.

``GET /spec-driven/config`` reports ``"lab": {"enabled": bool}``, false on
every instance that is not a lab; the editor shows the lab's chat only when it
is true.

Security
--------

A lab instance runs model-written code on the server with the server's LLM key
and the users' GitHub tokens nearby. Run it only for trusted users, on a host
of its own.

- **Who can use it.** Only the logins in ``BESSER_LAB_ALLOWED_LOGINS``, checked
  at sign-in and on every request against the list the backend read when it
  started. An empty list allows no one.
- **The LLM key** is the server's (``BESSER_LAB_LLM_API_KEY``, a
  ``SecretStr``). Child processes get ``safe_subprocess_env()``, which drops
  every variable whose name contains ``KEY``, ``SECRET``, ``TOKEN``,
  ``PASSWORD``, ``CREDENTIAL``, ``PRIVATE``, ``API_``, ``AUTH``, ``CERT`` or
  ``SESSION``, so the key and the GitHub client secret reach neither the
  shell, the checks nor the preview.
- **The GitHub token** is used only to push and open pull requests, given to
  git through the environment of that one command and redacted from every
  error message. The sign-in scopes (``repo``, ``gist``, ``user``) do not
  include ``workflow``, so the lab refuses changes to ``.github/workflows/``
  before it pushes.
- **The shell.** ``run_command`` runs in the Spec-Driven Agent's sandbox: on
  Linux in bubblewrap, with the workspace's top-level directory (``/srv`` for
  a workspace under ``/srv``) hidden except the workspace itself and the rest
  of the host visible read-only; on Windows unconfined, with a warning. Keep
  the GitHub session store and the lab's state (``BESSER_LAB_STATE_DIR``)
  under that top-level directory, outside the workspace, so a command can
  change neither; and the environment file readable only by root. With the
  state in ``.lab/`` of the workspace, a command can rewrite a change's notes,
  which later prompts read (the session writes the active change's
  ``state.json`` back after every build, but not the notes). A lab instance
  answers 404 to a new Spec-Driven run (``/spec-driven/generate`` and
  ``/spec-driven/resume``), so no other sandboxed workspace runs beside the
  lab's.
- **What runs outside the sandbox.** The checks (tests the model may have
  written) and the preview run as child processes of the backend, as the
  service user, without bubblewrap. Model-written code can therefore read
  what that user can read. The file tools refuse paths with a ``.git``
  component, but a shell command can still write each checkout's ``.git``
  folder. The lab's own git calls run with ``core.hooksPath`` set to
  ``/dev/null`` and ``core.fsmonitor`` off, and the token never meets the
  working copy's git config (it is used only in a scratch repository, see
  `Pull requests`_). Other config-driven commands (``filter``, ``textconv``,
  ``gpg``) that a command plants there stay possible when the lab runs
  ``git diff``, ``add`` or ``commit``, until the sandbox also keeps ``.git``
  read-only. Treat a lab host as one whose working copies the model controls.
- **The preview editor.** When users sign in and chat in the preview editor,
  editor code the model writes is hot-reloaded into their open tabs before
  anyone reviews it, and runs on the origin that holds their GitHub session.
  Only an editor that serves no working-copy code on that origin avoids this.
- **Pushes.** Only ``lab/`` branches are pushed, never forced, and only as
  draft pull requests; a change holding a credential in a ``.env*`` file is
  refused.
