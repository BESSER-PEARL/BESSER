.. _besser-lab:

BESSER Lab
==========

.. note::
   **Experimental, off by default.** BESSER Lab is a private instance for
   BESSER's own maintainers. Every lab route answers 404 unless
   ``BESSER_LAB_ENABLED`` is on, and a public editor build contains none of
   the lab's screens.

BESSER Lab is an instance of the web editor in which the BESSER team develops
BESSER itself. A team member describes a change, or starts from a GitHub issue;
the *lab agent* plans it, the person corrects the plan, the agent builds it on
``lab/`` branches of BESSER, the web editor and the modeling agent, runs those
repositories' own CI checks, and the person tries the change live and opens
draft pull requests. The lab then follows the pull requests' CI and review
comments until they are merged. Several people work at once, each change in
its own working trees with its own conversation.

Only the logins listed in ``BESSER_LAB_ALLOWED_LOGINS`` can sign in. The lab
runs on a host of its own (``scripts/lab/README.md`` sets one up) or, for one
person, in a local container (:doc:`lab_local`). How to use it is in the web
editor's user guide ("BESSER Lab").

.. toctree::
   :maxdepth: 1

   lab_local

The lab agent is the Spec-Driven Agent's shape with BESSER as the target: a
plan, a tool loop, then checks with a bounded number of fix rounds. It reuses
the Spec-Driven Agent's provider clients, file and shell tools, context
compaction and shell sandbox, but not its Phase 2 loop, which needs a B-UML
model and a generated application to validate. The engine is in
``besser/spec_driven_agent/lab/``; the editor backend's part is in
``besser/utilities/web_modeling_editor/backend/services/lab/`` (module list in
:doc:`../web_editor_backend`).

On disk
-------

Three folders, each a setting, plus the stores of the backend:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Folder
     - Holds
   * - ``BESSER_LAB_WORKSPACE_ROOT``
     - ``base/besser/`` and ``base/modeling-agent/``: the *base clones*, on
       their base branches and never edited; the editor's base clone sits at
       the submodule path ``besser/utilities/web_modeling_editor/frontend``
       inside ``base/besser``. ``changes/<id>/besser/`` and
       ``changes/<id>/modeling-agent/``: one git worktree per repository for
       each open change, the editor's again at the submodule path.
       ``changes/<id>.sandbox-home/``: the ``$HOME`` of the change's shell,
       checks and installs. ``.cache/uv/`` and
       ``.cache/node/<package-lock sha256>/``: the stores shared by every
       change.
   * - ``BESSER_LAB_GIT_ROOT`` (default ``<workspace root>-git``)
     - Every git directory: ``besser/``,
       ``besser/modules/besser/utilities/web_modeling_editor/frontend/`` (the
       editor), ``modeling-agent/``, and each worktree's admin directory under
       its repository's ``worktrees/``.
   * - ``BESSER_LAB_STATE_DIR`` (default ``<workspace root>-state``)
     - ``changes/<id>/``: ``state.json`` (the change), ``notes.md`` (what was
       asked, planned, built and reported, which later steps read back),
       ``thread.jsonl`` (the conversation), ``runs.jsonl`` (one record per
       run: who, which step, model, cost, turns, time, how it stopped),
       ``last_run.json``, ``plan.json`` and ``note_proposals.json``.
       ``audit/<YYYY-MM>.jsonl`` (the audit log), ``memory/notes.json`` (the
       team notes), ``maps/<repo>-<sha12>.md`` (cached repository maps) and
       ``preview-logs/slot-<i>/<service>.log``.

The lab runs every git command as an argument list, with the scrubbed child
environment and explicit ``GIT_DIR`` and ``GIT_WORK_TREE``. It finds a
worktree's git directory from the git root (the ``worktrees/*/gitdir`` entry
that names the tree), never from the ``.git`` file inside the tree, which
model-written code could rewrite. Its git calls set ``core.hooksPath`` to
``/dev/null`` and turn ``core.fsmonitor`` off. Writes to a git directory's
shared refs (worktrees, branches, fetches) take that repository's lock. At
start ``LabWorkspace.ensure_ready`` checks that every base clone exists, is on
its base branch and keeps its git directory outside the workspace root; a
layout that fails it (an older lab's, for one) names
``scripts/lab/setup_workspace.sh``, which lays the workspace out.

**Environments.** A change's venvs and ``node_modules`` are made when a check
or the preview first needs them, from the shared stores, and stamped with a
hash of the files they were made from, so they are remade only when those
change (a change that edits ``package-lock.json`` gets its own ``npm ci``).
How the files are shared is the *link mode*, probed once at start and shown on
the home and settings pages:

- ``reflink``: copy-on-write clones (XFS with reflink, Btrfs). A change can
  write into its ``node_modules`` without reaching any other.
- ``hardlink`` (ext4 and the like): hard links, with the shared
  ``node_modules`` snapshot made read-only and a warning logged, since an
  in-place write into a linked file would reach every change.

BESSER's environments use Python 3.12, the modeling agent's 3.11; the
interpreters live outside the workspace root, which every sandbox hides.

**Base refresh.** Every ``BESSER_LAB_BASE_REFRESH_MINUTES`` (and on
:guilabel:`Refresh now` in the settings, and whenever a change opens) the lab
fetches each base branch from its upstream, fast-forwards the base clone and
keeps the result as ``refs/remotes/upstream/<branch>``. Open changes never
move by themselves: each stays on the base commit it started from (its
``bases``), and the change shows how many commits its base has gained since.

Changes
-------

A **change** is one unit of work: one branch per repository it touches, its
working trees, its conversation and its records.

- **Draft.** ``POST /lab/change`` (the home's composer, or
  :guilabel:`New change`) makes a draft owned by the user: state and a thread,
  no branch. Its plan step reads the base clones, in a sandbox that cannot
  write them.
- **Open.** The draft's first plan that touches a repository gives it a branch
  ``lab/<owner>/<slug>`` (the owner's login and the plan's title, reduced to
  ``a-z``, ``0-9`` and ``-``, with ``-2``, ``-3``, ... when the name is taken
  locally, upstream or by a branch the lab pushed earlier) and a worktree in
  every repository, each from its base's tip, after a base refresh. A plan
  that touches nothing keeps the draft a draft. Later plans revise the change
  and keep its branches.
- **Closed.** Once its pull requests are merged, or closed unmerged, a change
  can be closed: its trees and branches are removed, its state folder and
  thread stay, and the list of changes shows it under "Recently merged" (the
  newest 10). It is refused while work is not on GitHub
  (``LabWorkspace.unpushed_repos``: uncommitted work, a merge in progress or
  commits the last push does not hold). The editor closes a change by itself
  when every pull request is merged and nothing is unpushed; the server writes
  "the lab closed it" into the thread, so every viewer sees it.
- **Dropped.** Dropping deletes everything: trees, branches, state and thread.
  Pull requests already opened stay on GitHub; the change's cost stays in the
  audit log.

Who may do what:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Action
     - Who
   * - See any change, send it messages, open and push its pull requests,
       update it from its base, close it, try it, keep or dismiss a proposed
       team note
     - Every allowed login
   * - Drop a change
     - Its owner (who started it) or an admin (``BESSER_LAB_ADMIN_LOGINS``)
   * - Stop a run (``POST /spec-driven/cancel/{run_id}``)
     - Who started the run, the change's owner, or an admin; anyone else gets
       403
   * - Refresh the bases, read the settings and the audit log
     - Every allowed login

Commits are authored as the login whose request made them
(``<login>@users.noreply.github.com``, set per command through ``git -c``);
each build is committed with the change's title as its message.

**One thread per change.** Every message, run, pull-request event and notice
of a change is appended to its ``thread.jsonl``: a message with its login,
action and the class diagram it carried (one over 512 KB is not kept, and a
notice says so); a run as ``running`` and then how it ended, with its result
(``LabDone``). The editor rebuilds the conversation from it, so a reload,
another browser or a teammate sees the same thread. A run's cost comes from
the change's run records. The run card a teammate rebuilds from the thread
offers only the proposed team notes still pending: one already kept or
dismissed is left out.

**Several people at once.** A message goes to the change named in the request
(the one in the editor's URL); the lab never picks a change, and a new change
starts only from the home or :guilabel:`New change`. Each change has a lock: a
run holds it from start to end, an action (drop, close, update, pull requests)
while it works, and anyone else is refused at once with who holds it
(``change_busy``); the lab never queues. Across the lab, at most
``BESSER_LAB_MAX_CONCURRENT_RUNS`` runs are in progress (``runs_full``, naming
each run in progress). A run that loses its stream goes on until no one has
followed it for ``BESSER_LLM_DISCONNECTED_GRACE_SECONDS``;
``/spec-driven/runs/{run_id}/events`` replays it, so a teammate who opens the
change follows it live.

**Caps.** Each run reserves its cost cap when it starts:
``min(BESSER_LAB_MAX_COST_USD, what the user has left this UTC month, what the
lab has left)``. Reservations count against both monthly caps
(``BESSER_LAB_USER_MONTHLY_CAP_USD``, ``BESSER_LAB_MONTHLY_CAP_USD``) until
the run ends and settles at what it spent, so runs at once cannot overshoot a
cap. Under 0.25 USD left, a run, and a new change, is refused (``user_cap``
or ``instance_cap``, naming the date the month ends). The header's meter
shows what the user's runs cost this month.

**A user's own key.** The header's key chip opens the editor's key dialog
(provider, key, model). A saved key stays in that browser tab
(``sessionStorage``) and is sent in the body of each message of that user
(``llm`` of ``POST /lab/message``): the run then uses that provider, key and
model for both steps, is paid by its owner, reserves nothing and is not held
back by the monthly caps. Its cost is still recorded on the change and in
the audit log (``ownKey``), where it does not count towards the caps; the
per-run cap still applies. The key is passed to the LLM client only: it is
in no thread, record or log line, and it is redacted from the stream. A
custom endpoint (the dialog's "Local") is refused unless
``BESSER_LLM_ALLOW_CUSTOM_BASE_URL`` is set, as for the Spec-Driven Agent.
Without ``BESSER_LAB_LLM_API_KEY`` the lab has no key of its own
(``labKey`` false on ``GET /lab/home``): every run needs the user's, and a
message without one is refused (``state``).

**Audit log.** ``audit/<YYYY-MM>.jsonl``, one append-only file per UTC month,
records who did what to which change: ``change_created``, ``change_opened``,
``run`` (step, cost, model, stop reason, time), ``run_refused``,
``pull_requests_opened``, ``base_updated``, ``bases_refreshed``,
``preview_slot_taken``, ``change_closed`` and ``change_dropped``. It outlives
dropped changes; its ``run`` lines are what the monthly caps count. The
settings page shows one month of it, with what each login spent.

**Refusals.** Every lab refusal answers ``{"detail": {"code", "message"}}``
with the message written for the user:

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Code
     - Status
     - When
   * - ``change_busy``
     - 409
     - A run or another action holds the change.
   * - ``runs_full``
     - 429
     - ``BESSER_LAB_MAX_CONCURRENT_RUNS`` runs are in progress.
   * - ``user_cap``, ``instance_cap``
     - 409
     - The user's, or the lab's, monthly budget is spent.
   * - ``not_owner``
     - 403
     - A drop by someone who is neither the owner nor an admin.
   * - ``preview_full``
     - 409
     - Every preview slot shows another change that was used in the last 10
       minutes; the message names each slot's change and user.
   * - ``not_found``
     - 404
     - No such change, slot, note or proposal.
   * - ``state``
     - 409
     - What the change's state refuses: a closed change, a draft's diff or
       pull request, the preview off, a close before the work is on GitHub,
       a failed git or GitHub step.

How a message is handled
------------------------

``POST /lab/message`` runs one step for the change it names
(``LabSession.handle``).

1. **Plan.** For a draft, a change whose plan is not approved yet (a reply to
   a plan revises it), or ``action: "plan"``. The lab agent works with
   read-only tools, starting from an orientation (see `Knowledge`_), and calls
   ``submit_plan`` with the kind of change, a title, a summary, the steps, the
   repositories it touches and the questions only the user can answer. A
   metamodel or diagram-type plan also carries a class diagram of the
   extension (``model_spec``), which the editor shows in the change's Model
   tab for the user to correct; the next plan revision and the build start
   from the diagram as the user left it. The issues of the request (the one
   picked on the home and those the message names, at most three) are read
   from GitHub before the loop and recorded on the change. The plan step may
   use its own model (``BESSER_LAB_PLAN_LLM_MODEL``). A plan step that ends
   with a question, or with the answer that nothing needs to change, records
   the exchange in the notes, so the next message continues it.
2. **Build.** When the user builds the plan (the plan card's
   :guilabel:`Build this plan` sends ``action: "build"``), when they send a
   message to a change whose plan is approved, or for a review task (see
   `Pull requests and the review loop`_). The lab agent works in the change's
   trees with the write and shell tools, following the guidance of the
   change's kind, runs the checks itself with ``run_checks`` and reports what
   it did with ``submit_build_report`` (a summary, departures from the plan,
   what it did not do). When its loop ends, the lab runs the checks of the
   changed files once more, after making each repository's environment; while
   a check fails and fix rounds remain (three), the failures go back to the
   model in the same conversation. A stop ends a running check with every
   process it started; a stopped build reports the checks that ran.
3. **Result.** The step's result is the ``lab`` field of the stream's ``done``
   event (``LabDone``): the changed files per repository, the checks, the
   checklist of the change's kind, a summary, the agent's report, the plan and
   proposed model of a plan step, the team notes the build proposes, and the
   change's pull requests, issues and cost so far.

``submit_plan`` (``lab/prompts.py``) refuses an incomplete plan with every
problem listed, and the model is asked again: a kind that is not one of the
kinds below, a blank title or summary, a title over 72 characters (it names
the branch and the pull request), no step, an unknown repository, or a
metamodel or diagram-type plan without ``model_spec``. A ``model_spec`` is
checked like a class diagram: names are Python identifiers and unique,
attribute types are B-UML primitives (``str``, ``int``, ``float``, ``bool``,
``date``, ``datetime``, ``time``, ``timedelta``, ``any``) or classes and
enumerations of the same diagram, association ends and generalizations name
its classes, multiplicities read ``1``, ``0..1``, ``0..*``, ``1..*`` or ``*``
with an upper bound above 0, association names are unique, and no class
generalizes itself, directly or through a cycle. ``composition: true`` makes
the target the whole and the source the part, as in the editor.

Change kinds and checklists
---------------------------

The kind of a change selects the guidance the build step reads and the
checklist of its result. Guidance and rules come from the repositories' own
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
judged* when its repository is not in the working trees or a file it reads
does not parse; the result says which and why.

The tools
---------

The lab agent's tools are the Spec-Driven Agent's file and shell tools,
offered per step by ``LabToolExecutor`` (``lab/tooling.py``), plus the lab's
own. A call to a tool that is not offered gets an error result, not an
exception; the file tools refuse a path with a ``.git`` component; a result
over 80,000 characters is cut, with a note on how to read the rest.

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
     - ``run_command`` (sandboxed, see `Confinement`_)
     - no
     - yes
   * - Knowledge
     - ``github_issue``, ``github_pull_request``, ``github_search`` (when
       GitHub can be read in the run), ``git_history``, ``read_lab_note``
       (when the team memory is on)
     - yes
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
     - ``submit_build_report`` -- the build's account for the user and the
       pull request: summary, departures from the plan, what was not done
     - no
     - yes
   * - Lab
     - ``propose_lab_note`` -- a team note for a person to approve (at most
       two per run)
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
It refuses a name that exists or is not lower-case snake case, and computes
every edit before it writes, so a refusal leaves the tree unchanged.

The loop (``lab/loop.py``) makes one model call per turn and runs that turn's
tool calls: consecutive read calls in parallel, every other call alone and in
order. It stops on the turn cap (150), the cost cap and the runtime cap, all
checked against the whole message's spend, or when the user stops the run. It
has no no-progress limit: reading for many turns is normal on a repository of
BESSER's size. A reply cut off at the output-token limit is discarded and the
model is asked for a smaller step, at most three times. The context is kept in
bounds with the Spec-Driven Agent's compaction. A rejected API key stops the
run; any other provider error ends the loop as ``api_error``. Edits to
existing files are refused when they break the file's syntax, as in the
Spec-Driven Agent.

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
     - A Python file under ``besser/`` changed, or the editor's
       ``besser-openapi.json``.
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
  test files of its top-level area that import it. A module the change adds
  maps only to ``test_<name>.py`` in its package's mirror and to the tests of
  its area that import it. A change to ``besser/generators/registry.py`` also
  runs the registry, regeneration and generator-tool tests.
- **Editor** -- a file under ``src/main/features/<x>/`` or
  ``src/main/shared/<y>/`` maps to every test under that folder; any other
  webapp file to the tests next to it and in its ``__tests__`` folder.
- A changed file that maps to no tests gets a skipped "No tests found for ..."
  result rather than nothing. A file of the editor package
  (``packages/editor/src``) gets a skipped result saying that the package has
  no test runner and that its tests go in ``packages/webapp``.

Each check runs with its repository's own environment (venv, ``node_modules``)
in the change's sandbox, with no network (a check that needs it would set
``CheckSpec.network``; none does). A check whose tool is not installed (exit
code 127, or a missing ``pytest``, ``sphinx``, ``sphinx_immaterial``,
``ruff`` or ``importlinter`` module) is reported as *skipped* with the reason
and logged as a warning; it does not start a fix round. Each check has its
own timeout (30 minutes for tests), cut to the time the run has left; a
timeout or a stop kills the whole process tree. The lab runs only the checks
of the touched areas before a push, never the full suites, and says so.

Confinement
-----------

Code the model writes runs only in bubblewrap, under one mount policy
(``lab/sandbox_policy.py``, ``lab_sandbox_mounts``) applied per change (on
Linux the lab refuses to start with ``BESSER_LLM_SHELL_SANDBOX=off``):

- **read-write**: the change's own trees (BESSER with the editor inside it,
  the modeling agent);
- **read-only**: the git root, so code can read history but never rewrite a
  git directory the lab's own git reads; and, for a draft's plan step, the
  base trees;
- **masked** (an empty tmpfs): the workspace root itself, which hides the
  base clones, the shared stores and every other change, plus the lab's state
  folder, the folders of the GitHub session and token stores, of the run store
  and of the run workspaces, and the lab's own checkout;
- the rest of the host read-only. Each sandbox has its own PID namespace, so
  ``/proc/<pid>/environ`` of the backend is out of reach.

It applies to:

- **the shell** (``run_command``), with a private network
  (``BESSER_LLM_SHELL_NETWORK=private`` in the deploy files);
- **the checks**, with no network;
- **installs** of a change's dependencies (a changed lock file or requirements
  file runs ``npm ci`` or ``uv pip install`` in the change's tree), with a
  private network, since the files may be model-written and installing runs
  their build scripts; base environments, made from reviewed upstream code by
  the setup script, are not sandboxed;
- **the preview** services (see `Try it: the preview`_).

A private network is pasta's (``passt``) NAT to the internet with no host
loopback; a preview service gets only its own port forwarded in and, for the
modeling agent, the LLM proxy's port out. Every child gets the scrubbed
environment (``safe_subprocess_env()``: no variable whose name contains
``KEY``, ``SECRET``, ``TOKEN``, ``PASSWORD``, ``CREDENTIAL``, ``PRIVATE``,
``API_``, ``AUTH``, ``CERT`` or ``SESSION``), so the lab's LLM key, the
preview key and the OAuth secret reach none of them. On Linux a command whose
sandbox cannot start is refused; it never runs unconfined. Only on a platform
without namespaces (Windows, macOS) does it run unconfined, with a warning,
which is why native Windows serves for unit tests only.

The lab refuses to start (its routes answer 503 with the reason) when its
folders overlap the policy, such as a masked folder inside a change's tree.

What remains, as accepted risks of a host the team trusts:

- **The shared uv cache during installs.** The install sandbox mounts the
  workspace's uv cache read-write, so a hostile install could plant a package
  file that another change's install then links. Changes come only from
  allowed logins; review what a change adds to its requirements.
- **Egress through NAT.** pasta's NAT leaves the host as the service user, so
  sandboxed code reaches whatever the host reaches, intranet hosts included.
  ``scripts/lab/README.md`` gives an nftables rule that drops private and
  link-local destinations for that user.
- **Hard links.** With the ``hardlink`` link mode, the shared files are hard
  links (see `On disk`_).

Try it: the preview
-------------------

With ``BESSER_LAB_PREVIEW_ENABLED``, the change's **Try it** tab runs the
change live: its backend, its editor's Vite dev server (hot reload) and its
modeling agent, each in the change's sandbox with a private network
(``services/lab/preview.py``). The backend concept is "preview"; the editor
calls it Try it.

**Slots.** ``BESSER_LAB_PREVIEW_URLS`` lists one origin per slot (two on the
host: ``https://preview.<LAB_HOST>`` and ``https://preview-2.<LAB_HOST>``).
Slot *i* uses ports ``9100+10i`` (backend), ``8180+10i`` (editor) and
``8865+10i`` (modeling agent) on ``BESSER_LAB_PREVIEW_BIND``. A slot shows one
change at a time and follows it after each build (restarting what the build
changed). Taking a slot for a change (``POST /lab/preview/slot``) gives the
slot already showing it, else a free one, else one nobody has used for 10
minutes; otherwise ``preview_full``. The tab touches its slot while it is open
(``POST /lab/preview/slot/touch``), so the slot is not taken over as idle. The
first preparation of a change makes its environments, which takes minutes; the
slot's ``detail`` names the current step. Each service's state (starting,
running, failed, stopped), its last log lines, a restart and the full log
(``GET /lab/preview/logs/{service}``, redacted) are in the tab; the logs are in
``preview-logs/slot-<i>/`` of the state folder. A backend file change restarts
the backend, a ``package-lock.json`` change the editor.

**Origins.** The lab's own editor runs on ``https://<LAB_HOST>``, which holds
the GitHub session; a change's editor runs only on a preview origin, which
holds none and which only the lab may frame (CSP ``frame-ancestors
https://<LAB_HOST>``). An origin equal to ``DEPLOYMENT_URL`` disables its slot,
with the reason. The preview backend gets ``BESSER_LAB_ENABLED=false`` (it is
a plain editor backend, not a second lab) and no ``GITHUB_CLIENT_ID``, so
sign-in is off in the preview. The preview editor gets ``VITE_DEV_BACKEND_URL``
(``<slot origin>/besser_api``) and ``UML_BOT_WS_URL`` (``wss://<slot
host>/agent``). The nginx site sends each slot's ``/besser_api/``, ``/agent``
and ``/`` to its three ports and never routes a preview to the lab backend.

**The assistant and its key.** The preview's modeling agent gets
``OPENAI_BASE_URL=http://127.0.0.1:<BESSER_LAB_PREVIEW_LLM_PROXY_PORT>/v1``
and a placeholder key. The LLM proxy (``services/lab/llm_proxy.py``) runs in
the lab backend, outside every sandbox: it replaces the caller's
``Authorization`` with ``BESSER_LAB_PREVIEW_OPENAI_API_KEY`` (a separate key
with its own budget), forwards only ``chat/completions``, ``responses``,
``embeddings`` and ``audio/transcriptions`` to
``BESSER_LAB_PREVIEW_OPENAI_BASE_URL``, answers 429 past
``BESSER_LAB_PREVIEW_LLM_MAX_REQUESTS_PER_DAY`` requests per UTC day across
all slots, and logs each request's path and status, never a header or a body.
Model-written agent code can spend the key, never read it. Without the key the
proxy does not run and the preview assistant has no model access.

.. important::
   **The preview runs without the assistant until the modeling agent declares
   ``BESSER_AGENT_WS_PORT``.** A slot starts a change's modeling agent only
   when the agent's ``src/common/settings.py`` reads its port from
   ``BESSER_AGENT_WS_PORT`` (and its allowed origins from
   ``BESSER_AGENT_WS_ORIGIN`` and ``BESSER_AGENT_WS_ORIGIN_ALT``). Today's
   modeling agent reads them only from ``config.yaml``, which a change's
   working tree does not have, so every slot runs the backend and the editor
   and its Try it tab says the assistant is not available in the preview. The
   ``config.yaml`` that ``setup_workspace.sh`` writes for the base agent is not
   read by the preview.

**Local mode** (``http://`` origins): the lab's editor frames only ``https:``
pages, so the Try it tab opens the preview in a new tab instead of a frame
(see :doc:`lab_local`).

Pull requests and the review loop
---------------------------------

**Opening pull requests.** ``POST /lab/pull-request``
(``services/lab/pull_requests.py``), from the change's header or the run card,
with the clicking user's GitHub token:

1. It refuses, before anything is committed or pushed, a change with no
   modifications, a branch that does not start with ``lab/``, a change to a
   file under ``.github/workflows/`` (the sign-in scopes ``repo``, ``gist``,
   ``user`` do not include ``workflow``), an added or modified ``.env*`` file
   holding a credential, and a ``closesIssues`` entry that is not one of the
   change's issues (400, code ``bad_request``).
2. It commits the uncommitted work as the user, with the pull request's title
   as the message.
3. It pushes each branch to the repository the change started from (its
   ``bases``: a fork when ``BESSER_LAB_REPO_*`` names one), the editor first;
   when the editor changed and BESSER has changes of its own, BESSER's branch
   first gets one commit that moves the submodule pointer to the pushed editor
   commit. The token reaches git only through the environment
   (``GIT_CONFIG_COUNT`` with an ``Authorization`` header and an empty
   ``credential.helper``), and only in a scratch bare repository that borrows
   the trees' objects, never with a tree's own git config. Commits added to
   the branch on GitHub are merged first. The push is never forced.
4. It opens a **draft** pull request against each base branch, titled
   ``feat: <title>`` (``fix:`` for a bug fix) unless the user gave a title, or
   pushes to the one already open.

A failure the user can act on is told in plain words (sign in again, no write
access, a branch protection rule, no connection to GitHub), with git's or
GitHub's own message only in the server log; a refused sign-in answers 401 so
the editor offers to sign in again. ``checksNote`` says which checks the change
passed in the lab, and that the full suites did not run.

**The pull-request body.** Each body carries a lab-owned block between
``<!-- besser-lab:begin -->`` and ``<!-- besser-lab:end -->``, rewritten at
every push; what reviewers wrote outside it is kept. It holds the agent's
build report (or its closing text), the plan, the checks run in the lab, the
issues (``Closes owner/name#n`` for those the user ticked in the dialog,
``Related to`` for the others; GitHub closes an issue when the change reaches
the release branch, not ``development``), the related pull requests in merge
order (the editor first) and who asked for the change and which model built
it. Never the cost: the repositories are public.

**Diff and line comments.** ``GET /lab/diff`` sends the change's working trees
against their merge-base with the base, file by file, every patch redacted
(with the count of what was replaced). A file over 200 KB, or past the
response's 2 MB, comes back without its patch, and the editor asks for that
file alone (``repo`` and ``path``, up to 2 MB). A draft has no diff; a closed
change's is refused (its branches are gone; the diff is on the pull request).
In the editor's diff view the user comments on a line, a range or a file and
sends the comments to the lab agent as one build message.

**CI and review comments.** ``GET /lab/review`` (``services/lab/review.py``)
reads each pull request with one GraphQL query: state, draft, mergeability,
review decision, checks with the redacted end of each failing GitHub Actions
job's log (from ``*.blob.core.windows.net``; without that egress the card
shows the check's summary and link), reviews, review threads and comments. It
says where the change stands (``lifecycle``: open, partly merged, merged,
closed; ``closable`` and why not) and when to ask again (30 s while checks run
or mergeability is unknown, 120 s otherwise). Answers are cached for 20 s, so
two tabs do not double the calls. The editor turns what is new into a review
card in the change's chat.

**Review tasks.** The card's :guilabel:`Fix CI`, :guilabel:`Address comments`
and :guilabel:`Resolve conflicts` send a build message with a ``review`` task
(``fix_ci``, ``address_comments``, ``resolve_conflicts``), optionally for one
repository. The lab reads the pull requests afresh and writes the build step's
message from them. Everything GitHub users wrote reaches the agent framed as
untrusted data (see `Knowledge`_). Only comments of owners, members and
collaborators are sent, plus any comment the user ticks (an outside
contributor's is marked so).

**Update from base.** ``POST /lab/change/update-base`` brings one repository of
the change up to its base. With an open pull request, GitHub merges the base
into the pull request's branch and the lab pulls the result back; otherwise
(or when GitHub reports a conflict) the lab fetches the bases and merges the
upstream base into the change's tree, leaving any conflict in progress for a
"Resolve conflicts" build step, which concludes the merge once the markers are
gone. A base that changed GitHub workflows is refused by GitHub for the lab's
token; the message says to use GitHub's :guilabel:`Update branch`. The outcome
goes into the thread.

**Closing.** When every pull request is merged and nothing is unpushed, the
editor closes the change by itself (``POST /lab/change/close``); a change whose
pull requests were closed unmerged is closed with :guilabel:`Close the change`.
Closing a change that is already closed answers its detail unchanged; a close
that overlaps another viewer's close waits for it (up to 30 seconds) and then
answers the same detail.

Knowledge
---------

What the lab agent is told before it reads a file, so it does not rediscover
BESSER on every run (``lab/knowledge.py``, ``lab/memory.py``,
``lab/github_tools.py``). No model call builds any of it.

- **The skill.** BESSER's contributor skill lives in the repository, at
  ``.claude/skills/besser-dev/``. The system prompt lists the
  ``.claude/skills/*/SKILL.md`` of every repository (name, path and its
  ``metadata.when_to_read`` line); the agent reads a skill with ``read_file``
  when its line fits the task, and each change kind names what to read first.
- **Orientation.** The first message of each step holds the ``CLAUDE.md`` of
  the repositories in scope (in full for BESSER in the plan step and for the
  plan's repositories in the build step, as a section index with line ranges
  for the others) and a map of each repository, built
  from ``git ls-tree`` and ``git log`` of the base commit the change started
  from (never the working tree): layout, landmarks and recent commits. Maps
  are cached in ``maps/`` of the state folder, keyed by commit.
- **GitHub tools and issue framing.** ``github_issue``,
  ``github_pull_request`` and ``github_search`` read GitHub with the
  requesting user's token (``services/lab/github_reader.py``; the token never
  reaches the agent, and every answer is redacted), at most 40 calls per run.
  Issues and search read the canonical ``BESSER-PEARL`` repositories, where the
  team's issues are; pull requests are read from the repositories the lab
  pushes to. ``git_history`` reads a repository's local history. Text people
  wrote on GitHub is wrapped in a ``<github kind=... author=...
  association=... untrusted="true">`` block (HTML comments removed, any
  ``<github`` tag inside escaped, so it cannot close the block), and the
  prompt says it is data to evaluate, not instructions. Review comments and CI
  log tails reach the agent the same way.
- **Team notes.** A short list of reviewed notes, ``memory/notes.json``: a
  decision the team made, a mistake the agent made and was corrected on, or a
  convention waiting to be written into the docs it names. The build step may
  propose at most two per run with ``propose_lab_note``; they wait on the run
  card, where any allowed user keeps (and may edit) or dismisses each, and the
  note records who approved it. The system prompt lists the notes that apply
  to the change's repositories and kind (at most 25 lines); the agent reads
  one with ``read_lab_note``. At most 60 notes, 1,200 characters each; a note
  no change has read for 90 days is marked stale. The settings page lists,
  edits and deletes them. The memory is off, and says why, when the state
  folder is inside the workspace root, where sandboxed code could plant a
  note.
- **The plan model.** ``BESSER_LAB_PLAN_LLM_MODEL`` runs the plan step on
  another model of the same provider (a stronger one, for instance); the
  build step keeps ``BESSER_LAB_LLM_MODEL``. The change's run records and the
  audit log name the model of each run.

Configuration
-------------

The lab's variables are fields of ``BackendSettings``
(``besser/utilities/web_modeling_editor/backend/settings.py``), read once per
process. A flag is on only for ``1``, ``true``, ``yes`` or ``on``; a number
that does not parse or is not positive falls back to its default. The
repositories and preview origins are the exceptions: a malformed value is
refused and the backend does not start, since a default would point the lab at
another repository, which it pushes to.

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

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
       Empty allows no one. Read when the backend starts.
   * - ``BESSER_LAB_ADMIN_LOGINS``
     - empty
     - Logins among the allowed ones who may drop any change and stop any run;
       one not on the allowed list is ignored with a warning.
   * - ``BESSER_LAB_WORKSPACE_ROOT``
     - unset
     - The working trees: ``base/``, ``changes/<id>/``, ``.cache/``. Hidden
       from every sandbox. Unset, the lab cannot start (503).
   * - ``BESSER_LAB_GIT_ROOT``
     - ``<workspace root>-git``
     - Every git directory, outside the working trees; read-only in every
       sandbox.
   * - ``BESSER_LAB_STATE_DIR``
     - ``<workspace root>-state``
     - Changes, threads, run records, audit log, team notes, repository maps,
       preview logs. Hidden from every sandbox.
   * - ``BESSER_LAB_REPO_BESSER``
     - ``BESSER-PEARL/BESSER@development``
     - BESSER's repository and base branch (``owner/name@branch``): where new
       changes start and their pull requests go. A fork, for testing.
   * - ``BESSER_LAB_REPO_FRONTEND``
     - ``BESSER-PEARL/BESSER-Web-Modeling-Editor@develop``
     - The web editor's repository and base branch.
   * - ``BESSER_LAB_REPO_MODELING_AGENT``
     - ``BESSER-PEARL/modeling-agent@develop``
     - The modeling agent's repository and base branch.
   * - ``BESSER_LAB_BASE_REFRESH_MINUTES``
     - ``60``
     - Minutes between two fetches of the base branches; ``0``: only by hand
       and when a change opens. Open changes never move by themselves.
   * - ``BESSER_LAB_LLM_PROVIDER``
     - ``anthropic``
     - LLM provider of the lab agent. A blank value gives the default.
   * - ``BESSER_LAB_LLM_MODEL``
     - unset
     - Model of the lab agent; unset or blank, the provider's default.
   * - ``BESSER_LAB_PLAN_LLM_MODEL``
     - unset
     - Model of the plan step, same provider; unset, ``BESSER_LAB_LLM_MODEL``.
   * - ``BESSER_LAB_LLM_API_KEY``
     - unset
     - The lab's own API key for that provider, used by runs whose user
       saved no key of their own. Unset: every run needs the user's key.
   * - ``BESSER_LAB_MAX_COST_USD``
     - ``5.0``
     - Cost cap of one run, in US dollars.
   * - ``BESSER_LAB_MAX_RUNTIME_SECONDS``
     - ``2400``
     - Runtime cap of one run, in seconds.
   * - ``BESSER_LAB_USER_MONTHLY_CAP_USD``
     - ``75.0``
     - What one login's runs may cost per UTC month.
   * - ``BESSER_LAB_MONTHLY_CAP_USD``
     - ``300.0``
     - What all runs together may cost per UTC month.
   * - ``BESSER_LAB_MAX_CONCURRENT_RUNS``
     - ``2``
     - Runs in progress at once across all changes; more are refused
       (``runs_full``), never queued.
   * - ``BESSER_LAB_PREVIEW_ENABLED``
     - off
     - Run a change's backend, editor and modeling agent in the Try it tab.
   * - ``BESSER_LAB_PREVIEW_URLS``
     - empty
     - Comma-separated public origins of the preview slots, one per slot,
       each other than ``DEPLOYMENT_URL``. Empty: one slot on
       ``http://localhost`` ports.
   * - ``BESSER_LAB_PREVIEW_BIND``
     - ``127.0.0.1``
     - Host address the preview ports are published on (``0.0.0.0`` in a
       container).
   * - ``BESSER_LAB_PREVIEW_OPENAI_API_KEY``
     - unset
     - The key the LLM proxy adds to the preview assistant's requests. Unset:
       no proxy, and the preview assistant has no model access.
   * - ``BESSER_LAB_PREVIEW_OPENAI_BASE_URL``
     - ``https://api.openai.com/v1``
     - The proxy's upstream (OpenAI or a compatible gateway).
   * - ``BESSER_LAB_PREVIEW_LLM_PROXY_PORT``
     - ``9300``
     - Loopback port of the LLM proxy.
   * - ``BESSER_LAB_PREVIEW_LLM_MAX_REQUESTS_PER_DAY``
     - ``2000``
     - Requests the proxy forwards per UTC day across all slots; more answer
       429.

The lab also relies on ``BESSER_LLM_SHELL_NETWORK=private`` (the shell's
private network, see :doc:`configuration`) and on the GitHub OAuth app,
configured as for any instance (``GITHUB_CLIENT_ID``, ``GITHUB_CLIENT_SECRET``,
``GITHUB_REDIRECT_URI``, ``DEPLOYMENT_URL``, see :ref:`the backend's
environment variables <web-editor-backend-env>`). The deploy files hold every
value: ``scripts/lab/lab.env.example`` on a host, ``.env.lab.example`` and
``docker-compose.lab.yml`` in local mode.

HTTP API
--------

The lab's routes, all under ``/besser_api``, are listed in
:doc:`../web_editor_backend` ("BESSER Lab"). Each needs the
``X-GitHub-Session`` header of a signed-in, allowed login (``LabUserDep``):
404 while ``BESSER_LAB_ENABLED`` is off (before the body is validated), 401
without a valid session, 403 for a login not on the list, and 503 with the
reason when the lab is on but cannot start. They are mounted on every
instance, so ``/openapi.json`` is the same with the lab on or off.

On a lab instance:

- the OAuth callback refuses a login that is not on the list: it stores no
  session, revokes the token, logs the login and redirects to
  ``<DEPLOYMENT_URL>?error=not_allowed#login=<login>``, which the lab's
  refusal screen reads; ``GET /github/auth/login?prompt=select_account`` shows
  GitHub's account picker, so another account can sign in;
- ``POST /spec-driven/generate`` and ``POST /spec-driven/resume/{run_id}``
  answer 404 (no other run beside the lab's);
- the run routes (``GET /spec-driven/runs/{run_id}``, its ``/events`` and
  ``/events.json``, ``POST /spec-driven/cancel/{run_id}``) need an allowed
  login, and a lab run stops only for its starter, the change's owner or an
  admin (403 otherwise);
- ``GET /spec-driven/config`` reports ``"lab": {"enabled": true}``.

The stream of ``POST /lab/message`` uses the Spec-Driven Agent's event types
(``start``, ``phase``, ``tool_call``, ``text``, ``cost``, ``error``, ``done``;
see :doc:`api`). ``start`` carries the 32-hex ``runId``; the phases are
``plan``, ``build`` and ``check``. A stopped run ends with ``done`` like any
other, with ``cancelled: true``, ``lab.stopReason: "cancelled"`` and the
checks that ran. ``done`` has ``mode: "modify"``, empty ``downloadUrl`` and
``fileName``, ``changedFiles`` (each path prefixed with its repository) and
``lab``:

.. code-block:: json

   {
     "step": "build",
     "changeId": "3f9c2a1b7d4e",
     "title": "Add a note element",
     "kind": "editor_element",
     "repoChanges": [{"repo": "frontend", "branch": "lab/jordi/add-a-note-element", "base": "develop",
                      "files": [{"path": "packages/editor/src/main/packages/components.ts",
                                 "status": "modified"}]}],
     "checks": [{"id": "frontend:tests", "repo": "frontend", "title": "Tests of the touched areas",
                 "passed": true, "skipped": false, "summary": "12 passed", "durationSeconds": 12.4}],
     "checklist": [{"id": "editor_element.registries", "text": "...", "done": true}],
     "summary": "What changed, per repository.",
     "stopReason": "completed",
     "report": {"summary": "...", "deviations": [], "notDone": []},
     "pullRequests": [],
     "cost": {"usd": 1.84, "turns": 41, "elapsedSeconds": 612.0, "runs": 2},
     "issues": [{"repo": "frontend", "number": 212, "title": "Note element", "closes": true}],
     "plan": null,
     "modelProposal": null,
     "finalText": "The model's own closing account of the build.",
     "noteProposals": []
   }

A plan step carries ``plan`` (``kind``, ``title``, ``summary``, ``steps``,
``repos``, ``questions``) and, for a metamodel or diagram-type change,
``modelProposal`` (``title``, ``diagramType: "ClassDiagram"`` and the editor's
class diagram JSON). A checklist item's ``done`` is ``null`` when it could not
be judged. ``cost`` is the change's so far, ``issues`` the GitHub issues it
started from with their "Closes" ticks.

Hosting
-------

``scripts/lab/README.md`` sets up a lab host from an empty Linux machine:
``provision.sh install|update``, the GitHub OAuth app, the TLS certificate for
the lab and every preview origin, the systemd units (the backend, and the
lab's own editor built with ``LAB_BUILD=true``), the nginx site, the sandbox
requirements (unprivileged user namespaces, ``/dev/net/tun``), the egress rule,
backups and the host checklist. For one machine without a host, see
:doc:`lab_local`.
