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
   * - Model queries
     - ``query_class``, ``list_classes_with``, ``get_constraints_for``
   * - Validation / bookkeeping
     - ``validate_app``, ``test_api``, ``validate_model``, ``check_syntax``, ``task_list``
   * - Generators
     - The 20 generator tools listed below.
   * - Shell
     - ``run_command``, ``install_dependencies``

The model-query tools give the LLM random access into your domain model
(attributes, flattened inherited methods, association ends, OCL constraints)
without holding the whole serialized model in context.

File edits and checklist evidence
---------------------------------

``modify_file`` refuses ambiguous anchors, identical no-ops and duplicate-looking
insertions. ``already_applied`` requires a receipt for that exact successful edit
and an unchanged post-edit file hash. After restart, matching replacement-shaped
content may instead produce ``possible_replay``: no write is made, but no success
is claimed. Missing-file responses suggest existing paths without substituting
them for the requested path.

After two rejected text edits on a file, the executor provides an explicit
recovery sequence: ``read_file`` followed by ``replace_file_lines``. This applies
in customization and validation repair. Repeated quotation failures do not
permanently freeze the file; an agent that ignores recovery is still bounded by
the loop and run budgets.

``read_file`` returns a ``read_id`` tied to the resolved path, whole-file content
hash, and actually displayed lines. ``replace_file_lines`` accepts that ID,
1-based inclusive ``start_line``/``end_line``, and complete ``new_text``. It does
not require reproducing the old text. Unread/truncated ranges, stale IDs,
no-ops, new elisions, and changes that break previously valid Python syntax are
refused without writing. After a successful edit, read again before another
same-file range edit. IDs are session-local and expire on resume. The tool shares
path containment, per-file locking, diagnostics, and successful-write evidence
with the other editors. An accepted edit is not proof of business correctness.

``task_list(action='done')`` runs an attached verifier. A verifier exception is a
failed check, not success. Without a verifier, supply
``evidence=[{"id": N, "path": "...", "quote": "..."}]`` from a successful current
write; this records implementation, with acceptance still unverified.
Already-correct scaffold code can instead use ``existing=true`` after reading
the file and citing executable evidence. This records an existing implementation,
not a fabricated write or a passing business test.
``action='blocked'`` requires a reason and retains required unresolved work.
A later successful check/evidence submission can clear the block. ``drop`` is
only for work outside the user's request. Task outcomes are retained in the recipe.

Application checks during editing
---------------------------------

``validate_app`` reports source contracts, startup and create-request failures
after a coherent edit. Validation runs after writes in the same tool batch.

``test_api`` executes up to 20 declarative requests against a disposable copy of
one generated FastAPI backend and a fresh SQLite database. Requests share state
within a scenario; later requests may reference response JSON using
``{{0.room.id}}``. Assertions specify expected statuses and dotted JSON fields.
Named scenarios are retained during the run and replayed after source changes;
failures block completion. Correcting a mistaken test requires an explicit
``correction_reason``. Model-authored scenarios are supplemental checks, not an
independent proof of specification coverage.

Use ``test_api(action="list")`` to discover retained workflows and
``test_api(action="get", scenario_id="...")`` to inspect their exact inputs,
expectations and last report without executing them. ``action="run"`` with
only a scenario ID replays its saved definition. The original specification
remains authoritative: a generated assertion can be wrong and must not force
correct application behavior to regress.

Both runtime probes honor ``enable_import_smoke_check``. The API tool accepts no
shell commands or external URLs. It runs generated Python code with ordinary
side-effect guards, **not an OS security sandbox**; hosted untrusted execution
still requires deployment-level isolation.

Generator tools
---------------

Each of these calls a BESSER :doc:`code generator <../generators>` directly —
the same deterministic generator you would run yourself from Python. They are
what makes the agent an orchestrator rather than a code writer: for anything
BESSER already templates, the LLM asks for the template output and edits it
instead of writing it from scratch.

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
   * - ``generate_flutter``
     - :doc:`Flutter <../generators/flutter>`
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

.. _spec-driven-shell-tools:

Shell tools
-----------

``run_command`` and ``install_dependencies`` run arbitrary commands. They are
what lets the agent close its own loop — run ``pytest``, run ``npm run build``,
read the failures and fix them — and they are the single largest capability
difference between a run with them and a run without. They are also arbitrary
code execution in the backend process, so who is allowed to switch them on is a
deployment decision, never a per-request one.

The policy
~~~~~~~~~~

**Off on a shared host, including the hosted editor.** There, users bring their
own API key and share one backend process; an agent that can be steered into
running commands is remote code execution against other tenants' runs and
against server-side configuration. Every static tool (``read_file``,
``write_file``, ``modify_file``, ``check_syntax``, the generators, the runtime
probes) stays available, so the agent still produces and validates a full
application — it just cannot shell out. The ``pip install --dry-run``
dependency check in :doc:`Phase 3 <validation>` is behind the same flag, since
resolving an sdist can execute its build backend.

**On, deliberately, for a local or on-prem install.** One machine, one tenant,
the operator's own data — the hosted objection does not apply, and withholding
the tools is pure loss.

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
- **Working directory inside the run workspace.** ``working_dir`` is resolved
  against the workspace and a path that escapes it is rejected.
- **A 120-second timeout** per command; the spawned shell is killed when it
  expires. Be aware of both edges: a cold ``npm install`` can exceed it, and a
  process the command detached can outlive it.
- **A stripped environment.** The child process gets an allowlist (``PATH``,
  ``HOME``, locale, temp dirs, a few Python/Node variables) with anything
  name-matching a secret removed, so provider API keys, OAuth secrets and SMTP
  credentials are not visible to a command or to a build backend it triggers.
- **A denylist** for the obvious catastrophes: ``sudo``, ``rm -rf /``,
  curl-pipe-shell, fork bombs, reads of ``~/.ssh`` and ``~/.aws``.
- **A bounded amount of output** in the model's context, with the full log
  still reachable (below).

What it does not:

- **It is not an OS sandbox.** The command runs as the backend user, with that
  user's filesystem and network access. The working-directory lock sets where a
  command *starts*; a command is free to ``cd`` elsewhere. The denylist stops
  mistakes, not a determined prompt-injection payload. Real isolation means a
  container or VM per run, which the agent does not create for you.
- **It does not confine dependency installs.** ``install_dependencies`` runs
  ``pip install`` with the interpreter that is running BESSER and ``npm
  install`` in the workspace. Run an on-prem install in its own virtualenv or
  container image.
- **It does not make generated code safe to execute.** That caveat applies to
  the runtime probes as well, and is unchanged by this flag.

Treat "enable shell tools" as "I am willing to run model-authored commands on
this machine, as this user". That is true and acceptable on a developer laptop
or a single-tenant on-prem box, and it is not true on a shared one.

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
stream is itself capped at 2 MB — far above anything a build log needs, and
only there so a runaway command cannot fill the disk.

.. note::
   Adding a tool is a two-line change in
   ``besser/spec_driven_agent/tools.py`` — the declaration *and* an entry in
   ``_TOOL_MODEL_REQUIREMENTS``, so the tool is only offered when the models it
   needs are present. See :doc:`../contributor_guide`.
