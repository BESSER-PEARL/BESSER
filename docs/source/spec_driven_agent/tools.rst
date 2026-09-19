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

Shell tools
-----------

.. warning::
   The shell tools (``run_command`` / ``install_dependencies``) run arbitrary
   commands. On a **server deployment** they are **disabled by default**: they
   are opt-in via ``BESSER_LLM_ENABLE_SHELL_TOOLS``, and the hosted editor
   withholds them — every static tool remains available. The
   dependency-resolution check in :doc:`Phase 3 <validation>` (a
   ``pip install --dry-run``) is gated behind the same flag, since resolving an
   sdist can execute its build backend.

   ``LLMOrchestrator(allow_shell_tools=...)`` defaults to ``False`` on every
   path, including a :doc:`Python library run <usage>`. Pass ``True`` to opt
   in. The gate is enforced at dispatch, so naming a shell tool that was never
   advertised is refused too — hiding a capability is not withholding it.

.. note::
   Adding a tool is a two-line change in
   ``besser/generators/llm/tools.py`` — the declaration *and* an entry in
   ``_TOOL_MODEL_REQUIREMENTS``, so the tool is only offered when the models it
   needs are present. See :doc:`../contributor_guide`.
