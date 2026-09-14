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
     - ``list_files``, ``read_file``, ``write_file``, ``modify_file``,
       ``search_in_files``, ``delete_file``
   * - Model queries
     - ``query_class``, ``list_classes_with``, ``get_constraints_for``
   * - Validation / bookkeeping
     - ``validate_model``, ``check_syntax``, ``task_list``
   * - Generators
     - The 20 generator tools listed below.
   * - Shell
     - ``run_command``, ``install_dependencies``

The model-query tools give the LLM random access into your domain model
(attributes, flattened inherited methods, association ends, OCL constraints)
without holding the whole serialized model in context.

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

   They are intended for trusted local or CLI runs, and a
   :doc:`Python library run <usage>` is treated as one:
   ``LLMOrchestrator(allow_shell_tools=...)`` defaults to ``True``, so a
   library run *can* execute shell commands on your machine unless you pass
   ``False``.

.. note::
   Adding a tool is a two-line change in
   ``besser/generators/llm/tools.py`` — the declaration *and* an entry in
   ``_TOOL_MODEL_REQUIREMENTS``, so the tool is only offered when the models it
   needs are present. See :doc:`../contributor_guide`.
