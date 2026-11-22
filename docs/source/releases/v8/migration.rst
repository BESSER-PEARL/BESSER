Upgrade to v8
=============

Review the sections that apply to your library code, generated applications,
or hosted editor. Export existing models and keep a copy of application files
before regenerating them.

Python and dependencies
-----------------------

BESSER requires Python **3.11 or newer**; CI tests 3.11 and 3.12.
``tiktoken`` is a new core dependency, and ``httpx`` is bounded to ``<0.28``.
The backend pins FastAPI to ``0.110.x`` and Starlette to ``0.36.x`` and adds
Anthropic, ruff, and pyflakes. Install the ``agents`` extra for direct
OpenAI/Anthropic library use:

.. code-block:: console

   python -m pip install "besser[agents]"

Generated backend requests
--------------------------

In backend mode, ``<Class>Create`` schemas used by ``POST`` and ``PUT`` no
longer accept these server-owned attributes: an ``id`` that is not a declared
primary key, ``createdAt`` / ``updatedAt``, and derived attributes. A declared
primary key (``is_id=True``) stays client-supplied regardless of its name.

Update clients to match the generated Create schema. If an ``id`` must come
from the client, mark it ``is_id`` in the model. Relationship fields now use
the referenced primary-key type, which may be a string rather than an integer.

On the non-owning side of a one-to-one association the Create schema no longer
contains a link field. Set that link from the side holding the foreign key.

Default values and method bodies
--------------------------------

The SQLAlchemy, Pydantic, Python, Django, and backend generators coerce defaults
to the attribute type and emit them as literals. Defaults such as ``"abc"``
for an integer, executable expressions, empty non-string defaults, or unknown
enumeration literals raise ``InvalidDefaultValueError`` during generation.
Use a literal value of the declared type.

Python and Django generators infer simple method bodies such as getters,
setters, boolean predicates, and ``__str__``. Other methods without a modeled
implementation now raise ``NotImplementedError`` when called. Provide an
implementation before depending on those methods at runtime.

Django generation
-----------------

``DjangoGenerator.generate()`` raises generation errors instead of printing
them and returning. Handle the exception in callers that need recovery.
Generation uses ``output_dir`` without changing the process working directory.
It refuses to replace a non-empty directory without ``manage.py``; choose
a fresh output directory for a new project.

Model behavior
--------------

* Renaming a class updates association-end names derived from it. Custom role
  names are preserved; a rename that introduces a collision is skipped.
* ``DomainModel.validate()`` warns about mandatory association cycles and
  multiple associations between the same classes, and reports duplicate
  association-end names as errors. Review these findings before generation.
* ``AssociationClass`` now forwards its timestamp and metadata correctly.
* ``GUIModel`` accepts ``stylesheet``; ``DataBinding`` accepts
  ``aggregation``. Unknown aggregation names raise ``ValueError``.
  Existing constructor calls can omit both options.

Imports and backend defaults
----------------------------

Uploaded B-UML files may contain only statements supported by the code
builders. Arbitrary Python in an import is rejected; an
``if __name__ == "__main__":`` block is ignored. Export your model with the
builders rather than adding executable setup logic to an import file.

``DEFAULT_SQL_DIALECT`` is now ``sqlite``. Requests without a dialect no
longer use the invalid ``standard`` SQLAlchemy dialect.

Spec-Driven Agent callers
-------------------------

Import ``LLMGenerator`` from ``besser.spec_driven_agent``. Code written against
the unpublished ``besser.generators.llm`` prototype must use the new path.

Shell tools and toolchain validation are **off by default** in the library
and backend. Python callers must explicitly pass ``allow_shell_tools=True``
and/or ``enable_toolchain_validation=True`` when needed. The web backend uses
deployment environment variables; a request cannot enable these permissions.

Runtime execution requires a working Linux bubblewrap sandbox unless a
development deployment has explicitly opted out. Unavailable checks are
reported as unverified. See :doc:`/spec_driven_agent/validation`.

Hosted editor deployment
------------------------

Before deploying v8 images:

1. Add ``besser-wme-smartgen`` using the ``smartgen_worker`` image.
2. Configure its four ``security_opt`` entries, LLM environment variables,
   shared workspace, and network as in the production runbook.
3. Route ``/besser_api/spec-driven/`` to the worker, keeping GitHub push and
   import endpoints on the backend.
4. Deploy compatible BESSER, WME, and Modeling Agent versions.
5. Verify build stamps, sandbox startup, public API, and a small generation run.

Use :doc:`/spec_driven_agent/production_deployment` for exact configuration.
The deploy workflow checks for the service and image; it does not establish
that every sandbox setting or proxy route is correct.
