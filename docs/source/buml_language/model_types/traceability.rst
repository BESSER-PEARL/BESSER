Traceability model
==================

The traceability metamodel records what happened to the artifacts of a project
and how they are related: which model elements were created or changed, by whom
and with which tool, and which generated code comes from which element. A trace
is stored apart from the models it describes, so recording traces never changes
a model file.

The metamodel follows `W3C PROV <https://www.w3.org/TR/prov-o/>`_: an
``Artifact`` is a ``prov:Entity``, an ``Operation`` is a ``prov:Activity`` and an
``Actor`` is a ``prov:Agent``.

Concepts
--------

- **TraceModel** -- the root: every artifact, operation, actor, tool and trace link
  recorded for one project. It is a ``Model``.
- **Artifact** -- something an operation consumes, produces or changes. An artifact
  can be a fragment of another one (``parent``).

  - **ModelArtifact** -- a model or a model element, referenced by the element's
    persistent id (``Element.id``, the editor element id) and its B-UML metaclass
    name (``DomainModel``, ``Class``, ``Property``, ``BinaryAssociation``, ...). The
    reference is an id, not an object, so a trace outlives deleted elements and loads
    without the model; ``ModelArtifact.resolve(model)`` finds the live element.
    The model artifacts of a project form one tree per model: a model (a
    diagram, of any type) has no ``parent``, its elements are its fragments,
    and their members (attributes, literals) are theirs. An operation can
    change or read several models, and a generator can read several.
  - **CodeArtifact** -- a generated file, or a line range of one, with the SHA-256 of
    its content. The content itself may be kept too.

- **Operation** -- an activity: a manual modelling session, an assistant edit, a code
  generation run, ... (``OperationType``). It is performed by one or more **Actors**
  (``Person``, ``Agent``) with **Tools** (``Editor``, ``CodeGenerator``, ``LLM``), has a
  status and start/end times, and holds its log records (``LogRecord``).
- **Change** -- one granular effect of an operation on one artifact: ``CREATED``,
  ``UPDATED`` or ``DELETED`` (``ChangeKind``). An update names the changed ``feature``
  (``name``, ``type``, ``multiplicity``, ...) with its old and new values. The
  operation type says *how* the work was done; the changes say *what* it did.
- **TraceLink** -- a directed link from a source artifact to a target artifact (a
  class and the lines generated from it), made by an operation. **Evidence** says why
  it holds: ``RuleEvidence`` (a deterministic rule, such as a generator template
  block), ``AIEvidence`` (an AI technique, backed by log records) or
  ``ChangeEvidence`` (inferred from changes). **Confidence** says how sure it is.

A link is *suspect* when its source, or a fragment of it, changed after the link was
made (``TraceModel.is_suspect``): the generated code may no longer match the model.
``TraceModel.validate()`` checks that every reference stays inside the trace and that
ids are unique.

Names are free text (a project owner is "BESSER User"), so trace elements inherit
``Element`` rather than ``NamedElement``. Every trace element has an ``id`` (a UUID
unless given) and a ``timestamp`` in UTC.

Example
-------

.. code-block:: python

    from besser.BUML.metamodel.traceability import (
        CodeArtifact, CodeGenerator, Confidence, ModelArtifact, Operation, OperationType,
        Person, RuleEvidence, TraceLink, TraceModel,
    )

    owner = Person("BESSER User")
    generator = CodeGenerator("sqlalchemy", "8.5.0")
    book = ModelArtifact("Book", element_id="class-1", element_type="Class")
    file = CodeArtifact("sql_alchemy.py", path="sql_alchemy.py", content_hash="...")
    lines = CodeArtifact("Book (108-140)", path="sql_alchemy.py", content_hash="...",
                         start_line=108, end_line=140, parent=file)

    generation = Operation(OperationType.CODE_GENERATION, [owner], uses=[generator],
                           consumes=[book], produces=[file, lines])
    evidence = RuleEvidence("sql_alchemy_template.py.j2#class")
    link = TraceLink(book, lines, generation, [evidence], Confidence(1.0, [evidence]))

    trace = TraceModel(project_id="my-project", artifacts=[book, file, lines],
                       operations=[generation], trace_links=[link],
                       actors=[owner], tools=[generator])
    trace.validate()

Traces from code generation
---------------------------

A generator records which lines come from which element by wrapping template blocks
in the ``{% trace %}`` tag of ``besser.utilities.generation_trace``::

    {% trace class, "class" %}
    class {{ class.name }}(Base):
        __tablename__ = "{{ class.name.lower() }}"
    {% endtrace %}

The tag leaves the generated text unchanged; ``extract_trace_regions`` returns the
line range of each block, and ``build_generation_trace`` turns them into the
``TraceModel`` of the run: one ``CODE_GENERATION`` operation and one trace link per
block, with the template rule as evidence and confidence 1. Elements without an
``id`` (a model built in Python rather than in the editor) are rendered untraced.
The :doc:`SQLAlchemy generator <../../generators/alchemy>` is the first generator
that records traces.

In the web editor
-----------------

The editor records traces per project when *Record traces for this project* is
turned on (off by default) on the project's *Traceability* page. Class
diagram edits are recorded as modelling sessions, SQLAlchemy generations as
generation runs. The trace is kept in the browser apart from the project, and
travels with a project export when asked for: in a JSON export as the JSON form of
this metamodel (see ``trace_converter.py`` / ``trace_processor.py`` in the backend
converters), in a B-UML export as its Python form (below). See
:doc:`../../web_editor`.

Python form
-----------

``trace_model_to_code`` (:doc:`../../utilities/buml_code_builder`) writes a trace
as Python that rebuilds it, ids and timestamps included. A project exported as
B-UML with its trace has it as a ``TRACEABILITY`` section bound to
``trace_model``; running the file gives you the ``TraceModel``::

    namespace = {}
    exec(open("library.py").read(), namespace)
    trace = namespace["trace_model"]
    for link in trace.trace_links:
        print(link.source.name, "->", link.target.name, trace.is_suspect(link))

The JSON and Python forms hold the same information and convert into each other
without loss.
