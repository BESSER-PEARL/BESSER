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

  - **ModelArtifact** -- a model or a model element. When the element is at hand (a
    trace loaded with its model, as in a project exported to Python), the artifact
    references it (``element``) and takes its name, metaclass and id from it.
    Otherwise it records them: the element's persistent id (``Element.id``, the
    editor element id), its name and its B-UML metaclass name (``DomainModel``,
    ``Class``, ``Property``, ``BinaryAssociation``, ...). A record is what lets a trace
    keep the history of deleted elements and live apart from the model, as the
    editor's JSON does; ``ModelArtifact.resolve(model)`` finds the live element.
    The model artifacts of a project form one tree per model: a model (a
    diagram, of any type) has no ``parent``, its elements are its fragments,
    and their members (attributes, literals) are theirs. An operation can
    change or read several models, and a generator can read several.
  - **CodeArtifact** -- a generated file, or a line range of one, with the SHA-256 of
    its content. The content itself may be kept too.
  - **PromptArtifact** -- a message given to an agent, with its SHA-256. A message is
    kept as an artifact only when something is derived from it (it is the source of a
    trace link); the other messages of a conversation stay in its log.

- **Operation** -- an activity, performed by one or more **Actors** (``Person``,
  ``Agent``) with **Tools** (``Editor``, ``CodeGenerator``, ``LLM``). It has a status and
  start/end times, consumes and produces artifacts, and holds its log records
  (``LogRecord``). ``Operation`` is abstract; its kind is its class:

  - **Modeling** -- edits made to models, recorded as its **Changes** (at least one).
    Whether a person or the assistant made them is told by the actors (a ``Person``,
    an ``Agent``, or both), not by the kind of operation.
  - **CodeGeneration** -- a generation run, deriving code from models; it uses at least
    one ``CodeGenerator`` or ``LLM`` (a deterministic generator, or an LLM writing the
    code; its links' evidence tells which). It is a **Transformation** (abstract): an operation that
    derives new artifacts and the trace links to them, and edits nothing.
  - **Conversation** -- a dialogue between a user and an agent (the BESSER assistant),
    with its messages as log records; at least one actor is an ``Agent``. What it leads
    to varies, so it is not a ``Transformation``: a clarification question produces
    nothing; stated requirements are artifacts it produces, linked from the
    ``PromptArtifact`` they came from; an edit or a generation it asks for is a separate
    ``Modeling`` or ``CodeGeneration``. A chain such as prompt -> requirement -> class ->
    code is made of trace links, each made by the operation that took that step.

- **Change** -- one granular effect of a ``Modeling`` on one artifact: ``CREATED``,
  ``UPDATED`` or ``DELETED`` (``ChangeKind``). An update names the changed ``feature``
  (``name``, ``type``, ``multiplicity``, ...) with its old and new values.
- **TraceLink** -- a directed link from a source artifact to a target artifact (a
  class and the lines generated from it), made by an operation. **Evidence** says why
  it holds: ``RuleEvidence`` (a deterministic rule, such as a generator template
  block), ``AIEvidence`` (an AI technique, backed by log records) or
  ``ChangeEvidence`` (inferred from changes). **Confidence** says how sure it is.

A link is *suspect* when its source, or a fragment of it, changed after the link was
made (``TraceModel.is_suspect``): the generated code may no longer match the model. A link is
*stale* -- the derived attribute ``TraceLink./stale``, computed by
``TraceModel.is_stale`` and never stored -- when it is suspect or a link into its
source, further upstream, is stale: something it depends on changed.
``TraceModel.validate()`` checks that every reference stays inside the trace and that
ids are unique.

Names are free text (a project owner is "BESSER User"), so trace elements inherit
``Element`` rather than ``NamedElement``. Every trace element has an ``id`` (a UUID
unless given) and a ``timestamp`` in UTC.

Example
-------

.. code-block:: python

    from besser.BUML.metamodel.traceability import (
        CodeArtifact, CodeGeneration, CodeGenerator, Confidence, ModelArtifact, Person,
        RuleEvidence, TraceLink, TraceModel,
    )

    owner = Person("BESSER User")
    generator = CodeGenerator("sqlalchemy", "8.5.0")
    book = ModelArtifact("Book", element_id="class-1", element_type="Class")
    file = CodeArtifact("sql_alchemy.py", path="sql_alchemy.py", content_hash="...")
    lines = CodeArtifact("Book (108-140)", path="sql_alchemy.py", content_hash="...",
                         start_line=108, end_line=140, parent=file)

    generation = CodeGeneration([owner], uses=[generator], consumes=[book],
                                produces=[file, lines])
    evidence = RuleEvidence("sql_alchemy_template.py.j2#class")
    link = TraceLink(book, lines, generation, [evidence], Confidence(1.0))

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
``TraceModel`` of the run: one ``CodeGeneration`` operation and one trace link per
block, with the template rule as evidence and confidence 1. Elements without an
``id`` (a model built in Python rather than in the editor) are rendered untraced.
The :doc:`SQLAlchemy generator <../../generators/alchemy>` is the first generator
that records traces.

In the web editor
-----------------

The editor records traces per project when *Record traces for this project* is
turned on (off by default) on the project's *Traceability* page. Class
diagram edits are recorded as modeling sessions, SQLAlchemy generations as
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
``trace_model``, after the models it is about. It references their elements
instead of repeating them -- ``ModelArtifact(element=domain_model)``,
``ModelArtifact(element=Library)``, ``ModelArtifact(element=Library_name)`` -- for
the models, classes, enumerations, attributes and methods, which the structural
sections bind to variables. The other elements (associations, generalizations,
literals, constraints) and deleted ones stay records. Running the file gives you
the ``TraceModel``, whose artifacts point at the real B-UML objects::

    namespace = {}
    exec(open("library.py").read(), namespace)
    trace = namespace["trace_model"]
    for link in trace.trace_links:
        print(link.source.name, "->", link.target.name, trace.is_suspect(link))

The structural sections carry no element ids, so importing the file gives every
element a new id. The importer loads the trace after the structural sections and
matches each model artifact to the imported diagrams by qualified name
(``Library``, ``Library.name``, an association's name, ``Genre.NOVEL``), which is
exact inside one diagram; a deleted element matches nothing and keeps its recorded
id.
