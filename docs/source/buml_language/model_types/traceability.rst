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
- **Artifact** -- something an operation reads, writes or changes. An artifact
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
    trace link); the other messages of a conversation stay in the logs of its tool calls.

- **Operation** -- an activity, performed by one or more **Actors** (``Person``,
  ``Agent``). It has a status (``ExecutionStatus``) and start/end times, reads and
  writes artifacts, and reaches its **Tools** (``Editor``, ``CodeGenerator``, ``LLM``)
  only through its **ToolCalls**; the tools it used (``Operation.tools``) are derived
  from them. It may have a **Confidence** in its result, as a trace link may (none is
  computed yet). ``Operation`` is abstract; its kind is its class:

  - **ModelingSession** -- edits made to models, recorded as its **ChangeEvents** (at least one).
    Whether a person or the assistant made them is told by the actors (a ``Person``,
    an ``Agent``, or both), not by the kind of operation.
  - **CodeGeneration** -- a generation run, deriving code from models; it calls at least
    one ``CodeGenerator`` or ``LLM`` (a deterministic generator, or an LLM writing the
    code; its links' evidence tells which). It is a **Transformation** (abstract): an operation that
    derives new artifacts and the trace links to them, and edits nothing.
  - **Conversation** -- a dialogue between a user and an agent (the BESSER assistant);
    each turn is a tool call to the agent's ``LLM``, holding the turn's messages. At
    least one actor is an ``Agent``. What it leads
    to varies, so it is not a ``Transformation``: a clarification question writes
    nothing; stated requirements are artifacts it writes, linked from the
    ``PromptArtifact`` they came from; an edit or a generation it asks for is a separate
    ``ModelingSession`` or ``CodeGeneration``. A chain such as prompt -> requirement -> class ->
    code is made of trace links, each made by the operation that took that step.

- **ToolCall** -- one use of a tool within an operation: an LLM prompted, a generator
  run, an editor session. It calls one tool, has its own ``ExecutionStatus``, start/end
  times and position in the operation (``sequence_number``), and holds the messages
  exchanged with the tool, in order (**LogRecord**: a request, an intermediate message,
  an answer, an error, ... by ``LogType``; none for an editor session). A call to an
  ``LLM`` can record the tokens it read and wrote (``input_tokens``,
  ``output_tokens``), what the call cost; only a call to an LLM records them.
- **ChangeEvent** -- one granular effect of a ``ModelingSession`` on one artifact: ``CREATED``,
  ``UPDATED`` or ``DELETED`` (``ChangeKind``). An update names the changed ``feature``
  (``name``, ``type``, ``multiplicity``, ...) with its old and new values.
- **TraceLink** -- a directed link from a source artifact to a target artifact (a
  class and the lines generated from it), made by an operation. **Evidence** says why
  it holds: ``RuleEvidence`` (a deterministic rule, such as a generator template
  block) or ``AIEvidence`` (an AI technique, such as the LLM behind an assistant
  edit). Any evidence can rest on log records, the messages it is based on: an AI
  decision's prompt and answer, a generator's log of the run. **Confidence** says how
  sure it is.

Whether a link still holds is the derived attribute ``TraceLink./outOfDate``, a set of
``OutOfDateReason`` values computed by ``TraceModel.out_of_date`` and never stored; it is
empty while the link holds (other tools call an out-of-date link *suspect*):

- ``SOURCE_CHANGED`` -- the source, or a fragment of it, changed after the link was made: the
  target (the generated code) may no longer match it.
- ``TARGET_CHANGED`` -- the target, or a fragment of it, was edited after the link was made:
  it may no longer be what was derived (a class the assistant made, then edited by hand).
- ``UPSTREAM_CHANGED`` -- neither end changed, but a link into the source has a changed
  source, or is itself upstream changed: the source may need updating, and this link with it.

Changes made by the operation that made the link are part of making it and do not count.
Only a changed source travels downstream: an edited target is the source of the next link,
which sees the edit as its own changed source. Generated code is never edited in the
editor, so a link into code is never ``TARGET_CHANGED``.
``TraceModel.validate()`` checks that every reference stays inside the trace (a tool
call's tool, an ``AIEvidence``'s log records), that ids are unique and that the tool
calls of an operation have distinct sequence numbers.

Names are free text (a project owner is "BESSER User"), so trace elements inherit
``Element`` rather than ``NamedElement``. Every trace element has an ``id`` (a UUID
unless given) and a ``timestamp`` in UTC.

Example
-------

.. code-block:: python

    from besser.BUML.metamodel.traceability import (
        CodeArtifact, CodeGeneration, CodeGenerator, Confidence, ModelArtifact, Person,
        RuleEvidence, ToolCall, TraceLink, TraceModel,
    )

    owner = Person("BESSER User")
    generator = CodeGenerator("sqlalchemy", "8.5.0")
    book = ModelArtifact("Book", element_id="class-1", element_type="Class")
    file = CodeArtifact("sql_alchemy.py", path="sql_alchemy.py", content_hash="...")
    lines = CodeArtifact("Book (108-140)", path="sql_alchemy.py", content_hash="...",
                         start_line=108, end_line=140, parent=file)

    generation = CodeGeneration([owner], tool_calls=[ToolCall(generator, 0)],
                                reads=[book], writes=[file, lines])
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
line range of each block (and the element's owner, so an attribute's or method's
block nests under its class). A specialisation is recorded under the metaclass the
editor names it by: an ``OCLConstraint`` parsed from an OCL box is a ``Constraint``. A file generated as a whole from one element -- a backend router from
its class -- records it as a ``TraceSource`` instead (``trace_source(element, rule)``).
A generator exposes both per file as ``traced_files``, a ``FileTrace`` by path in
its output folder.

``build_generation_trace`` takes the files of a run (``GeneratedFile``: path,
content, ``FileTrace``) and builds its ``TraceModel``: one ``CodeGeneration``
operation, a link from the model to every file, from each whole-file source to its
file, and from each element to each of its blocks, with the template rule as
evidence and confidence 1. A file's content is kept when it is under the size
cap, so the views can show it and its blocks' lines; above it only its hash is.
Elements without an ``id`` (a model built in Python rather than in the editor) are
rendered untraced. An OCL box of the editor holding one constraint gives the
constraint the box's ``id``; a box holding several is split into constraints without
one. The :doc:`SQLAlchemy <../../generators/alchemy>` and
:doc:`backend <../../generators/backend>` generators record traces of classes,
enumerations, associations, attributes, methods and OCL constraints.

In the web editor
-----------------

The editor records traces per project when *Record traces for this project* is
turned on (off by default) on the project's *Traceability* page. Class
diagram edits are recorded as modeling sessions (each with one call to the
editor), SQLAlchemy and backend generations as generation runs (one call to the
generator), and the exchange with the modeling assistant as conversations. The trace is kept in the browser apart from the project, and
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
        print(link.source.name, "->", link.target.name, trace.out_of_date(link))

The structural sections carry no element ids, so importing the file gives every
element a new id. The importer loads the trace after the structural sections and
matches each model artifact to the imported diagrams by qualified name
(``Library``, ``Library.name``, an association's name, ``Genre.NOVEL``), which is
exact inside one diagram; a deleted element matches nothing and keeps its recorded
id.
