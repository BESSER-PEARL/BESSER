Making a Generator Traceable
============================

Every generator is traced at file level without any work: a traced generation (the web
editor's ``/besser_api/traceability/generate-output*`` endpoints, or
``build_generation_trace`` called yourself) links each model the generator read to each text
file it wrote. *Instrumenting* a generator adds which lines of a file come from which model
element, so the editor's *Traceability* page can show, for a class, the lines generated from it.
Instrumenting never changes the generated code: the trace markers are stripped before a file is
written. The pieces live in ``besser.utilities.generation_trace`` and ``GeneratorInterface``; the
:doc:`traceability model <../buml_language/model_types/traceability>` describes what a trace is.

What a traceable generator provides
-----------------------------------

``traced_files`` (a ``GeneratorInterface`` property, always present, empty by default): after
``generate()``, for each file the generator wrote in that run with ``_write_traced`` or ``_write_file``, its path relative to the
generator's output directory (``/``-separated, e.g. ``routers/book.py``) mapped to a ``FileTrace``:

- ``regions`` -- ``TraceRegion`` line ranges (1-based, inclusive) rendered from one element by one
  template rule, recorded by the ``{% trace %}`` tag;
- ``sources`` -- ``TraceSource`` elements the *whole* file was generated from (a router from its
  class), recorded with ``trace_source(element, rule)``.

``ai_files`` and ``ai_llm`` (also ``GeneratorInterface`` properties, empty by default): the files, by
the same paths, an optional LLM step of the generator wrote or changed in the run, and the LLM that did
(the BAF generator's personalization). A trace links them by AI evidence of that LLM, not by a rule.

Only elements with an ``id`` are recorded. The web editor's processors give every element a trace can
refer to the id the editor records it by (its editor element id; a quantum gate, whose cell has none,
``<diagram id>:gate:<column>:<wire>``); what the editor draws but the model has no element for, or what
the metamodel cannot give an id (an ``Object``, an agent state's body, a network's ``Configuration``,
a GUI's text and layout containers, a circuit's registers), has none. A model built in Python usually
has no ids, and then its blocks are rendered untraced, with the same output.

Instrumenting a generator, step by step
---------------------------------------

1. **Make the environment traceable.** Build the Jinja environment with ``traced_environment``
   instead of ``Environment(loader=FileSystemLoader(...))``; it takes the templates folder (or a
   loader) and any other ``Environment`` option, and adds ``TraceExtension`` to the extensions:

   .. code-block:: python

       from besser.utilities.generation_trace import traced_environment

       env = traced_environment(templates_path, trim_blocks=True, lstrip_blocks=True,
                                extensions=["jinja2.ext.do"])

2. **Mark the blocks.** In a template, wrap what is rendered for one element in
   ``{% trace element, "label" %}`` ... ``{% endtrace %}``. The rule recorded is
   ``<template name>#<label>``; the label defaults to the element's metaclass name. Blocks nest
   (a class block holding attribute blocks), and an element's ``owner`` (a method's class) is
   recorded too, so its artifact nests under the owner's. A block that renders only whitespace
   gives no region. Mark the elements a reader would look for (classes, attributes, methods,
   associations, enumerations, constraints; screens and components of a GUI; states of a state
   machine; layers of a network); leave boilerplate unmarked.

   .. code-block:: jinja

       {% for class in classes %}{% trace class, "class" %}
       class {{ class.name }}(Base):
           {% for attr in class.attributes %}{% trace attr, "attribute" %}
           {{ attr.name }} = Column(...)
           {% endtrace %}{% endfor %}
       {% endtrace %}{% endfor %}

   The tag goes around the text a block renders, not inside a Python string or comment the
   template builds: its markers are invisible control characters, removed before writing. Each marker
   carries a random nonce of the run that rendered it (``generate()`` runs inside ``trace_run()``, which
   ``GeneratorInterface`` opens), and only markers with that nonce are removed: text of the model that
   holds the same control characters (a method body) is written as it is and cannot forge a region.

3. **Write through the helper.** Write each render with ``self._write_traced`` instead of
   ``open(...).write(...)``:

   .. code-block:: python

       def _write_traced(self, relative_path: str, rendered: str, *sources: Optional[TraceSource],
                         output_dir: Optional[str] = None) -> str

   It removes the markers, writes the file (UTF-8, ``\n`` line endings, creating folders),
   records ``traced_files[relative_path]`` and returns the text written. ``output_dir`` defaults to
   ``build_generation_dir()``; pass it when the generator writes elsewhere (the backend generator's
   own default folder). For a file generated as a whole from one element, pass its source:
   ``self._write_traced(f"routers/{name}.py", rendered, trace_source(cls, "router.py.j2#router"))``.
   A helper function outside the class takes the bound method as a parameter
   (``functools.partial(self._write_traced, output_dir=...)``), as
   ``besser.generators.backend.api_generator.generate_modular_api`` does. Text that is not a template
   render (a user's own config file or skill, a JSON dump) goes through ``self._write_file`` (same
   arguments): it is written as it is, with its whole-file sources and no regions.

4. **Composites merge their parts.** A generator that runs others into its output calls
   ``self._merge_traced_files(child, prefix)`` after each ``child.generate()``; ``prefix`` is the
   folder, relative to its own output directory, the child wrote into (``"backend"``), empty when
   the same. The Web App generator merges its React, backend and agent parts this way, and the Docker
   Compose generator the agents it bakes. A child's ``ai_files`` join the composite's, without their
   regions. A child whose output is not delivered as is has nothing to merge: the SQL generator runs the
   SQLAlchemy code it generates in a scratch folder and records the traces of its own statements instead.

5. **Test it.** Assert that ``traced_files`` holds the regions you marked and that they point at
   the right lines, and that tracing changes nothing: the output of a model whose elements have ids
   equals the output of the same model without ids (``tests/generators/backend/test_backend_trace.py``
   and ``tests/generators/sqlalchemy/test_sqlalchemy_trace.py`` are the models).
   ``tests/utilities/web_modeling_editor/backend/test_traced_generation_all.py`` runs every
   registered generator through the traced endpoints on the editor's own diagrams and checks the trace
   is valid, the delivery unchanged, and the elements it traces are those of the diagrams sent, by
   their editor ids; add your generator to its element-level checks.

Nothing else is needed: the editor backend reads ``traced_files`` from the generator instance after
the run, whatever generator it is.

Building the trace of a run yourself
------------------------------------

.. code-block:: python

    from datetime import datetime, timezone
    from besser.utilities.generation_trace import (
        build_generation_trace, collect_generated_files, input_model)

    started = datetime.now(timezone.utc)
    generator = SQLAlchemyGenerator(domain_model, output_dir="out")
    generator.generate()
    trace = build_generation_trace(
        collect_generated_files("out", generator.traced_files),  # every text file, with its regions
        models=[input_model(domain_model, model_id="diagram-1")],  # each model read: id, metaclass, name
        generator="sqlalchemy", generator_version=None, actor_name="Ada",
        max_stored_content_bytes=1_000_000, started_at=started, ended_at=datetime.now(timezone.utc))

``collect_generated_files(output_dir, traced_files)`` returns each UTF-8 file under the folder as a
``GeneratedFile`` (path, content, ``content_hash`` SHA-256, trace); binary files are left out.
``generated_text_files(entries, traced_files)`` does the same for ``(path, bytes)`` pairs (the
entries of a ZIP). ``input_model(model, model_id=None, name=None)`` records a model by its id, its
metaclass (``DomainModel``, ``GUIModel``, ``Agent``, ...) and its name, with the ids of its elements
so each traced element is placed under the model that holds it; an element two models reach (a class a
GUI's table shows, a network a method runs) goes to the one that holds it most directly.
``build_generation_trace`` links every input model to every file (rule ``generator:<generator>``), each
source to its file and each element to its regions, all with confidence 1; a file in ``ai_files`` is
linked by AI evidence of ``llm`` instead. When a header is added above a delivered file, move its regions
with ``shift_file_trace(trace, lines)``; ``prefixed_traces(traced_files, prefix)`` puts a set of
traces under a folder.

.. _traceable-generator-coverage:

Coverage
--------

What each generator traces, for a model drawn in the web editor (the traced endpoints) or built in
Python with ids. *Element* means line ranges (or whole files) linked to the model elements listed;
every generator also links each model it read to each text file it wrote. Copied static files
(runtime components, images, ``package.json``-like scaffolding) and files written by a tool rather
than a template (Django's ``startproject`` and ``startapp`` files) are file level in every case.
``test_traced_generation_all.py`` checks every row the editor's endpoints can run.

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Generator
     - Model(s) read
     - Granularity
   * - ``python``, ``java``, ``test_case``
     - DomainModel
     - Element: classes, attributes, methods, enumerations and literals, associations
   * - ``pydantic``, ``rest_api``
     - DomainModel
     - Element: classes, attributes, enumerations, OCL constraints
   * - ``sqlalchemy``, ``backend``
     - DomainModel (``backend``: and the NNs its methods run)
     - Element: classes, attributes, methods, enumerations, associations (``backend``: OCL constraints too)
   * - ``sql``, ``supabase``, ``jsonschema``, ``rdf``
     - DomainModel
     - Element: tables or types of classes and enumerations, associations (``supabase``, ``jsonschema``,
       ``rdf``: attributes and literals too)
   * - ``django``, ``spring``
     - DomainModel
     - Element: models, forms, views, URLs and admin of each class (``spring``: entities, finders and
       class files); Django's own project files are file level
   * - ``web_app``
     - DomainModel, GUIModel, Agent
     - Element: its React, backend and agent parts as for those generators; ``docker-compose.yml`` and
       the Dockerfiles are file level
   * - ``react``, ``flutter``
     - DomainModel, GUIModel
     - Element: screens (by page id) and the components the editor records (React), classes and
       attributes (Flutter); not run alone by the editor (``react`` runs inside ``web_app``)
   * - ``agent`` (BAF)
     - Agent
     - Element: states, intents, transitions, tools, skills (a state's body is part of its state);
       files the personalization LLM wrote: file level, AI evidence
   * - ``docker_compose``
     - DeploymentModel (with its Agents and BPMN models)
     - Element: services (artifacts) and networks (nodes); the agents it bakes as for ``agent``
   * - ``terraform``
     - DeploymentModel of the ``deployment`` metamodel
     - Element when its elements have ids (built in Python: no editor diagram holds this model)
   * - ``bpmn``
     - BPMNModel
     - Element: flow nodes, flows, data objects and associations, participants, lanes, annotations, groups
   * - ``qiskit``
     - QuantumCircuit
     - Element: operations and function gates (registers have no editor element)
   * - ``pytorch``, ``tensorflow``, ``forecaster``
     - NN
     - Element: layers, tensor operations, sub-networks, datasets (not the ``Configuration``)
   * - ``jsonobject``
     - ObjectModel
     - File level: an ``Object`` cannot carry an id

