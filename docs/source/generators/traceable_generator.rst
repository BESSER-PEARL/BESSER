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
``generate()``, for each file the generator wrote with ``_write_traced``, its path relative to the
generator's output directory (``/``-separated, e.g. ``routers/book.py``) mapped to a ``FileTrace``:

- ``regions`` -- ``TraceRegion`` line ranges (1-based, inclusive) rendered from one element by one
  template rule, recorded by the ``{% trace %}`` tag;
- ``sources`` -- ``TraceSource`` elements the *whole* file was generated from (a router from its
  class), recorded with ``trace_source(element, rule)``.

Only elements with an ``id`` are recorded. Every element the web editor converts has one (its
editor element id); a model built in Python usually has none, and then its blocks are rendered
untraced, with the same output.

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
   template builds: its markers are invisible control characters, removed before writing.

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
   ``besser.generators.backend.api_generator.generate_modular_api`` does.

4. **Composites merge their parts.** A generator that runs others into its output calls
   ``self._merge_traced_files(child, prefix)`` after each ``child.generate()``; ``prefix`` is the
   folder, relative to its own output directory, the child wrote into (``"backend"``), empty when
   the same. The Web App generator merges its React, backend and agent parts this way. A child whose
   output is not delivered as is (the SQL generator runs the SQLAlchemy code it generates in a scratch
   folder) has nothing to merge: its file is still linked to the model at file level.

5. **Test it.** Assert that ``traced_files`` holds the regions you marked and that they point at
   the right lines, and that tracing changes nothing: the output of a model whose elements have ids
   equals the output of the same model without ids (``tests/generators/backend/test_backend_trace.py``
   and ``tests/generators/sqlalchemy/test_sqlalchemy_trace.py`` are the models).
   ``tests/utilities/web_modeling_editor/backend/test_traced_generation_all.py`` runs every
   registered generator through the traced endpoint and checks the trace is valid and the delivery
   unchanged.

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
so each traced element is placed under the model that holds it. ``build_generation_trace`` links every
input model to every file (rule ``generator:<generator>``), each source to its file and each element
to its regions, all with confidence 1. When a header is added above a delivered file, move its regions
with ``shift_file_trace(trace, lines)``; ``prefixed_traces(traced_files, prefix)`` puts a set of
traces under a folder.
