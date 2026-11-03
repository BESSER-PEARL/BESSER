Build Your Own Generator
========================

At BESSER, you can also build your own code generator. Code generators consist of M2T model-to-text transformations
to automatically generate software artifacts from an input model (could be any :doc:`type of model <../buml_language/model_types>`).

BESSER provides an interface (abstract class) called ``GeneratorInterface`` that you can inherit to build your code
generator. This way, we standardize the use of BESSER code generators, improve maintainability, and usability (you can
check the code of 
`the GeneratorInterface in the repository <https://github.com/BESSER-PEARL/BESSER/blob/master/besser/generators/generator_interface.py>`_).

As an example, let's look at our Python class code generator below. Notice how this generator inherits from the 
``GeneratorInterface`` class and defines two methods: 

* Constructor method ``__init__()`` 
    this method contains the parameters ``model`` (indicating the B-UML model) and ``output_dir`` (the directory where 
    the generated code will be stored) which is optional.
* ``generate()`` method 
    to generate the ``classes.py`` file. The M2T transformation (lines 30 to 34) is performed using 
    `Jinja <https://jinja.palletsprojects.com/>`_, a templating engine for generating template-based documents. However,
    you could use the tool of your choice for these transformations.

.. literalinclude:: ../../../besser/generators/python_classes/python_classes_generator.py
   :language: python
   :linenos:

Remember that in BESSER, B-UML models have a set of methods to facilitate their traversal. For example, for structural models,
the ``class.attributes`` method gets the list of attributes of the class, ``class.all_attributes`` gets the list of attributes
including the inherited ones (if the class inherits from another one), ``model.classes_sorted_by_inheritance()`` gets the classes
of the model sorted according to the inheritance hierarchy, and so on. You can consult the :doc:`API documentation <../api>` for
more information.


Registering Your Generator
--------------------------

BESSER keeps one list of its generators: ``GENERATORS`` in ``besser/generators/registry.py``.
The :doc:`BESSER Web Modeling Editor <../web_editor>` backend (``SUPPORTED_GENERATORS``) and the
:doc:`Spec-Driven Agent <../spec_driven_agent/index>` (its ``generate_*`` tools) are both derived
from it, so registering a generator is one ``GeneratorSpec`` entry:

.. code-block:: python

    GeneratorSpec(
        id="my_generator",                 # the generator key the editor sends
        class_path="besser.generators.my_generator:MyGenerator",  # imported on first use
        category="web_framework",          # logical grouping (object_oriented, web_framework, database, etc.)
        output_type="zip",                 # "file" for single-file output, "zip" for archives
        file_extension=".zip",             # extension of the generated artifact
        requires_class_diagram=True,       # True if the generator needs a ClassDiagram as input
        download_filename="my_app.zip",    # name of the downloaded file
    ),

Set ``requires_class_diagram=False`` for generators that consume other diagram types
(e.g., agent diagrams, quantum circuits, deployment models), and ``required_diagram_type`` to the
diagram the generator reads (e.g. ``"NNDiagram"``).

``download_filename`` is the name the browser saves the download under; ``get_filename_for_generator()``
reads it from the registry, and an entry without one falls back to ``<generator_key>_output<file_extension>``.
A single-file generator whose own output filename is meaningful (the Supabase generator's
``<timestamp>_<slug>.sql`` migration name, for example) can set ``keep_generated_filename=True`` instead,
so the download keeps that name.

A generator that needs a second model besides its diagram (the Web App, React and Flutter generators
need a GUI model next to the class diagram) sets ``needs_beyond_one_diagram`` to what it needs:
``/generate-output``, which carries one diagram, then answers 400 with that text instead of running it.

Every entry is offered in the editor. ``agent_tool`` (default ``None``) also makes it a
Spec-Driven Agent tool, with the tool's name, the short ``label`` the run preview shows, its
description, input parameters and the models it needs (``requires``). An agent tool then needs:

* its generator ``id`` in ``AGENT_TOOL_ORDER`` (same file), the order the tools appear in the
  agent's prompt; ``agent_tools()`` raises on a tool missing from it;
* a handler in ``ToolExecutor._handlers`` (``besser/spec_driven_agent/agent/tool_executor.py``);
* a row in the generator-tools table of ``docs/source/spec_driven_agent/tools.rst``.

``tests/generators/test_generator_registry.py`` pins both lists (the tool names, and snapshots of
the two derived payloads under ``tests/generators/snapshots/``), so a change to either shows up in
review: update them with the new entry.
