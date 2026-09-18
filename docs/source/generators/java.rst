Java Classes Generator
========================

This code generator produces the Java domain model, i.e. the set of Java classes that represent the entities and 
relationships of a :doc:`../buml_language/model_types/structural`.

Let's generate the code for the Java domain model of our :doc:`../examples/library_example` structural model example. 
You should create a ``JavaGenerator`` object, provide the :doc:`../buml_language/model_types/structural`, and use 
the ``generate`` method as follows:

.. code-block:: python

    from besser.generators.java_classes import JavaGenerator

    generator: JavaGenerator = JavaGenerator(model=library_model)
    generator.generate()

The generator writes **one file per type**: a ``<ClassName>.java`` for every class in the model and a ``<EnumName>.java``
for every enumeration, all in the ``<<current_directory>>/output`` folder. By default no ``package`` declaration is emitted;
the classes look as follows.

.. note::
   The three listings below are checked-in fixtures that predate the current
   template: they still carry a ``package tests.structural.library.output.java;``
   line and a leading comment that the generator no longer emits. Read them for
   the class/field/method shape, not for the file header.

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/java/Author.java
   :language: java
   :linenos:

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/java/Book.java
   :language: java
   :linenos:

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/java/Library.java
   :language: java
   :linenos:

To emit a ``package`` declaration, pass the ``package_name`` parameter. It is independent of ``output_dir``:
the output directory decides *where* the files are written, ``package_name`` decides what the classes declare.

.. code-block:: python

    from besser.generators.java_classes import JavaGenerator

    generator: JavaGenerator = JavaGenerator(
        model=library_model,
        output_dir="my_java_project",
        package_name="com.example.library",
    )
    generator.generate()

Will result in the following line being added to the beginning of every generated class and enum:

.. code-block:: java

    package com.example.library;

Parameters
----------

- ``model``: The input B-UML structural model (required).
- ``output_dir``: Output directory (optional, default: ``output/`` in the current directory).
- ``package_name``: Java package declared at the top of every generated file (optional).
  When omitted, no ``package`` line is emitted.