Java Classes Generator
========================

This code generator produces the Java domain model, i.e. the set of Java classes that represent the entities and 
relationships of a :doc:`../buml_language/model_types/structural`.

Let's generate the code for the Java domain model of our :doc:`../examples/library_example` structural model example. 
You should create a ``JavaGenerator`` object, provide the :doc:`../buml_language/model_types/structural`, and use 
the ``generate`` method as follows:

.. code-block:: java
    
    from besser.generators.java_classes import JavaGenerator
    
    generator: Java_Generator = JavaGenerator(model=library_model)
    generator.generate()

The ``classes.py`` file with the Java domain model (i.e., the set of classes) will be generated in the ``<<current_directory>>/output`` 
folder. Note that in this case, the package name will be set to the default generation directory "output" and this will be reflected in the Java classes and will look as follows. 

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/java/Author.java
   :language: java
   :linenos:

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/java/Book.java
   :language: java
   :linenos:

.. literalinclude:: ../../../tests/BUML/metamodel/structural/library/output/java/Library.java
   :language: java
   :linenos:

Note that in case the output_dir is set to a specific name, the Java classes will take over the given name as the package name:

.. code-block:: java
    
    from besser.generators.java_classes import JavaGenerator
    
    generator: Java_Generator = JavaGenerator(model=library_model, output_dir="my_java_project")
    generator.generate()

Will result in the following line being added to the beginning of the classes:

.. code-block:: java

    package my_java_project;