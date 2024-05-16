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