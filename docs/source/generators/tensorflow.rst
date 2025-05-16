TensorFlow Generator
====================

BESSER provides a code generator for `TensorFlow <https://www.tensorflow.org/>`_, which is a
popular open-source library for deep learning. This generator transforms
:doc:`B-UML Neural Network models <../buml_language/model_types/nn>` into TensorFlow code,
allowing you to create neural networks based on your B-UML specifications.

To use the TensorFlow generator, you need to create a ``TFGenerator`` object, provide the
:doc:`B-UML Neural Network model <../buml_language/model_types/nn>`, and use the ``generate`` 
method as follows:

.. code-block:: python
    
    from besser.generators.nn.tf.tf_code_generator import TFGenerator
    
    tf_model = TFGenerator(
        model=nn_model, output_dir="output_folder", generation_type="subclassing"
    )
    tf_model.generate()



The configuration parameters for the `TFGenerator` are as follows:

- **model**: The neural network model.
- **output_dir**: The name of the output directory where the ``tf_nn_subclassing.py`` file will be generated.
- **generation_type**: The type of NN architecture. Either ``subclassing`` or ``sequential``.


The ``tf_nn_subclassing.py`` file will be generated inside ``output_folder`` and it will look as follows:



.. literalinclude:: ../../../tests/BUML/metamodel/nn/output/tutorial_example/tf_nn_subclassing.py
   :language: python
   :linenos: