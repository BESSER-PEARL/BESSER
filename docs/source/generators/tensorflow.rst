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



Parameters
----------

- **model**: The neural network model.
- **output_dir**: The name of the output directory where the generated file will be placed.
- **generation_type**: The type of NN architecture. Either ``subclassing`` or ``sequential``.

The filename embeds the generation type, so a ``TFGenerator`` invoked with
``generation_type="subclassing"`` produces ``tf_nn_subclassing.py``, and
``generation_type="sequential"`` produces ``tf_nn_sequential.py``.

Web Modeling Editor Support
---------------------------

Neural networks can also be designed visually in the
:doc:`BESSER Web Modeling Editor <../web_editor>` using the ``NNDiagram`` type.
The backend converts the diagram into an ``NN`` metamodel instance and passes
it to the TensorFlow generator. From the editor's **Generate** menu you can
choose between the **Subclassing** and **Sequential** variants; the diagram
is validated through ``NN.validate()`` before code is produced.

Output
------

The generated file ``tf_nn_<generation_type>.py`` contains:

- **Imports**: TensorFlow and supporting modules required by the generated code.
- **Network architecture**: in ``subclassing`` mode, a
  ``NeuralNetwork(tf.keras.Model)`` class with an ``__init__`` that
  instantiates the layers and a ``call`` method that chains them. In
  ``sequential`` mode, a ``tf.keras.Sequential`` definition.
- **Training and evaluation block** (emitted only when a Training Dataset is
  attached to the NN): dataset loading, loss function and optimizer setup,
  the training loop, evaluation against the test dataset, and saving the
  trained model.

The generated output for the tutorial example is shown below.

.. literalinclude:: ../../../tests/BUML/metamodel/nn/output/tutorial_example/tf_nn_subclassing.py
   :language: python
   :linenos: