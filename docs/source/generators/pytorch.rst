Pytorch Generator
=================

A code generator for `PyTorch <https://pytorch.org/>`_ is also provided by BESSER. PyTorch
is an open-source machine learning framework developed by Meta (Facebook) that is widely used for
deep learning. Our Generator transforms
:doc:`B-UML Neural Network models <../buml_language/model_types/nn>` into PyTorch code,
allowing you to create neural networks based on your B-UML specifications.

To use the PyTorch generator, you need to create a ``PytorchGenerator`` object, provide the
:doc:`B-UML Neural Network model <../buml_language/model_types/nn>`, and use the ``generate`` 
method as follows:

.. code-block:: python
    
    from besser.generators.nn.pytorch.pytorch_code_generator import PytorchGenerator
    
    pytorch_model = PytorchGenerator(
        model=nn_model, output_dir="output_folder", generation_type="subclassing"
    )
    pytorch_model.generate()


Parameters
----------

- **model**: The neural network model.
- **output_dir**: The name of the output directory where the generated file will be placed.
- **generation_type**: The type of NN architecture. Either ``subclassing`` or ``sequential``.
- **channel_last**: When ``True``, PyTorch convolutional layers permute their
  input and output to match the TensorFlow channel-last convention. Default
  ``False``.

The filename embeds the generation type, so a ``PytorchGenerator`` invoked with
``generation_type="subclassing"`` produces ``pytorch_nn_subclassing.py``, and
``generation_type="sequential"`` produces ``pytorch_nn_sequential.py``.

Web Modeling Editor Support
---------------------------

Neural networks can also be designed visually in the
:doc:`BESSER Web Modeling Editor <../web_editor>` using the ``NNDiagram`` type.
The backend converts the diagram into an ``NN`` metamodel instance and passes
it to the PyTorch generator. From the editor's **Generate** menu you can choose
between the **Subclassing** and **Sequential** variants; the diagram is
validated through ``NN.validate()`` before code is produced.

Output
------

The generated file ``pytorch_nn_<generation_type>.py`` contains:

- **Imports**: PyTorch and supporting modules required by the generated code.
- **Network architecture**: in ``subclassing`` mode, a
  ``NeuralNetwork(nn.Module)`` class with an ``__init__`` that instantiates
  the layers and a ``forward`` method that chains them. In ``sequential``
  mode, an ``nn.Sequential`` definition.
- **Training and evaluation block** (emitted only when a Training Dataset is
  attached to the NN): dataset loading, loss function and optimizer setup,
  the training loop, evaluation against the test dataset, and saving the
  trained model.

The generated output for the tutorial example is shown below.

.. literalinclude:: ../../../tests/BUML/metamodel/nn/output/tutorial_example/pytorch_nn_subclassing.py
   :language: python
   :linenos: