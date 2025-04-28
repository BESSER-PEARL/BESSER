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


The configuration parameters for the `PytorchGenerator` are as follows:

- **model**: The neural network model.
- **output_dir**: The name of the output directory where the ``pytorch_nn_subclassing.py`` file will be generated.
- **generation_type**: The type of NN architecture. Either ``subclassing`` or ``sequential``.


The ``pytorch_nn_subclassing.py`` file will be generated inside ``output_folder`` and it will look as follows:



.. literalinclude:: ../../../tests/BUML/metamodel/nn/output/tutorial_example/pytorch_nn_subclassing.py
   :language: python
   :linenos: