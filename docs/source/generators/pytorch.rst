Pytorch Generator
=================

A code generator for `Pytorch <https://pytorch.org/>`_ is also provided by BESSER. Pytorch
is an open-source machine learning framework developed by Meta (Facebook) that is widely used for
deep learning and artificial intelligence (AI) applications. Our Generator transform
:doc:`B-UML Neural Network models <../buml_language/model_types/nn>` into Pytorch code,
allowing you to create neural networks based on your B-UML specifications.

To use the Pytorch generator, you need to create a ``TFGenerator`` object, provide the
:doc:`B-UML Neural Network model <../buml_language/model_types/nn>`, and use the ``generate`` 
method as follows:

.. code-block:: python
    
    from besser.generators.nn.pytorch.pytorch_code_generator import PytorchGenerator
    
    pytorch_model = PytorchGenerator(
        model=nn_model, output_dir="output_folder", generation_type="subclassing"
    )
    pytorch_model.generate()


To complete...
