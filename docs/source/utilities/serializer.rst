Model Serializer
================

The serializer component enables serialization (convert into a byte stream) and deserialization of models using the 
`Pickle <https://docs.python.org/3/library/pickle.html>`_ tool. 

Model serialization
-------------------

Any model within BESSER can be serialized as follows.

.. code-block:: python

    from besser.BUML.metamodel.structural import DomainModel
    from besser.utilities import ModelSerializer

    # Create an instance of DomainModel
    test_model: DomainModel = DomainModel(name = "testModel")
    # Create an instance of ModelSerializer
    serializer: ModelSerializer = ModelSerializer()
    # test_model serialization
    serializer.dump(model=test_model)

The model is serialized and stored in a file named ``<<model_name>>.buml`` in the current directory.
However, you can also specify the output directory and name of the serialized model file as follows.

.. code-block:: python

    # Serialize test_model providing output directory and filename
    serializer.dump(model=test_model, output_dir="/directory/", output_file_name="filename")

Model deserialization
---------------------

To deserialize a model (for example a Model Domain of B-UML) you can use the ``load()`` method providing the model path 
(including the file name) in the following way.

.. code-block:: python

    from besser.BUML.metamodel.structural import DomainModel
    from besser.utilities import ModelSerializer

    # Create an instance of ModelSerializer
    serializer: ModelSerializer = ModelSerializer()
    # Load the model
    model: DomainModel = serializer.load(model_path="/directory/filename")

.. note::
    
    For a detailed description of the model serializer please refer to the :doc:`API documentation <../api/api_utilities>` documentation.