Grammar for NN
=============================

Neural networks (NNs) can be described using various notations.
we have designed a textual notation for their definition, supported by a grammar we
developed to instantiate the concepts of the metamodel.

A textual example of the neural network (NN) model is shown below. 
The model definition begins by specifying the NN’s name (my_model). 
Next, the layers are defined outlining three layers (l1, l2, and l3), 
with l1 and l3 being 2D Convolutional layers, and l2 as a Pooling layer. Then, 
the modules definition specifies the order of the layers.
Finally, hyperparameters are defined, such as the “adam” optimiser.

.. code-block:: console

    my_model:
        layers:
            - l1:
                type=Conv2D
                actv_func=relu
                in_channels=3
                out_channels=32
                kernel_dim=[3, 3]
            - l2:
                type=Pooling
                pooling_type=max
                dimension=2D
                kernel_dim=[2, 2]
            - l3:
                type=Conv2D
                actv_func=relu
                in_channels=32
                out_channels=64
                kernel_dim=[3, 3]
        modules:
            - l1 - l2 - l3 - l4 - l5 - l6 - l7 - l8
        config:
            batch_size=32
            epochs=10
            learning_rate=0.001
            optimiser="adam"
            metrics=["f1-score"]
            loss_function=crossentropy
            weight_decay=0.2
            momentum=0.1

Save this model as a textual file, e.g. ``nn_model.txt``.

Then, load and process the model using our grammar and apply the transformation to obtain the B-UML based model.

.. code-block:: python

    # Import methods and classes
    from besser.BUML.notations.nn import buml_neural_network

    # PlantUML to B-UML model
    nn_buml_model = buml_neural_network("nn_model.txt")

``nn_buml_model`` is the BUML model containing the neural network specification.
