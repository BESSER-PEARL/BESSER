"""A starter B-UML model: a small convolutional network that classifies 28x28 grayscale images.

    besser validate MODEL_FILE                    check the model
    besser info MODEL_FILE                        what it holds, and the generators that read it
    besser generate pytorch MODEL_FILE -o out     generate training code ('besser generators' lists them all)
    besser convert MODEL_FILE -o model.json       open it in https://editor.besser-pearl.org (import the project)

B-UML reference: https://besser.readthedocs.io/en/latest/buml_language/model_types/nn.html
"""

from besser.BUML.metamodel.nn import (
    NN,
    Configuration,
    Conv2D,
    Dataset,
    DropoutLayer,
    FlattenLayer,
    Image,
    LinearLayer,
    PoolingLayer,
)

# The network. The besser command reads every model bound to a top-level name of this file.
digit_classifier = NN(name="digit_classifier")

# Layers, in the order the data goes through them; each layer has a unique name
digit_classifier.add_layer(Conv2D(name="conv1", actv_func="relu", in_channels=1, out_channels=8, kernel_dim=[3, 3]))
digit_classifier.add_layer(PoolingLayer(name="pool1", pooling_type="max", dimension="2D", kernel_dim=[2, 2]))
digit_classifier.add_layer(FlattenLayer(name="flatten"))
# in_features is what the layers above deliver: 8 channels of 13x13 values
digit_classifier.add_layer(LinearLayer(name="hidden", actv_func="relu", in_features=1352, out_features=64))
digit_classifier.add_layer(DropoutLayer(name="dropout", rate=0.2))
digit_classifier.add_layer(LinearLayer(name="output", in_features=64, out_features=10))

# Training configuration: the generators write the training and evaluation loops from it
digit_classifier.add_configuration(Configuration(
    batch_size=32,
    epochs=10,
    learning_rate=0.001,
    optimizer="adam",
    loss_function="crossentropy",
    metrics=["accuracy"],
))

# Datasets: where the generated code reads the training and test data (one folder of images per class)
digit_classifier.add_train_data(Dataset(
    name="train_data",
    path_data="dataset/digits/train",
    task_type="multi_class",
    input_format="images",
    image=Image(shape=[28, 28, 1], normalize=False),
))
digit_classifier.add_test_data(Dataset(name="test_data", path_data="dataset/digits/test"))
