"""B-UML networks that fit the time-series forecaster, shared by the generator and the web editor
backend tests (the networks are built by functions, so tests can parametrize over them)."""
from besser.BUML.metamodel.nn import NN, Configuration, DropoutLayer, GRULayer, LinearLayer

WINDOW, HORIZON = 24, 12


def configuration(epochs: int = 2, loss: str = "mse") -> Configuration:
    """Two quick epochs of AdamW on the mean squared error."""
    return Configuration(batch_size=32, epochs=epochs, learning_rate=0.003, optimizer="adamW",
                         loss_function=loss, metrics=["mae"], weight_decay=0.01)


def mlp(window: int = WINDOW, horizon: int = HORIZON) -> NN:
    """The window plus the sine and cosine of the month in, ``horizon`` values out."""
    model = NN(name="mlp")
    model.add_layer(LinearLayer(name="hidden", in_features=window + 2, out_features=32, actv_func="relu"))
    model.add_layer(DropoutLayer(name="dropout", rate=0.1))
    model.add_layer(LinearLayer(name="output", in_features=32, out_features=horizon))
    model.add_configuration(configuration())
    return model


def gru(horizon: int = HORIZON) -> NN:
    """One value and the season position per step in, ``horizon`` values out."""
    model = NN(name="gru")
    model.add_layer(GRULayer(name="recurrent", input_size=3, hidden_size=16, return_type="last",
                             batch_first=True, output_var="state"))
    model.add_layer(LinearLayer(name="output", in_features=16, out_features=horizon, input_var="state"))
    model.add_configuration(configuration())
    return model
