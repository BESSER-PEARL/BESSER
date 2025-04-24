import os
import importlib.util
from antlr4 import CommonTokenStream, FileStream, ParseTreeWalker
from besser.BUML.metamodel.nn import NN
from .NNLexer import NNLexer
from .NNParser import NNParser
from .nn_buml_listener import NeuralNetworkASTListener

def buml_neural_network(nn_path: str, output_dir: str = "buml", buml_model_file_name: str = "nn_model"):
    """Transforms a neural network textual model into a B-UML model.

    Args:
        nn_path (str): The path to the file containing the NN text.
        output_dir (str, optional): The directory where the generated file will be stored. Defaults to "buml".
        buml_model_file_name (str, optional): The name of the generated Python file without extension. Defaults to "nn_model".

    Returns:
        tuple: (BUML_model, train_data, test_data) - The B-UML model objects.
    """
    lexer = NNLexer(FileStream(nn_path))
    parser = NNParser(CommonTokenStream(lexer))
    parse_tree = parser.neuralNetwork()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(output_dir, buml_model_file_name + ".py")

    # Write the generated file
    with open(file_path, "w+", encoding="utf-8") as output:
        listen = NeuralNetworkASTListener(output)
        walker = ParseTreeWalker()
        walker.walk(listen, parse_tree)

    # Load the generated Python file as a module
    spec = importlib.util.spec_from_file_location("nn_model", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Retrieve the BUML model from the loaded module
    buml_model: NN = module.nn_model
    train_data: NN = module.train_data
    test_data: NN = module.test_data

    return buml_model, train_data, test_data
