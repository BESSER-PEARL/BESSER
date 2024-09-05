import os
import importlib.util
from antlr4 import CommonTokenStream, FileStream, ParseTreeWalker
from besser.BUML.metamodel.nn import NN
from .NNLexer import NNLexer
from .NNParser import NNParser
from .nn_buml_listener import NeuralNetworkASTListener

def buml_neural_network(nn_path:str, buml_model_file_name:str = "nn_model"):
    """Transforms a neural network textual model into a B-UML model.

    Args:
        nn_path (str): The path to the file containing the NN text.
        buml_model_file_name (str, optional): the name of the file produced with the base 
        code to build the B-UML model.

    Returns:
        BUML_model (NN): the B-UML model object.
    """
    lexer = NNLexer(FileStream(nn_path))
    parser = NNParser(CommonTokenStream(lexer))
    parse_tree = parser.neuralNetwork()
    # file creation
    if not os.path.exists("buml"):
        os.makedirs("buml")
    with open("buml/" + buml_model_file_name + ".py", "w+", encoding="utf-8") as output:
        listen = NeuralNetworkASTListener(output)
        walker = ParseTreeWalker()
        walker.walk(listen, parse_tree)
    # Load the generated Python file as a module
    file_path = "buml/" + buml_model_file_name + ".py"
    spec = importlib.util.spec_from_file_location("nn_model", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Retrieve the BUML model from the loaded module
    buml_model: NN = module.nn_model
    train_data: NN = module.train_data
    test_data: NN = module.test_data
    return buml_model, train_data, test_data
