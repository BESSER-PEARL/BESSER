from antlr4 import CommonTokenStream, FileStream, ParseTreeWalker
from .deploymentLexer import deploymentLexer
from .deploymentParser import deploymentParser
from .depl_to_buml_listener import Deployment_BUML_Listener
from besser.BUML.metamodel.deployment import DeploymentModel
import os

def buml_deployment_model(deployment_textfile:str, buml_model_file_name:str = "deployment_buml_model"):
    """.

    Args:
        plantUML_model_path (str): The path to the file containing the deployment architecture textual definition.
        buml_model_file_name (str, optional): the name of the file produced with the base code to build the B-UML model.

    Returns:
        BUML_model (DeploymentModel): the B-UML model object.
    """
    lexer = deploymentLexer(FileStream(deployment_textfile))
    parser = deploymentParser(CommonTokenStream(lexer))
    parse_tree = parser.architecture()
    # file creation
    if not os.path.exists("buml"):
        os.makedirs("buml")
    output = open("buml/" + buml_model_file_name + ".py","w+")
    listen = Deployment_BUML_Listener(output)
    walker = ParseTreeWalker()
    walker.walk(listen, parse_tree)
    output.close()
    # model creation
    namespace = {}
    with open("buml/" + buml_model_file_name + ".py", 'r') as model_code:
        code = model_code.read()
        exec(code, namespace)
    BUML_model: DeploymentModel = namespace.get('deployment_model')
    return(BUML_model)