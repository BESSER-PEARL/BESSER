from antlr4 import CommonTokenStream, FileStream, ParseTreeWalker
from .deploymentLexer import deploymentLexer
from .deploymentParser import deploymentParser
from .depl_to_buml_listener import Deployment_BUML_Listener
from besser.BUML.metamodel.deployment import DeploymentModel
import os

def buml_deployment_model(deployment_textfile: str, buml_model_file_name: str = "deployment_buml_model", output_dir: str = "buml"):
    """
    Args:
        deployment_textfile (str): The path to the file containing the deployment architecture textual definition.
        buml_model_file_name (str, optional): the name of the file produced with the base code to build the B-UML model.
        output_dir (str, optional): directory where the intermediate file will be stored. Defaults to "buml".

    Returns:
        BUML_model (DeploymentModel): the B-UML model object.
    """
    lexer = deploymentLexer(FileStream(deployment_textfile))
    parser = deploymentParser(CommonTokenStream(lexer))
    parse_tree = parser.architecture()

    # file creation with proper path handling
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"{buml_model_file_name}.py") if output_dir else f"{buml_model_file_name}.py"

    with open(output_file, "w+") as output:
        listen = Deployment_BUML_Listener(output)
        walker = ParseTreeWalker()
        walker.walk(listen, parse_tree)

    # model creation
    namespace = {}
    with open(output_file, 'r') as model_code:
        code = model_code.read()
        exec(code, namespace)
    
    BUML_model: DeploymentModel = namespace.get('deployment_model')
    return BUML_model