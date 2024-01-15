from antlr4 import CommonTokenStream, FileStream, ParseTreeWalker
from .PlantUMLLexer import PlantUMLLexer
from .PlantUMLParser import PlantUMLParser
from .plantUML_buml_listener import BUMLGenerationListener
from BUML.metamodel.structural import DomainModel
import os

def plantuml_to_buml(plantUML_model_path:str, buml_model_file_name:str = "buml_model"):
    """Transforms a PlantUML model into a B-UML model.

    Args:
        plantUML_model_path (str): The path to the file containing the PlantUML code.
        buml_model_file_name (str, optional): the name of the file produced with the base code to build the B-UML model.

    Returns:
        BUML_model (DomainModel): the B-UML model object.
    """
    lexer = PlantUMLLexer(FileStream(plantUML_model_path))
    parser = PlantUMLParser(CommonTokenStream(lexer))
    parse_tree = parser.domainModel()
    # file creation
    if not os.path.exists("buml"):
        os.makedirs("buml")
    output = open("buml/" + buml_model_file_name + ".py","w+")
    listen = BUMLGenerationListener(output)
    walker = ParseTreeWalker()
    walker.walk(listen, parse_tree)
    output.close()
    # model creation
    namespace = {}
    with open("buml/" + buml_model_file_name + ".py", 'r') as model_code:
        code = model_code.read()
        exec(code, namespace)
    BUML_model: DomainModel = namespace.get('domain')
    return(BUML_model)