from antlr4 import CommonTokenStream, FileStream, ParseTreeWalker
from besser.utilities.buml_code_builder import domain_model_to_code
from besser.BUML.metamodel.structural import DomainModel
from .PlantUMLLexer import PlantUMLLexer
from .PlantUMLParser import PlantUMLParser
from .plantUML_buml_listener import BUMLGenerationListener

def plantuml_to_buml(plantUML_model_path:str, buml_file_path:str = None):
    """Transforms a PlantUML model into a B-UML model.

    Args:
        plantUML_model_path (str): The path to the file containing the PlantUML code.
        buml_file_path (str, optional): the path of the file produced with the base 
                code to build the B-UML model (None as default).

    Returns:
        BUML_model (DomainModel): the B-UML model object.
    """
    lexer = PlantUMLLexer(FileStream(plantUML_model_path))
    parser = PlantUMLParser(CommonTokenStream(lexer))
    parse_tree = parser.domainModel()
    listen = BUMLGenerationListener()
    walker = ParseTreeWalker()
    walker.walk(listen, parse_tree)
    domain_model: DomainModel = listen.get_buml_model()
    if buml_file_path is not None:
        domain_model_to_code(model=domain_model, file_path=buml_file_path)
    return domain_model
