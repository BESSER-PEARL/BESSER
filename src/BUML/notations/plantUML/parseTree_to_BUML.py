from antlr4 import CommonTokenStream, FileStream, ParseTreeWalker
from .PlantUMLLexer import PlantUMLLexer
from .PlantUMLParser import PlantUMLParser
from .plantUML_buml_listener import BUMLGenerationListener
from BUML.metamodel.structural import DomainModel
import os

class PlantUMLToBUML:

    def __init__(self, plantUML_model:str):
        self.plantUML_model: str = plantUML_model
        self.BUML_model: DomainModel = None

    def generate_BUML_model(self):
        lexer = PlantUMLLexer(FileStream(self.plantUML_model))
        parser = PlantUMLParser(CommonTokenStream(lexer))
        parse_tree = parser.domainModel()
        # file creation
        if not os.path.exists("buml"):
            os.makedirs("buml")
        output = open("buml/buml_model.py","w+")
        listen = BUMLGenerationListener(output)
        walker = ParseTreeWalker()
        walker.walk(listen, parse_tree)
        output.close()
        # model creation
        namespace = {}
        with open("buml/buml_model.py", 'r') as model_code:
            code = model_code.read()
            exec(code, namespace)
        self.BUML_model = namespace.get('domain')
        return(self.BUML_model)

    @property
    def BUML_model(self) -> DomainModel:
        return self.__BUML_model

    @BUML_model.setter
    def BUML_model(self, BUML_model: DomainModel):
        self.__BUML_model = BUML_model
