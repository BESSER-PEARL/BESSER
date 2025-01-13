from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker
from besser.BUML.notations.ocl.BOCLLexer import BOCLLexer
from besser.BUML.notations.ocl.BOCLParser import BOCLParser
from besser.BUML.notations.ocl.BOCLListener import BOCLListener
from besser.BUML.notations.ocl.RootHandler import Root_Handler
from besser.BUML.metamodel.structural import DomainModel
from besser.BUML.metamodel.object import ObjectModel


def ocl_parser(ocl, dm: DomainModel, om: ObjectModel):
    """
    Check OCL constraints against domain model
    
    Args:
        ocl: OCL constraints to check
        dm: Domain model
        om: Object model
    
    Returns:
        bool: True if parsing successful, False otherwise
    """
    input_stream = InputStream(ocl.expression)
    root_handler = Root_Handler(ocl, dm, om)
    lexer = BOCLLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = BOCLParser(stream)
    tree = parser.oclFile()
    listener = BOCLListener(root_handler)
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
