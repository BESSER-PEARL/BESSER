from besser.BUML.notations.ocl.BOCLLexer import BOCLLexer
from besser.BUML.notations.ocl.BOCLParser import BOCLParser
from besser.BUML.notations.ocl.BOCLListener import BOCLListener
from besser.BUML.notations.ocl.RootHandler import Root_Handler
from antlr4 import *

class OCLParserWrapper:
    def __init__(self, dm, om):
        self.dm = dm
        self.om = om


    def parse(self, ocl):
        input_stream = InputStream(ocl.expression)
        rootHandler = Root_Handler(self.dm, self.om)
        rootHandler.set_context(ocl.context)
        lexer = BOCLLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = BOCLParser(stream)
        tree = parser.oclFile()
        listener = BOCLListener(rootHandler)
        listener.preprocess(ocl.expression)
        walker = ParseTreeWalker()
        walker.walk(listener, tree)


        return True