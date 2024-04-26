from besser.BUML.notations.ocl.BOCLLexer import BOCLLexer
from besser.BUML.notations.ocl.BOCLParser import BOCLParser
from besser.BUML.notations.ocl.BOCLListener import BOCLListener
from besser.BUML.notations.ocl.RootHandler import Root_Handler
from antlr4 import *

class OCLWrapper:

    def __init__(self,dm,om):
        self.dm = dm
        self.om = om

    # this method  parse the OCL expression and create the Tree
    # Arguments:
    # OCL Object that contains the expressions and the class
    def parse(self,ocl):

        try:
            input_stream = InputStream(ocl.expression)
            rootHandler = Root_Handler(ocl,self.dm,self.om)
            lexer = BOCLLexer(input_stream)
            stream = CommonTokenStream(lexer)
            parser = BOCLParser(stream)
            tree = parser.oclFile()
            listener = BOCLListener(rootHandler)
            walker = ParseTreeWalker()
            walker.walk(listener,tree)
            return True
        except Exception as e:
            print(e)
            return False
