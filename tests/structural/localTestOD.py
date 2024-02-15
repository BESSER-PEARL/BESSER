import sys
from antlr4 import *
from besser.BUML.notations.od.ODLexer import ODLexer
from besser.BUML.notations.od.ODParser import ODParser
from besser.BUML.notations.od.ODListener import ODListener
import unittest

if __name__ == '__main__':
    # main(sys.argv)
    od = "../../besser/BUML/notations/od/libraryObjectDiagram.plantuml"
    all_objs=[]
    input_stream = FileStream(od)
    lexer = ODLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = ODParser(stream)
    tree = parser.objectDiagram()
    listener = ODListener(all_objs)
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
 