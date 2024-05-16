from antlr4 import *
from besser.BUML.notations.objectPlantUML.ODLexer import ODLexer
from besser.BUML.notations.objectPlantUML.ODParser import ODParser
from besser.BUML.notations.objectPlantUML.ODListener import ODListener
def test_simple_OD():
    od = "libraryObjectDiagram.plantuml"
    all_objs = []
    input_stream = FileStream(od)
    lexer = ODLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = ODParser(stream)
    tree = parser.objectDiagram()
    listener = ODListener(all_objs)
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    assert parser.getNumberOfSyntaxErrors() == 0

def test_number_of_objects():
    od = "libraryObjectDiagram.plantuml"
    all_objs = []
    input_stream = FileStream(od)
    lexer = ODLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = ODParser(stream)
    tree = parser.objectDiagram()
    listener = ODListener(all_objs)
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    assert len(all_objs) == 5
def test_number_of_slots_for_libaray():
    od = "libraryObjectDiagram.plantuml"
    all_objs = []
    input_stream = FileStream(od)
    lexer = ODLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = ODParser(stream)
    tree = parser.objectDiagram()
    listener = ODListener(all_objs)
    walker = ParseTreeWalker()
    walker.walk(listener, tree)

    assert len(all_objs[0].slots) == 2

def test_number_of_slots_for_libaray1():
    od = "libraryObjectDiagram.plantuml"
    all_objs = []
    input_stream = FileStream(od)
    lexer = ODLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = ODParser(stream)
    tree = parser.objectDiagram()
    listener = ODListener(all_objs)
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    assert len(all_objs[1].slots) == 3

def test_number_of_slots_for_book():
    od = "libraryObjectDiagram.plantuml"
    all_objs = []
    input_stream = FileStream(od)
    lexer = ODLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = ODParser(stream)
    tree = parser.objectDiagram()
    listener = ODListener(all_objs)
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    assert len(all_objs[2].slots) == 2

if __name__ == '__main__':
    pass
