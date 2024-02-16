from antlr4 import *
from besser.BUML.notations.od.ODLexer import ODLexer
from besser.BUML.notations.od.ODParser import ODParser
from besser.BUML.notations.od.ODListener import ODListener
def test_simple_OD():
    od = "../../besser/BUML/notations/od/libraryObjectDiagram.plantuml"
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
    od = "../../besser/BUML/notations/od/libraryObjectDiagram.plantuml"
    all_objs = []
    input_stream = FileStream(od)
    lexer = ODLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = ODParser(stream)
    tree = parser.objectDiagram()
    listener = ODListener(all_objs)
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    assert len(all_objs) == 4
def test_number_of_slots_for_libaray():
    od = "../../besser/BUML/notations/od/libraryObjectDiagram.plantuml"
    all_objs = []
    input_stream = FileStream(od)
    lexer = ODLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = ODParser(stream)
    tree = parser.objectDiagram()
    listener = ODListener(all_objs)
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    assert len(all_objs[0].get_slots()) == 3

def test_number_of_slots_for_libaray1():
    od = "../../besser/BUML/notations/od/libraryObjectDiagram.plantuml"
    all_objs = []
    input_stream = FileStream(od)
    lexer = ODLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = ODParser(stream)
    tree = parser.objectDiagram()
    listener = ODListener(all_objs)
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    assert len(all_objs[1].get_slots()) == 3

def test_number_of_slots_for_book():
    od = "../../besser/BUML/notations/od/libraryObjectDiagram.plantuml"
    all_objs = []
    input_stream = FileStream(od)
    lexer = ODLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = ODParser(stream)
    tree = parser.objectDiagram()
    listener = ODListener(all_objs)
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    assert len(all_objs[2].get_slots()) == 3

if __name__ == '__main__':
    pass
