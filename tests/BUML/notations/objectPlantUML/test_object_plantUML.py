import os
import pytest
from antlr4 import *
from besser.BUML.notations.objectPlantUML.ODLexer import ODLexer
from besser.BUML.notations.objectPlantUML.ODParser import ODParser
from besser.BUML.notations.objectPlantUML.ODListener import ODListener

model_dir = os.path.dirname(os.path.abspath(__file__))
od = os.path.join(model_dir, "libraryObjectDiagram.plantuml")

@pytest.mark.skip(reason="The PlantUML parser for Object models dont create DataValue objects, we need to fix it")
def test_simple_OD(): 
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

@pytest.mark.skip(reason="The PlantUML parser for Object models dont create DataValue objects, we need to fix it")
def test_number_of_objects():
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

@pytest.mark.skip(reason="The PlantUML parser for Object models dont create DataValue objects, we need to fix it")
def test_number_of_slots_for_libaray():
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

@pytest.mark.skip(reason="The PlantUML parser for Object models dont create DataValue objects, we need to fix it")
def test_number_of_slots_for_libaray1():
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

@pytest.mark.skip(reason="The PlantUML parser for Object models dont create DataValue objects, we need to fix it")
def test_number_of_slots_for_book():
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
