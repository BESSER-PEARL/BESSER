import sys
from antlr4 import *
from besser.BUML.notations.ocl.OCLsLexer import OCLsLexer
from besser.BUML.notations.ocl.OCLsParser import OCLsParser
from besser.BUML.notations.ocl.OCLsListener import OCLsListener
from besser.BUML.notations.ocl.RootHandler import Root_Handler
import unittest

class TestOclInterpreter(unittest.TestCase):

    def test_one(self):
        ocl = "context meeting inv: self.start < self.end and self.start < 5  or self.end > 10"
        input_stream = InputStream(ocl)
        rootHandler = Root_Handler()
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        listener = OCLsListener(rootHandler)
        walker = ParseTreeWalker()
        walker.walk(listener, tree)
        assert rootHandler.get_root() !=None
    def test_two(self):
        ocl = "context meeting inv: self.start < 5 "
        input_stream = InputStream(ocl)
        rootHandler = Root_Handler()
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        listener = OCLsListener(rootHandler)
        walker = ParseTreeWalker()
        walker.walk(listener, tree)
        assert rootHandler.get_root() !=None

    def test_three(self):
        ocl = "context meeting inv: self.end > 5 "
        input_stream = InputStream(ocl)
        rootHandler = Root_Handler()
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        listener = OCLsListener(rootHandler)
        walker = ParseTreeWalker()
        walker.walk(listener, tree)
        assert rootHandler.get_root() !=None
    def test_four(self):
        ocl = "context temp inv invariant_LoyaltyProgram16 : self.participants->forAll( i_Customer : Customer | i_Customer.age() <= 70 )"
        input_stream = InputStream(ocl)
        rootHandler = Root_Handler()
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        listener = OCLsListener(rootHandler)
        walker = ParseTreeWalker()
        walker.walk(listener, tree)
        assert rootHandler.get_root() !=None
    def test_five(self):
        ocl = "context temp inv invariant_LoyaltyProgram16 : self.employee->select(age > 50)"
        input_stream = InputStream(ocl)
        rootHandler = Root_Handler()
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        listener = OCLsListener(rootHandler)
        walker = ParseTreeWalker()
        walker.walk(listener, tree)
        assert rootHandler.get_root() !=None

    def test_six(self):
        ocl = "context temp inv invariant_LoyaltyProgram16 : self.employee->select( p | p.age>50 )"
        input_stream = InputStream(ocl)
        rootHandler = Root_Handler()
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        listener = OCLsListener(rootHandler)
        walker = ParseTreeWalker()
        walker.walk(listener, tree)
        assert rootHandler.get_root() !=None
    def test_seven(self):
        ocl = "context temp inv invariant_LoyaltyProgram16 : self.employee->select( p : Personne | p.age>50)"
        input_stream = InputStream(ocl)
        rootHandler = Root_Handler()
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        listener = OCLsListener(rootHandler)
        walker = ParseTreeWalker()
        walker.walk(listener, tree)
        assert rootHandler.get_root() !=None
    def test_eight(self):
        ocl = "context temp inv invariant_LoyaltyProgram16 : self.employee->forAll(age<10)"
        input_stream = InputStream(ocl)
        rootHandler = Root_Handler()
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        listener = OCLsListener(rootHandler)
        walker = ParseTreeWalker()
        walker.walk(listener, tree)
        assert rootHandler.get_root() !=None

    def test_nine(self):
        ocl = "context temp inv invariant_LoyaltyProgram16 : self.employee->exists(age<10)"
        input_stream = InputStream(ocl)
        rootHandler = Root_Handler()
        lexer = OCLsLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = OCLsParser(stream)
        tree = parser.oclFile()
        listener = OCLsListener(rootHandler)
        walker = ParseTreeWalker()
        walker.walk(listener, tree)
        assert rootHandler.get_root() !=None

if __name__ == '__main__':
    pass