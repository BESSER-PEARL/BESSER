import sys
from antlr4 import *
from besser.BUML.notations.ocl.OCLsLexer import OCLsLexer
from besser.BUML.notations.ocl.OCLsParser import OCLsParser
from besser.BUML.notations.ocl.OCLsListener import OCLsListener
from besser.BUML.notations.ocl.RootHandler import Root_Handler

import unittest

if __name__ == '__main__':
    # main(sys.argv)
    # ocl = "context meeting inv: self.start < self.end and self.start < 5  or self.end > 10"
    # ocl = "context meeting inv: self.start < self.end"
    # ocl = "context temp inv invariant_LoyaltyProgram16 : self.participants->forAll( i_Customer : Customer | i_Customer.age() <= 70 )"
    ocl = "context temp inv invariant_LoyaltyProgram16 : self.employee->select(age > 50)"
    ocl = "context temp inv invariant_LoyaltyProgram16 : self.employee->select( p | p.age>50 )"
    ocl = "context temp inv invariant_LoyaltyProgram16 : self.employee->select( p : Personne | p.age>50)"
    ocl = "context temp inv invariant_LoyaltyProgram16 : self.employee->forAll(age<10)"
    ocl = "context temp inv invariant_LoyaltyProgram16 : self.employee->exists(age<10)"
    ocl = "context temp inv invariant_LoyaltyProgram16 : self.employee->collect(age)  = Bag{10,5,10,7} "

    input_stream = InputStream(ocl)
    rootHandler = Root_Handler()
    lexer = OCLsLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = OCLsParser(stream)
    tree = parser.oclFile()
    listener = OCLsListener(rootHandler)
    walker = ParseTreeWalker()
    walker.walk(listener,tree)
    listener.print()
    print("in main")