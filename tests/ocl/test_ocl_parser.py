from tests.object.library_object import library_model, object_model
from besser.BUML.notations.ocl.BOCLLexer import BOCLLexer
from besser.BUML.notations.ocl.BOCLParser import BOCLParser
from besser.BUML.notations.ocl.BOCLListener import BOCLListener
from besser.BUML.notations.ocl.RootHandler import Root_Handler
from antlr4 import *

class OCLParser():
      def __init__(self, dm, om) :
            self.dm = dm
            self.om = om
            
      def parse (self, ocl):
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

def test1():
    parser = OCLParser(library_model, object_model)
    cons = list(library_model.constraints)
    constraint = cons[0]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test2():
    parser = OCLParser(library_model, object_model)
    cons = list(library_model.constraints)
    constraint = cons[1]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test3():
    parser = OCLParser(library_model, object_model)

    cons = list(library_model.constraints)
    constraint = cons[2]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test4():
    parser = OCLParser(library_model, object_model)

    cons = list(library_model.constraints)
    constraint = cons[3]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test5():
    parser = OCLParser(library_model, object_model)

    cons = list(library_model.constraints)
    constraint = cons[4]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test6():
    parser = OCLParser(library_model, object_model)

    cons = list(library_model.constraints)
    constraint = cons[4]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test7():
    parser = OCLParser(library_model, object_model)

    cons = list(library_model.constraints)
    constraint = cons[5]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test8():
    parser = OCLParser(library_model, object_model)

    cons = list(library_model.constraints)
    constraint = cons[6]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None

    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test9():
    parser = OCLParser(library_model, object_model)

    cons = list(library_model.constraints)
    constraint = cons[7]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test10():
    parser = OCLParser(library_model, object_model)

    cons = list(library_model.constraints)
    constraint = cons[8]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test11():
    parser = OCLParser(library_model, object_model)

    cons = list(library_model.constraints)
    constraint = cons[9]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test12():
    parser = OCLParser(library_model, object_model)

    cons = list(library_model.constraints)
    constraint = cons[10]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

if __name__ == '__main__':
    pass