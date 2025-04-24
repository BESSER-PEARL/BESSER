from tests.BUML.metamodel.object.library_object import library_model, object_model
from besser.BUML.notations.ocl.OCLParserWrapper import OCLParserWrapper

def test1():
    parser = OCLParserWrapper(library_model, object_model)
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
    parser = OCLParserWrapper(library_model, object_model)
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
    parser = OCLParserWrapper(library_model, object_model)

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
    parser = OCLParserWrapper(library_model, object_model)

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
    parser = OCLParserWrapper(library_model, object_model)

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
    parser = OCLParserWrapper(library_model, object_model)

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
    parser = OCLParserWrapper(library_model, object_model)

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
    parser = OCLParserWrapper(library_model, object_model)

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
    parser = OCLParserWrapper(library_model, object_model)

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
    parser = OCLParserWrapper(library_model, object_model)

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
    parser = OCLParserWrapper(library_model, object_model)

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
    parser = OCLParserWrapper(library_model, object_model)

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