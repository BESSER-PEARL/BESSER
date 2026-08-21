from tests.BUML.metamodel.object.library_object import library_model, object_model
from bocl.OCLWrapper import OCLWrapper

def test1():
    parser = OCLWrapper(library_model, object_model)
    cons = sorted(list(library_model.constraints), key=lambda c: c.name)
    constraint = cons[0]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.evaluate(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res is not None:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    else:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res is not None

def test2():
    parser = OCLWrapper(library_model, object_model)
    cons = sorted(list(library_model.constraints), key=lambda c: c.name)
    constraint = cons[1]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.evaluate(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res is not None:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    else:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res is not None

def test3():
    parser = OCLWrapper(library_model, object_model)

    cons = sorted(list(library_model.constraints), key=lambda c: c.name)
    constraint = cons[2]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.evaluate(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res is not None:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    else:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res is not None

def test4():
    parser = OCLWrapper(library_model, object_model)

    cons = sorted(list(library_model.constraints), key=lambda c: c.name)
    constraint = cons[3]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.evaluate(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res is not None:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    else:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res is not None

def test5():
    parser = OCLWrapper(library_model, object_model)

    cons = sorted(list(library_model.constraints), key=lambda c: c.name)
    constraint = cons[4]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.evaluate(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res is not None:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    else:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res is not None

def test6():
    parser = OCLWrapper(library_model, object_model)

    cons = sorted(list(library_model.constraints), key=lambda c: c.name)
    constraint = cons[4]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.evaluate(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res is not None:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    else:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res is not None

def test7():
    parser = OCLWrapper(library_model, object_model)

    cons = sorted(list(library_model.constraints), key=lambda c: c.name)
    constraint = cons[5]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.evaluate(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res is not None:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    else:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res is not None

def test8():
    parser = OCLWrapper(library_model, object_model)

    cons = sorted(list(library_model.constraints), key=lambda c: c.name)
    constraint = cons[6]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.evaluate(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None

    if res is not None:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    else:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res is not None

def test9():
    parser = OCLWrapper(library_model, object_model)

    cons = sorted(list(library_model.constraints), key=lambda c: c.name)
    constraint = cons[7]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.evaluate(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res is not None:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    else:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res is not None

def test10():
    parser = OCLWrapper(library_model, object_model)

    cons = sorted(list(library_model.constraints), key=lambda c: c.name)
    constraint = cons[8]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.evaluate(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res is not None:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    else:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res is not None

def test11():
    parser = OCLWrapper(library_model, object_model)

    cons = sorted(list(library_model.constraints), key=lambda c: c.name)
    constraint = cons[9]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.evaluate(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res is not None:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    else:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res is not None

def test12():
    parser = OCLWrapper(library_model, object_model)

    cons = sorted(list(library_model.constraints), key=lambda c: c.name)
    constraint = cons[10]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = parser.evaluate(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res is not None:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    else:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res is not None

if __name__ == '__main__':
    pass
