from tests.object.library_object import library_model,object_model
from besser.BUML.notations.ocl.OCLWrapper import OCLWrapper

def test1():
    wrapper = OCLWrapper(library_model,object_model)

    cons = list(library_model.constraints)
    constraint = cons[0]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = wrapper.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')

    assert res == True
def test2():
    wrapper = OCLWrapper(library_model,object_model)

    cons = list(library_model.constraints)
    constraint = cons[1]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = wrapper.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')

    assert res == True
def test3():
    wrapper = OCLWrapper(library_model,object_model)

    cons = list(library_model.constraints)
    constraint = cons[2]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = wrapper.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test4():
    wrapper = OCLWrapper(library_model,object_model)

    cons = list(library_model.constraints)
    constraint = cons[3]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = wrapper.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')

    assert res == True
def test5():
    wrapper = OCLWrapper(library_model,object_model)

    cons = list(library_model.constraints)
    constraint = cons[4]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = wrapper.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')

    assert res == True

def test5():
    wrapper = OCLWrapper(library_model,object_model)

    cons = list(library_model.constraints)
    constraint = cons[4]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = wrapper.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')

    assert res == True
def test6():
    wrapper = OCLWrapper(library_model,object_model)

    cons = list(library_model.constraints)
    constraint = cons[5]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = wrapper.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test7():
    wrapper = OCLWrapper(library_model,object_model)

    cons = list(library_model.constraints)
    constraint = cons[6]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = wrapper.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None

    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test8():
    wrapper = OCLWrapper(library_model,object_model)

    cons = list(library_model.constraints)
    constraint = cons[7]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = wrapper.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True
def test9():
    wrapper = OCLWrapper(library_model,object_model)

    cons = list(library_model.constraints)
    constraint = cons[8]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = wrapper.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test10():
    wrapper = OCLWrapper(library_model,object_model)

    cons = list(library_model.constraints)
    constraint = cons[9]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = wrapper.parse(constraint)
    except Exception as error:
            print('\x1b[0;30;41m' + 'Exception Occured! Info:' +str(error)  + '\x1b[0m')
            res = None
    if res:
            print('\x1b[6;30;42m' + 'Parsed Correctly' + '\x1b[0m')
    elif res == False:
            print('\x1b[0;30;41m' + 'Cannot be Parsed' + '\x1b[0m')
    assert res == True

def test11():
    wrapper = OCLWrapper(library_model,object_model)

    cons = list(library_model.constraints)
    constraint = cons[10]
    print("Query: " + str(constraint.expression),end = ": ")
    res = None
    try:
        res = wrapper.parse(constraint)
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