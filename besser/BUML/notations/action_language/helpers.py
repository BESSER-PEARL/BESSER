from besser.BUML.metamodel.structural import Class

from besser.BUML.metamodel.action_language.action_language import Type, FunctionType, IntType, StringType, RealType, \
    BoolType, SequenceType, ObjectType, AnyType, OptionalType, EnumType, TypeUnion, Nothing, NoType


class UnknownClassifier(ObjectType, EnumType):
    def __init__(self, name: str):
        ObjectType.__init__(self, None)
        EnumType.__init__(self, None)
        self.__name = name

    def name(self) -> str:
        return self.__name


base_classes = {
    'int':          IntType(),
    'float':        RealType(),
    'str':          StringType(),
    'bool':         BoolType(),
    'time':         AnyType(),
    'date':         AnyType(),
    'datetime':     AnyType(),
    'timedelta':    AnyType(),
    'any':          AnyType()
}

def functions_for_sequence_type(sequence_type: SequenceType):
    elem = sequence_type.elementsType
    elem_to_bool = FunctionType([elem],BoolType())
    elem_to_any = FunctionType([elem], AnyType())
    reduce = FunctionType([elem, elem], elem)
    sequence_functions = {
        'size':         FunctionType([],                IntType()),
        'is_empty':     FunctionType([],                BoolType()),
        'add':          FunctionType([elem],  None),
        'remove':       FunctionType([elem],  None),
        'contains':     FunctionType([elem],            BoolType()),
        'filter':       FunctionType([elem_to_bool],    sequence_type),
        'forall':       FunctionType([elem_to_bool],    BoolType()),
        'exists':       FunctionType([elem_to_bool],    BoolType()),
        'one':          FunctionType([elem_to_bool],    BoolType()),
        'is_unique':    FunctionType([elem_to_any],     BoolType()),
        'map':          FunctionType([elem_to_any],     BoolType()),
        'reduce':       FunctionType([reduce, elem],    AnyType())
    }
    return sequence_functions

def get_typename(t: Type) -> str:
    if t == AnyType():
        return "any"
    if t in base_classes.values():
        for name in base_classes:
            if base_classes[name] == t:
                return name
    if isinstance(t, ObjectType):
        return t.clazz.name
    if isinstance(t, EnumType):
        return t.enum.name
    if isinstance(t, SequenceType):
        return f"[{get_typename(t.elementsType)}]"
    if isinstance(t, OptionalType):
        return f"{get_typename(t.type)}?"
    if isinstance(t, FunctionType):
        param_typenames = list(map(lambda param_type: get_typename(param_type), t.params_type))
        return f"({", ".join(param_typenames)}) -> {get_typename(t.return_type)}"
    if isinstance(t, TypeUnion):
        return f"({get_typename(t.a)}|{get_typename(t.b)})"
    if isinstance(t, Nothing):
        return f"Nothing"
    return f"No Type Infered"