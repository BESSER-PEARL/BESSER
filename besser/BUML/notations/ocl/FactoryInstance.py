from besser.BUML.metamodel.ocl.ocl import *

class Factory:

    def __init__(self,context):
        self.context = context

    def checkInAttributes(self,name,context):
        for attrib in context.attributes:
            if name == attrib.name:
                return attrib
    def checkInAssociation(self,name,context):
        for end in context.all_association_ends():
            # for end in association.ends:
                if name == end.name:
                    return end

    def handleProp(self,name,iterator):
        if "." in name:
            name = name.split('.')[1]

        prop = self.checkInAttributes(name,self.context)
        if prop is None:
            prop = self.checkInAssociation(name,self.context)
        if iterator is not None and prop is None and len(iterator)>0:
            prop = self.checkInAttributes(name,iterator[-1].get_iterator[-1].type)
            if prop is None:
                prop = self.checkInAssociation(name, iterator[-1].get_iterator[-1].type)
            pass
        return prop

    def create_property_Call_Expression(self,name, type, iterators=None):
        prop = self.handleProp(name,iterators)
        if prop is not None:
            return prop
        else:
            raise Exception("Property "+name+ " not found in class "+str(self.context.name))

    def create_date_literal_expression(self,name,value):
        return DateLiteralExpression(name,value)
    def create_variable_expression(self,name,type):
        var = VariableExp(name,type)
        return var

    def create_boolean_literal_expression(self, name, val):
        return BooleanLiteralExpression(name, val)
    def create_integer_literal_expression(self, name, val):
        return IntegerLiteralExpression(name, val)
    def create_string_literal_expression(self,name,val):
        return StringLiteralExpression(name,val)

    def create_real_literal_expression(self, name, val):
        return RealLiteralExpression(name, val)
        pass

    def create_if_else_exp(self,name, type):
        return IfExp(name,type)
    def create_set_type(self):
        return SetType("Set")
    def create_ordered_set_type(self):
        return OrderedSetType("OrderedSetType")
    def create_sub_ordered_set_type(self):
        return OrderedSetType("SubOrderedSetType")

    def create_operation_call_expression(self, leftpart = None, rightpart= None, infixOperator= None, inBetweenOp=None,beforeOp = None,isleftNone=False,name = None):
        if inBetweenOp is None and isleftNone is False and name is None:
            if beforeOp is None:
                return OperationCallExpression("Operation", infixOperator.get_infix_operator(),
                                           [leftpart, infixOperator, rightpart])
            else:
                return OperationCallExpression("Operation", infixOperator.get_infix_operator(),
                                           [infixOperator, rightpart])
        elif inBetweenOp is None and isleftNone is True and name is None:
            return OperationCallExpression("Operation", infixOperator.get_infix_operator(),
                                           [infixOperator, rightpart])
        elif inBetweenOp is not None and name is None:
            if beforeOp is None:
                oce = OperationCallExpression("Operation", infixOperator.get_infix_operator(),
                                               [leftpart, infixOperator, rightpart])
                oce.referredOperation=inBetweenOp
                return oce
            else:
                oce = OperationCallExpression("Operation", infixOperator.get_infix_operator(),
                                              [beforeOp,leftpart, infixOperator, rightpart])
                oce.referredOperation=inBetweenOp
                return oce
        else:
            return OperationCallExpression(name=name, operation =name,arguments=[] )

    def create_type_exp(self,classifier):
        return TypeExp(classifier,classifier)

    def create_loop_expression(self,collectionOperator):
        return LoopExp(collectionOperator,None)
        pass
    def create_bag_type(self):
        return BagType("BagType")
    def create_collection_literal_expression(self,name,type):
        return CollectionLiteralExp(name = name,type=type)
    def create_sequence_type(self):
        return SequenceType("SequenceType")
    def create_sub_sequence_type(self):
        return SequenceType("SubSequenceType")
    def create_collection_item(self,name ,item):
        return CollectionItem(name,item)
    def create_iterator_expression(self,name,type):
        return IteratorExp(name,type)
        pass

    def create_infix_operator(self,op):
        return InfixOperator(op)
