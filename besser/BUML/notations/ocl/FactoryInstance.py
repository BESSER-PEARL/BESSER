from besser.BUML.metamodel.ocl.rules import *

class Factory:

    def create_variable_expression(self,name,type):
        var = VariableExp(name,type)
        return var

    def create_integer_literal_expression(self, name, val):
        return IntegerLiteralExpression(name, val)
        pass

    def create_real_literal_expression(self, name, val):
        return RealLiteralExpression(name, val)
        pass

    def create_operation_call_expression(self, leftpart, rightpart, infixOperator, inBetweenOp=None):
        if inBetweenOp is None:
            return OperationCallExpression("Operation", infixOperator.get_infix_operator(),
                                           [leftpart, infixOperator, rightpart])
        else:
            return OperationCallExpression("Operation", infixOperator.get_infix_operator(),
                                           [inBetweenOp, leftpart, infixOperator, rightpart])




    def create_loop_expression(self,collectionOperator):
        return LoopExp(collectionOperator,None)
        pass
    def create_collection_literal_expression(self,type):
        return CollectionLiteralExp(name = "NP",type=type)
    def create_collection_item(self,name ,item):
        return CollectionItem(name,item)
    def create_iterator_expression(self,name,type):
        return IteratorExp(name,type)
        pass

    def create_infix_operator(self,op):
        return InfixOperator(op)
