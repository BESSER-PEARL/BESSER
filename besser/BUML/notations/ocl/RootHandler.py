from besser.BUML.notations.ocl.FactoryInstance import Factory
from besser.BUML.notations.ocl.InitRoot import InitRoot
from besser.BUML.notations.ocl.PreConditionRoot import PreConditionRoot
from besser.BUML.notations.ocl.PostConditionRoot import PostConditionRoot
from besser.BUML.notations.ocl.ConstraintRoot import ConstraintRoot


class Root_Handler:
    def __init__(self,  dm=None, om=None):

        self.root = ConstraintRoot(None,None)
        self.init_roots: [InitRoot] = []
        self.pre_conditions_roots: [PreConditionRoot] = []
        self.post_conditions_roots: [PostConditionRoot] = []

        self.dm = dm
        self.om = om
        self.factory = Factory(None)
        self.all = []
        self.context = None
        self.invariant = False
        self.pre = False
        self.post = False
        self.if_else_roots = []
        self.context_name = ""
    def set_context(self,context):
        self.factory.context= context
        self.context = context
        self.context_name = context

    def handle_post_root(self,context,function_name,root):
        self.post_conditions_roots.append(PostConditionRoot(context,function_name,root))
        self.reset_root()

    def handle_pre_root(self,context,function_name,root):
        self.pre_conditions_roots.append(PreConditionRoot(context,function_name,root))
        self.reset_root()

    def handle_init_root(self,context, variable_name,root,type):
        self.add_to_init_root(InitRoot(context,variable_name,root,type))
        self.reset_root()

    def set_root_context(self, context):
        self.root.context= context
    def add_to_init_root(self, root):
        self.init_roots.append(root)

    def add_to_pre_condition_root(self, root):
        self.pre_conditions_roots.append(root)

    def add_to_post_condition_root(self, root):
        self.post_conditions_roots.append(root)

    def reset_root(self):
        self.root = ConstraintRoot(None,None)

    def get_root(self):
        return self.root.root

    def get_inv(self):
        return self.invariant

    def set_inv(self, inv):
        self.invariant = inv

    def set_post(self, post):
        self.post = post

    def get_post(self):
        return self.post

    def set_body(self, body):
        self.body = body

    def get_body(self):
        return self.body

    def set_pre(self, pre):
        self.pre = pre

    def get_pre(self):
        return self.pre

    def create_property(self, prop):
        return self.factory.create_property_Call_Expression(prop, "NP")

    def handle_property(self, prop):
        self.add_to_root(self.factory.create_property_Call_Expression(prop, "NP"))

    def get_context_name(self):
        return self.context_name

    def set_context_name(self, name):
        self.context_name = name

    def create_if_else_exp(self, name, type):
        self.if_else_roots.append(None)
        return self.factory.create_if_else_exp(name, type)

    def pop(self):
        if len(self.all) != 0:
            self.add_to_root(self.all.pop())

    def checkNumberOrVariable(self, txt):
        if txt.isnumeric():
            if "." in txt:
                return "real"
            else:
                return "int"
        elif "'" in txt:
            return "str"
        elif txt == "True" or txt == "False":
            return "bool"
        elif "Date::" in txt:
            return "date"
        else:
            return "var"

    def _add_root(self, root, op):
        if len(self.all) == 0:
            if root is None:
                root = op
            else:
                op.source = root
                root = op
        else:
            self.all[-1].add_body(op)
        return root

    def getTop(self, last=False):
        currentHead = self.if_else_roots.pop()
        if not last:
            self.if_else_roots.append(None)
        return currentHead

    def add_to_root(self, op):
        self.last_added = op
        if len(self.if_else_roots) != 0:
            self.if_else_roots.append(self._add_root(self.if_else_roots.pop(), op))
            pass
        else:
            self.root.root = self._add_root(self.root.root, op)
        pass

    def print(self):
        self.handlePrint(self.root.root)

    def handle_ID(self, id):
        varID = self.factory.create_variable_expression(id, None)
        self.add_to_root(varID)

    def create_ordered_set(self):
        type = self.factory.create_ordered_set_type()
        return self.factory.create_collection_literal_expression("orderedSet", type)

    def create_set(self):
        type = self.factory.create_set_type()
        return self.factory.create_collection_literal_expression("set", type)

    def create_sub_ordered_set(self):
        type = self.factory.create_ordered_set_type()
        return self.factory.create_collection_literal_expression("subOrderedSet", type)

    def create_sequence(self):
        type = self.factory.create_sequence_type()
        return self.factory.create_collection_literal_expression("sequence", type)

    def create_sub_sequence(self):
        type = self.factory.create_sub_sequence_type()
        return self.factory.create_collection_literal_expression("subsequence", type)

    def create_bag(self):
        type = self.factory.create_bag_type()
        return self.factory.create_collection_literal_expression("bag", type=type)

    def create_type_exp(self, classifier):
        return self.factory.create_type_exp(classifier)

    def create_infinix_op(self, op):
        self.factory.create_infix_operator(op)

    def handle_and_with_function_call(self, text):
        op = None
        inF = None
        if text[0:3] == "and":
            inF = self.factory.create_infix_operator("AND")
            op = self.factory.create_operation_call_expression(name="AND")
        if text[0:2] == "or":
            inF = self.factory.create_infix_operator("OR")
            op = self.factory.create_operation_call_expression(name="OR")
        op.arguments.append(inF)
        self.add_to_root(op)

    def create_operation_call_exp(self, name):
        return self.factory.create_operation_call_expression(name=name)

    def get_factory(self):
        return self.factory

    def handle_bag(self, collectionLiteral, operator):
        infixOperator = None
        if operator is not None:
            infixOperator = self.factory.create_infix_operator(operator)
        if infixOperator is not None:
            operationCallExp = self.factory.create_operation_call_expression(None, collectionLiteral, infixOperator,
                                                                             None, True)
        self.add_to_root(operationCallExp)

    def handle_adding_to_root(self, expression, op=None):
        if op is not None:
            expression.referredOperation(op)
        self.add_to_root(expression)

    def handlePrimaryExp(self, primaryExp, operator):
        pass

    def handle_collection(self, oclExp):
        collectionOperator = None
        if "forAll" in oclExp[0:8]:
            collectionOperator = "forAll"
        elif "exists" in oclExp[0:8]:
            collectionOperator = "exists"
        elif "collect" in oclExp[0:9]:
            collectionOperator = "collect"
        elif "select" in oclExp[0:8]:
            collectionOperator = "select"
        elif "reject" in oclExp[0:8]:
            collectionOperator = "reject"

        # print("Collection Operator: " + collectionOperator)
        self.handleColl(oclExp, collectionOperator)

    def handle_single_variable(self, variable_name, sign):
        op = self.factory.create_operation_call_expression(name="operation")
        infinix_op = self.factory.create_infix_operator(sign)
        prop = self.factory.create_property_Call_Expression(variable_name, "NI")
        op.arguments.append(infinix_op)
        op.arguments.append(prop)
        self.add_to_root(op)

    def verify(self, item):
        referredOP = None
        if 'and' in item[0:3]:
            item = item[3:]
            referredOP = 'AND'
        if 'or' in item[0:2]:
            item = item[2:]
            referredOP = 'OR'
        if 'allInstances' in item[0:12]:
            return
        if referredOP is None:
            prop = self.factory.create_property_Call_Expression(item, 'NI')
            # prop= self.factory.create_property_Call_Expression(item,'NI')
            self.add_to_root(prop)
        else:
            opCallExp = self.factory.create_operation_call_expression(name=referredOP)
            opCallExp.referredOperation = self.factory.create_infix_operator(referredOP)
            self.add_to_root(opCallExp)

    def getClass(self, name):
        for type in self.dm.types:
            if name == type.name:
                return type
        raise Exception("Class not found")

    def handleColl(self, forAllExp, collectionOperator):
        self.all.append(self.factory.create_loop_expression(collectionOperator))
        without_arrow = forAllExp.replace("->", '')
        without_collOp = without_arrow.replace(collectionOperator + "(", '')
        if "|" in without_collOp:
            iterator = without_collOp.split("|")[0]
            multiple_variable = iterator.split(',')
            for variable in multiple_variable:
                iteratorParts = variable.split(':')
                iteratorVariableName = iteratorParts[0]
                if ":" in variable:
                    iteratorclass = self.getClass(iteratorParts[1])
                else:
                    iteratorclass = "NotMentioned"
                iteratorExp = self.factory.create_iterator_expression(iteratorVariableName, iteratorclass)
                self.all[-1].addIterator(iteratorExp)

    def handle_binary_expression(self, expression, operator, inbetween=None, beforeSign=None):
        # print("handling right part")
        expressionParts = expression.split(operator)
        leftside = self.checkNumberOrVariable(expressionParts[0])
        rightside = self.checkNumberOrVariable(expressionParts[1])

        leftPart = None
        rightPart = None
        if "var" in leftside:
            leftPart = self.factory.create_property_Call_Expression(expressionParts[0], type="NP", iterators=self.all)
        elif "int" in leftside:
            leftPart = self.factory.create_integer_literal_expression("NP", int(expressionParts[0]))
        elif "real" in leftside:
            leftPart = self.factory.create_real_literal_expression("NP", float(expressionParts[0]))
        elif "bool" in leftside:
            leftPart = self.factory.create_boolean_literal_expression("NP", (expressionParts[0]))
        elif "str" in leftside:
            leftPart = self.factory.create_string_literal_expression("str", expressionParts[0].replace("'", ""))
        elif "date" in leftside:
            leftPart = self.factory.create_date_literal_expression("date", expressionParts[0].replace("'", ""))
        if "var" in rightside:
            rightPart = self.factory.create_property_Call_Expression(expressionParts[1], type="NP", iterators=self.all)
        elif "int" in rightside:
            rightPart = self.factory.create_integer_literal_expression("NP", int(expressionParts[1]))
        elif "real" in rightside:
            rightPart = self.factory.create_real_literal_expression("NP", float(expressionParts[1]))
        elif "bool" in rightside:
            rightPart = self.factory.create_boolean_literal_expression("NP", (expressionParts[1]))
        elif "str" in rightside:
            rightPart = self.factory.create_string_literal_expression("str", expressionParts[1].replace("'", ""))
        elif "date" in rightside:
            rightPart = self.factory.create_date_literal_expression("date", expressionParts[1].replace("'", ""))

        infixOperator = self.factory.create_infix_operator(operator)
        beforeOp = None
        if beforeSign is not None:
            beforeOp = self.factory.create_infix_operator(beforeSign)
        inBetweenOp = None
        if inbetween is not None and len(inbetween) > 0:
            inBetweenOp = self.factory.create_infix_operator(inbetween)
        opeartion_call_exp = self.factory.create_operation_call_expression(leftPart, rightPart, infixOperator,
                                                                           inBetweenOp, beforeOp)
        self.add_to_root(opeartion_call_exp)

    def handleBinaryFunc(self, operator, number):
        # import inspect
        # print(inspect.stack()[0][3])

        num = self.checkNumberOrVariable(number)
        if "int" in num:
            rightPart = self.factory.create_integer_literal_expression("NP", int(number))

        if "real" in num:
            rightPart = self.factory.create_real_literal_expression("NP", float(number))
        op = self.factory.create_infix_operator(operator)
        self.last_added.arguments.append(op)
        self.last_added.arguments.append(rightPart)

    def handle_last_opnum(self, operator, number):
        op = self.factory.create_operation_call_expression(name='callExp')
        op.arguments.append(self.factory.create_infix_operator(operator))
        num = self.checkNumberOrVariable(number)
        if "int" in num:
            rightPart = self.factory.create_integer_literal_expression("NP", int(number))
        if "real" in num:
            rightPart = self.factory.create_real_literal_expression("NP", float(number))

        op.arguments.append(
            rightPart
        )
        self.add_to_root(op)

    def handlePrint(self, root):
        if root is None:
            return
        if hasattr(root, 'arguments'):
            print(root.arguments)
            print(root.referredOperation())
            self.handlePrint(root.source())

        if hasattr(root, 'body'):
            print(root.name)
            print(root.iterator)
            for item in root.body:
                print(item)
