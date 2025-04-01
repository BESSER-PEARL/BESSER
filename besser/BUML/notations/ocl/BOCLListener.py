# Generated from BOCL.g4 by ANTLR 4.13.1
from antlr4 import *

if "." in __name__:
    from .BOCLParser import BOCLParser
else:
    from BOCLParser import BOCLParser
import inspect
from besser.BUML.metamodel.ocl.ocl import OperationCallExpression
from besser.BUML.notations.ocl.comparison_operator_checker import comparison_verifier

# This class defines a complete listener for a parse tree produced by BOCLParser.
class BOCLListener(ParseTreeListener):
    operator = None

    def __init__(self, rh):
        self.rootHandler = rh
        self.forAllBody = False
        self.operator = []
        self.initItems = []
        self.coll_data = []
        self.primaryExp =None
        self.debug = False
        self.debug_print = False
        self.all_if_else = []
        self.types_of = []
        self.userDefined = None
        self.unary = ""
        self.sign = []
        self.num = []
        self.functions = []
    def preprocess(self,ocl):
        if ocl.count('(') != ocl.count(')'):
            raise Exception(" Incorrect Syntax: Number of Brackets Mismatch")
        if ocl.count('{') != ocl.count('}'):
            raise Exception(" Incorrect Syntax: Number of Brackets Mismatch")
        if ocl.count('[') != ocl.count(']'):
            raise Exception(" Incorrect Syntax: Number of Brackets Mismatch")
        cc = comparison_verifier()

        # print(cc.verify_ocl(ocl))
        if not cc.verify_ocl(ocl):
            raise Exception("Incorrect Syntax for comparison operators")
    # Enter a parse tree produced by BOCLParser#oclFile.
    def enterOclFile(self, ctx: BOCLParser.OclFileContext):
        # print(inspect.stack()[0][3])
        self.preprocess(ctx.getText())
        if self.debug:
            print(inspect.stack()[0][3])
        if self.debug_print:
            print(ctx.getText())
        if 'inv' in ctx.getText().split(':')[0]:
            self.rootHandler.set_inv(True)
        if 'pre:' in ctx.getText():
            self.rootHandler.set_pre(True)
        if 'post:' in ctx.getText():
            self.rootHandler.set_post(True)
        if 'body:' in ctx.getText():
            self.rootHandler.set_post(True)

        pass

    # Exit a parse tree produced by BOCLParser#oclFile.
    def exitOclFile(self, ctx: BOCLParser.OclFileContext):
        if len(self.operator) == 1:
            if self.userDefined is not None:
                self.rootHandler.handle_last_opnum(self.operator.pop(),self.userDefined)
            else:
                pass #throw exception
        else:
            pass  # throw exception


        pass

        # Enter a parse tree produced by BOCLParser#preCondition.
    def enterPreCondition(self, ctx: BOCLParser.PreConditionContext):
        if self.debug:
            print(inspect.stack()[0][3])
        if self.debug_print:
            print(ctx.getText())
        # Exit a parse tree produced by BOCLParser#preCondition.
    def exitPreCondition(self, ctx: BOCLParser.PreConditionContext):
        if self.debug:
            print(inspect.stack()[0][3])
        if self.debug_print:
            print(ctx.getText())
        context = ctx.getText().split("context")[1].split(":")[0]
        function_name = ctx.getText().split("::")[1].split("pre:")[0]
        self.rootHandler.handle_pre_root(context, function_name, self.rootHandler.get_root())

        # Enter a parse tree produced by BOCLParser#postCondition.
    def enterPostCondition(self, ctx: BOCLParser.PostConditionContext):
        if self.debug:
            print(inspect.stack()[0][3])
        if self.debug_print:
            print(ctx.getText())
        # Exit a parse tree produced by BOCLParser#postCondition.
    def exitPostCondition(self, ctx: BOCLParser.PostConditionContext):
        if self.debug:
            print(inspect.stack()[0][3])
        if self.debug_print:
            print(ctx.getText())
        context = ctx.getText().split("context")[1].split(":")[0]
        function_name = ctx.getText().split("::")[1].split("post:")[0]
        self.rootHandler.handle_post_root(context, function_name, self.rootHandler.get_root())

        # Enter a parse tree produced by BOCLParser#initConstraints.
    def enterInitConstraints(self, ctx: BOCLParser.InitConstraintsContext):
        if self.debug:
            print(inspect.stack()[0][3])
        if self.debug_print:
            print(ctx.getText())
        # Exit a parse tree produced by BOCLParser#initConstraints.
    def exitInitConstraints(self, ctx: BOCLParser.InitConstraintsContext):
        if self.debug:
            print(inspect.stack()[0][3])
        if self.debug_print:
            print(ctx.getText())

        context = ctx.getText().split("context")[1].split(":")[0]
        variable_name = ctx.getText().split("::")[1].split(":")[0]
        variable_type = ctx.getText().split("init:")[0].split(":")[-1]
        # print(ctx.getText())
        if len(self.initItems)>0 and self.rootHandler.get_root() is None:
            self.rootHandler.handle_init_root(context, variable_name, self.initItems.pop(-1), variable_type)
        else:
            self.rootHandler.handle_init_root(context, variable_name, self.rootHandler.get_root(),variable_type)
    # Enter a parse tree produced by BOCLParser#ContextExp.
    def enterContextExp(self, ctx: BOCLParser.ContextExpContext):
        self.context = (ctx.getText())

        pass

    # Exit a parse tree produced by BOCLParser#ContextExp.
    def exitContextExp(self, ctx: BOCLParser.ContextExpContext):
        # print(inspect.stack()[0][3])

        pass

    # Enter a parse tree produced by BOCLParser#constraint.
    def enterConstraint(self, ctx: BOCLParser.ConstraintContext):

        if self.debug:
            print(inspect.stack()[0][3])
        if self.debug_print:
            print(ctx.getText())
        self.context_name = (self.context.split(ctx.getText()))[0].replace('context', '')
        self.rootHandler.set_context_name(self.context_name)
        # print(ctx.getText())

    # Exit a parse tree produced by BOCLParser#constraint.
    def exitConstraint(self, ctx: BOCLParser.ConstraintContext):
        pass

    # Enter a parse tree produced by BOCLParser#functionCall.
    def enterFunctionCall(self, ctx: BOCLParser.FunctionCallContext):
        if self.debug:
            print(inspect.stack()[0][3])
        if self.debug_print:
            print(ctx.getText())
        op_exp = self.rootHandler.create_operation_call_exp(ctx.getText())
        op_exp.referredOperation=self.rootHandler.create_operation_call_exp('FunctionCall')
        self.types_of.append(op_exp)
        self.functions.append(op_exp)
        # print(ctx.getText())
        pass

    # Exit a parse tree produced by BOCLParser#functionCall.
    def exitFunctionCall(self, ctx: BOCLParser.FunctionCallContext):
        # print(inspect.stack()[0][3])
        pass

    # Enter a parse tree produced by BOCLParser#type.
    def enterType(self, ctx: BOCLParser.TypeContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        pass
        type_exp = self.rootHandler.create_type_exp(ctx.getText())
        self.types_of[-1].add(type_exp)
        if isinstance(self.types_of[-1].referredOperation, OperationCallExpression):
            if self.types_of[-1].referredOperation.name == "FunctionCall":
                self.types_of.pop()

    # Exit a parse tree produced by BOCLParser#type.
    def exitType(self, ctx: BOCLParser.TypeContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        pass

    # Enter a parse tree produced by BOCLParser#collectionType.
    def enterCollectionType(self, ctx: BOCLParser.CollectionTypeContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        pass

    # Exit a parse tree produced by BOCLParser#collectionType.
    def exitCollectionType(self, ctx: BOCLParser.CollectionTypeContext):
        pass

    # Enter a parse tree produced by BOCLParser#userDefinedType.
    def enterUserDefinedType(self, ctx: BOCLParser.UserDefinedTypeContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        pass

    # Exit a parse tree produced by BOCLParser#userDefinedType.
    def exitUserDefinedType(self, ctx: BOCLParser.UserDefinedTypeContext):
        pass

    # Enter a parse tree produced by BOCLParser#binary.
    def enterBinary(self, ctx: BOCLParser.BinaryContext):
        # print(inspect.stack()[0][3])
        self.binary = ctx.getText()

        if self.unary:
            self.sign.append(self.unary.split(self.binary)[0])
            # print(self.sign)
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        # print(ctx.getText())
        pass

    # Exit a parse tree produced by BOCLParser#binary.
    def exitBinary(self, ctx: BOCLParser.BinaryContext):

        pass

    # Enter a parse tree produced by BOCLParser#unary.
    def enterUnary(self, ctx: BOCLParser.UnaryContext):
        # print(inspect.stack()[0][3])4
        self.unary = ctx.getText()
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        pass

    # Exit a parse tree produced by BOCLParser#unary.
    def exitUnary(self, ctx: BOCLParser.UnaryContext):
        pass


    # Enter a parse tree produced by BOCLParser#OCLISTYPEOF.
    def enterOCLISTYPEOF(self, ctx: BOCLParser.OCLISTYPEOFContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        if ctx.parentCtx is not None:

            propertyName = ctx.parentCtx.getText().split('.')[-2]
            self.rootHandler.handle_property(propertyName)
        op_exp = self.rootHandler.create_operation_call_exp("OCLISTYPEOF")
        op_exp.referredOperation=self.rootHandler.create_operation_call_exp('OCLISTYPEOF')
        self.types_of.append(op_exp)
        pass

    # Exit a parse tree produced by BOCLParser#OCLISTYPEOF.
    def exitOCLISTYPEOF(self, ctx: BOCLParser.OCLISTYPEOFContext):
        self.rootHandler.add_to_root(self.types_of.pop())
        pass

    # Enter a parse tree produced by BOCLParser#OCLASTYPE.
    def enterOCLASTYPE(self, ctx: BOCLParser.OCLASTYPEContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        op_exp = self.rootHandler.create_operation_call_exp("OCLASTYPE")
        op_exp.referredOperation=self.rootHandler.create_operation_call_exp('OCLASTYPE')
        self.types_of.append(op_exp)

        pass

    # Exit a parse tree produced by BOCLParser#OCLASTYPE.
    def exitOCLASTYPE(self, ctx: BOCLParser.OCLASTYPEContext):
        self.rootHandler.add_to_root(self.types_of.pop())
        pass

    # Enter a parse tree produced by BOCLParser#OCLISKINDOF.
    def enterOCLISKINDOF(self, ctx: BOCLParser.OCLISKINDOFContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        op_exp = self.rootHandler.create_operation_call_exp("OCLISKINDOF")
        op_exp.referredOperation=self.rootHandler.create_operation_call_exp('OCLISKINDOF')
        self.types_of.append(op_exp)

        pass

    # Exit a parse tree produced by BOCLParser#OCLISKINDOF.
    def exitOCLISKINDOF(self, ctx: BOCLParser.OCLISKINDOFContext):
        self.rootHandler.add_to_root(self.types_of.pop())
        pass

    # Enter a parse tree produced by BOCLParser#ISEMPTY.
    def enterISEMPTY(self, ctx: BOCLParser.ISEMPTYContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        op_call_exp = self.rootHandler.create_operation_call_exp('IsEmpty')
        if len(self.coll_data) != 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop())
        else:
            if ctx.parentCtx is not None:
                propertyName = ctx.parentCtx.getText().split('->')[0]
                self.rootHandler.handle_property(propertyName)
        self.rootHandler.handle_adding_to_root(op_call_exp)

        pass

    # Exit a parse tree produced by BOCLParser#ISEMPTY.
    def exitISEMPTY(self, ctx: BOCLParser.ISEMPTYContext):
        pass

    # Enter a parse tree produced by BOCLParser#SUM.
    def enterSUM(self, ctx: BOCLParser.SUMContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        op_call_exp = self.rootHandler.create_operation_call_exp('Sum')
        if len(self.coll_data) != 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop())
        else:
            if ctx.parentCtx is not None:
                propertyName = ctx.parentCtx.getText().split('->')[0]
                self.rootHandler.handle_property(propertyName)

        self.rootHandler.handle_adding_to_root(op_call_exp)

        pass

    # Exit a parse tree produced by BOCLParser#SUM.
    def exitSUM(self, ctx: BOCLParser.SUMContext):
        pass

    # Enter a parse tree produced by BOCLParser#SIZE.
    def enterSIZE(self, ctx: BOCLParser.SIZEContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        op_call_exp = self.rootHandler.create_operation_call_exp('Size')
        if len(self.coll_data) != 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop())
        else:
            if ctx.parentCtx is not None:
                propertyName = ctx.parentCtx.getText().split('->')[0]
                if len(propertyName)>1:
                    self.rootHandler.handle_property(propertyName)
            # self.rootHandler.handle_property(propertyName)
        self.rootHandler.handle_adding_to_root(op_call_exp)

        pass

    # Exit a parse tree produced by BOCLParser#SIZE.
    def exitSIZE(self, ctx: BOCLParser.SIZEContext):
        pass

    # Enter a parse tree produced by BOCLParser#INCLUDES.
    def enterINCLUDES(self, ctx: BOCLParser.INCLUDESContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        op_call_exp = self.rootHandler.create_operation_call_exp('INCLUDES')
        if len(self.coll_data) != 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop())
        else:
            if ctx.parentCtx is not None:
                propertyName = ctx.parentCtx.getText().split('->')[0]
                self.rootHandler.handle_property(propertyName)
        self.coll_data.append(op_call_exp)

        pass

    # Exit a parse tree produced by BOCLParser#INCLUDES.
    def exitINCLUDES(self, ctx: BOCLParser.INCLUDESContext):

        exp = self.coll_data.pop()
        if self.primaryExp is not None:
            if len(exp.arguments) == 0:
                prop = self.rootHandler.create_property(self.primaryExp)
                exp.arguments.append(prop)
        self.rootHandler.add_to_root(exp)
        pass

    # Enter a parse tree produced by BOCLParser#EXCLUDES.
    def enterEXCLUDES(self, ctx: BOCLParser.EXCLUDESContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        op_call_exp = self.rootHandler.create_operation_call_exp('EXCLUDES')
        if len(self.coll_data) != 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop())
        else:
            temp = ctx.getText()
            if ctx.parentCtx is not None:
                propertyName = ctx.parentCtx.getText().split('->')[0]
                self.rootHandler.handle_property(propertyName)
        self.coll_data.append(op_call_exp)

        pass

    # Exit a parse tree produced by BOCLParser#EXCLUDES.
    def exitEXCLUDES(self, ctx: BOCLParser.EXCLUDESContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        exp = self.coll_data.pop()
        if self.primaryExp is not None:
            if len(exp.arguments)==0:
                prop = self.rootHandler.create_property(self.primaryExp)
                exp.arguments.append(prop)
        self.rootHandler.add_to_root(exp)

        pass

    # Enter a parse tree produced by BOCLParser#SEQUENCE.
    def enterSEQUENCE(self, ctx: BOCLParser.SEQUENCEContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        self.coll_data.append(self.rootHandler.create_sequence())

        pass

    # Exit a parse tree produced by BOCLParser#SEQUENCE.
    def exitSEQUENCE(self, ctx: BOCLParser.SEQUENCEContext):
        op = None
        # print(ctx.getText())
        if len(self.operator) != 0:
            op = self.operator.pop()
            if len(self.coll_data) > 0:
                self.rootHandler.handle_adding_to_root(self.coll_data.pop(), op)

        pass

    # Enter a parse tree produced by BOCLParser#SUBSEQUENCE.
    def enterSUBSEQUENCE(self, ctx: BOCLParser.SUBSEQUENCEContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        self.coll_data.append(self.rootHandler.create_sub_sequence())

        pass

    # Exit a parse tree produced by BOCLParser#SUBSEQUENCE.
    def exitSUBSEQUENCE(self, ctx: BOCLParser.SUBSEQUENCEContext):
        self.rootHandler.add_to_root(self.coll_data.pop())
        pass

    # Enter a parse tree produced by BOCLParser#ALLINSTANCES.
    def enterALLINSTANCES(self, ctx: BOCLParser.ALLINSTANCESContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        if self.userDefined is not None:
            self.rootHandler.add_to_root(self.rootHandler.create_type_exp(self.userDefined))
            self.userDefined = None
        op_call_exp = self.rootHandler.create_operation_call_exp("ALLInstances")
        op_call_exp.referredOperation=self.rootHandler.create_operation_call_exp("ALLInstances")
        self.rootHandler.add_to_root(op_call_exp)
        pass

    # Exit a parse tree produced by BOCLParser#ALLINSTANCES.
    def exitALLINSTANCES(self, ctx: BOCLParser.ALLINSTANCESContext):
        pass

    # Enter a parse tree produced by BOCLParser#ORDEREDSET.
    def enterORDEREDSET(self, ctx: BOCLParser.ORDEREDSETContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        txt = ctx.getText()
        self.coll_data.append(self.rootHandler.create_ordered_set())

        pass

    # Exit a parse tree produced by BOCLParser#ORDEREDSET.
    def exitORDEREDSET(self, ctx: BOCLParser.ORDEREDSETContext):
        op = None
        if len(self.operator) != 0:
            op = self.operator.pop()
        if len(self.coll_data) > 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop(), op)
        pass

    # Enter a parse tree produced by BOCLParser#SUBORDEREDSET.
    def enterSUBORDEREDSET(self, ctx: BOCLParser.SUBORDEREDSETContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        if len(self.coll_data) > 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop())
        self.coll_data.append(self.rootHandler.create_sub_ordered_set())
        pass

    # Exit a parse tree produced by BOCLParser#SUBORDEREDSET.
    def exitSUBORDEREDSET(self, ctx: BOCLParser.SUBORDEREDSETContext):
        op = None
        if len(self.operator) != 0:
            op = self.operator.pop()
        if len(self.coll_data) > 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop(), op)
        pass

    # Enter a parse tree produced by BOCLParser#SET.
    def enterSET(self, ctx: BOCLParser.SETContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        if len(self.coll_data) > 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop())
        self.coll_data.append(self.rootHandler.create_set())

        pass

    # Exit a parse tree produced by BOCLParser#SET.
    def exitSET(self, ctx: BOCLParser.SETContext):
        op = None
        if len(self.operator) != 0:
            op = self.operator.pop()
        if len(self.coll_data) > 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop(), op)
        pass

    # Enter a parse tree produced by BOCLParser#BAG.
    def enterBAG(self, ctx: BOCLParser.BAGContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        self.coll_data.append(self.rootHandler.create_bag())
        pass

    # Exit a parse tree produced by BOCLParser#BAG.
    def exitBAG(self, ctx: BOCLParser.BAGContext):
        op = None
        if len(self.operator) != 0:
            op = self.operator.pop()
        if len(self.coll_data) > 0:
            self.rootHandler.handle_bag(self.coll_data.pop(), op)

        pass

    # Enter a parse tree produced by BOCLParser#PREPEND.
    def enterPREPEND(self, ctx: BOCLParser.PREPENDContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        op_call_exp = self.rootHandler.create_operation_call_exp('PREPEND')
        if len(self.coll_data) > 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop())
        self.coll_data.append(op_call_exp)

        pass

    # Exit a parse tree produced by BOCLParser#PREPEND.
    def exitPREPEND(self, ctx: BOCLParser.PREPENDContext):

        self.rootHandler.add_to_root(self.coll_data.pop())
        pass

    # Enter a parse tree produced by BOCLParser#LAST.
    def enterLAST(self, ctx: BOCLParser.LASTContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        op_call_exp = self.rootHandler.create_operation_call_exp('Last')
        if len(self.coll_data) > 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop())
        self.rootHandler.handle_adding_to_root(op_call_exp)

        # print(inspect.stack()[0][3])
        pass

    # Exit a parse tree produced by BOCLParser#LAST.
    def exitLAST(self, ctx: BOCLParser.LASTContext):

        pass

    # Enter a parse tree produced by BOCLParser#APPEND.
    def enterAPPEND(self, ctx: BOCLParser.APPENDContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        op_call_exp = self.rootHandler.create_operation_call_exp('APPEND')
        if len(self.coll_data) > 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop())
        self.coll_data.append(op_call_exp)

        # print(inspect.stack()[0][3])
        pass

    # Exit a parse tree produced by BOCLParser#APPEND.
    def exitAPPEND(self, ctx: BOCLParser.APPENDContext):
        self.rootHandler.add_to_root(self.coll_data.pop())
        pass

    # Enter a parse tree produced by BOCLParser#COLLECTION.
    def enterCOLLECTION(self, ctx: BOCLParser.COLLECTIONContext):

        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
                print(ctx.parentCtx.getText())
            # print(ctx.getText())

        if ctx.parentCtx is not None:
            self.rootHandler.verify(ctx.parentCtx.getText().split(ctx.getText())[0])
        self.rootHandler.handle_collection(ctx.getText())
        # print(inspect.stack()[0][3])

    # Exit a parse tree produced by BOCLParser#COLLECTION.
    def exitCOLLECTION(self, ctx: BOCLParser.COLLECTIONContext):
        if self.debug:
            print("exitCOLLECTION")
            if self.debug_print:
                print(ctx.getText())

        self.rootHandler.pop()

        pass

    # Enter a parse tree produced by BOCLParser#CollectionExpressionVariable.
    def enterCollectionExpressionVariable(self, ctx: BOCLParser.CollectionExpressionVariableContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
                print(ctx.parentCtx.getText())
        if ctx.parentCtx is not None:
            self.rootHandler.verify(ctx.parentCtx.getText().split(ctx.getText())[0])
        self.rootHandler.handle_collection(ctx.getText())
        pass

    # Exit a parse tree produced by BOCLParser#CollectionExpressionVariable.
    def exitCollectionExpressionVariable(self, ctx: BOCLParser.CollectionExpressionVariableContext):
        # print("exitCollectionExpressionVariable")
        # self.rootHandler.pop()
        pass

    # Enter a parse tree produced by BOCLParser#SYMMETRICDIFFERENCE.
    def enterSYMMETRICDIFFERENCE(self, ctx: BOCLParser.SYMMETRICDIFFERENCEContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        op_call_exp = self.rootHandler.create_operation_call_exp('SYMMETRICDIFFERENCE')
        if len(self.coll_data) > 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop())
        self.coll_data.append(op_call_exp)
        # print(inspect.stack()[0][3])
        pass

    # Exit a parse tree produced by BOCLParser#SYMMETRICDIFFERENCE.
    def exitSYMMETRICDIFFERENCE(self, ctx: BOCLParser.SYMMETRICDIFFERENCEContext):
        self.rootHandler.add_to_root(self.coll_data.pop())

    # Enter a parse tree produced by BOCLParser#FIRST.
    def enterFIRST(self, ctx: BOCLParser.FIRSTContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        op_call_exp = self.rootHandler.create_operation_call_exp('First')
        if len(self.coll_data) > 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop())
        self.rootHandler.handle_adding_to_root(op_call_exp)
        # print(inspect.stack()[0][3])

    # Exit a parse tree produced by BOCLParser#FIRST.
    def exitFIRST(self, ctx: BOCLParser.FIRSTContext):
        pass

    # Enter a parse tree produced by BOCLParser#DERIVE.
    def enterDERIVE(self, ctx: BOCLParser.DERIVEContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        # print(inspect.stack()[0][3])
        pass

    # Exit a parse tree produced by BOCLParser#DERIVE.
    def exitDERIVE(self, ctx: BOCLParser.DERIVEContext):
        pass

    # Enter a parse tree produced by BOCLParser#UNION.
    def enterUNION(self, ctx: BOCLParser.UNIONContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        op_call_exp = self.rootHandler.create_operation_call_exp('UNION')
        if len(self.coll_data) > 0:
            self.rootHandler.handle_adding_to_root(self.coll_data.pop())
        self.coll_data.append(op_call_exp)

        # print(inspect.stack()[0][3])
        pass

    # Exit a parse tree produced by BOCLParser#UNION.
    def exitUNION(self, ctx: BOCLParser.UNIONContext):
        self.rootHandler.add_to_root(self.coll_data.pop())

        pass

    # Enter a parse tree produced by BOCLParser#defExp.
    def enterDefExp(self, ctx: BOCLParser.DefExpContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        # print(inspect.stack()[0][3])
        pass

    # Exit a parse tree produced by BOCLParser#defExp.
    def exitDefExp(self, ctx: BOCLParser.DefExpContext):
        pass

    # Enter a parse tree produced by BOCLParser#defIDAssignmentexpression.
    def enterDefIDAssignmentexpression(self, ctx: BOCLParser.DefIDAssignmentexpressionContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        # print(inspect.stack()[0][3])
        pass

    # Exit a parse tree produced by BOCLParser#defIDAssignmentexpression.
    def exitDefIDAssignmentexpression(self, ctx: BOCLParser.DefIDAssignmentexpressionContext):
        pass

    # Enter a parse tree produced by BOCLParser#PrimaryExp.
    def enterPrimaryExp(self, ctx: BOCLParser.PrimaryExpContext):
        if self.unary:
            self.sign.append(self.unary.split(ctx.getText())[0])
            print(self.sign)
            if self.unary == self.sign[-1] + ctx.getText():
                self.rootHandler.handle_single_variable(ctx.getText(), self.sign.pop())
                #
        self.primaryExp = ctx.getText()
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())


        # print(inspect.stack()[0][3])
        # print(ctx.getText())

        pass

    # Exit a parse tree produced by BOCLParser#PrimaryExp.
    def exitPrimaryExp(self, ctx: BOCLParser.PrimaryExpContext):
        # #print("exitPrimaryExp")
        # #print(ctx.getText())
        # input()
        # if len(self.operator) !=0 and self.operator[-1] in ctx.getText():
        #
        #     self.rootHandler.handlePrimaryExp(ctx.getText(),self.operator[-1])
        pass

    # Enter a parse tree produced by BOCLParser#funcCall.
    def enterFuncCall(self, ctx: BOCLParser.FuncCallContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        # print(inspect.stack()[0][3])
        pass

    # Exit a parse tree produced by BOCLParser#funcCall.
    def exitFuncCall(self, ctx: BOCLParser.FuncCallContext):
        pass

    # Enter a parse tree produced by BOCLParser#op.
    def enterOp(self, ctx: BOCLParser.OpContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        # self.operator.append(ctx.getText())
        pass

    # Exit a parse tree produced by BOCLParser#op.
    def exitOp(self, ctx: BOCLParser.OpContext):
        pass

    # Enter a parse tree produced by BOCLParser#arrowexp.
    def enterArrowexp(self, ctx: BOCLParser.ArrowexpContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        # print(inspect.stack()[0][3])
        pass

    # Exit a parse tree produced by BOCLParser#arrowexp.
    def exitArrowexp(self, ctx: BOCLParser.ArrowexpContext):
        pass

    # Enter a parse tree produced by BOCLParser#number.
    def enterNumber(self, ctx: BOCLParser.NumberContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        item = self.rootHandler.get_factory().create_collection_item("item", ctx.getText())
        self.initItems.append(item)
        if len(self.coll_data) > 0:
            self.coll_data[-1].add(item)

        pass

    # Exit a parse tree produced by BOCLParser#number.
    def exitNumber(self, ctx: BOCLParser.NumberContext):
        pass

    # Enter a parse tree produced by BOCLParser#PredefinedfunctionCall.
    def enterPredefinedfunctionCall(self, ctx: BOCLParser.PredefinedfunctionCallContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        pass

    # Exit a parse tree produced by BOCLParser#PredefinedfunctionCall.
    def exitPredefinedfunctionCall(self, ctx: BOCLParser.PredefinedfunctionCallContext):
        pass

    # Enter a parse tree produced by BOCLParser#ID.
    def enterID(self, ctx: BOCLParser.IDContext):
        # print(inspect.stack()[0][3])
        # print(ctx.getText())
        txt = ctx.getText()

        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        if len(self.coll_data) != 0:
            item = self.rootHandler.get_factory().create_collection_item("item", ctx.getText())
            self.coll_data[-1].add(item)
        else:
            if self.rootHandler.get_root() is not None:
                self.rootHandler.handle_ID(ctx.getText())
            if len(self.rootHandler.all)>0:
                self.rootHandler.handle_ID(ctx.getText())
        pass

    # Exit a parse tree produced by BOCLParser#ID.
    def exitID(self, ctx: BOCLParser.IDContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        item = self.rootHandler.get_factory().create_collection_item("item", ctx.getText())
        self.initItems.append(item)

    # Enter a parse tree produced by BOCLParser#SingleQuoteExp.
    def enterSingleQuoteExp(self, ctx: BOCLParser.SingleQuoteExpContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        pass

    # Exit a parse tree produced by BOCLParser#SingleQuoteExp.
    def exitSingleQuoteExp(self, ctx: BOCLParser.SingleQuoteExpContext):
        pass

    # Enter a parse tree produced by BOCLParser#doubleDots.
    def enterDoubleDots(self, ctx: BOCLParser.DoubleDotsContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        pass

    # Exit a parse tree produced by BOCLParser#doubleDots.
    def exitDoubleDots(self, ctx: BOCLParser.DoubleDotsContext):
        pass

    # Enter a parse tree produced by BOCLParser#binaryExpression.
    def enterBinaryExpression(self, ctx: BOCLParser.BinaryExpressionContext):

        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        pass

    # Exit a parse tree produced by BOCLParser#binaryExpression.
    def exitBinaryExpression(self, ctx: BOCLParser.BinaryExpressionContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        and_or = self.binary.split(ctx.getText())[0]
        # print(and_or)
        beforeSign = None
        if len(self.sign) != 0:
            beforeSign = self.sign.pop()
        inBetween_op = None
        if "and" or "or" in and_or:
            inBetween_op = and_or
        if len(self.operator) != 0:
            text_to_process = ctx.getText()
            if "()" in text_to_process:
                text_to_process = ctx.parentCtx.parentCtx.getText()
            self.rootHandler.handle_binary_expression(text_to_process, self.operator.pop(), inBetween_op, beforeSign)
            # self.operator.pop()

        pass

    # Enter a parse tree produced by BOCLParser#unaryExpression.
    def enterUnaryExpression(self, ctx: BOCLParser.UnaryExpressionContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        pass

    # Exit a parse tree produced by BOCLParser#unaryExpression.
    def exitUnaryExpression(self, ctx: BOCLParser.UnaryExpressionContext):
        pass

    # Enter a parse tree produced by BOCLParser#operator.
    def enterOperator(self, ctx: BOCLParser.OperatorContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        self.operator.append(ctx.getText())

        pass

    # Exit a parse tree produced by BOCLParser#operator.
    def exitOperator(self, ctx: BOCLParser.OperatorContext):
        pass

    # Enter a parse tree produced by BOCLParser#numberORUserDefined.
    def enterNumberORUserDefined(self, ctx: BOCLParser.NumberORUserDefinedContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        self.userDefined = ctx.getText()

        pass

    # Exit a parse tree produced by BOCLParser#numberORUserDefined.
    def exitNumberORUserDefined(self, ctx: BOCLParser.NumberORUserDefinedContext):
        pass

    # Enter a parse tree produced by BOCLParser#primaryExpression.
    def enterPrimaryExpression(self, ctx: BOCLParser.PrimaryExpressionContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        # print(ctx.getText())
        # print(ctx.getText())
        if not ctx.getText() == "self":
            if len(self.types_of) != 0:
                self.rootHandler.handle_property(ctx.getText())

        self.userDefined = ctx.getText()
        pass

    # Exit a parse tree produced by BOCLParser#primaryExpression.
    def exitPrimaryExpression(self, ctx: BOCLParser.PrimaryExpressionContext):
        # if len(self.operator) != 0:
        #     self.rootHandler.handlePrimaryExp(ctx.getText(),self.operator[-1])
        #     self.operator.pop()
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        pass

    # Enter a parse tree produced by BOCLParser#literal.
    def enterLiteral(self, ctx: BOCLParser.LiteralContext):
        # print(inspect.stack()[0][3])
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        pass

    # Exit a parse tree produced by BOCLParser#literal.
    def exitLiteral(self, ctx: BOCLParser.LiteralContext):
        pass

    def print(self):
        self.rootHandler.print()

    def enterIfExp(self, ctx: BOCLParser.IfExpContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        self.all_if_else.append(self.rootHandler.create_if_else_exp('if', 'IfExpression'))

        pass

    # Exit a parse tree produced by BOCLParser#ifExp.
    def exitIfExp(self, ctx: BOCLParser.IfExpContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        pass

    # Enter a parse tree produced by BOCLParser#thenExp.
    def enterThenExp(self, ctx: BOCLParser.ThenExpContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        top = self.rootHandler.getTop()
        self.all_if_else[-1].ifCondition = top

    # Exit a parse tree produced by BOCLParser#thenExp.
    def exitThenExp(self, ctx: BOCLParser.ThenExpContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        pass

    # Enter a parse tree produced by BOCLParser#elseExp.
    def enterElseExp(self, ctx: BOCLParser.ElseExpContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        self.all_if_else[-1].thenExpression = self.rootHandler.getTop()
        pass

    # Exit a parse tree produced by BOCLParser#elseExp.
    def exitElseExp(self, ctx: BOCLParser.ElseExpContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        pass

    # Enter a parse tree produced by BOCLParser#endIfExp.
    def enterEndIfExp(self, ctx: BOCLParser.EndIfExpContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        self.all_if_else[-1].elseCondition = self.rootHandler.getTop(True)
        self.rootHandler.add_to_root(self.all_if_else.pop())

    # Exit a parse tree produced by BOCLParser#endIfExp.
    def exitEndIfExp(self, ctx: BOCLParser.EndIfExpContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        pass

    def enterDoubleCOLONs(self, ctx: BOCLParser.DoubleCOLONsContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())

        self.userDefined = ctx.getText().split('::')[0]
        pass

    # Exit a parse tree produced by BOCLParser#doubleCOLONs.
    def exitDoubleCOLONs(self, ctx: BOCLParser.DoubleCOLONsContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        pass

    def enterEndExpression(self, ctx: BOCLParser.EndExpressionContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        self.rootHandler.pop()
        if len(ctx.getText())>2:
            if ctx.getText()[0:3] == "and" or ctx.getText()[0:2] == "or":
                self.rootHandler.handle_and_with_function_call(ctx.getText())
        # elif ctx.getText()[0:1] in ['+','-','*','/','=']:
        #     self.rootHandler.create_infinix_op(ctx.getText()[0:1])

        pass
# Exit a parse tree produced by BOCLParser#endExpression.
    def exitEndExpression(self, ctx: BOCLParser.EndExpressionContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        pass
    def enterBinaryFunctionCall(self, ctx:BOCLParser.BinaryFunctionCallContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        pass

    # Exit a parse tree produced by BOCLParser#binaryFunctionCall.
    def exitBinaryFunctionCall(self, ctx:BOCLParser.BinaryFunctionCallContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        operator = self.operator.pop()
        num = ctx.getText().replace(operator,"")
        self.rootHandler.handleBinaryFunc(operator,num)
# Enter a parse tree produced by BOCLParser#dateLiteral.
    def enterDateLiteral(self, ctx:BOCLParser.DateLiteralContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        pass

    # Exit a parse tree produced by BOCLParser#dateLiteral.
    def exitDateLiteral(self, ctx:BOCLParser.DateLiteralContext):
        if self.debug:
            print(inspect.stack()[0][3])
            if self.debug_print:
                print(ctx.getText())
        pass
del BOCLParser