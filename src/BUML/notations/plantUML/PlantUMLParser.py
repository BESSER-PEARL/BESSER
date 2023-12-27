# Generated from ./PlantUML.g4 by ANTLR 4.13.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,40,200,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,1,0,
        1,0,1,0,5,0,44,8,0,10,0,12,0,47,9,0,1,0,1,0,1,1,1,1,1,1,3,1,54,8,
        1,1,2,1,2,1,2,1,2,1,2,1,3,1,3,3,3,63,8,3,1,3,1,3,3,3,67,8,3,1,3,
        1,3,1,3,1,3,5,3,73,8,3,10,3,12,3,76,9,3,1,3,1,3,1,3,1,4,1,4,3,4,
        83,8,4,1,5,1,5,3,5,87,8,5,1,6,1,6,3,6,91,8,6,1,6,1,6,1,6,1,6,3,6,
        97,8,6,1,6,3,6,100,8,6,1,6,1,6,1,6,3,6,105,8,6,1,6,1,6,1,7,1,7,1,
        8,3,8,112,8,8,1,8,1,8,3,8,116,8,8,1,9,3,9,119,8,9,1,9,3,9,122,8,
        9,3,9,124,8,9,1,9,1,9,3,9,128,8,9,1,9,3,9,131,8,9,3,9,133,8,9,1,
        10,3,10,136,8,10,1,10,3,10,139,8,10,3,10,141,8,10,1,10,1,10,3,10,
        145,8,10,1,10,3,10,148,8,10,3,10,150,8,10,1,11,1,11,1,11,3,11,155,
        8,11,1,11,1,11,1,11,1,12,1,12,1,12,1,13,1,13,1,13,1,13,3,13,167,
        8,13,1,13,1,13,1,14,1,14,1,15,3,15,174,8,15,1,15,1,15,1,15,1,15,
        1,15,1,16,3,16,182,8,16,1,16,3,16,185,8,16,1,16,3,16,188,8,16,1,
        16,1,16,1,16,1,16,1,17,1,17,1,18,1,18,1,19,1,19,1,19,0,0,20,0,2,
        4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,0,4,1,0,36,37,
        1,0,19,22,1,0,23,30,1,0,31,32,214,0,40,1,0,0,0,2,53,1,0,0,0,4,55,
        1,0,0,0,6,62,1,0,0,0,8,80,1,0,0,0,10,86,1,0,0,0,12,88,1,0,0,0,14,
        108,1,0,0,0,16,111,1,0,0,0,18,123,1,0,0,0,20,140,1,0,0,0,22,151,
        1,0,0,0,24,159,1,0,0,0,26,162,1,0,0,0,28,170,1,0,0,0,30,173,1,0,
        0,0,32,181,1,0,0,0,34,193,1,0,0,0,36,195,1,0,0,0,38,197,1,0,0,0,
        40,41,5,33,0,0,41,45,5,40,0,0,42,44,3,2,1,0,43,42,1,0,0,0,44,47,
        1,0,0,0,45,43,1,0,0,0,45,46,1,0,0,0,46,48,1,0,0,0,47,45,1,0,0,0,
        48,49,5,34,0,0,49,1,1,0,0,0,50,54,3,4,2,0,51,54,3,6,3,0,52,54,3,
        10,5,0,53,50,1,0,0,0,53,51,1,0,0,0,53,52,1,0,0,0,54,3,1,0,0,0,55,
        56,5,1,0,0,56,57,5,2,0,0,57,58,5,36,0,0,58,59,5,40,0,0,59,5,1,0,
        0,0,60,63,3,8,4,0,61,63,5,3,0,0,62,60,1,0,0,0,62,61,1,0,0,0,63,64,
        1,0,0,0,64,66,5,35,0,0,65,67,3,24,12,0,66,65,1,0,0,0,66,67,1,0,0,
        0,67,68,1,0,0,0,68,69,5,4,0,0,69,74,5,40,0,0,70,73,3,30,15,0,71,
        73,3,32,16,0,72,70,1,0,0,0,72,71,1,0,0,0,73,76,1,0,0,0,74,72,1,0,
        0,0,74,75,1,0,0,0,75,77,1,0,0,0,76,74,1,0,0,0,77,78,5,5,0,0,78,79,
        5,40,0,0,79,7,1,0,0,0,80,82,5,6,0,0,81,83,5,3,0,0,82,81,1,0,0,0,
        82,83,1,0,0,0,83,9,1,0,0,0,84,87,3,12,6,0,85,87,3,22,11,0,86,84,
        1,0,0,0,86,85,1,0,0,0,87,11,1,0,0,0,88,90,5,35,0,0,89,91,3,26,13,
        0,90,89,1,0,0,0,90,91,1,0,0,0,91,96,1,0,0,0,92,97,3,14,7,0,93,97,
        3,16,8,0,94,97,3,18,9,0,95,97,3,20,10,0,96,92,1,0,0,0,96,93,1,0,
        0,0,96,94,1,0,0,0,96,95,1,0,0,0,97,99,1,0,0,0,98,100,3,26,13,0,99,
        98,1,0,0,0,99,100,1,0,0,0,100,101,1,0,0,0,101,104,5,35,0,0,102,103,
        5,7,0,0,103,105,5,35,0,0,104,102,1,0,0,0,104,105,1,0,0,0,105,106,
        1,0,0,0,106,107,5,40,0,0,107,13,1,0,0,0,108,109,5,8,0,0,109,15,1,
        0,0,0,110,112,5,9,0,0,111,110,1,0,0,0,111,112,1,0,0,0,112,113,1,
        0,0,0,113,115,5,8,0,0,114,116,5,10,0,0,115,114,1,0,0,0,115,116,1,
        0,0,0,116,17,1,0,0,0,117,119,5,11,0,0,118,117,1,0,0,0,118,119,1,
        0,0,0,119,124,1,0,0,0,120,122,5,9,0,0,121,120,1,0,0,0,121,122,1,
        0,0,0,122,124,1,0,0,0,123,118,1,0,0,0,123,121,1,0,0,0,124,125,1,
        0,0,0,125,132,5,8,0,0,126,128,5,10,0,0,127,126,1,0,0,0,127,128,1,
        0,0,0,128,133,1,0,0,0,129,131,5,11,0,0,130,129,1,0,0,0,130,131,1,
        0,0,0,131,133,1,0,0,0,132,127,1,0,0,0,132,130,1,0,0,0,133,19,1,0,
        0,0,134,136,5,37,0,0,135,134,1,0,0,0,135,136,1,0,0,0,136,141,1,0,
        0,0,137,139,5,9,0,0,138,137,1,0,0,0,138,139,1,0,0,0,139,141,1,0,
        0,0,140,135,1,0,0,0,140,138,1,0,0,0,141,142,1,0,0,0,142,149,5,8,
        0,0,143,145,5,10,0,0,144,143,1,0,0,0,144,145,1,0,0,0,145,150,1,0,
        0,0,146,148,5,37,0,0,147,146,1,0,0,0,147,148,1,0,0,0,148,150,1,0,
        0,0,149,144,1,0,0,0,149,147,1,0,0,0,150,21,1,0,0,0,151,154,5,35,
        0,0,152,155,5,12,0,0,153,155,5,13,0,0,154,152,1,0,0,0,154,153,1,
        0,0,0,155,156,1,0,0,0,156,157,5,35,0,0,157,158,5,40,0,0,158,23,1,
        0,0,0,159,160,5,14,0,0,160,161,5,35,0,0,161,25,1,0,0,0,162,163,5,
        15,0,0,163,166,3,28,14,0,164,165,5,16,0,0,165,167,3,28,14,0,166,
        164,1,0,0,0,166,167,1,0,0,0,167,168,1,0,0,0,168,169,5,15,0,0,169,
        27,1,0,0,0,170,171,7,0,0,0,171,29,1,0,0,0,172,174,3,34,17,0,173,
        172,1,0,0,0,173,174,1,0,0,0,174,175,1,0,0,0,175,176,5,35,0,0,176,
        177,5,7,0,0,177,178,3,36,18,0,178,179,5,40,0,0,179,31,1,0,0,0,180,
        182,3,34,17,0,181,180,1,0,0,0,181,182,1,0,0,0,182,184,1,0,0,0,183,
        185,3,38,19,0,184,183,1,0,0,0,184,185,1,0,0,0,185,187,1,0,0,0,186,
        188,5,17,0,0,187,186,1,0,0,0,187,188,1,0,0,0,188,189,1,0,0,0,189,
        190,5,35,0,0,190,191,5,18,0,0,191,192,5,40,0,0,192,33,1,0,0,0,193,
        194,7,1,0,0,194,35,1,0,0,0,195,196,7,2,0,0,196,37,1,0,0,0,197,198,
        7,3,0,0,198,39,1,0,0,0,32,45,53,62,66,72,74,82,86,90,96,99,104,111,
        115,118,121,123,127,130,132,135,138,140,144,147,149,154,166,173,
        181,184,187
    ]

class PlantUMLParser ( Parser ):

    grammarFileName = "PlantUML.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'skinparam'", "'groupInheritance'", "'class'", 
                     "'{'", "'}'", "'abstract'", "':'", "'--'", "'<'", "'>'", 
                     "'o'", "'<|--'", "'--|>'", "'extends'", "'\"'", "'..'", 
                     "'void'", "'()'", "'#'", "'-'", "'~'", "'+'", "'int'", 
                     "'float'", "'str'", "'bool'", "'time'", "'date'", "'datetime'", 
                     "'timedelta'", "'{static}'", "'{abstract}'", "'@startuml'", 
                     "'@enduml'", "<INVALID>", "<INVALID>", "'*'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "Start", "End", "ID", "INT", "ASTK", 
                      "DOUBLE_QUOTE", "WS", "NL" ]

    RULE_domainModel = 0
    RULE_element = 1
    RULE_skinParam = 2
    RULE_class = 3
    RULE_abstract = 4
    RULE_relationship = 5
    RULE_association = 6
    RULE_bidirectional = 7
    RULE_unidirectional = 8
    RULE_aggregation = 9
    RULE_composition = 10
    RULE_inheritance = 11
    RULE_extends = 12
    RULE_cardinality = 13
    RULE_cardinalityVal = 14
    RULE_attribute = 15
    RULE_method = 16
    RULE_visibility = 17
    RULE_primitiveData = 18
    RULE_modifier = 19

    ruleNames =  [ "domainModel", "element", "skinParam", "class", "abstract", 
                   "relationship", "association", "bidirectional", "unidirectional", 
                   "aggregation", "composition", "inheritance", "extends", 
                   "cardinality", "cardinalityVal", "attribute", "method", 
                   "visibility", "primitiveData", "modifier" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    T__9=10
    T__10=11
    T__11=12
    T__12=13
    T__13=14
    T__14=15
    T__15=16
    T__16=17
    T__17=18
    T__18=19
    T__19=20
    T__20=21
    T__21=22
    T__22=23
    T__23=24
    T__24=25
    T__25=26
    T__26=27
    T__27=28
    T__28=29
    T__29=30
    T__30=31
    T__31=32
    Start=33
    End=34
    ID=35
    INT=36
    ASTK=37
    DOUBLE_QUOTE=38
    WS=39
    NL=40

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class DomainModelContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Start(self):
            return self.getToken(PlantUMLParser.Start, 0)

        def NL(self):
            return self.getToken(PlantUMLParser.NL, 0)

        def End(self):
            return self.getToken(PlantUMLParser.End, 0)

        def element(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PlantUMLParser.ElementContext)
            else:
                return self.getTypedRuleContext(PlantUMLParser.ElementContext,i)


        def getRuleIndex(self):
            return PlantUMLParser.RULE_domainModel

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDomainModel" ):
                listener.enterDomainModel(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDomainModel" ):
                listener.exitDomainModel(self)




    def domainModel(self):

        localctx = PlantUMLParser.DomainModelContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_domainModel)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 40
            self.match(PlantUMLParser.Start)
            self.state = 41
            self.match(PlantUMLParser.NL)
            self.state = 45
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while (((_la) & ~0x3f) == 0 and ((1 << _la) & 34359738442) != 0):
                self.state = 42
                self.element()
                self.state = 47
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 48
            self.match(PlantUMLParser.End)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ElementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def skinParam(self):
            return self.getTypedRuleContext(PlantUMLParser.SkinParamContext,0)


        def class_(self):
            return self.getTypedRuleContext(PlantUMLParser.ClassContext,0)


        def relationship(self):
            return self.getTypedRuleContext(PlantUMLParser.RelationshipContext,0)


        def getRuleIndex(self):
            return PlantUMLParser.RULE_element

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterElement" ):
                listener.enterElement(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitElement" ):
                listener.exitElement(self)




    def element(self):

        localctx = PlantUMLParser.ElementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_element)
        try:
            self.state = 53
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [1]:
                self.enterOuterAlt(localctx, 1)
                self.state = 50
                self.skinParam()
                pass
            elif token in [3, 6]:
                self.enterOuterAlt(localctx, 2)
                self.state = 51
                self.class_()
                pass
            elif token in [35]:
                self.enterOuterAlt(localctx, 3)
                self.state = 52
                self.relationship()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SkinParamContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INT(self):
            return self.getToken(PlantUMLParser.INT, 0)

        def NL(self):
            return self.getToken(PlantUMLParser.NL, 0)

        def getRuleIndex(self):
            return PlantUMLParser.RULE_skinParam

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSkinParam" ):
                listener.enterSkinParam(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSkinParam" ):
                listener.exitSkinParam(self)




    def skinParam(self):

        localctx = PlantUMLParser.SkinParamContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_skinParam)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 55
            self.match(PlantUMLParser.T__0)
            self.state = 56
            self.match(PlantUMLParser.T__1)
            self.state = 57
            self.match(PlantUMLParser.INT)
            self.state = 58
            self.match(PlantUMLParser.NL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ClassContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(PlantUMLParser.ID, 0)

        def NL(self, i:int=None):
            if i is None:
                return self.getTokens(PlantUMLParser.NL)
            else:
                return self.getToken(PlantUMLParser.NL, i)

        def abstract(self):
            return self.getTypedRuleContext(PlantUMLParser.AbstractContext,0)


        def extends(self):
            return self.getTypedRuleContext(PlantUMLParser.ExtendsContext,0)


        def attribute(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PlantUMLParser.AttributeContext)
            else:
                return self.getTypedRuleContext(PlantUMLParser.AttributeContext,i)


        def method(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PlantUMLParser.MethodContext)
            else:
                return self.getTypedRuleContext(PlantUMLParser.MethodContext,i)


        def getRuleIndex(self):
            return PlantUMLParser.RULE_class

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterClass" ):
                listener.enterClass(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitClass" ):
                listener.exitClass(self)




    def class_(self):

        localctx = PlantUMLParser.ClassContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_class)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 62
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [6]:
                self.state = 60
                self.abstract()
                pass
            elif token in [3]:
                self.state = 61
                self.match(PlantUMLParser.T__2)
                pass
            else:
                raise NoViableAltException(self)

            self.state = 64
            self.match(PlantUMLParser.ID)
            self.state = 66
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==14:
                self.state = 65
                self.extends()


            self.state = 68
            self.match(PlantUMLParser.T__3)
            self.state = 69
            self.match(PlantUMLParser.NL)
            self.state = 74
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while (((_la) & ~0x3f) == 0 and ((1 << _la) & 40810184704) != 0):
                self.state = 72
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,4,self._ctx)
                if la_ == 1:
                    self.state = 70
                    self.attribute()
                    pass

                elif la_ == 2:
                    self.state = 71
                    self.method()
                    pass


                self.state = 76
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 77
            self.match(PlantUMLParser.T__4)
            self.state = 78
            self.match(PlantUMLParser.NL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AbstractContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return PlantUMLParser.RULE_abstract

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAbstract" ):
                listener.enterAbstract(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAbstract" ):
                listener.exitAbstract(self)




    def abstract(self):

        localctx = PlantUMLParser.AbstractContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_abstract)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 80
            self.match(PlantUMLParser.T__5)
            self.state = 82
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==3:
                self.state = 81
                self.match(PlantUMLParser.T__2)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class RelationshipContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def association(self):
            return self.getTypedRuleContext(PlantUMLParser.AssociationContext,0)


        def inheritance(self):
            return self.getTypedRuleContext(PlantUMLParser.InheritanceContext,0)


        def getRuleIndex(self):
            return PlantUMLParser.RULE_relationship

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterRelationship" ):
                listener.enterRelationship(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitRelationship" ):
                listener.exitRelationship(self)




    def relationship(self):

        localctx = PlantUMLParser.RelationshipContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_relationship)
        try:
            self.state = 86
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,7,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 84
                self.association()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 85
                self.inheritance()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AssociationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(PlantUMLParser.ID)
            else:
                return self.getToken(PlantUMLParser.ID, i)

        def NL(self):
            return self.getToken(PlantUMLParser.NL, 0)

        def bidirectional(self):
            return self.getTypedRuleContext(PlantUMLParser.BidirectionalContext,0)


        def unidirectional(self):
            return self.getTypedRuleContext(PlantUMLParser.UnidirectionalContext,0)


        def aggregation(self):
            return self.getTypedRuleContext(PlantUMLParser.AggregationContext,0)


        def composition(self):
            return self.getTypedRuleContext(PlantUMLParser.CompositionContext,0)


        def cardinality(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PlantUMLParser.CardinalityContext)
            else:
                return self.getTypedRuleContext(PlantUMLParser.CardinalityContext,i)


        def getRuleIndex(self):
            return PlantUMLParser.RULE_association

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAssociation" ):
                listener.enterAssociation(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAssociation" ):
                listener.exitAssociation(self)




    def association(self):

        localctx = PlantUMLParser.AssociationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_association)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 88
            self.match(PlantUMLParser.ID)
            self.state = 90
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==15:
                self.state = 89
                self.cardinality()


            self.state = 96
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,9,self._ctx)
            if la_ == 1:
                self.state = 92
                self.bidirectional()
                pass

            elif la_ == 2:
                self.state = 93
                self.unidirectional()
                pass

            elif la_ == 3:
                self.state = 94
                self.aggregation()
                pass

            elif la_ == 4:
                self.state = 95
                self.composition()
                pass


            self.state = 99
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==15:
                self.state = 98
                self.cardinality()


            self.state = 101
            self.match(PlantUMLParser.ID)
            self.state = 104
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==7:
                self.state = 102
                self.match(PlantUMLParser.T__6)
                self.state = 103
                self.match(PlantUMLParser.ID)


            self.state = 106
            self.match(PlantUMLParser.NL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BidirectionalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return PlantUMLParser.RULE_bidirectional

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBidirectional" ):
                listener.enterBidirectional(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBidirectional" ):
                listener.exitBidirectional(self)




    def bidirectional(self):

        localctx = PlantUMLParser.BidirectionalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_bidirectional)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 108
            self.match(PlantUMLParser.T__7)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class UnidirectionalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.nav_l = None # Token
            self.nav_r = None # Token


        def getRuleIndex(self):
            return PlantUMLParser.RULE_unidirectional

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterUnidirectional" ):
                listener.enterUnidirectional(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitUnidirectional" ):
                listener.exitUnidirectional(self)




    def unidirectional(self):

        localctx = PlantUMLParser.UnidirectionalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_unidirectional)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 111
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==9:
                self.state = 110
                localctx.nav_l = self.match(PlantUMLParser.T__8)


            self.state = 113
            self.match(PlantUMLParser.T__7)
            self.state = 115
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==10:
                self.state = 114
                localctx.nav_r = self.match(PlantUMLParser.T__9)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AggregationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.aggr_l = None # Token
            self.aggr_r = None # Token


        def getRuleIndex(self):
            return PlantUMLParser.RULE_aggregation

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAggregation" ):
                listener.enterAggregation(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAggregation" ):
                listener.exitAggregation(self)




    def aggregation(self):

        localctx = PlantUMLParser.AggregationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_aggregation)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 123
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,16,self._ctx)
            if la_ == 1:
                self.state = 118
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==11:
                    self.state = 117
                    localctx.aggr_l = self.match(PlantUMLParser.T__10)


                pass

            elif la_ == 2:
                self.state = 121
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==9:
                    self.state = 120
                    self.match(PlantUMLParser.T__8)


                pass


            self.state = 125
            self.match(PlantUMLParser.T__7)
            self.state = 132
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,19,self._ctx)
            if la_ == 1:
                self.state = 127
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==10:
                    self.state = 126
                    self.match(PlantUMLParser.T__9)


                pass

            elif la_ == 2:
                self.state = 130
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==11:
                    self.state = 129
                    localctx.aggr_r = self.match(PlantUMLParser.T__10)


                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CompositionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.comp_l = None # Token
            self.comp_r = None # Token

        def ASTK(self, i:int=None):
            if i is None:
                return self.getTokens(PlantUMLParser.ASTK)
            else:
                return self.getToken(PlantUMLParser.ASTK, i)

        def getRuleIndex(self):
            return PlantUMLParser.RULE_composition

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterComposition" ):
                listener.enterComposition(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitComposition" ):
                listener.exitComposition(self)




    def composition(self):

        localctx = PlantUMLParser.CompositionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_composition)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 140
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,22,self._ctx)
            if la_ == 1:
                self.state = 135
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==37:
                    self.state = 134
                    localctx.comp_l = self.match(PlantUMLParser.ASTK)


                pass

            elif la_ == 2:
                self.state = 138
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==9:
                    self.state = 137
                    self.match(PlantUMLParser.T__8)


                pass


            self.state = 142
            self.match(PlantUMLParser.T__7)
            self.state = 149
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,25,self._ctx)
            if la_ == 1:
                self.state = 144
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==10:
                    self.state = 143
                    self.match(PlantUMLParser.T__9)


                pass

            elif la_ == 2:
                self.state = 147
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==37:
                    self.state = 146
                    localctx.comp_r = self.match(PlantUMLParser.ASTK)


                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class InheritanceContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.inh_left = None # Token

        def ID(self, i:int=None):
            if i is None:
                return self.getTokens(PlantUMLParser.ID)
            else:
                return self.getToken(PlantUMLParser.ID, i)

        def NL(self):
            return self.getToken(PlantUMLParser.NL, 0)

        def getRuleIndex(self):
            return PlantUMLParser.RULE_inheritance

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInheritance" ):
                listener.enterInheritance(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInheritance" ):
                listener.exitInheritance(self)




    def inheritance(self):

        localctx = PlantUMLParser.InheritanceContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_inheritance)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 151
            self.match(PlantUMLParser.ID)
            self.state = 154
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [12]:
                self.state = 152
                localctx.inh_left = self.match(PlantUMLParser.T__11)
                pass
            elif token in [13]:
                self.state = 153
                self.match(PlantUMLParser.T__12)
                pass
            else:
                raise NoViableAltException(self)

            self.state = 156
            self.match(PlantUMLParser.ID)
            self.state = 157
            self.match(PlantUMLParser.NL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExtendsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(PlantUMLParser.ID, 0)

        def getRuleIndex(self):
            return PlantUMLParser.RULE_extends

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExtends" ):
                listener.enterExtends(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExtends" ):
                listener.exitExtends(self)




    def extends(self):

        localctx = PlantUMLParser.ExtendsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_extends)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 159
            self.match(PlantUMLParser.T__13)
            self.state = 160
            self.match(PlantUMLParser.ID)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CardinalityContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.min_ = None # CardinalityValContext
            self.max_ = None # CardinalityValContext

        def cardinalityVal(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(PlantUMLParser.CardinalityValContext)
            else:
                return self.getTypedRuleContext(PlantUMLParser.CardinalityValContext,i)


        def getRuleIndex(self):
            return PlantUMLParser.RULE_cardinality

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCardinality" ):
                listener.enterCardinality(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCardinality" ):
                listener.exitCardinality(self)




    def cardinality(self):

        localctx = PlantUMLParser.CardinalityContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_cardinality)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 162
            self.match(PlantUMLParser.T__14)
            self.state = 163
            localctx.min_ = self.cardinalityVal()
            self.state = 166
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==16:
                self.state = 164
                self.match(PlantUMLParser.T__15)
                self.state = 165
                localctx.max_ = self.cardinalityVal()


            self.state = 168
            self.match(PlantUMLParser.T__14)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CardinalityValContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INT(self):
            return self.getToken(PlantUMLParser.INT, 0)

        def ASTK(self):
            return self.getToken(PlantUMLParser.ASTK, 0)

        def getRuleIndex(self):
            return PlantUMLParser.RULE_cardinalityVal

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCardinalityVal" ):
                listener.enterCardinalityVal(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCardinalityVal" ):
                listener.exitCardinalityVal(self)




    def cardinalityVal(self):

        localctx = PlantUMLParser.CardinalityValContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_cardinalityVal)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 170
            _la = self._input.LA(1)
            if not(_la==36 or _la==37):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AttributeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(PlantUMLParser.ID, 0)

        def primitiveData(self):
            return self.getTypedRuleContext(PlantUMLParser.PrimitiveDataContext,0)


        def NL(self):
            return self.getToken(PlantUMLParser.NL, 0)

        def visibility(self):
            return self.getTypedRuleContext(PlantUMLParser.VisibilityContext,0)


        def getRuleIndex(self):
            return PlantUMLParser.RULE_attribute

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAttribute" ):
                listener.enterAttribute(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAttribute" ):
                listener.exitAttribute(self)




    def attribute(self):

        localctx = PlantUMLParser.AttributeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_attribute)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 173
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & 7864320) != 0):
                self.state = 172
                self.visibility()


            self.state = 175
            self.match(PlantUMLParser.ID)
            self.state = 176
            self.match(PlantUMLParser.T__6)
            self.state = 177
            self.primitiveData()
            self.state = 178
            self.match(PlantUMLParser.NL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MethodContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(PlantUMLParser.ID, 0)

        def NL(self):
            return self.getToken(PlantUMLParser.NL, 0)

        def visibility(self):
            return self.getTypedRuleContext(PlantUMLParser.VisibilityContext,0)


        def modifier(self):
            return self.getTypedRuleContext(PlantUMLParser.ModifierContext,0)


        def getRuleIndex(self):
            return PlantUMLParser.RULE_method

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMethod" ):
                listener.enterMethod(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMethod" ):
                listener.exitMethod(self)




    def method(self):

        localctx = PlantUMLParser.MethodContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_method)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 181
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & 7864320) != 0):
                self.state = 180
                self.visibility()


            self.state = 184
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==31 or _la==32:
                self.state = 183
                self.modifier()


            self.state = 187
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==17:
                self.state = 186
                self.match(PlantUMLParser.T__16)


            self.state = 189
            self.match(PlantUMLParser.ID)
            self.state = 190
            self.match(PlantUMLParser.T__17)
            self.state = 191
            self.match(PlantUMLParser.NL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class VisibilityContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return PlantUMLParser.RULE_visibility

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVisibility" ):
                listener.enterVisibility(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVisibility" ):
                listener.exitVisibility(self)




    def visibility(self):

        localctx = PlantUMLParser.VisibilityContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_visibility)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 193
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 7864320) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PrimitiveDataContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return PlantUMLParser.RULE_primitiveData

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPrimitiveData" ):
                listener.enterPrimitiveData(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPrimitiveData" ):
                listener.exitPrimitiveData(self)




    def primitiveData(self):

        localctx = PlantUMLParser.PrimitiveDataContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_primitiveData)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 195
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 2139095040) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ModifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return PlantUMLParser.RULE_modifier

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterModifier" ):
                listener.enterModifier(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitModifier" ):
                listener.exitModifier(self)




    def modifier(self):

        localctx = PlantUMLParser.ModifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_modifier)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 197
            _la = self._input.LA(1)
            if not(_la==31 or _la==32):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





