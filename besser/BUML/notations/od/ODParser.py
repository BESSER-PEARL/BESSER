# Generated from OD.g4 by ANTLR 4.13.1
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
        4,1,17,80,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,1,0,1,0,1,0,4,0,28,
        8,0,11,0,12,0,29,1,0,1,0,1,1,1,1,1,1,1,1,3,1,38,8,1,1,1,3,1,41,8,
        1,1,2,1,2,1,3,1,3,1,4,1,4,5,4,49,8,4,10,4,12,4,52,9,4,1,4,1,4,1,
        5,1,5,1,5,1,5,1,6,1,6,1,7,4,7,63,8,7,11,7,12,7,64,1,8,1,8,1,8,1,
        8,1,8,3,8,72,8,8,1,9,1,9,1,10,1,10,1,11,1,11,1,11,0,0,12,0,2,4,6,
        8,10,12,14,16,18,20,22,0,3,2,0,13,15,17,17,1,0,5,10,1,0,13,14,74,
        0,24,1,0,0,0,2,33,1,0,0,0,4,42,1,0,0,0,6,44,1,0,0,0,8,46,1,0,0,0,
        10,55,1,0,0,0,12,59,1,0,0,0,14,62,1,0,0,0,16,66,1,0,0,0,18,73,1,
        0,0,0,20,75,1,0,0,0,22,77,1,0,0,0,24,27,5,11,0,0,25,28,3,2,1,0,26,
        28,3,16,8,0,27,25,1,0,0,0,27,26,1,0,0,0,28,29,1,0,0,0,29,27,1,0,
        0,0,29,30,1,0,0,0,30,31,1,0,0,0,31,32,5,12,0,0,32,1,1,0,0,0,33,34,
        5,1,0,0,34,37,3,4,2,0,35,36,5,2,0,0,36,38,3,6,3,0,37,35,1,0,0,0,
        37,38,1,0,0,0,38,40,1,0,0,0,39,41,3,8,4,0,40,39,1,0,0,0,40,41,1,
        0,0,0,41,3,1,0,0,0,42,43,5,13,0,0,43,5,1,0,0,0,44,45,5,13,0,0,45,
        7,1,0,0,0,46,50,5,3,0,0,47,49,3,10,5,0,48,47,1,0,0,0,49,52,1,0,0,
        0,50,48,1,0,0,0,50,51,1,0,0,0,51,53,1,0,0,0,52,50,1,0,0,0,53,54,
        5,4,0,0,54,9,1,0,0,0,55,56,3,12,6,0,56,57,5,2,0,0,57,58,3,14,7,0,
        58,11,1,0,0,0,59,60,5,13,0,0,60,13,1,0,0,0,61,63,7,0,0,0,62,61,1,
        0,0,0,63,64,1,0,0,0,64,62,1,0,0,0,64,65,1,0,0,0,65,15,1,0,0,0,66,
        67,3,18,9,0,67,68,3,20,10,0,68,71,3,18,9,0,69,70,5,2,0,0,70,72,3,
        22,11,0,71,69,1,0,0,0,71,72,1,0,0,0,72,17,1,0,0,0,73,74,5,13,0,0,
        74,19,1,0,0,0,75,76,7,1,0,0,76,21,1,0,0,0,77,78,7,2,0,0,78,23,1,
        0,0,0,7,27,29,37,40,50,64,71
    ]

class ODParser ( Parser ):

    grammarFileName = "OD.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'Object'", "':'", "'{'", "'}'", "'<|--'", 
                     "'o--'", "'--'", "'..'", "'-->'", "'..>'", "'@startuml'", 
                     "'@enduml'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "Start", "End", 
                      "IDENTIFIER", "STRING", "NUMBER", "WHITESPACE", "DATE" ]

    RULE_objectDiagram = 0
    RULE_objectDeclaration = 1
    RULE_objectName = 2
    RULE_className = 3
    RULE_propertiesBlock = 4
    RULE_property = 5
    RULE_propertyName = 6
    RULE_propertyValue = 7
    RULE_linkDeclaration = 8
    RULE_linkObjectName = 9
    RULE_linkType = 10
    RULE_linkName = 11

    ruleNames =  [ "objectDiagram", "objectDeclaration", "objectName", "className", 
                   "propertiesBlock", "property", "propertyName", "propertyValue", 
                   "linkDeclaration", "linkObjectName", "linkType", "linkName" ]

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
    Start=11
    End=12
    IDENTIFIER=13
    STRING=14
    NUMBER=15
    WHITESPACE=16
    DATE=17

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ObjectDiagramContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Start(self):
            return self.getToken(ODParser.Start, 0)

        def End(self):
            return self.getToken(ODParser.End, 0)

        def objectDeclaration(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ODParser.ObjectDeclarationContext)
            else:
                return self.getTypedRuleContext(ODParser.ObjectDeclarationContext,i)


        def linkDeclaration(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ODParser.LinkDeclarationContext)
            else:
                return self.getTypedRuleContext(ODParser.LinkDeclarationContext,i)


        def getRuleIndex(self):
            return ODParser.RULE_objectDiagram

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterObjectDiagram" ):
                listener.enterObjectDiagram(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitObjectDiagram" ):
                listener.exitObjectDiagram(self)




    def objectDiagram(self):

        localctx = ODParser.ObjectDiagramContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_objectDiagram)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 24
            self.match(ODParser.Start)
            self.state = 27 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 27
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [1]:
                    self.state = 25
                    self.objectDeclaration()
                    pass
                elif token in [13]:
                    self.state = 26
                    self.linkDeclaration()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 29 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==1 or _la==13):
                    break

            self.state = 31
            self.match(ODParser.End)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ObjectDeclarationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def objectName(self):
            return self.getTypedRuleContext(ODParser.ObjectNameContext,0)


        def className(self):
            return self.getTypedRuleContext(ODParser.ClassNameContext,0)


        def propertiesBlock(self):
            return self.getTypedRuleContext(ODParser.PropertiesBlockContext,0)


        def getRuleIndex(self):
            return ODParser.RULE_objectDeclaration

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterObjectDeclaration" ):
                listener.enterObjectDeclaration(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitObjectDeclaration" ):
                listener.exitObjectDeclaration(self)




    def objectDeclaration(self):

        localctx = ODParser.ObjectDeclarationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_objectDeclaration)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 33
            self.match(ODParser.T__0)
            self.state = 34
            self.objectName()
            self.state = 37
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==2:
                self.state = 35
                self.match(ODParser.T__1)
                self.state = 36
                self.className()


            self.state = 40
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==3:
                self.state = 39
                self.propertiesBlock()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ObjectNameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDENTIFIER(self):
            return self.getToken(ODParser.IDENTIFIER, 0)

        def getRuleIndex(self):
            return ODParser.RULE_objectName

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterObjectName" ):
                listener.enterObjectName(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitObjectName" ):
                listener.exitObjectName(self)




    def objectName(self):

        localctx = ODParser.ObjectNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_objectName)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 42
            self.match(ODParser.IDENTIFIER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ClassNameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDENTIFIER(self):
            return self.getToken(ODParser.IDENTIFIER, 0)

        def getRuleIndex(self):
            return ODParser.RULE_className

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterClassName" ):
                listener.enterClassName(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitClassName" ):
                listener.exitClassName(self)




    def className(self):

        localctx = ODParser.ClassNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_className)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 44
            self.match(ODParser.IDENTIFIER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PropertiesBlockContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def property_(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ODParser.PropertyContext)
            else:
                return self.getTypedRuleContext(ODParser.PropertyContext,i)


        def getRuleIndex(self):
            return ODParser.RULE_propertiesBlock

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPropertiesBlock" ):
                listener.enterPropertiesBlock(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPropertiesBlock" ):
                listener.exitPropertiesBlock(self)




    def propertiesBlock(self):

        localctx = ODParser.PropertiesBlockContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_propertiesBlock)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 46
            self.match(ODParser.T__2)
            self.state = 50
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==13:
                self.state = 47
                self.property_()
                self.state = 52
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 53
            self.match(ODParser.T__3)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PropertyContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def propertyName(self):
            return self.getTypedRuleContext(ODParser.PropertyNameContext,0)


        def propertyValue(self):
            return self.getTypedRuleContext(ODParser.PropertyValueContext,0)


        def getRuleIndex(self):
            return ODParser.RULE_property

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterProperty" ):
                listener.enterProperty(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitProperty" ):
                listener.exitProperty(self)




    def property_(self):

        localctx = ODParser.PropertyContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_property)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 55
            self.propertyName()
            self.state = 56
            self.match(ODParser.T__1)
            self.state = 57
            self.propertyValue()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PropertyNameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDENTIFIER(self):
            return self.getToken(ODParser.IDENTIFIER, 0)

        def getRuleIndex(self):
            return ODParser.RULE_propertyName

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPropertyName" ):
                listener.enterPropertyName(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPropertyName" ):
                listener.exitPropertyName(self)




    def propertyName(self):

        localctx = ODParser.PropertyNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_propertyName)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 59
            self.match(ODParser.IDENTIFIER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PropertyValueContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDENTIFIER(self, i:int=None):
            if i is None:
                return self.getTokens(ODParser.IDENTIFIER)
            else:
                return self.getToken(ODParser.IDENTIFIER, i)

        def STRING(self, i:int=None):
            if i is None:
                return self.getTokens(ODParser.STRING)
            else:
                return self.getToken(ODParser.STRING, i)

        def NUMBER(self, i:int=None):
            if i is None:
                return self.getTokens(ODParser.NUMBER)
            else:
                return self.getToken(ODParser.NUMBER, i)

        def DATE(self, i:int=None):
            if i is None:
                return self.getTokens(ODParser.DATE)
            else:
                return self.getToken(ODParser.DATE, i)

        def getRuleIndex(self):
            return ODParser.RULE_propertyValue

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPropertyValue" ):
                listener.enterPropertyValue(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPropertyValue" ):
                listener.exitPropertyValue(self)




    def propertyValue(self):

        localctx = ODParser.PropertyValueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_propertyValue)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 62 
            self._errHandler.sync(self)
            _alt = 1
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 61
                    _la = self._input.LA(1)
                    if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 188416) != 0)):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()

                else:
                    raise NoViableAltException(self)
                self.state = 64 
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,5,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LinkDeclarationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def linkObjectName(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ODParser.LinkObjectNameContext)
            else:
                return self.getTypedRuleContext(ODParser.LinkObjectNameContext,i)


        def linkType(self):
            return self.getTypedRuleContext(ODParser.LinkTypeContext,0)


        def linkName(self):
            return self.getTypedRuleContext(ODParser.LinkNameContext,0)


        def getRuleIndex(self):
            return ODParser.RULE_linkDeclaration

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLinkDeclaration" ):
                listener.enterLinkDeclaration(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLinkDeclaration" ):
                listener.exitLinkDeclaration(self)




    def linkDeclaration(self):

        localctx = ODParser.LinkDeclarationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_linkDeclaration)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 66
            self.linkObjectName()
            self.state = 67
            self.linkType()
            self.state = 68
            self.linkObjectName()
            self.state = 71
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==2:
                self.state = 69
                self.match(ODParser.T__1)
                self.state = 70
                self.linkName()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LinkObjectNameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IDENTIFIER(self):
            return self.getToken(ODParser.IDENTIFIER, 0)

        def getRuleIndex(self):
            return ODParser.RULE_linkObjectName

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLinkObjectName" ):
                listener.enterLinkObjectName(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLinkObjectName" ):
                listener.exitLinkObjectName(self)




    def linkObjectName(self):

        localctx = ODParser.LinkObjectNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_linkObjectName)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 73
            self.match(ODParser.IDENTIFIER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class LinkTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return ODParser.RULE_linkType

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLinkType" ):
                listener.enterLinkType(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLinkType" ):
                listener.exitLinkType(self)




    def linkType(self):

        localctx = ODParser.LinkTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_linkType)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 75
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 2016) != 0)):
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


    class LinkNameContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STRING(self):
            return self.getToken(ODParser.STRING, 0)

        def IDENTIFIER(self):
            return self.getToken(ODParser.IDENTIFIER, 0)

        def getRuleIndex(self):
            return ODParser.RULE_linkName

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLinkName" ):
                listener.enterLinkName(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLinkName" ):
                listener.exitLinkName(self)




    def linkName(self):

        localctx = ODParser.LinkNameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_linkName)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 77
            _la = self._input.LA(1)
            if not(_la==13 or _la==14):
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





