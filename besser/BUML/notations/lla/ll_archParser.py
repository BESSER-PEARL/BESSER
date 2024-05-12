# Generated from ./ll_arch.g4 by ANTLR 4.13.1
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
        4,1,29,122,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,1,0,1,0,1,0,1,0,1,0,4,0,20,8,0,11,0,12,0,21,1,0,1,0,1,0,1,0,4,
        0,28,8,0,11,0,12,0,29,1,0,1,0,1,0,1,0,4,0,36,8,0,11,0,12,0,37,1,
        0,1,0,1,0,1,1,1,1,3,1,45,8,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,3,1,3,
        1,3,1,3,1,3,1,3,1,3,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,
        1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,4,4,82,8,4,11,4,12,4,83,
        1,4,1,4,1,5,1,5,1,5,1,5,5,5,92,8,5,10,5,12,5,95,9,5,1,5,1,5,1,6,
        1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,6,
        1,6,1,6,1,6,1,6,1,6,1,6,1,6,0,0,7,0,2,4,6,8,10,12,0,0,120,0,14,1,
        0,0,0,2,44,1,0,0,0,4,46,1,0,0,0,6,53,1,0,0,0,8,60,1,0,0,0,10,87,
        1,0,0,0,12,98,1,0,0,0,14,15,5,1,0,0,15,16,5,2,0,0,16,17,5,3,0,0,
        17,19,5,2,0,0,18,20,3,2,1,0,19,18,1,0,0,0,20,21,1,0,0,0,21,19,1,
        0,0,0,21,22,1,0,0,0,22,23,1,0,0,0,23,24,5,4,0,0,24,25,5,5,0,0,25,
        27,5,2,0,0,26,28,3,8,4,0,27,26,1,0,0,0,28,29,1,0,0,0,29,27,1,0,0,
        0,29,30,1,0,0,0,30,31,1,0,0,0,31,32,5,4,0,0,32,33,5,6,0,0,33,35,
        5,2,0,0,34,36,3,12,6,0,35,34,1,0,0,0,36,37,1,0,0,0,37,35,1,0,0,0,
        37,38,1,0,0,0,38,39,1,0,0,0,39,40,5,4,0,0,40,41,5,4,0,0,41,1,1,0,
        0,0,42,45,3,4,2,0,43,45,3,6,3,0,44,42,1,0,0,0,44,43,1,0,0,0,45,3,
        1,0,0,0,46,47,5,7,0,0,47,48,5,8,0,0,48,49,5,9,0,0,49,50,5,10,0,0,
        50,51,5,28,0,0,51,52,5,11,0,0,52,5,1,0,0,0,53,54,5,12,0,0,54,55,
        5,8,0,0,55,56,5,9,0,0,56,57,5,10,0,0,57,58,5,28,0,0,58,59,5,11,0,
        0,59,7,1,0,0,0,60,61,5,13,0,0,61,62,5,8,0,0,62,63,5,9,0,0,63,64,
        5,10,0,0,64,65,5,28,0,0,65,66,5,14,0,0,66,67,5,15,0,0,67,68,5,10,
        0,0,68,69,5,27,0,0,69,70,5,14,0,0,70,71,5,16,0,0,71,72,5,10,0,0,
        72,73,5,27,0,0,73,74,5,14,0,0,74,75,5,17,0,0,75,76,5,10,0,0,76,77,
        5,29,0,0,77,78,5,14,0,0,78,79,5,18,0,0,79,81,5,10,0,0,80,82,3,10,
        5,0,81,80,1,0,0,0,82,83,1,0,0,0,83,81,1,0,0,0,83,84,1,0,0,0,84,85,
        1,0,0,0,85,86,5,11,0,0,86,9,1,0,0,0,87,88,5,8,0,0,88,93,5,29,0,0,
        89,90,5,14,0,0,90,92,5,29,0,0,91,89,1,0,0,0,92,95,1,0,0,0,93,91,
        1,0,0,0,93,94,1,0,0,0,94,96,1,0,0,0,95,93,1,0,0,0,96,97,5,11,0,0,
        97,11,1,0,0,0,98,99,5,19,0,0,99,100,5,8,0,0,100,101,5,13,0,0,101,
        102,5,10,0,0,102,103,5,29,0,0,103,104,5,14,0,0,104,105,5,20,0,0,
        105,106,5,10,0,0,106,107,5,29,0,0,107,108,5,14,0,0,108,109,5,21,
        0,0,109,110,5,10,0,0,110,111,5,27,0,0,111,112,5,14,0,0,112,113,5,
        22,0,0,113,114,5,10,0,0,114,115,5,27,0,0,115,116,5,14,0,0,116,117,
        5,23,0,0,117,118,5,10,0,0,118,119,5,27,0,0,119,120,5,11,0,0,120,
        13,1,0,0,0,6,21,29,37,44,83,93
    ]

class ll_archParser ( Parser ):

    grammarFileName = "ll_arch.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'LLA'", "'{'", "'clusters'", "'}'", "'apps'", 
                     "'containers'", "'public_cluster'", "'('", "'name'", 
                     "':'", "')'", "'private_cluster'", "'application'", 
                     "','", "'cpu_required'", "'memory_required'", "'image'", 
                     "'components'", "'container'", "'cluster'", "'cpu_limit'", 
                     "'memory_limit'", "'instances'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "WS", "ML_COMMENT", "SL_COMMENT", "INT", "ID", "STRING" ]

    RULE_architecture = 0
    RULE_cluster = 1
    RULE_privateCluster = 2
    RULE_publicCluster = 3
    RULE_application = 4
    RULE_component = 5
    RULE_container = 6

    ruleNames =  [ "architecture", "cluster", "privateCluster", "publicCluster", 
                   "application", "component", "container" ]

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
    WS=24
    ML_COMMENT=25
    SL_COMMENT=26
    INT=27
    ID=28
    STRING=29

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ArchitectureContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def cluster(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ll_archParser.ClusterContext)
            else:
                return self.getTypedRuleContext(ll_archParser.ClusterContext,i)


        def application(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ll_archParser.ApplicationContext)
            else:
                return self.getTypedRuleContext(ll_archParser.ApplicationContext,i)


        def container(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ll_archParser.ContainerContext)
            else:
                return self.getTypedRuleContext(ll_archParser.ContainerContext,i)


        def getRuleIndex(self):
            return ll_archParser.RULE_architecture

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterArchitecture" ):
                listener.enterArchitecture(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitArchitecture" ):
                listener.exitArchitecture(self)




    def architecture(self):

        localctx = ll_archParser.ArchitectureContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_architecture)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 14
            self.match(ll_archParser.T__0)
            self.state = 15
            self.match(ll_archParser.T__1)
            self.state = 16
            self.match(ll_archParser.T__2)
            self.state = 17
            self.match(ll_archParser.T__1)
            self.state = 19 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 18
                self.cluster()
                self.state = 21 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==7 or _la==12):
                    break

            self.state = 23
            self.match(ll_archParser.T__3)
            self.state = 24
            self.match(ll_archParser.T__4)
            self.state = 25
            self.match(ll_archParser.T__1)
            self.state = 27 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 26
                self.application()
                self.state = 29 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==13):
                    break

            self.state = 31
            self.match(ll_archParser.T__3)
            self.state = 32
            self.match(ll_archParser.T__5)
            self.state = 33
            self.match(ll_archParser.T__1)
            self.state = 35 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 34
                self.container()
                self.state = 37 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==19):
                    break

            self.state = 39
            self.match(ll_archParser.T__3)
            self.state = 40
            self.match(ll_archParser.T__3)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ClusterContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def privateCluster(self):
            return self.getTypedRuleContext(ll_archParser.PrivateClusterContext,0)


        def publicCluster(self):
            return self.getTypedRuleContext(ll_archParser.PublicClusterContext,0)


        def getRuleIndex(self):
            return ll_archParser.RULE_cluster

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCluster" ):
                listener.enterCluster(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCluster" ):
                listener.exitCluster(self)




    def cluster(self):

        localctx = ll_archParser.ClusterContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_cluster)
        try:
            self.state = 44
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [7]:
                self.enterOuterAlt(localctx, 1)
                self.state = 42
                self.privateCluster()
                pass
            elif token in [12]:
                self.enterOuterAlt(localctx, 2)
                self.state = 43
                self.publicCluster()
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


    class PrivateClusterContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(ll_archParser.ID, 0)

        def getRuleIndex(self):
            return ll_archParser.RULE_privateCluster

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPrivateCluster" ):
                listener.enterPrivateCluster(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPrivateCluster" ):
                listener.exitPrivateCluster(self)




    def privateCluster(self):

        localctx = ll_archParser.PrivateClusterContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_privateCluster)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 46
            self.match(ll_archParser.T__6)
            self.state = 47
            self.match(ll_archParser.T__7)
            self.state = 48
            self.match(ll_archParser.T__8)
            self.state = 49
            self.match(ll_archParser.T__9)
            self.state = 50
            self.match(ll_archParser.ID)
            self.state = 51
            self.match(ll_archParser.T__10)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PublicClusterContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(ll_archParser.ID, 0)

        def getRuleIndex(self):
            return ll_archParser.RULE_publicCluster

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPublicCluster" ):
                listener.enterPublicCluster(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPublicCluster" ):
                listener.exitPublicCluster(self)




    def publicCluster(self):

        localctx = ll_archParser.PublicClusterContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_publicCluster)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 53
            self.match(ll_archParser.T__11)
            self.state = 54
            self.match(ll_archParser.T__7)
            self.state = 55
            self.match(ll_archParser.T__8)
            self.state = 56
            self.match(ll_archParser.T__9)
            self.state = 57
            self.match(ll_archParser.ID)
            self.state = 58
            self.match(ll_archParser.T__10)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ApplicationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(ll_archParser.ID, 0)

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(ll_archParser.INT)
            else:
                return self.getToken(ll_archParser.INT, i)

        def STRING(self):
            return self.getToken(ll_archParser.STRING, 0)

        def component(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(ll_archParser.ComponentContext)
            else:
                return self.getTypedRuleContext(ll_archParser.ComponentContext,i)


        def getRuleIndex(self):
            return ll_archParser.RULE_application

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterApplication" ):
                listener.enterApplication(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitApplication" ):
                listener.exitApplication(self)




    def application(self):

        localctx = ll_archParser.ApplicationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_application)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 60
            self.match(ll_archParser.T__12)
            self.state = 61
            self.match(ll_archParser.T__7)
            self.state = 62
            self.match(ll_archParser.T__8)
            self.state = 63
            self.match(ll_archParser.T__9)
            self.state = 64
            self.match(ll_archParser.ID)
            self.state = 65
            self.match(ll_archParser.T__13)
            self.state = 66
            self.match(ll_archParser.T__14)
            self.state = 67
            self.match(ll_archParser.T__9)
            self.state = 68
            self.match(ll_archParser.INT)
            self.state = 69
            self.match(ll_archParser.T__13)
            self.state = 70
            self.match(ll_archParser.T__15)
            self.state = 71
            self.match(ll_archParser.T__9)
            self.state = 72
            self.match(ll_archParser.INT)
            self.state = 73
            self.match(ll_archParser.T__13)
            self.state = 74
            self.match(ll_archParser.T__16)
            self.state = 75
            self.match(ll_archParser.T__9)
            self.state = 76
            self.match(ll_archParser.STRING)
            self.state = 77
            self.match(ll_archParser.T__13)
            self.state = 78
            self.match(ll_archParser.T__17)
            self.state = 79
            self.match(ll_archParser.T__9)
            self.state = 81 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 80
                self.component()
                self.state = 83 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==8):
                    break

            self.state = 85
            self.match(ll_archParser.T__10)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ComponentContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STRING(self, i:int=None):
            if i is None:
                return self.getTokens(ll_archParser.STRING)
            else:
                return self.getToken(ll_archParser.STRING, i)

        def getRuleIndex(self):
            return ll_archParser.RULE_component

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterComponent" ):
                listener.enterComponent(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitComponent" ):
                listener.exitComponent(self)




    def component(self):

        localctx = ll_archParser.ComponentContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_component)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 87
            self.match(ll_archParser.T__7)
            self.state = 88
            self.match(ll_archParser.STRING)
            self.state = 93
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==14:
                self.state = 89
                self.match(ll_archParser.T__13)
                self.state = 90
                self.match(ll_archParser.STRING)
                self.state = 95
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 96
            self.match(ll_archParser.T__10)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ContainerContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STRING(self, i:int=None):
            if i is None:
                return self.getTokens(ll_archParser.STRING)
            else:
                return self.getToken(ll_archParser.STRING, i)

        def INT(self, i:int=None):
            if i is None:
                return self.getTokens(ll_archParser.INT)
            else:
                return self.getToken(ll_archParser.INT, i)

        def getRuleIndex(self):
            return ll_archParser.RULE_container

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterContainer" ):
                listener.enterContainer(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitContainer" ):
                listener.exitContainer(self)




    def container(self):

        localctx = ll_archParser.ContainerContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_container)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 98
            self.match(ll_archParser.T__18)
            self.state = 99
            self.match(ll_archParser.T__7)
            self.state = 100
            self.match(ll_archParser.T__12)
            self.state = 101
            self.match(ll_archParser.T__9)
            self.state = 102
            self.match(ll_archParser.STRING)
            self.state = 103
            self.match(ll_archParser.T__13)
            self.state = 104
            self.match(ll_archParser.T__19)
            self.state = 105
            self.match(ll_archParser.T__9)
            self.state = 106
            self.match(ll_archParser.STRING)
            self.state = 107
            self.match(ll_archParser.T__13)
            self.state = 108
            self.match(ll_archParser.T__20)
            self.state = 109
            self.match(ll_archParser.T__9)
            self.state = 110
            self.match(ll_archParser.INT)
            self.state = 111
            self.match(ll_archParser.T__13)
            self.state = 112
            self.match(ll_archParser.T__21)
            self.state = 113
            self.match(ll_archParser.T__9)
            self.state = 114
            self.match(ll_archParser.INT)
            self.state = 115
            self.match(ll_archParser.T__13)
            self.state = 116
            self.match(ll_archParser.T__22)
            self.state = 117
            self.match(ll_archParser.T__9)
            self.state = 118
            self.match(ll_archParser.INT)
            self.state = 119
            self.match(ll_archParser.T__10)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





