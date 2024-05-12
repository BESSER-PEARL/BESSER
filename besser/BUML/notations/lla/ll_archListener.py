# Generated from ./ll_arch.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .ll_archParser import ll_archParser
else:
    from ll_archParser import ll_archParser

# This class defines a complete listener for a parse tree produced by ll_archParser.
class ll_archListener(ParseTreeListener):

    # Enter a parse tree produced by ll_archParser#architecture.
    def enterArchitecture(self, ctx:ll_archParser.ArchitectureContext):
        pass

    # Exit a parse tree produced by ll_archParser#architecture.
    def exitArchitecture(self, ctx:ll_archParser.ArchitectureContext):
        pass


    # Enter a parse tree produced by ll_archParser#cluster.
    def enterCluster(self, ctx:ll_archParser.ClusterContext):
        pass

    # Exit a parse tree produced by ll_archParser#cluster.
    def exitCluster(self, ctx:ll_archParser.ClusterContext):
        pass


    # Enter a parse tree produced by ll_archParser#privateCluster.
    def enterPrivateCluster(self, ctx:ll_archParser.PrivateClusterContext):
        pass

    # Exit a parse tree produced by ll_archParser#privateCluster.
    def exitPrivateCluster(self, ctx:ll_archParser.PrivateClusterContext):
        pass


    # Enter a parse tree produced by ll_archParser#publicCluster.
    def enterPublicCluster(self, ctx:ll_archParser.PublicClusterContext):
        pass

    # Exit a parse tree produced by ll_archParser#publicCluster.
    def exitPublicCluster(self, ctx:ll_archParser.PublicClusterContext):
        pass


    # Enter a parse tree produced by ll_archParser#application.
    def enterApplication(self, ctx:ll_archParser.ApplicationContext):
        pass

    # Exit a parse tree produced by ll_archParser#application.
    def exitApplication(self, ctx:ll_archParser.ApplicationContext):
        pass


    # Enter a parse tree produced by ll_archParser#component.
    def enterComponent(self, ctx:ll_archParser.ComponentContext):
        pass

    # Exit a parse tree produced by ll_archParser#component.
    def exitComponent(self, ctx:ll_archParser.ComponentContext):
        pass


    # Enter a parse tree produced by ll_archParser#container.
    def enterContainer(self, ctx:ll_archParser.ContainerContext):
        pass

    # Exit a parse tree produced by ll_archParser#container.
    def exitContainer(self, ctx:ll_archParser.ContainerContext):
        pass



del ll_archParser