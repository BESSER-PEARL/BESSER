# Generated from ./deployment.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .deploymentParser import deploymentParser
else:
    from deploymentParser import deploymentParser

# This class defines a complete listener for a parse tree produced by deploymentParser.
class deploymentListener(ParseTreeListener):

    # Enter a parse tree produced by deploymentParser#architecture.
    def enterArchitecture(self, ctx:deploymentParser.ArchitectureContext):
        pass

    # Exit a parse tree produced by deploymentParser#architecture.
    def exitArchitecture(self, ctx:deploymentParser.ArchitectureContext):
        pass


    # Enter a parse tree produced by deploymentParser#application.
    def enterApplication(self, ctx:deploymentParser.ApplicationContext):
        pass

    # Exit a parse tree produced by deploymentParser#application.
    def exitApplication(self, ctx:deploymentParser.ApplicationContext):
        pass


    # Enter a parse tree produced by deploymentParser#service.
    def enterService(self, ctx:deploymentParser.ServiceContext):
        pass

    # Exit a parse tree produced by deploymentParser#service.
    def exitService(self, ctx:deploymentParser.ServiceContext):
        pass


    # Enter a parse tree produced by deploymentParser#container.
    def enterContainer(self, ctx:deploymentParser.ContainerContext):
        pass

    # Exit a parse tree produced by deploymentParser#container.
    def exitContainer(self, ctx:deploymentParser.ContainerContext):
        pass


    # Enter a parse tree produced by deploymentParser#deployment.
    def enterDeployment(self, ctx:deploymentParser.DeploymentContext):
        pass

    # Exit a parse tree produced by deploymentParser#deployment.
    def exitDeployment(self, ctx:deploymentParser.DeploymentContext):
        pass


    # Enter a parse tree produced by deploymentParser#region.
    def enterRegion(self, ctx:deploymentParser.RegionContext):
        pass

    # Exit a parse tree produced by deploymentParser#region.
    def exitRegion(self, ctx:deploymentParser.RegionContext):
        pass


    # Enter a parse tree produced by deploymentParser#cluster.
    def enterCluster(self, ctx:deploymentParser.ClusterContext):
        pass

    # Exit a parse tree produced by deploymentParser#cluster.
    def exitCluster(self, ctx:deploymentParser.ClusterContext):
        pass


    # Enter a parse tree produced by deploymentParser#publicCluster.
    def enterPublicCluster(self, ctx:deploymentParser.PublicClusterContext):
        pass

    # Exit a parse tree produced by deploymentParser#publicCluster.
    def exitPublicCluster(self, ctx:deploymentParser.PublicClusterContext):
        pass


    # Enter a parse tree produced by deploymentParser#privateCluster.
    def enterPrivateCluster(self, ctx:deploymentParser.PrivateClusterContext):
        pass

    # Exit a parse tree produced by deploymentParser#privateCluster.
    def exitPrivateCluster(self, ctx:deploymentParser.PrivateClusterContext):
        pass


    # Enter a parse tree produced by deploymentParser#service_list.
    def enterService_list(self, ctx:deploymentParser.Service_listContext):
        pass

    # Exit a parse tree produced by deploymentParser#service_list.
    def exitService_list(self, ctx:deploymentParser.Service_listContext):
        pass


    # Enter a parse tree produced by deploymentParser#deployment_list.
    def enterDeployment_list(self, ctx:deploymentParser.Deployment_listContext):
        pass

    # Exit a parse tree produced by deploymentParser#deployment_list.
    def exitDeployment_list(self, ctx:deploymentParser.Deployment_listContext):
        pass


    # Enter a parse tree produced by deploymentParser#region_list.
    def enterRegion_list(self, ctx:deploymentParser.Region_listContext):
        pass

    # Exit a parse tree produced by deploymentParser#region_list.
    def exitRegion_list(self, ctx:deploymentParser.Region_listContext):
        pass


    # Enter a parse tree produced by deploymentParser#protocol.
    def enterProtocol(self, ctx:deploymentParser.ProtocolContext):
        pass

    # Exit a parse tree produced by deploymentParser#protocol.
    def exitProtocol(self, ctx:deploymentParser.ProtocolContext):
        pass


    # Enter a parse tree produced by deploymentParser#service_type.
    def enterService_type(self, ctx:deploymentParser.Service_typeContext):
        pass

    # Exit a parse tree produced by deploymentParser#service_type.
    def exitService_type(self, ctx:deploymentParser.Service_typeContext):
        pass


    # Enter a parse tree produced by deploymentParser#provider.
    def enterProvider(self, ctx:deploymentParser.ProviderContext):
        pass

    # Exit a parse tree produced by deploymentParser#provider.
    def exitProvider(self, ctx:deploymentParser.ProviderContext):
        pass


    # Enter a parse tree produced by deploymentParser#boolean.
    def enterBoolean(self, ctx:deploymentParser.BooleanContext):
        pass

    # Exit a parse tree produced by deploymentParser#boolean.
    def exitBoolean(self, ctx:deploymentParser.BooleanContext):
        pass



del deploymentParser