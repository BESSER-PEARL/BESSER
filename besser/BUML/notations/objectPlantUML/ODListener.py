# Generated from OD.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .ODParser import ODParser
else:
    from ODParser import ODParser
from besser.BUML.metamodel.object.object import *
# This class defines a complete listener for a parse tree produced by ODParser.
class ODListener(ParseTreeListener):

    def __init__(self,objs):
        self.objs = objs

    # Enter a parse tree produced by ODParser#objectDiagram.
    def enterObjectDiagram(self, ctx:ODParser.ObjectDiagramContext):
        pass

    # Exit a parse tree produced by ODParser#objectDiagram.
    def exitObjectDiagram(self, ctx:ODParser.ObjectDiagramContext):
        pass


    # Enter a parse tree produced by ODParser#objectDeclaration.
    def enterObjectDeclaration(self, ctx:ODParser.ObjectDeclarationContext):
        pass

    # Exit a parse tree produced by ODParser#objectDeclaration.
    def exitObjectDeclaration(self, ctx:ODParser.ObjectDeclarationContext):
        pass


    # Enter a parse tree produced by ODParser#objectName.
    def enterObjectName(self, ctx:ODParser.ObjectNameContext):

        self.obj = Object(name = ctx.getText(),classifier= "not found", slots = [])
        self.objs.append(self.obj)

        # print(ctx.getText())

        #

        pass

    # Exit a parse tree produced by ODParser#objectName.
    def exitObjectName(self, ctx:ODParser.ObjectNameContext):

        pass


    # Enter a parse tree produced by ODParser#className.
    def enterClassName(self, ctx:ODParser.ClassNameContext):
        pass

    # Exit a parse tree produced by ODParser#className.
    def exitClassName(self, ctx:ODParser.ClassNameContext):
        pass


    # Enter a parse tree produced by ODParser#propertiesBlock.
    def enterPropertiesBlock(self, ctx:ODParser.PropertiesBlockContext):
        pass

    # Exit a parse tree produced by ODParser#propertiesBlock.
    def exitPropertiesBlock(self, ctx:ODParser.PropertiesBlockContext):
        pass


    # Enter a parse tree produced by ODParser#property.
    def enterProperty(self, ctx:ODParser.PropertyContext):

        pass

    # Exit a parse tree produced by ODParser#property.
    def exitProperty(self, ctx:ODParser.PropertyContext):
        pass


    # Enter a parse tree produced by ODParser#propertyName.
    def enterPropertyName(self, ctx:ODParser.PropertyNameContext):
        self.property = Property(name = ctx.getText(),type='NP')
        pass

    # Exit a parse tree produced by ODParser#propertyName.
    def exitPropertyName(self, ctx:ODParser.PropertyNameContext):
        pass


    # Enter a parse tree produced by ODParser#propertyValue.
    def enterPropertyValue(self, ctx:ODParser.PropertyValueContext):

        prop = AttributeLink(value = ctx.getText(),attribute= self.property )
        self.obj.add_slot(prop)
        pass

    # Exit a parse tree produced by ODParser#propertyValue.
    def exitPropertyValue(self, ctx:ODParser.PropertyValueContext):
        pass


    # Enter a parse tree produced by ODParser#linkDeclaration.
    def enterLinkDeclaration(self, ctx:ODParser.LinkDeclarationContext):

        pass

    # Exit a parse tree produced by ODParser#linkDeclaration.
    def exitLinkDeclaration(self, ctx:ODParser.LinkDeclarationContext):
        linkParts = ctx.getText().split(":")[0].split(self.linkType)
        linkName = ctx.getText().split(":")[1]
        obj1 = self.getObject(linkParts[0])
        obj2 = self.getObject(linkParts[1])
        prop = Property(name = linkName, type= 'NF')
        linkEndLeft = LinkEnd(linkName,prop, obj1)
        linkEndRight = LinkEnd(linkName,prop, obj2)
        link = Link(linkName, None, [linkEndLeft,linkEndRight])
        obj1._add_link(link)
        obj2._add_link(link)


        pass


    # Enter a parse tree produced by ODParser#linkObjectName.
    def enterLinkObjectName(self, ctx:ODParser.LinkObjectNameContext):
        # print(ctx.getText())
        pass

    # Exit a parse tree produced by ODParser#linkObjectName.
    def exitLinkObjectName(self, ctx:ODParser.LinkObjectNameContext):
        pass


    # Enter a parse tree produced by ODParser#linkType.
    def enterLinkType(self, ctx:ODParser.LinkTypeContext):
        # print(ctx.getText())
        self.linkType = ctx.getText()
        pass

    # Exit a parse tree produced by ODParser#linkType.
    def exitLinkType(self, ctx:ODParser.LinkTypeContext):
        pass


    # Enter a parse tree produced by ODParser#linkName.
    def enterLinkName(self, ctx:ODParser.LinkNameContext):
        pass

    # Exit a parse tree produced by ODParser#linkName.
    def exitLinkName(self, ctx:ODParser.LinkNameContext):
        pass

    def getObject(self, param):
        for object in self.objs:
            if object.name == param:
                        return object



del ODParser