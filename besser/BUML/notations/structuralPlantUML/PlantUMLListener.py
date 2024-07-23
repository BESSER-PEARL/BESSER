# Generated from ./PlantUML.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .PlantUMLParser import PlantUMLParser
else:
    from PlantUMLParser import PlantUMLParser

# This class defines a complete listener for a parse tree produced by PlantUMLParser.
class PlantUMLListener(ParseTreeListener):

    # Enter a parse tree produced by PlantUMLParser#domainModel.
    def enterDomainModel(self, ctx:PlantUMLParser.DomainModelContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#domainModel.
    def exitDomainModel(self, ctx:PlantUMLParser.DomainModelContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#element.
    def enterElement(self, ctx:PlantUMLParser.ElementContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#element.
    def exitElement(self, ctx:PlantUMLParser.ElementContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#skinParam.
    def enterSkinParam(self, ctx:PlantUMLParser.SkinParamContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#skinParam.
    def exitSkinParam(self, ctx:PlantUMLParser.SkinParamContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#class.
    def enterClass(self, ctx:PlantUMLParser.ClassContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#class.
    def exitClass(self, ctx:PlantUMLParser.ClassContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#abstract.
    def enterAbstract(self, ctx:PlantUMLParser.AbstractContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#abstract.
    def exitAbstract(self, ctx:PlantUMLParser.AbstractContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#relationship.
    def enterRelationship(self, ctx:PlantUMLParser.RelationshipContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#relationship.
    def exitRelationship(self, ctx:PlantUMLParser.RelationshipContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#association.
    def enterAssociation(self, ctx:PlantUMLParser.AssociationContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#association.
    def exitAssociation(self, ctx:PlantUMLParser.AssociationContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#bidirectional.
    def enterBidirectional(self, ctx:PlantUMLParser.BidirectionalContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#bidirectional.
    def exitBidirectional(self, ctx:PlantUMLParser.BidirectionalContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#unidirectional.
    def enterUnidirectional(self, ctx:PlantUMLParser.UnidirectionalContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#unidirectional.
    def exitUnidirectional(self, ctx:PlantUMLParser.UnidirectionalContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#aggregation.
    def enterAggregation(self, ctx:PlantUMLParser.AggregationContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#aggregation.
    def exitAggregation(self, ctx:PlantUMLParser.AggregationContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#composition.
    def enterComposition(self, ctx:PlantUMLParser.CompositionContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#composition.
    def exitComposition(self, ctx:PlantUMLParser.CompositionContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#inheritance.
    def enterInheritance(self, ctx:PlantUMLParser.InheritanceContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#inheritance.
    def exitInheritance(self, ctx:PlantUMLParser.InheritanceContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#extends.
    def enterExtends(self, ctx:PlantUMLParser.ExtendsContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#extends.
    def exitExtends(self, ctx:PlantUMLParser.ExtendsContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#cardinality.
    def enterCardinality(self, ctx:PlantUMLParser.CardinalityContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#cardinality.
    def exitCardinality(self, ctx:PlantUMLParser.CardinalityContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#cardinalityVal.
    def enterCardinalityVal(self, ctx:PlantUMLParser.CardinalityValContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#cardinalityVal.
    def exitCardinalityVal(self, ctx:PlantUMLParser.CardinalityValContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#attribute.
    def enterAttribute(self, ctx:PlantUMLParser.AttributeContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#attribute.
    def exitAttribute(self, ctx:PlantUMLParser.AttributeContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#method.
    def enterMethod(self, ctx:PlantUMLParser.MethodContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#method.
    def exitMethod(self, ctx:PlantUMLParser.MethodContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#parameter.
    def enterParameter(self, ctx:PlantUMLParser.ParameterContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#parameter.
    def exitParameter(self, ctx:PlantUMLParser.ParameterContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#value.
    def enterValue(self, ctx:PlantUMLParser.ValueContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#value.
    def exitValue(self, ctx:PlantUMLParser.ValueContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#dType.
    def enterDType(self, ctx:PlantUMLParser.DTypeContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#dType.
    def exitDType(self, ctx:PlantUMLParser.DTypeContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#enumeration.
    def enterEnumeration(self, ctx:PlantUMLParser.EnumerationContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#enumeration.
    def exitEnumeration(self, ctx:PlantUMLParser.EnumerationContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#enumLiteral.
    def enterEnumLiteral(self, ctx:PlantUMLParser.EnumLiteralContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#enumLiteral.
    def exitEnumLiteral(self, ctx:PlantUMLParser.EnumLiteralContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#visibility.
    def enterVisibility(self, ctx:PlantUMLParser.VisibilityContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#visibility.
    def exitVisibility(self, ctx:PlantUMLParser.VisibilityContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#primitiveData.
    def enterPrimitiveData(self, ctx:PlantUMLParser.PrimitiveDataContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#primitiveData.
    def exitPrimitiveData(self, ctx:PlantUMLParser.PrimitiveDataContext):
        pass


    # Enter a parse tree produced by PlantUMLParser#modifier.
    def enterModifier(self, ctx:PlantUMLParser.ModifierContext):
        pass

    # Exit a parse tree produced by PlantUMLParser#modifier.
    def exitModifier(self, ctx:PlantUMLParser.ModifierContext):
        pass



del PlantUMLParser