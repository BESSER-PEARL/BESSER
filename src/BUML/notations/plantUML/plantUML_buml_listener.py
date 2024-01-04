from .PlantUMLParser import PlantUMLParser
from .PlantUMLListener import PlantUMLListener

class BUMLGenerationListener(PlantUMLListener):
    visibility = {"+": "public",
                  "-": "private",
                  "#": "protected",
                  "~": "package"}

    def __init__(self, output):
        self.output = output
        self.__attr_list: list = list()
        self.__abstract_class: bool = False
        self.__dtypes: set = set()
        self.__classes: list = list()
        self.__relations: dict = dict()
        self.__ends: list = list()
        self.__inheritances: dict = dict()
        self.__relation_classes: list = list()
        self.__group_inh: int = 0
        self.__parent_classes: dict = dict()
        
    def enterClass(self, ctx: PlantUMLParser.ClassContext):
        text = "# " + ctx.ID().getText() + " class definition \n"
        self.output.write(text)
        self.__attr_list = []
        self.__abstract_class = False
    
    def exitClass(self, ctx: PlantUMLParser.ClassContext):
        attributes = list_to_str(self.__attr_list)
        text = ctx.ID().getText() + ": Class = Class(name=\"" + ctx.ID().getText() + "\", attributes=" + attributes
        if self.__abstract_class:
            text += ", is_abstract=True"
        text += ")\n\n"
        self.output.write(text)
        self.__classes.append(ctx.ID().getText())

    def enterAttribute(self, ctx: PlantUMLParser.AttributeContext):
        attribute_name = ctx.parentCtx.ID().getText() + "_" + ctx.ID().getText()
        text = attribute_name + ": Property = Property(name=\"" + ctx.ID().getText() + \
            "\", property_type="+ ctx.primitiveData().getText() +"_type"
        if ctx.visibility():
            text += ", visibility=\"" + self.visibility[ctx.visibility().getText()] + "\""
        text += ")\n"
        self.output.write(text)
        self.__attr_list.append(attribute_name)
        self.__dtypes.add(ctx.primitiveData().getText())

    def enterAbstract(self, ctx: PlantUMLParser.AbstractContext):
        self.__abstract_class = True
    
    def enterAssociation(self, ctx: PlantUMLParser.AssociationContext):
        self.__ends = []
            
    def exitAssociation(self, ctx: PlantUMLParser.AssociationContext):
        cl_name_1 = ctx.ID(0).getText()
        cl_name_2 = ctx.ID(1).getText()
        if ctx.ID(2) is None:
            association_name = cl_name_1 + "_" + cl_name_2
            end1 = "end_" + cl_name_1 + "_" + cl_name_2
            end2 = "end_" + cl_name_2 + "_" + cl_name_1
        else:
            association_name = ctx.ID(2).getText()
            end1 = "end_" + cl_name_1 + "_" + association_name
            end2 = "end_" + cl_name_2 + "_" + association_name
        text = association_name + ": BinaryAssociation = BinaryAssociation(name=\"" + association_name + "\", ends={\n\
        Property(name=\"" + end1 + "\", property_type=" + cl_name_1 + ", multiplicity=" + getMultiplicity(ctx.cardinality(0)) + self.__ends[0] + "),\n\
        Property(name=\"" + end2 + "\", property_type=" + cl_name_2 + ", multiplicity=" + getMultiplicity(ctx.cardinality(1)) + self.__ends[1] + ")})\n"
        self.__relations[association_name] = text
        self.__relation_classes.append(cl_name_1)
        self.__relation_classes.append(cl_name_2)

    def enterBidirectional(self, ctx: PlantUMLParser.BidirectionalContext):
        self.__ends = ["", ""]
    
    def enterUnidirectional(self, ctx: PlantUMLParser.UnidirectionalContext):
        end_1 = ", is_navigable=True" if ctx.nav_l is not None else ""
        end_2 = ", is_navigable=True" if ctx.nav_r is not None else ""
        self.__ends.append(end_1)
        self.__ends.append(end_2)

    def enterComposition(self, ctx: PlantUMLParser.CompositionContext):
        end_1 = ", is_composite=True" if ctx.comp_l is not None else ""
        end_2 = ", is_composite=True" if ctx.comp_r is not None else ""
        self.__ends.append(end_1)
        self.__ends.append(end_2)

    def enterAggregation(self, ctx: PlantUMLParser.AggregationContext):
        end_1 = ", is_aggregation=True" if ctx.aggr_l is not None else ""
        end_2 = ", is_aggregation=True" if ctx.aggr_r is not None else ""
        self.__ends.append(end_1)
        self.__ends.append(end_2)
    
    def enterInheritance(self, ctx: PlantUMLParser.InheritanceContext):
        if ctx.inh_left:
            general = ctx.ID(0).getText()
            specific = ctx.ID(1).getText()
        else:
            general = ctx.ID(1).getText()
            specific = ctx.ID(0).getText()
        inheritance_name = "gen_" + general + "_" + specific
        text = inheritance_name + ": Generalization = Generalization(general=" + general + ", specific=" + specific + ")\n"
        self.__inheritances[inheritance_name] = text
        self.__relation_classes.append(general)
        self.__relation_classes.append(specific)

        if general not in self.__parent_classes:
            self.__parent_classes[general] = []
        self.__parent_classes[general].append(inheritance_name)

    def enterExtends(self, ctx: PlantUMLParser.ExtendsContext):
        general = ctx.ID().getText()
        specific = ctx.parentCtx.ID().getText()
        inheritance_name = "gen_" + general + "_" + specific
        text = inheritance_name + ": Generalization = Generalization(general=" + general + ", specific=" + specific + ")\n"
        self.__inheritances[inheritance_name] = text
    
    def enterSkinParam(self, ctx: PlantUMLParser.SkinParamContext):
        self.__group_inh = int(ctx.INT().getText())

    def exitDomainModel(self, ctx: PlantUMLParser.DomainModelContext):
        self.check_classes_definition()
        self.output.write("# Relationships\n")
        for relation in self.__relations.values():
            self.output.write(relation)
        self.output.write("\n# Generalizations\n")
        for inheritance in self.__inheritances.values():
            self.output.write(inheritance)
        if self.__group_inh > 1:
            self.create_generalization_set()
        classes = list_to_str(self.__classes)
        associations = list_to_str(list(self.__relations.keys()))
        generalizations = list_to_str(list(self.__inheritances.keys()))
        self.output.write("\n\n# Domain Model\n")
        self.output.write("domain: DomainModel = DomainModel(name=\"Domain Model\", types=" + classes + ", associations=" + associations + ", generalizations=" + generalizations + ")")
        text = '''from BUML.metamodel.structural import NamedElement, DomainModel, Type, Class, \\
        Property, PrimitiveDataType, Multiplicity, Association, BinaryAssociation, Generalization, \\
        GeneralizationSet, AssociationClass \n\n'''
        text += "# Primitive Data Types \n"
        for dtype in self.__dtypes:
            text += dtype + "_type = PrimitiveDataType(\"" + dtype + "\")\n"
        text += "\n"
        self.output.seek(0)
        content = self.output.read()
        self.output.seek(0)
        self.output.write(text + content)

    def check_classes_definition(self):
        for cls in list(set(self.__relation_classes) - set(self.__classes)):
            text = "# " + cls + " class definition \n"
            text += cls + ": Class = Class(name=\"" + cls + "\", attributes={})\n\n"
            self.output.write(text)

    def create_generalization_set(self):
        for key, value in self.__parent_classes.items():
            if len(value) >= self.__group_inh:
                generalizations = ", ".join(value)
                text = key + "_generalization_set: GeneralizationSet = GeneralizationSet(name=\"" + key + \
                     "_gen_set\", generalizations={" + generalizations + "}, is_disjoint=True, is_complete=True)\n"
                self.output.write(text)
    
def list_to_str(list:list):
    if len(list) == 0:
        str_list = "set()"
    else:
        str_list = ", ".join(list)
        str_list = "{" + str_list + "}"
    return str_list

def getMultiplicity(car:PlantUMLParser.CardinalityContext):
    min = ""
    max = ""
    cardinality = ""
    if car is None:
        min = "1"
        max = "1"
    if car.cardinalityVal(0).INT():
        min = car.cardinalityVal(0).INT().getText()
    elif car.cardinalityVal(0).ASTK():
        min = "\"*\""
    if car.cardinalityVal(1) and car.cardinalityVal(1).INT():
        max = car.cardinalityVal(1).INT().getText()
    elif car.cardinalityVal(1) and car.cardinalityVal(1).ASTK():
        max = "\"*\""
    if max == "":
        max = min
    if max == "\"*\"" == min:
        min = "0"

    cardinality = "Multiplicity(" + min + ", " + max + ")"
    return (cardinality)
