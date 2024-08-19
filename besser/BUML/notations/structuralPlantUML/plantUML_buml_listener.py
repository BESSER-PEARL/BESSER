from .PlantUMLParser import PlantUMLParser
from .PlantUMLListener import PlantUMLListener

class BUMLGenerationListener(PlantUMLListener):
    """
       This listener class generates a B-UML structural model from a parse-tree that 
       representing a plantUML textual model

       Args:
           output (file): The file to be written with the code for the B-UML model creation.
    """

    visibility = {"+": "public",
                  "-": "private",
                  "#": "protected",
                  "~": "package"}

    def __init__(self, output):
        self.output = output
        self.__attributes: list = []
        self.__methods: list = []
        self.__parameters: list = []
        self.__dtypes: set = set()
        self.__enums: dict = {}
        self.__e_literals: list = []
        self.__classes: dict = {}
        self.__relations: dict = {}
        self.__ends: list = []
        self.__inheritances: dict = {}
        self.__relation_classes: list = []
        self.__group_inh: int = 0
        self.__parent_classes: dict = {}

    def enterClass(self, ctx: PlantUMLParser.ClassContext):
        text = "\n# " + ctx.ID().getText() + " class attributes and methods\n"
        self.output.write(text)
        self.__attributes = []
        self.__methods = []

    def exitClass(self, ctx: PlantUMLParser.ClassContext):
        text=""
        textd = ctx.ID().getText() + ": Class = Class(name=\"" + ctx.ID().getText() + "\""
        if len(self.__attributes) > 0:
            text += ctx.ID().getText() + ".attributes=" + list_to_str(self.__attributes) + "\n"
        if len(self.__methods) > 0:
            text += ctx.ID().getText() + ".methods=" + list_to_str(self.__methods) + "\n"
        if ctx.abstract():
            textd += ", is_abstract=True"
        textd += ")\n"
        self.output.write(text)
        self.__classes[(ctx.ID().getText())] = textd

    def enterAttribute(self, ctx: PlantUMLParser.AttributeContext):
        attribute_name = ctx.parentCtx.ID().getText() + "_" + ctx.ID().getText()
        text = attribute_name + ": Property = Property(name=\"" + ctx.ID().getText() + \
            "\", type=" + self.get_type(ctx.dType())
        if ctx.visibility():
            text += ", visibility=\"" + self.visibility[ctx.visibility().getText()] + "\""
        text += ")\n"
        self.output.write(text)
        self.__attributes.append(attribute_name)

    def get_type(self, ctx: PlantUMLParser.DTypeContext):
        """
        Return the data type.

        Args:
            ctx (PlantUMLParser.DTypeContext): The context containing type information.

        Returns:
            str: The formatted type.
        """
        if ctx.primitiveData():
            attr_type = ctx.primitiveData().getText()
            if attr_type == 'string':
                attr_type = 'str'
            self.__dtypes.add(attr_type)
            attr_type = attr_type + '_type'
        else:
            attr_type = ctx.ID().getText()
        return attr_type

    def enterMethod(self, ctx: PlantUMLParser.MethodContext):
        self.__parameters = []
        method_name = ctx.parentCtx.ID().getText() + "_m_" + ctx.ID().getText()
        text = method_name + ": Method = Method(name=\"" + ctx.ID().getText() + "\", "
        if ctx.visibility():
            text += "visibility=\"" + self.visibility[ctx.visibility().getText()] + "\", "
        if ctx.modifier():
            if ctx.modifier().getText() == "{abstract}":
                text += "is_abstract=True, "
        self.output.write(text)
        self.__methods.append(method_name)

    def enterParameter(self, ctx: PlantUMLParser.ParameterContext):
        text = "Parameter(name=\"" + ctx.ID().getText() + "\", type=" + self.get_type(ctx.dType())
        if ctx.value():
            text += ", default_value="
            if ctx.value().D_QUOTE(0):
                text += "\""
            if ctx.value().ID():
                text += ctx.value().ID().getText()
            if ctx.value().INT():
                text += ctx.value().INT().getText()
            if ctx.value().FLOAT():
                text += ctx.value().FLOAT().getText()
            if ctx.value().D_QUOTE(1):
                text += "\""
        text += ")"
        self.__parameters.append(text)

    def exitMethod(self, ctx: PlantUMLParser.MethodContext):
        text = ""
        if ctx.parameter():
            parameters = list_to_str(self.__parameters)
            text = "parameters=" + parameters + ", "
        if ctx.dType():
            text += "type=" + self.get_type(ctx.dType()) + ")\n"
        else:
            text += "type=None)\n"
        self.output.write(text)

    def enterAssociation(self, ctx: PlantUMLParser.AssociationContext):
        self.__ends = []

    def exitAssociation(self, ctx: PlantUMLParser.AssociationContext):
        cl_name_1 = ctx.ID(0).getText()
        cl_name_2 = ctx.ID(1).getText()
        if ctx.ID(2) is None:
            raise ValueError("All the associations in the model must have a name")
        assoc_name = ctx.ID(2).getText()
        if assoc_name in self.__relations:
            raise ValueError("The model cannot have two associations with the same name")
        text = assoc_name + ": BinaryAssociation = BinaryAssociation(name=\"" + assoc_name + "\", ends={\n\
        Property(name=\"" + assoc_name + "\", type=" + cl_name_1 + ", multiplicity=" + get_multiplicity(ctx.c_left) + self.__ends[0] + "),\n\
        Property(name=\"" + assoc_name + "\", type=" + cl_name_2 + ", multiplicity=" + get_multiplicity(ctx.c_right) + self.__ends[1] + ")})\n"
        self.__relations[assoc_name] = text
        self.__relation_classes.append(cl_name_1)
        self.__relation_classes.append(cl_name_2)

    def enterBidirectional(self, ctx: PlantUMLParser.BidirectionalContext):
        self.__ends = ["", ""]

    def enterUnidirectional(self, ctx: PlantUMLParser.UnidirectionalContext):
        end_1 = ", is_navigable=True" if ctx.nav_l is not None else ", is_navigable=False"
        end_2 = ", is_navigable=True" if ctx.nav_r is not None else ", is_navigable=False"
        self.__ends.append(end_1)
        self.__ends.append(end_2)

    def enterComposition(self, ctx: PlantUMLParser.CompositionContext):
        end_1 = ", is_navigable=False, is_composite=True" if ctx.comp_l is not None else ""
        end_2 = ", is_navigable=False, is_composite=True" if ctx.comp_r is not None else ""
        self.__ends.append(end_1)
        self.__ends.append(end_2)

    def enterAggregation(self, ctx: PlantUMLParser.AggregationContext):
        end_1 = ", is_navigable=False, is_aggregation=True" if ctx.aggr_l is not None else ""
        end_2 = ", is_navigable=False, is_aggregation=True" if ctx.aggr_r is not None else ""
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

    def enterEnumLiteral(self, ctx: PlantUMLParser.EnumLiteralContext):
        self.__e_literals.append("EnumerationLiteral(name=\"" + ctx.ID().getText() + "\")")

    def exitEnumeration(self, ctx: PlantUMLParser.EnumerationContext):
        self.__enums[ctx.ID().getText()] = self.__e_literals
        self.__e_literals = []

    def exitDomainModel(self, ctx: PlantUMLParser.DomainModelContext):
        if len(self.__relations) != 0:
            self.output.write("\n# Relationships\n")
            for relation in self.__relations.values():
                self.output.write(relation)
        if len(self.__inheritances) != 0:
            self.output.write("\n# Generalizations\n")
            for inheritance in self.__inheritances.values():
                self.output.write(inheritance)
            if self.__group_inh > 1:
                self.generalization_set_definition()
        classes = list_to_str(list(self.__classes.keys()))
        associations = list_to_str(list(self.__relations.keys()))
        generalizations = list_to_str(list(self.__inheritances.keys()))
        enumerations = list_to_str(list(self.__enums.keys()))
        space = "\n" + ("\t"*4)
        self.output.write("\n\n# Domain Model\n")
        self.output.write("domain: DomainModel = DomainModel(" + space +
                          "name=\"Domain Model\","+ space +
                          "types=" + classes + "," + space +
                          "associations=" + associations + "," + space + 
                          "generalizations=" + generalizations + "," + space +
                          "enumerations=" + enumerations + space + ")\n")
        text = '''from besser.BUML.metamodel.structural import *  \n\n'''

        # Primitive data types definition
        text += "# Primitive Data Types\n"
        for dtype in self.__dtypes:
            text += dtype + "_type = PrimitiveDataType(\"" + dtype + "\")\n"
        text += "\n"

        # Enumeration definition
        if len(self.__enums) > 0:
            text += "# Enumerations\n"
            for key, value in self.__enums.items():
                literals = ",\n\t\t\t".join(value)
                text += key + " = Enumeration(name=\"" + key + "\", literals = {\n\t\t\t" + literals + "})\n\n"

        # Classes definition
        if len(self.__classes) > 0:
            text += "# Classes\n"
            for key, value in self.__classes.items():
                text += value
        for cls in list(set(self.__relation_classes) - set(self.__classes.keys())):
            text += cls + ": Class = Class(name=\"" + cls + "\")\n"

        self.output.seek(0)
        content = self.output.read()
        self.output.seek(0)
        self.output.write(text + content)

    def generalization_set_definition(self):
        """
            Method to write the generalization definition code
        """
        for key, value in self.__parent_classes.items():
            if len(value) >= self.__group_inh:
                generalizations = ", ".join(value)
                text = key + "_generalization_set: GeneralizationSet = GeneralizationSet(name=\"" \
                        + key + "_gen_set\", generalizations={" + generalizations \
                        + "}, is_disjoint=True, is_complete=True)\n"
                self.output.write(text)

def list_to_str(elements:list):
    """
        Method to transform a list of elements to string

        Args:
           elements (list): The list to transform.
    """
    if len(elements) == 0:
        str_list = "set()"
    else:
        str_list = ", ".join(elements)
        str_list = "{" + str_list + "}"
    return str_list

def get_multiplicity(cardinality:PlantUMLParser.CardinalityContext):
    """
        Method to get the multiplicity or cardinality in str format

        Args:
           cardinality (CardinalityContext): The cardinality context from parse-tree to analyze.
    """
    min_value = ""
    max_value = ""
    multiplicity = ""
    if cardinality is None:
        min_value = "1"
        max_value = "1"
    else:
        if cardinality.cardinalityVal(0).INT():
            min_value = cardinality.cardinalityVal(0).INT().getText()
        elif cardinality.cardinalityVal(0).ASTK():
            min_value = "\"*\""
        if cardinality.cardinalityVal(1) and cardinality.cardinalityVal(1).INT():
            max_value = cardinality.cardinalityVal(1).INT().getText()
        elif cardinality.cardinalityVal(1) and cardinality.cardinalityVal(1).ASTK():
            max_value = "\"*\""
        if max_value == "":
            max_value = min_value
        if max_value == "\"*\"" == min_value:
            min_value = "0"
    multiplicity = "Multiplicity(" + min_value + ", " + max_value + ")"
    return multiplicity
