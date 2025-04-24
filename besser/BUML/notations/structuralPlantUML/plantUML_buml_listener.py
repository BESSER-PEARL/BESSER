import warnings
from besser.BUML.metamodel.structural import DomainModel, Class, Multiplicity, Property, \
    BinaryAssociation, Generalization, Enumeration, EnumerationLiteral, Method, Parameter
from .PlantUMLParser import PlantUMLParser
from .PlantUMLListener import PlantUMLListener

class BUMLGenerationListener(PlantUMLListener):
    """
       This listener class generates a B-UML structural model from a parse-tree that 
       representing a plantUML textual model
    """

    VISIBILITY = {"+": "public", "-": "private", "#": "protected", "~": "package"}

    def __init__(self):
        self.__buml_model = None

    def get_buml_model(self):
        """DomainModel: Retrieves the B-UML model instance."""
        return self.__buml_model

    def enterDomainModel(self, ctx: PlantUMLParser.DomainModelContext):
        self.__buml_model = DomainModel(name="DomainModel")
        classes = self.find_descendant_nodes_by_type(node=ctx,
                                                target_type=PlantUMLParser.ClassContext)
        for cl in classes:
            abstract = True if cl.abstract() else False
            new_class: Class = Class(name=cl.ID().getText(), is_abstract=abstract)
            self.__buml_model.add_type(new_class)
            # check extends
            if cl.extends():
                general_cl_name = cl.extends().ID().getText()
                general_class = self.__buml_model.get_class_by_name(class_name=general_cl_name)
                if general_class is None:
                    general_class = Class(name=general_cl_name)
                self.__buml_model.add_type(general_class)
                # create generalization
                new_generalization: Generalization = Generalization(general=general_class,
                                                                    specific=new_class)
                self.__buml_model.add_generalization(new_generalization)

        enums = self.find_descendant_nodes_by_type(node=ctx,
                                                target_type=PlantUMLParser.EnumerationContext)
        for enum in enums:
            new_enum: Enumeration = Enumeration(name=enum.ID().getText())
            self.__buml_model.add_type(new_enum)

    def find_descendant_nodes_by_type(self, node, target_type):
        """
        Recursively finds and returns all descendant nodes of a specified type.

        Args:
            node: The node to search for descendants. This can be any node in the parse tree.
            target_type: The type of node to match against. This should be a class type that 
                        the nodes are expected to be instances of.

        Returns:
            A list of nodes that are instances of the specified target type. 
            If no matching nodes are found, an empty list is returned.
        """
        matching_nodes = []

        if isinstance(node, target_type):
            matching_nodes.append(node)

        for i in range(node.getChildCount()):
            child = node.getChild(i)
            matching_nodes.extend(self.find_descendant_nodes_by_type(child, target_type))

        return matching_nodes

    def enterAttribute(self, ctx: PlantUMLParser.AttributeContext):
        cl_name = ctx.parentCtx.ID().getText()
        attr_name = ctx.ID().getText()

        visibility = self.VISIBILITY[ctx.visibility().getText()] if ctx.visibility() \
            else "public"

        if ctx.dType().primitiveData():
            primitive_type = "str" if ctx.dType().primitiveData().getText() == "string" \
                else ctx.dType().primitiveData().getText()
            attr_type = self.__buml_model.get_type_by_name(primitive_type)
        else:
            attr_type = self.__buml_model.get_type_by_name(ctx.dType().ID().getText())
        if attr_type is None:
            raise ValueError("Use a valid type for the \"" + attr_name +  "\" attribute")

        new_attr: Property = Property(name=attr_name, type=attr_type, visibility=visibility)
        cl = self.__buml_model.get_class_by_name(class_name=cl_name)
        cl.add_attribute(attribute=new_attr)


    def enterMethod(self, ctx: PlantUMLParser.MethodContext):
        cl_name = ctx.parentCtx.ID().getText()
        method_name = ctx.ID().getText()

        visibility = self.VISIBILITY[ctx.visibility().getText()] if ctx.visibility() \
            else "public"
        abstract = False
        if ctx.modifier():
            abstract = True if ctx.modifier().getText() == "{abstract}" else False

        method_type = None
        if ctx.dType():
            if ctx.dType().primitiveData():
                primitive_type = "str" if ctx.dType().primitiveData().getText() == "string" \
                    else ctx.dType().primitiveData().getText()
                method_type = self.__buml_model.get_type_by_name(primitive_type)
            else:
                method_type = self.__buml_model.get_type_by_name(ctx.dType().ID().getText())
                if method_type is None:
                    raise ValueError("Use a valid type for the \"" + method_name +  "\" method")

        new_method: Method = Method(name=method_name,
                                    visibility=visibility,
                                    is_abstract=abstract,
                                    type=method_type)

        cl = self.__buml_model.get_class_by_name(class_name=cl_name)
        cl.add_method(method=new_method)

    def enterParameter(self, ctx: PlantUMLParser.ParameterContext):
        cl_name = ctx.parentCtx.parentCtx.ID().getText()
        method_name = ctx.parentCtx.ID().getText()
        param_name = ctx.ID().getText()

        if ctx.dType().primitiveData():
            primitive_type = "str" if ctx.dType().primitiveData().getText() == "string" \
                else ctx.dType().primitiveData().getText()
            param_type = self.__buml_model.get_type_by_name(primitive_type)
        else:
            param_type = self.__buml_model.get_type_by_name(ctx.dType().ID().getText())
        if param_type is None:
            raise ValueError("Use a valid type for the \"" + param_name +  "\" parameter")

        default_value = None
        if ctx.value():
            if ctx.value().ID():
                default_value = ctx.value().ID().getText()
            elif ctx.value().INT():
                default_value = int(ctx.value().INT().getText())
            elif ctx.value().FLOAT():
                default_value = float(ctx.value().FLOAT().getText())
            if ctx.value().D_QUOTE(0) and ctx.value().D_QUOTE(1):
                default_value = str(default_value)

        new_param: Parameter = Parameter(name=param_name,
                                        type= param_type,
                                        default_value=default_value)
        methods = self.__buml_model.get_class_by_name(class_name=cl_name).methods
        for method in methods:
            if method.name == method_name:
                method.add_parameter(new_param)

    def enterInheritance(self, ctx: PlantUMLParser.InheritanceContext):
        if ctx.inh_left:
            general = self.__buml_model.get_class_by_name(ctx.ID(0).getText())
            specific = self.__buml_model.get_class_by_name(ctx.ID(1).getText())
        else:
            general = self.__buml_model.get_class_by_name(ctx.ID(1).getText())
            specific = self.__buml_model.get_class_by_name(ctx.ID(0).getText())
        new_generalization: Generalization = Generalization(general=general, specific=specific)
        self.__buml_model.add_generalization(new_generalization)

    def enterAssociation(self, ctx: PlantUMLParser.AssociationContext):
        cl_left = self.__buml_model.get_class_by_name(ctx.ID(0).getText())
        cl_right = self.__buml_model.get_class_by_name(ctx.ID(1).getText())
        navigation = [True, True]
        mult_left: Multiplicity = Multiplicity(min_multiplicity=1, max_multiplicity=1)
        mult_right: Multiplicity = Multiplicity(min_multiplicity=1, max_multiplicity=1)
        composition = [False, False]

        if ctx.ID(2) is None:
            assoc_name = cl_left.name + "_" + cl_right.name
            warnings.warn(
                f"No name was provided for the association between '{cl_left.name}' and "
                f"'{cl_right.name}'. A default name '{assoc_name}' will be auto-assigned.",
                UserWarning
            )
        else:
            assoc_name = ctx.ID(2).getText()
            for type_ in self.__buml_model.types:
                if type_.name.lower() == assoc_name.lower():
                    warnings.warn(
                        f"The association name '{assoc_name}' is identical or similar to "
                        f"the class/enumeration/type '{type_.name}', which may cause issues "
                        f"with some code generators.", UserWarning
                    )

        if ctx.unidirectional():
            unidirectional_ctx = ctx.unidirectional()
            navigation[0] = bool(unidirectional_ctx.nav_l)
            navigation[1] = bool(unidirectional_ctx.nav_r)

        if ctx.composition():
            composition_ctx = ctx.composition()
            composition[0] = bool(composition_ctx.comp_l)
            composition[1] = bool(composition_ctx.comp_r)

        if ctx.c_left:
            mult_left = self.get_cardinality(ctx.c_left)
        if ctx.c_right:
            mult_right = self.get_cardinality(ctx.c_right)

        end_left: Property = Property(name=ctx.ID(0).getText(),
                                    type=cl_left,
                                    multiplicity=mult_left,
                                    is_composite=composition[0],
                                    is_navigable=navigation[0])
        end_right: Property = Property(name=ctx.ID(1).getText(),
                                    type=cl_right,
                                    multiplicity=mult_right,
                                    is_composite=composition[1],
                                    is_navigable=navigation[1])
        new_association:BinaryAssociation = BinaryAssociation(name=assoc_name,
                                    ends={end_left, end_right})

        self.__buml_model.add_association(new_association)

    def get_cardinality(self, ctx: PlantUMLParser.CardinalityContext):
        """
        Extracts and constructs the cardinality from the given context.

        Args:
            ctx (PlantUMLParser.CardinalityContext): The context object that 
                contains the cardinality values to be parsed.

        Returns:
            Multiplicity: An B-UML Multiplicity object representing the minimum 
            and maximum cardinality derived from the context.
        """
        if ctx.cardinalityVal(0).INT():
            min_cardinality = max_cardinality = int(ctx.cardinalityVal(0).INT().getText())
        else:
            min_cardinality = 0
            max_cardinality = "*"

        if ctx.cardinalityVal(1):
            if ctx.cardinalityVal(1).INT():
                max_cardinality = int(ctx.cardinalityVal(1).INT().getText())
            else:
                max_cardinality = "*"

        multiplicity: Multiplicity = Multiplicity(min_multiplicity=min_cardinality,
                                                  max_multiplicity=max_cardinality)
        return multiplicity

    def enterEnumLiteral(self, ctx: PlantUMLParser.EnumLiteralContext):
        new_literal: EnumerationLiteral = EnumerationLiteral(name=ctx.ID().getText())
        enum = self.__buml_model.get_type_by_name(ctx.parentCtx.ID().getText())
        enum.add_literal(literal=new_literal)
