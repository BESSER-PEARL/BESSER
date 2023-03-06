from MyUML.core.structural.structural import DomainModel, Class, Property, PrimitiveDataType, \
     BinaryAssociation, Multiplicity, Constraint, Generalization

# Function transformting textX model to core model
def textx_to_core(textx_model) -> DomainModel:
    model: DomainModel = DomainModel(name="StructuralModel")
    model.umlElements = set()

    # Class definition
    for element in textx_model.umlElements:
        element_type: str = element.__class__.__name__
        if element_type == "Class":
            new_class: Class = Class(name=element.name, is_abstract=element.isAbstract, attributes=set())
            model.types.add(new_class)
            # Attributes and operations definition
            attrs: set[Property] = set()
            opers: set[Property] = set()
            for content in element.classContents:
                # Attributes
                if content.__class__.__name__ == "Attribute":
                    attrs.add(Property(name=content.name, visibility="public", owner=new_class, 
                                       property_type=PrimitiveDataType(name=content.type)))
                # Operations
                # if content.__class__.__name__ == "Method":
            new_class.attributes = attrs
    
    # Association definition
    for element in textx_model.umlElements:
        element_type: str = element.__class__.__name__
        if element_type == "Bidirectional" or element_type == "Unidirectional" or \
            element_type == "Aggregation" or element_type == "Composition":
            # reference from
            class_from: Class = model.get_class_by_name(element.fromClass.name)
            min_from = element.fromCar.min
            max_from = min_from if element.fromCar.max is None else element.fromCar.max
            navigable_from: bool = True
            composition_from: bool = False
            aggregation_from: bool = False
            # reference to
            class_to: Class = model.get_class_by_name(element.toClass.name)
            min_to = element.toCar.min
            max_to = min_to if element.toCar.max is None else element.toCar.max
            navigable_to: bool = True
            composition_to: bool = False
            aggregation_to: bool = False
            ends: set[Property] = set()
            if element.__class__.__name__ == "Unidirectional":
                navigable_from = element.fromNav
                navigable_to = element.toNav
            if element.__class__.__name__ == "Aggregation":
                aggregation_from = element.fromAgg
                aggregation_to = element.toAgg            
            if element.__class__.__name__ == "Composition":
                composition_from = element.fromComp
                composition_to = element.toComp
            ends.add(Property(name=element.name, visibility="public", owner=class_from, property_type=class_from, 
                              multiplicity=Multiplicity(min_multiplicity=min_from,max_multiplicity=max_from), 
                              is_composite=composition_from, is_navigable=navigable_from, is_aggregation=aggregation_from))
            ends.add(Property(name=element.name, visibility="public", owner=class_to, property_type=class_to, 
                              multiplicity=Multiplicity(min_multiplicity=min_to, max_multiplicity=max_to), 
                              is_composite=composition_to, is_navigable=navigable_to, is_aggregation=aggregation_to))
            new_association: BinaryAssociation = BinaryAssociation(name=element.name, ends=ends)
            model.associations.add(new_association)
        
        # Generalization definition
        if element_type == "Inheritance":
            if element.fromInh == True:
                generalClass: Class = model.get_class_by_name(element.fromClass.name)
                specificClass: Class = model.get_class_by_name(element.toClass.name)
            else:
                generalClass: Class = model.get_class_by_name(element.toClass.name)
                specificClass: Class = model.get_class_by_name(element.fromClass.name)
            new_generalization: Generalization = Generalization(general=generalClass, specific=specificClass)
            model.generalizations.add(new_generalization)

    # Constraint definition
    for element in textx_model.oclConstraints:
        context: Class = model.get_class_by_name(element.context.name)
        new_constraint: Constraint = Constraint(name=element.name, context=context, expression=element.expression, language="OCL")
        model.constraints.add(new_constraint)
    return model



