from MyUML.core.structural.structural import DomainModel, Class, Property, PrimitiveDataType, BinaryAssociation

# Function transformting textX model to core model
def textx_to_core(textx_model) -> DomainModel:
    model: DomainModel = DomainModel(name="StructuralModel")
    model.umlElements = set()
    # Class definition
    for element in textx_model.umlElements:
        if element.__class__.__name__ == "Class":
            new_class: Class = Class(name=element.name, is_abstract=element.isAbstract, attributes=set())
            model.types.add(new_class)
            # Attributes and operations definition
            attrs: set[Property] = set()
            opers: set[Property] = set()
            for content in element.classContents:
                # Attributes
                if content.__class__.__name__ == "Attribute":
                    attrs.add(Property(name=content.name, visibility="public", owner=new_class, property_type=PrimitiveDataType(name=content.type)))
            new_class.attributes = attrs
    # Association definition
    # for element in textx_model.associations:
    #     ends: set[Property] = set()
    #     for memberEnd in element.ends:
    #         ends.add(next((x for x in model.elements if x.name == memberEnd.name), None))
    #     new_association: BinaryAssociation = BinaryAssociation(name=element.name, ends=ends)
    #     model.elements.add(new_association)
    return model



