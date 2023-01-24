from core.structural.structural import DomainModel, Class, Property, PrimitiveDataType


# Function transformting textX model to core model
def textx_to_core(textx_model) -> DomainModel:
    model: DomainModel = DomainModel(name=textx_model.name)
    model.elements = set()
    for element in textx_model.classes:
        new_class: Class = Class(name=element.name, attributes=set())
        model.elements.add(new_class)
        attrs: set[Property] = set()
        for attribute in element.attributes:
            attrs.add(Property(name=attribute.name, owner=new_class, property_type=PrimitiveDataType(name=attribute.type)))
        new_class.attributes = attrs

    return model



