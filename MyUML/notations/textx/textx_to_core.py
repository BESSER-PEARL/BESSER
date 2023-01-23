from core.structural.structural import DomainModel, Class


# Function transformting textX model to core model
def textx_to_core(textx_model) -> DomainModel:
    model: DomainModel = DomainModel(name=textx_model.name)
    model.elements = set()
    for element in textx_model.classes:
        model.elements.add( Class(name=element.name, attributes=set()))
    return model



