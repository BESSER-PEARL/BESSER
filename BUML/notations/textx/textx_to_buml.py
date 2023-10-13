from metamodel.structural.structural import DomainModel, Class, Property, PrimitiveDataType, \
     BinaryAssociation, Multiplicity, Constraint, Generalization, GeneralizationSet
from textx import metamodel_from_file
import os

# Function to build the buml metamodel from the grammar
def build_buml_mm_from_grammar():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    grammar_path = os.path.join(script_dir, 'buml.tx')
    buml_mm = metamodel_from_file(grammar_path)
    return buml_mm

def handle_inheritance(general_class: Class, specific_class: Class, model: DomainModel, inherited_classes: dict):
    # Check if we have already inherited attributes from the parent class
    if general_class.name in inherited_classes[specific_class.name]:
        return    
    # Check whether there are already common parent classes of general and specific to not duplicate atttributes
    common_parents = set(inherited_classes[general_class.name]) & set(inherited_classes[specific_class.name])
    inherited_attr = []
    for c in common_parents:
        for a in model.get_class_by_name(c).attributes:
            inherited_attr.append(a)
    # Add attributes of general class, ignore already added attributes
    for attribute in general_class.attributes:
        if attribute not in inherited_attr:
            specific_class.add_attribute(attribute)                
    # Add current parent and parents of parent class to list of parent classes of child class
    inherited_classes[specific_class.name].append(general_class.name)
    for parent in inherited_classes[general_class.name]:
        inherited_classes[specific_class.name].append(parent)
    # Look for the next parent class (if any)
    for generalization in model.generalizations:
        if generalization.specific == general_class:
            if generalization.general.name not in inherited_classes[general_class.name]:
                ancestor_class = generalization.general
                handle_inheritance(ancestor_class, specific_class, model, inherited_classes)

# Function transforming textX model to core model
def textx_to_buml(textx_model) -> DomainModel:
    model: DomainModel = DomainModel(name="StructuralModel")
    inheritanceGroup: int = 0

    # Class transformation
    for element in textx_model.elements:
        element_type: str = element.__class__.__name__
        if element_type == "Class":
            new_class: Class = Class(name=element.name, is_abstract=element.isAbstract, attributes=set())
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
            # Add new class to the model
            model.types.add(new_class)
        if element_type == "SkinParam":
            inheritanceGroup = element.group
    
    # Association definition
    for element in textx_model.elements:
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
    
    # Generalization group definition
    if inheritanceGroup > 1:
        gen_classes_list = []
        gen_classes_set = set()
        for generalization in model.generalizations:
            gen_classes_list.append(generalization.general)
            gen_classes_set.add(generalization.general)
        for general in gen_classes_set:
            if gen_classes_list.count(general) >= inheritanceGroup:
                generalizations: set = []
                for generalization in model.generalizations:
                    if general == generalization.general:
                        generalizations.append(generalization)
                new_generalizationSet: GeneralizationSet = GeneralizationSet(name="gen-set-" + general.name, generalizations=generalizations, 
                                                                             is_disjoint=True, is_complete=True)
    # prepare inherited classes list
    inherited_classes = dict()
    for c in model.get_classes():
        inherited_classes[c.name] = []
   # Iterate through the generalizations in model
    for generalization in model.generalizations:
        specific_class = generalization.specific
        general_class = generalization.general
        # Start the inheritance process
        handle_inheritance(general_class, specific_class, model, inherited_classes) 
        
    # Constraint definition
    for element in textx_model.oclConstraints:
        context: Class = model.get_class_by_name(element.context.name)
        new_constraint: Constraint = Constraint(name=element.name, context=context, expression=element.expression, language="OCL")
        model.constraints.add(new_constraint)
    return model



