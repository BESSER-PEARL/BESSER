from besser.BUML.metamodel.object import *
from besser.BUML.metamodel.structural import *

########################
#   Structural model   #
########################

attribute1: Property = Property(name="A1", type=IntegerType)
attribute2: Property = Property(name="A2", type=StringType)
class1: Class = Class(name="class1", attributes={attribute1, attribute2})
class2: Class = Class(name="class2", attributes=set())
aend1: Property = Property(name="end1", type=class1, multiplicity=Multiplicity(0, 1))
aend2: Property = Property(name="end2", type=class2, multiplicity=Multiplicity(0, "*"))
association1: BinaryAssociation = BinaryAssociation(name="association1", ends={aend1, aend2})

##########################
#   Object model tests   #
##########################

# Testing model initialization
def test_model_initialization():
    obj1: Object = Object(name="object1", classifier=class1, slots=[])
    obj2: Object = Object(name="object2", classifier=class2, slots=[])
    model: ObjectModel = ObjectModel(name="mymodel", objects={obj1,obj2})
    assert len(model.objects) == 2
    model_empty: ObjectModel = ObjectModel(name="mymodel", objects={})
    assert len(model_empty.objects) == 0

# Testing object initialization
def test_object_initialization():
    attr1: AttributeLink = AttributeLink(value=DataValue(classifier=IntegerType, value=100), attribute=attribute1)
    obj1: Object = Object(name="object1", classifier=class1, slots=[attr1])
    assert len(obj1.slots) == 1
    assert obj1.classifier.name == "class1"

# Testing attribute link
def test_attributeLink_initialization():
    attr1: AttributeLink = AttributeLink(value=DataValue(classifier=IntegerType, value=100), attribute=attribute1)
    attr2: AttributeLink = AttributeLink(value=DataValue(classifier=StringType, value="test value"), attribute=attribute2)
    assert attr1.value.value == 100
    assert attr2.value.value == "test value"

# Testing Link inicialization
def test_link_initialization():
    obj1: Object = Object(name="object1", classifier=class1, slots=[])
    obj2: Object = Object(name="object2", classifier=class2, slots=[])
    link_end1: LinkEnd = LinkEnd(name="l_end1", association_end=aend1, object=obj1)
    link_end2: LinkEnd = LinkEnd(name="l_end2", association_end=aend2, object=obj2)
    link1: Link = Link(name="link1", association=association1, connections=[link_end1, link_end2])
    assert len(link1.connections) == 2
    assert link1.association.name == "association1"
    assert len(link1.association.ends) == 2

# Testing Link inicialization
def test_linkEnd_initialization():
    obj1: Object = Object(name="object1", classifier=class1, slots=[])
    obj2: Object = Object(name="object2", classifier=class2, slots=[])
    link_end1: LinkEnd = LinkEnd(name="l_end1", association_end=aend1, object=obj1)
    link_end2: LinkEnd = LinkEnd(name="l_end2", association_end=aend2, object=obj2)
    assert link_end1.association_end.name == "end1"
    assert link_end2.object.name == "object2"

# Testing link_ends() method
def test_link_ends_method():
    obj1: Object = Object(name="object1", classifier=class1, slots=[])
    obj2: Object = Object(name="object2", classifier=class2, slots=[])
    link_end1: LinkEnd = LinkEnd(name="l_end1", association_end=aend1, object=obj1)
    link_end2: LinkEnd = LinkEnd(name="l_end2", association_end=aend2, object=obj2)
    link1: Link = Link(name="link1", association=association1, connections=[link_end1, link_end2])
    assert len(obj1.link_ends()) == 1
    for link_end in obj1.link_ends():
        assert link_end.object.name == "object2"
    for link_end in obj2.link_ends():
        assert link_end.object.name == "object1"