import pytest
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


def test_object_model_validation_passes_for_valid_links():
    worker = Class(name="Worker")
    task = Class(name="Task")
    works_on = Property(name="works_on", type=task, multiplicity=Multiplicity(0, 2))
    assigned_workers = Property(name="assigned_workers", type=worker, multiplicity=Multiplicity(0, 5))
    BinaryAssociation(name="Assignment", ends={works_on, assigned_workers})

    worker_1 = Object(name="worker_1", classifier=worker)
    worker_2 = Object(name="worker_2", classifier=worker)
    task_1 = Object(name="task_1", classifier=task)

    worker_1.works_on = task_1
    worker_2.works_on = task_1

    model = ObjectModel(name="AssignmentModel", objects={worker_1, worker_2, task_1})
    assert model.validate()["success"]


def test_object_model_validation_detects_multiplicity_violation():
    account = Class(name="Account")
    user = Class(name="User")
    owns = Property(name="owns", type=account, multiplicity=Multiplicity(1, 1))
    owner = Property(name="owner", type=user, multiplicity=Multiplicity(1, 1))
    BinaryAssociation(name="Ownership", ends={owns, owner})

    account_a = Object(name="account_a", classifier=account)
    user_alpha = Object(name="user_alpha", classifier=user)
    user_beta = Object(name="user_beta", classifier=user)

    user_alpha.owns = account_a
    user_beta.owns = account_a

    model = ObjectModel(name="OwnershipModel", objects={account_a, user_alpha, user_beta})
    with pytest.raises(ValueError) as excinfo:
        model.validate()
    message = str(excinfo.value)
    assert "multiplicity" in message
    assert "owns" in message or "owner" in message
