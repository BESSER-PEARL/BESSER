import pytest

from core.structural.structural import NamedElement, DomainModel, Type


def test_named_element():
    named_element: NamedElement = NamedElement(name="element1")
    assert named_element.name == "element1"


def test_model_initialization():
    class1: Type = Type(name="element1")
    class2: Type = Type(name="element2")
    model: DomainModel = DomainModel(name="mymodel", elements={class1, class2})
    assert len(model.elements) == 2


# Testing the WFR for duplicate names in a model
def test_model_duplicated_names():
    with pytest.raises(ValueError) as excinfo:
        class1: Type = Type(name="name1")
        class2: Type = Type(name="name1")
        model: DomainModel = DomainModel(name="mymodel", elements={class1, class2})
        assert "same name" in str(excinfo.value)



