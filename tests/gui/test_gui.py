import pytest

from besser.BUML.metamodel.structural import *
from besser.BUML.metamodel.gui import *


def test_named_element():
    named_element: NamedElement = NamedElement(name="element1")
    assert named_element.name == "element1"

# Testing the WFR for duplicate names in a model
def test_model_duplicated_names():
    with pytest.raises(ValueError) as excinfo:
        class1: Type = Type(name="name1")
        class2: Type = Type(name="name1")
        model: DomainModel = DomainModel(name="mymodel", types={class1, class2}, associations = None, packages = None, constraints = None)
        assert "same name" in str(excinfo.value)

# Test: Do not have two modules with the same name in an application.
def test_unique_module_names():
    module1: Module = Module(name="module1", screens=[])
    module2: Module = Module(name="module2", screens=[])
    my_app: Application = Application(name="application1", package="", versionCode="", versionName="", description="", screenCompatibility=False, modules=[module1, module2])
    with pytest.raises(ValueError) as excinfo:
        # Try to create a module with the same name as module1
        module_duplicate: Module = Module(name="module1", screens=[])
        my_app.modules = {module1, module2, module_duplicate}
        assert "An app cannot have two modules with the same name" in str(excinfo.value)

# Test: Do not have two screens with the same name in an application.
def test_unique_screen_names():
    screen1: Screen = Screen(name="screen1", description="", x_dpi="", y_dpi="", size="SmallScreen", components={})
    screen2: Screen = Screen(name="screen2", description="", x_dpi="", y_dpi="", size="SmallScreen", components={})
    module1: Module = Module(name="module1", screens={screen1, screen2})
    my_app: Application = Application(name="application1", package="", versionCode="", versionName="", description="", screenCompatibility=False, modules={module1})
    with pytest.raises(ValueError) as excinfo:
        # Try to create a screen with the same name as screen1
        screen_duplicate: Screen = Screen(name="screen1", description="", x_dpi="", y_dpi="", size="SmallScreen", components={})
        module1.screens = {screen1, screen2, screen_duplicate}
        assert "A module cannot have two screens with the same name" in str(excinfo.value)

# Test: Do not have two lists with the same name on the same screen.
def test_unique_list_names():
    list1: List = List(name="list1", description="", list_sources={})
    list2: List = List(name="list2", description="", list_sources={})
    screen1: Screen = Screen(name="screen1", description="", x_dpi="", y_dpi="", size="SmallScreen", components={list1, list2})
    with pytest.raises(ValueError) as excinfo:
        # Try to create a list with the same name as list1
        list_duplicate: List = List(name="list1", description="", list_sources={})
        screen1.components={list1, list2, list_duplicate}
        module1.screens = {screen1}
        assert "A screen cannot have two lists with the same name" in str(excinfo.value)

# Test: Do not have two fields with the same name in a list item.
def test_unique_field_names():
    field1: Property = Property(name="field1", type="")
    field2: Property = Property(name="field2", type="")
    modelElement: ModelElement = ModelElement(name="item1", dataSourceClass="", fields={field1, field2})
    with pytest.raises(ValueError) as excinfo:
        # Try to create a field with the same name as field1
        field_duplicate: Property = Property(name="field1", type="")
        modelElement.fields={field1, field2, field_duplicate}
        assert "A list item cannot have two fields with the same name" in str(excinfo.value)

# Test: Do not have two items with the same name in a list.
def test_unique_item_names():
    item1: ModelElement = ModelElement(name="item1", dataSourceClass="", fields={})
    item2: ModelElement = ModelElement(name="item2", dataSourceClass="", fields={})
    list1: List = List(name="list1", description="", list_sources={item1, item2})
    with pytest.raises(ValueError) as excinfo:
        # Try to create a item with the same name as item1
        item_duplicate: ModelElement = ModelElement(name="item1", dataSourceClass="", fields={})
        list1.list_sources={item1, item2, item_duplicate}
        assert "A list cannot have two items with the same name" in str(excinfo.value)

# Test: There should not be two buttons with the same name on the same screen.
def test_unique_button_names():
    button1: Button=Button(name="button1", Label="View List", description="")
    button2: Button=Button(name="button2", Label="Cancel", description="")
    screen1: Screen = Screen(name="screen1", description="", x_dpi="", y_dpi="", size="SmallScreen", components={button1, button2})
    with pytest.raises(ValueError) as excinfo:
        # Try to create a button with the same name as button1
        button_duplicate: Button=Button(name="button1", Label="View List", description="")
        screen1.components={button1, button2, button_duplicate}
        assert "A screen cannot have two buttons with the same name" in str(excinfo.value)
