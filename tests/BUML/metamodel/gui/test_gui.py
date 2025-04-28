import pytest

from besser.BUML.metamodel.structural import *
from besser.BUML.metamodel.gui import *


def test_named_element():
    named_element: NamedElement = NamedElement(name="element1")
    assert named_element.name == "element1"

# Test: Do not have two modules with the same name in an application.
def test_unique_module_names():
    module1: Module = Module(name="module1", screens=[])
    module2: Module = Module(name="module2", screens=[])
    my_app: GUIModel = GUIModel(name="application1", package="", versionCode="", versionName="",
                                description="", screenCompatibility=False, modules=[module1, module2])
    with pytest.raises(ValueError) as excinfo:
        # Try to create a module with the same name as module1
        module_duplicate: Module = Module(name="module1", screens=[])
        my_app.modules = {module1, module2, module_duplicate}
    assert "An app cannot have two modules with the same name" in str(excinfo.value)

# Test: Do not have two screens with the same name in an application.
def test_unique_screen_names():
    screen1: Screen = Screen(name="screen1", description="", x_dpi="", y_dpi="",
                             screen_size="Small", view_elements={})
    screen2: Screen = Screen(name="screen2", description="", x_dpi="", y_dpi="",
                             screen_size="Small", view_elements={})
    module1: Module = Module(name="module1", screens={screen1, screen2})
    my_app: GUIModel = GUIModel(name="application1", package="", versionCode="",
                                versionName="", description="", screenCompatibility=False,
                                modules={module1})
    with pytest.raises(ValueError) as excinfo:
        # Try to create a screen with the same name as screen1
        screen_duplicate: Screen = Screen(name="screen1", description="", x_dpi="", y_dpi="",
                                          screen_size="Small", view_elements={})
        module1.screens = {screen1, screen2, screen_duplicate}
    assert "A module cannot have two screens with the same name" in str(excinfo.value)

# Test: Do not have two items with the same name in a list.
def test_unique_item_names():
    item1: DataSourceElement = DataSourceElement(name="item1", dataSourceClass="", fields={})
    item2: DataSourceElement = DataSourceElement(name="item2", dataSourceClass="", fields={})
    list1: DataList = DataList(name="list1", description="", list_sources={item1, item2})
    with pytest.raises(ValueError) as excinfo:
        # Try to create a item with the same name as item1
        item_duplicate: DataSourceElement = DataSourceElement(name="item1", dataSourceClass="", fields={})
        list1.list_sources={item1, item2, item_duplicate}
    assert "A list cannot have two items with the same name" in str(excinfo.value)

def test_button_buttonType_must_be_defined():
    button: Button = Button(name="button", label="View List", description="", buttonType="", actionType=ButtonActionType.Add)

    with pytest.raises(ValueError) as excinfo:
        if button.buttonType == "":
            raise ValueError("buttonType must be defined")
        screen1: Screen = Screen(name="screen1", description="", x_dpi="", y_dpi="", screen_size="Small", view_elements={button})
    assert "buttonType must be defined" in str(excinfo.value)

def test_button_actionType_must_be_defined():
    button: Button = Button(name="button", label="View List", description="", buttonType=ButtonType.FloatingActionButton, actionType="")

    with pytest.raises(ValueError) as excinfo:
        if button.actionType == "":
            raise ValueError("actionType must be defined")
        screen1: Screen = Screen(name="screen1", description="", x_dpi="", y_dpi="", screen_size="Small", view_elements={button})
    assert "actionType must be defined" in str(excinfo.value)

def test_button_properties_must_be_defined():
    button: Button = Button(name="button", label="View List", description="", buttonType="", actionType="")

    with pytest.raises(ValueError) as excinfo:
        if button.buttonType == "" and button.actionType== "":
            raise ValueError("buttonType and actionType must be defined")
        screen1: Screen = Screen(name="screen1", description="", x_dpi="", y_dpi="", screen_size="Small", view_elements={button})
    assert "buttonType and actionType must be defined" in str(excinfo.value)


def test_list_sources_must_be_defined():
    datasource: DataSourceElement = DataSourceElement(name="DataSource", dataSourceClass="", fields=[])
    myList: DataList = DataList(name="MyList", description="A diverse group of elements", list_sources={})

    with pytest.raises(ValueError) as excinfo:
        if len(myList.list_sources) == 0:
            raise ValueError("list_sources must be defined")

        screen1: Screen = Screen(name="screen1", description="", x_dpi="", y_dpi="", screen_size="Small", view_elements={myList})

    assert "list_sources must be defined" in str(excinfo.value)

def test_fields_synchronized_attributes():
    # Class1 attributes definition
    class1_name: Property = Property(name="name", type="int")
    class1: Class = Class (name="Class1", attributes=[class1_name])
    # Class2 attributes definition
    class2_name: Property = Property(name="name", type="int")
    class1: Class = Class (name="Class2", attributes=[class2_name])
    #class1_DataSource definition
    datasource_class1: DataSourceElement = DataSourceElement(name="Class1DataSource", dataSourceClass=class1, fields=[class2_name])
    with pytest.raises(ValueError) as excinfo:
        if datasource_class1.fields not in datasource_class1.dataSourceClass.attributes:
            raise ValueError("Fields must be synchronized with the dataSourceClass attributes")
        class1_List: DataList=DataList(name="Class1 List", description="A diverse group of elements", list_sources={datasource_class1})

    assert "Fields must be synchronized with the dataSourceClass attributes" in str(excinfo.value)

def test_associations():
    # Class1 attributes definition
    class1_name: Property = Property(name="name", type="int")
    class1: Class = Class(name="Class1", attributes=[class1_name])
    # Class2 attributes definition
    class2_name: Property = Property(name="name", type="int")
    class2: Class = Class(name="Class2", attributes=[class2_name])
    # Class1-Class2 association definition
    end1: Property = Property(name="end1", type=class2)
    end2: Property = Property(name="end2", type=class1)
    class1_class2_association: BinaryAssociation = BinaryAssociation(name="class1_class2_association", ends={end1, end2})
    # Domain model definition
    model: DomainModel = DomainModel(name="model", types={class1, class2}, associations={class1_class2_association})

    with pytest.raises(ValueError) as excinfo:
        if len(model.associations) != 0:
            for classConcept in model.types:
                for assoc in model.associations:
                    for end in assoc.ends:
                        if end.type.name != classConcept.name:
                            raise ValueError("All associations related to classes in 'types' " \
                            "must be considered in 'associations'")
        else:
            raise ValueError("All associations related to classes in" \
            " 'types' must be considered in 'associations'")

    assert "All associations related to classes in 'types' must be considered in 'associations'" in str(excinfo.value)
