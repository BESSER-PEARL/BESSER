import pytest

from besser.BUML.metamodel.structural import *
from besser.BUML.metamodel.gui import *
from besser.BUML.metamodel.gui.dashboard import Map, MapLayer, MapLayerType, WorldMap, LocationMap


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


# ---------------------------------------------------------------------------
# Map component metamodel tests
# ---------------------------------------------------------------------------

def test_map_basic_construction():
    """Map can be constructed with center/zoom; layers defaults to empty list."""
    m = Map(name="StoreMap", title="Store Locations", center_latitude=48.8566,
            center_longitude=2.3522, zoom=13)
    assert m.name == "StoreMap"
    assert m.title == "Store Locations"
    assert m.center_latitude == 48.8566
    assert m.center_longitude == 2.3522
    assert m.zoom == 13
    assert m.layers == []  # new API: Map holds a list of MapLayer objects


def test_map_with_data_binding_and_geo_fields():
    """Map accepts a list of MapLayer objects with DataBinding + Property field refs."""
    lat_prop = Property(name="latitude", type=FloatType)
    lng_prop = Property(name="longitude", type=FloatType)
    label_prop = Property(name="store_name", type=StringType)
    location_class = Class(name="Location", attributes={lat_prop, lng_prop, label_prop})

    binding = DataBinding(name="location_binding", domain_concept=location_class)
    layer = MapLayer(
        name="stores",
        layer_type=MapLayerType.points,
        data_binding=binding,
        latitude_field=lat_prop,
        longitude_field=lng_prop,
        label_field=label_prop,
    )
    m = Map(name="LocationMap", center_latitude=0.0, center_longitude=0.0,
            zoom=5, layers=[layer])
    assert len(m.layers) == 1
    lyr = m.layers[0]
    assert lyr.latitude_field is lat_prop
    assert lyr.longitude_field is lng_prop
    assert lyr.label_field is label_prop
    assert lyr.data_binding is binding
    assert lyr.data_binding.domain_concept is location_class


def test_map_geo_field_type_validation():
    """MapLayer field setters raise TypeError for non-Property values."""
    layer = MapLayer(name="TestLayer")
    with pytest.raises(TypeError):
        layer.latitude_field = "not_a_property"


def test_map_geo_longitude_type_validation():
    """MapLayer.longitude_field setter raises TypeError for non-Property values."""
    layer = MapLayer(name="TestLayer")
    with pytest.raises(TypeError):
        layer.longitude_field = 42


def test_map_geo_label_field_type_validation():
    """MapLayer.label_field setter raises TypeError for non-Property values."""
    layer = MapLayer(name="TestLayer")
    with pytest.raises(TypeError):
        layer.label_field = object()


def test_world_map_inherits_map():
    """WorldMap is a Map subclass and forwards kwargs."""
    wm = WorldMap(name="World")
    assert isinstance(wm, Map)
    assert wm.name == "World"


def test_location_map_inherits_map():
    """LocationMap is a Map subclass and forwards kwargs."""
    lm = LocationMap(name="CityMap", center_latitude=51.5, center_longitude=-0.1, zoom=11)
    assert isinstance(lm, Map)
    assert lm.zoom == 11
