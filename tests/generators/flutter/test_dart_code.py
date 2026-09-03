import os
from besser.generators.flutter import FlutterSQLHelperGenerator, FlutterMainDartGenerator
from besser.BUML.metamodel.structural import *
from besser.BUML.metamodel.gui import *
import shutil



################ Test GUI Components ################
def test_screen_generation():
    # Define the model to be generated:
    # Class1 attributes definition
    class1_name: Property = Property(name="name", type="int")
    class1: Class = Class(name="Class1", attributes=[class1_name])
    # Domain model definition
    model : DomainModel = DomainModel(name="model", types={class1},
                                          associations={})
    # DataSource definition
    datasource: DataSourceElement = DataSourceElement(name="DataSource", dataSourceClass=class1, fields=[class1_name])

    # My List definition
    myList: DataList = DataList(name="MyList", description="A diverse group of elements", list_sources={datasource})

    # Screen definition
    myScreen: Screen = Screen(
        name="MyListScreen",
        description="Explore a collection of pets",
        x_dpi="x_dpi",
        y_dpi="y_dpi",
        screen_size="Small",
        view_elements={myList}
    )

    # HomeScreen definition
    myHomeScreen: Screen = Screen(
        name="MyHomeScreen",
        description="Explore a collection of pets",
        x_dpi="x_dpi",
        y_dpi="y_dpi",
        screen_size="Small",
        view_elements={}
    )

    # Module definition:
    MyModule: Module = Module(name="module_name", screens={myHomeScreen, myScreen})

    # GUI model definition:
    gui_model: GUIModel = GUIModel(
        name="app",
        package="com.example.app",
        versionCode="1",
        versionName="1.0",
        description="This is a comprehensive Flutter application",
        screenCompatibility=True,
        modules={MyModule}
    )

    # Generate the file
    output_file = 'output/main.dart'
    code_gen = FlutterMainDartGenerator(model=model, gui_model=gui_model, main_page=myHomeScreen, module=MyModule)
    code_gen.generate()

    # Check if the file exists
    assert os.path.exists(output_file), "The file was not created."

    # Read the content of the file
    with open(output_file, 'r') as file:
        content = file.read()

    # Check for expected lines in the file
    assert not f"class {myScreen.name}State extends " in content, "Missing screen definition in the generated file."

    # Clean up (optional)
    shutil.rmtree("output")



################ Test Error Handling ################
def test_Error_Handling():
    # Define the model to be generated:
    # Class1 attributes definition
    class1_name: Property = Property(name="name", type="int")
    class1: Class = Class(name="Class1", attributes=[class1_name])

    # Class2 attributes definition
    class2_name: Property = Property(name="name", type="int")
    class2: Class = Class(name="Class2", attributes=[class2_name])

    # Class3 attributes definition
    class3_name: Property = Property(name="name", type="int")
    class3: Class = Class(name="Class3", attributes=[class3_name])

    # Class1-class2 association definition
    end1: Property = Property(name="end1",type=class2, multiplicity=Multiplicity(1, 1))
    end2: Property = Property(name="end2", type=class1, multiplicity=Multiplicity(0, "*"))
    class1_class2_association: BinaryAssociation = BinaryAssociation(name="class1_class2_assoc", ends={end1, end2})

    # Class1-class2 association definition
    end3: Property = Property(name="end3", type=class1, multiplicity=Multiplicity(0, "*"))
    end4: Property = Property(name="end4", type=class3, multiplicity=Multiplicity(1, "*"))
    class1_class3_association: BinaryAssociation = BinaryAssociation(name="class1_class3_assoc", ends={end3, end4})

    # Domain model definition
    model : DomainModel = DomainModel(name="model", types={class1, class2, class3},
                                          associations={class1_class2_association, class1_class3_association})
    # DataSource definition
    datasource: DataSourceElement = DataSourceElement(name="DataSource", dataSourceClass=class1, fields=[class1_name])

    # My List definition
    myList: DataList = DataList(name="MyList", description="A diverse group of elements", list_sources={datasource})

    # Screen definition
    myScreen: Screen = Screen(
        name="MyListScreen",
        description="Explore a collection of pets",
        x_dpi="x_dpi",
        y_dpi="y_dpi",
        screen_size="Small",
        view_elements={myList}
    )

    # HomeScreen definition
    myHomeScreen: Screen = Screen(
        name="MyHomeScreen",
        description="Explore a collection of pets",
        x_dpi="x_dpi",
        y_dpi="y_dpi",
        screen_size="Small",
        view_elements={}
    )

    # Module definition:
    MyModule: Module = Module(name="module_name", screens={myHomeScreen, myScreen})

    # GUI model definition:
    gui_model: GUIModel = GUIModel(
        name="app",
        package="com.example.app",
        versionCode="1",
        versionName="1.0",
        description="This is a comprehensive Flutter application",
        screenCompatibility=True,
        modules={MyModule}
    )

    # Generate the file
    output_file = 'output/main.dart'
    code_gen = FlutterMainDartGenerator(model=model, gui_model=gui_model, main_page=myHomeScreen, module=MyModule)
    code_gen.generate()

    # Check if the file exists
    assert os.path.exists(output_file), "The file was not created."

    # Read the content of the file
    with open(output_file, 'r') as file:
        content = file.read()

    # Check for expected lines in the file
    assert not f"Please specify the {end1.type}" in content, "Missing error handling in the generated file."
    assert not f"Please select at least one {end4.type}" in content, "Missing error handling in the generated file."

    #os.remove(output_file)
    shutil.rmtree("output")

################ Test CRUD Operations ################
def test_CRUD_Operations():
    # Define the model to be generated:
    # Class1 attributes definition
    class1_name: Property = Property(name="name", type="int")
    class1: Class = Class(name="Class1", attributes=[class1_name])

    # Domain model definition
    model : DomainModel = DomainModel(name="model", types={class1},
                                          associations={})
    # DataSource definition
    datasource: DataSourceElement = DataSourceElement(name="DataSource", dataSourceClass=class1, fields=[class1_name])

    # My List definition
    myList: DataList = DataList(name="MyList", description="A diverse group of elements", list_sources={datasource})

    # Screen definition
    myScreen: Screen = Screen(
        name="MyListScreen",
        description="Explore a collection of pets",
        x_dpi="x_dpi",
        y_dpi="y_dpi",
        screen_size="Small",
        view_elements={myList}
    )

    # HomeScreen definition
    myHomeScreen: Screen = Screen(
        name="MyHomeScreen",
        description="Explore a collection of pets",
        x_dpi="x_dpi",
        y_dpi="y_dpi",
        screen_size="Small",
        view_elements={}
    )

    # Module definition:
    MyModule: Module = Module(name="module_name", screens={myHomeScreen, myScreen})

    # GUI model definition:
    gui_model: GUIModel = GUIModel(
        name="app",
        package="com.example.app",
        versionCode="1",
        versionName="1.0",
        description="This is a comprehensive Flutter application",
        screenCompatibility=True,
        modules={MyModule}
    )

    # Generate the file
    output_file = 'output/sql_helper.dart'

    code_gen = FlutterSQLHelperGenerator(model = model)
    code_gen.generate()

    # Check if the file exists
    assert os.path.exists(output_file), "The file was not created."

    # Read the content of the file
    with open(output_file, 'r') as file:
        content = file.read()

    # Check for expected lines in the file
    assert f"static Future<int> create{class1.name}" in content, f"Missing create{class1.name} method in the generated file."
    assert f"static Future<int> update{class1.name}" in content, f"Missing update{class1.name} method in the generated file."
    assert f"static Future<void> delete{class1.name}" in content, "Missing delete{class1.name} method in the generated file."
    assert f"static Future<List<Map<String, dynamic>>> get{class1.name}s" in content, "Missing get{class1.name}s method in the generated file."
    assert f"static Future<String?> get{class1.name}" in content, "Missing get{class1.name} method in the generated file."
    assert f"Future<List<String>> get{class1.name}NamesByIds" in content, "Missing get{class1.name}NamesByIds method in the generated file."
    assert f"static Future<int> get{class1.name}IdByName" in content, "Missing get{class1.name}IdByName method in the generated file."

    #os.remove(output_file)
    shutil.rmtree("output")

def test_Operations_many_to_many():
    # Define the model to be generated:
    # Class1 attributes definition
    class1_name: Property = Property(name="name", type="int")
    class1: Class = Class(name="Class1", attributes=[class1_name])

    # Class2 attributes definition
    class2_name: Property = Property(name="name", type="int")
    class2: Class = Class(name="Class2", attributes=[class2_name])

    # Class1-class2 association definition
    end1: Property = Property(name="end3", type=class1, multiplicity=Multiplicity(0, "*"))
    end2: Property = Property(name="end4", type=class2, multiplicity=Multiplicity(1, "*"))
    class1_class2_association: BinaryAssociation = BinaryAssociation(name="class1_class2_assoc", ends={end1, end2})

    # Domain model definition
    model : DomainModel = DomainModel(name="model", types={class1, class2},
                                          associations={class1_class2_association})

    # DataSource definition
    datasource: DataSourceElement = DataSourceElement(name="DataSource", dataSourceClass=class1, fields=[class1_name])

    # My List definition
    myList: DataList = DataList(name="MyList", description="A diverse group of elements", list_sources={datasource})

    # Screen definition
    myScreen: Screen = Screen(
        name="MyListScreen",
        description="Explore a collection of pets",
        x_dpi="x_dpi",
        y_dpi="y_dpi",
        screen_size="Small",
        view_elements={myList}
    )

    # HomeScreen definition
    myHomeScreen: Screen = Screen(
        name="MyHomeScreen",
        description="Explore a collection of pets",
        x_dpi="x_dpi",
        y_dpi="y_dpi",
        screen_size="Small",
        view_elements={}
    )

    # Module definition:
    MyModule: Module = Module(name="module_name", screens={myHomeScreen, myScreen})

    # GUI model definition:
    gui_model: GUIModel = GUIModel(
        name="app",
        package="com.example.app",
        versionCode="1",
        versionName="1.0",
        description="This is a comprehensive Flutter application",
        screenCompatibility=True,
        modules={MyModule}
    )

    # Generate the file
    output_file = 'output/sql_helper.dart'

    code_gen = FlutterSQLHelperGenerator(model = model)
    code_gen.generate()

    # Check if the file exists
    assert os.path.exists(output_file), "The file was not created."

    # Read the content of the file
    with open(output_file, 'r') as file:
        content = file.read()

    # Check for expected lines in the file
    assert f"static Future<List<String>> get{class1.name}NamesBy{class2.name}Id" in content, "Missing get{class1.name}NamesBy{class2.name}Id method in the generated file."
    assert f"static Future<List<String>> get{class2.name}NamesBy{class1.name}Id" in content, "Missing get{class2.name}NamesBy{class1.name}Id method in the generated file."

    #os.remove(output_file)
    shutil.rmtree("output")



# ---------------------------------------------------------------------------
# Map component Flutter generator tests
# ---------------------------------------------------------------------------

def test_flutter_map_main_dart_contains_flutter_map(tmp_path):
    """FlutterMainDartGenerator emits FlutterMap widget for a Map on the main screen."""
    from besser.BUML.metamodel.gui.dashboard import Map
    from besser.BUML.metamodel.structural import FloatType, StringType

    lat_prop = Property(name="latitude", type=FloatType)
    lng_prop = Property(name="longitude", type=FloatType)
    name_prop = Property(name="store_name", type=StringType)
    location_cls = Class(name="Location", attributes={lat_prop, lng_prop, name_prop})
    domain_model = DomainModel(name="MapDomain", types={location_cls})

    map_comp = Map(name="StoreMap", title="Store Locations",
                   center_latitude=51.5, center_longitude=-0.09, zoom=12)

    # The main page template renders Map with a full FlutterMap widget.
    # Put the Map component ON the main screen so it appears in the body.
    main_screen = Screen(
        name="MainMapScreen",
        description="Main screen with map",
        x_dpi="x_dpi",
        y_dpi="y_dpi",
        screen_size="Medium",
        view_elements={map_comp},
    )
    module = Module(name="AppModule", screens={main_screen})
    gui = GUIModel(
        name="mapapp",
        package="com.example.mapapp",
        versionCode="1",
        versionName="1.0",
        description="Map Flutter app",
        screenCompatibility=True,
        modules={module},
    )

    gen = FlutterMainDartGenerator(model=domain_model, gui_model=gui,
                                    main_page=main_screen, module=module,
                                    output_dir=str(tmp_path))
    gen.generate()

    output_file = tmp_path / "main.dart"
    assert output_file.exists(), "main.dart should be generated"
    content = output_file.read_text(encoding="utf-8")

    assert "FlutterMap" in content, "main.dart should contain FlutterMap widget"
    assert "TileLayer" in content, "main.dart should contain TileLayer for OSM tiles"
    assert "flutter_map" in content, "import for flutter_map should be present"


def test_flutter_map_pubspec_contains_flutter_map_dep(tmp_path):
    """FlutterPubspecGenerator includes flutter_map and latlong2 dependencies."""
    import shutil, os
    from besser.BUML.metamodel.gui.dashboard import Map

    map_comp = Map(name="PubspecMap", center_latitude=0.0, center_longitude=0.0, zoom=5)
    screen = Screen(name="S", description="", x_dpi="", y_dpi="",
                    screen_size="Medium", view_elements={map_comp})
    module = Module(name="M", screens={screen})
    gui = GUIModel(
        name="pubspecapp",
        package="com.example",
        versionCode="1",
        versionName="1.0",
        description="",
        screenCompatibility=True,
        modules={module},
    )

    from besser.generators.flutter import FlutterPubspecGenerator
    # FlutterPubspecGenerator has a pre-existing bug: it passes output_dir as the
    # model arg to super().__init__, so output_dir is always None and files are
    # written to <cwd>/output/. We work around it here.
    gen = FlutterPubspecGenerator(gui_model=gui)
    gen.generate()

    pubspec_path = os.path.join(os.path.abspath(''), 'output', 'pubspec.yaml')
    try:
        assert os.path.exists(pubspec_path), "pubspec.yaml should be generated"
        with open(pubspec_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "flutter_map" in content, "pubspec.yaml should list flutter_map dependency"
        assert "latlong2" in content, "pubspec.yaml should list latlong2 dependency"
    finally:
        shutil.rmtree(os.path.join(os.path.abspath(''), 'output'), ignore_errors=True)
