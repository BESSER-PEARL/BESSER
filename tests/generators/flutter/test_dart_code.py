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

