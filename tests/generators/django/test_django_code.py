import os
import shutil

from besser.generators.django import DjangoGenerator
from besser.BUML.metamodel.structural import DomainModel, Property, Class, \
    BinaryAssociation, Multiplicity
from besser.BUML.metamodel.gui import GUIModel, Module, DataList, \
    DataSourceElement, Screen


@staticmethod
def is_list(value):
    """Check if the given value is an instance of DataList class."""
    return isinstance(value, DataList)

@staticmethod
def is_model_element(value):
    """Check if the given value is an instance of DataSourceElement class."""
    return isinstance(value, DataSourceElement)

################ Test GUI Components ################
def test_screen_generation():
    # Define the model to be generated:

    # Class1 attributes definition
    class1_name: Property = Property(name="name", type="int")
    class1: Class = Class(name="Class1", attributes=[class1_name])

    # Domain model definition
    model: DomainModel = DomainModel(
        name="model",
        types={class1},
        associations={}
    )

    # DataSource definition
    datasource: DataSourceElement = DataSourceElement(
        name="DataSource",
        dataSourceClass=class1,
        fields=[class1_name]
    )

    # My List definition
    myList: DataList = DataList(
        name="MyList",
        description="A diverse group of elements",
        list_sources={datasource}
    )

    # Screen definition
    myScreen: Screen = Screen(
        name="MyListScreen",
        description="",
        x_dpi="",
        y_dpi="",
        screen_size="Small",
        view_elements={myList}
    )

    # HomeScreen definition
    myHomeScreen: Screen = Screen(
        name="MyHomeScreen",
        description="",
        x_dpi="",
        y_dpi="",
        screen_size="Small",
        view_elements={},
        is_main_page=True
    )

    # Module definition:
    MyModule: Module = Module(
        name="module_name",
        screens={myHomeScreen, myScreen}
    )

    # GUI model definition:
    gui_model: GUIModel = GUIModel(
        name="app",
        package="com.example.app",
        versionCode="1",
        versionName="1.0",
        description="This is a comprehensive Django application",
        screenCompatibility=True,
        modules={MyModule}
    )

    # Define file paths
    project_name = "my_project"
    app_name = "my_app"
    base_path = f"{project_name}/{app_name}"

    models_file = f"{base_path}/models.py"
    forms_file = f"{base_path}/forms.py"
    views_file = f"{base_path}/views.py"
    urls_file = f"{base_path}/urls.py"
    home_page_file = f"{base_path}/templates/home.html"

    # Generate the project
    code_gen = DjangoGenerator(
        model=model,
        project_name=project_name,
        app_name=app_name,
        gui_model=gui_model,
        containerization=False
    )

    code_gen.generate()

    # Verify generated files
    project_folder = code_gen.project_name

    assert os.path.exists(models_file), "The models file was not created."
    assert os.path.exists(forms_file), "The forms file was not created."
    assert os.path.exists(views_file), "The views file was not created."
    assert os.path.exists(urls_file), "The urls file was not created."
    assert os.path.exists(home_page_file), "The home page file was not created."

    # Get all screens from the modules
    screens = set()
    for module in gui_model.modules:
        screens.update(module.screens)

    main_page_screen = None

    # Check for main page condition
    for scr in screens:
        if getattr(scr, "is_main_page", False):
            main_page_screen = scr
            break  # Stop looping once found

    if main_page_screen:
        assert os.path.exists(home_page_file), "The main page file was not created."

    # Remove the main page screen from screens
    screens.discard(main_page_screen)

    # Verify template files for each screen
    for scr in screens:
        for component in scr.view_elements:
            if is_list(component):  # Ensure it's a List component
                for source in component.list_sources:
                    if is_model_element(source):
                        source_name = source.dataSourceClass.name
                        source_name = source_name[0].lower() + source_name[1:]

                        base_page_file = f"{base_path}/templates/{source_name}.html"
                        form_page_file = f"{base_path}/templates/{source_name}_form.html"
                        list_page_file = f"{base_path}/templates/{source_name}_list.html"

                        assert os.path.exists(base_page_file), (
                            f"The base page file for screen {scr.name} was not created."
                        )
                        assert os.path.exists(form_page_file), (
                            f"The form page file for screen {scr.name} was not created."
                        )
                        assert os.path.exists(list_page_file), (
                            f"The list page file for screen {scr.name} was not created."
                        )

    shutil.rmtree(project_folder)


################ Test CRUD Operations ################
def test_CRUD_Operations():
    """Test CRUD operations for generated Django application."""

    # Define the model to be generated:

    # Class1 attributes definition
    class1_name: Property = Property(name="name", type="int")
    class1: Class = Class(name="Class1", attributes=[class1_name])

    # Domain model definition
    model: DomainModel = DomainModel(
        name="model",
        types={class1},
        associations={}
    )

    # DataSource definition
    datasource: DataSourceElement = DataSourceElement(
        name="DataSource",
        dataSourceClass=class1,
        fields=[class1_name]
    )

    # My List definition
    my_list: DataList = DataList(
        name="MyList",
        description="A diverse group of elements",
        list_sources={datasource}
    )

    # Screen definition
    my_screen: Screen = Screen(
        name="MyListScreen",
        description="",
        x_dpi="",
        y_dpi="",
        screen_size="Small",
        view_elements={my_list}
    )

    # HomeScreen definition
    my_home_screen: Screen = Screen(
        name="MyHomeScreen",
        description="",
        x_dpi="",
        y_dpi="",
        screen_size="Small",
        view_elements={},
        is_main_page=True
    )

    # Module definition
    my_module: Module = Module(
        name="module_name",
        screens={my_home_screen, my_screen}
    )

    # GUI model definition
    gui_model: GUIModel = GUIModel(
        name="app",
        package="com.example.app",
        versionCode="1",
        versionName="1.0",
        description="This is a comprehensive Django application",
        screenCompatibility=True,
        modules={my_module}
    )

    # Define file paths
    project_name = "my_project"
    app_name = "my_app"
    base_path = f"{project_name}/{app_name}"
    views_file = f"{base_path}/views.py"

    # Generate project
    code_gen = DjangoGenerator(
        model=model,
        project_name=project_name,
        app_name=app_name,
        gui_model=gui_model,
        containerization=False
    )
    code_gen.generate()

    # Verify generated files
    project_folder = code_gen.project_name

    assert os.path.exists(views_file), "The views file was not created."

    # Read the content of the file
    with open(views_file, "r", encoding="utf-8") as file:
        content = file.read()

    # Check for expected methods in the file
    assert (
        "def handle_create(request, form_class, success_message)" in content
    ), "Missing handle_create method in the generated file."

    assert (
        "def handle_edit(request, form_class, instance, success_message, redirect_url)"
        in content
    ), "Missing handle_edit method in the generated file."

    # Get all screens from the modules
    screens = set()
    for module in gui_model.modules:
        screens.update(module.screens)

    main_page_screen = None

    # Check for main page condition
    for scr in screens:
        if getattr(scr, "is_main_page", False):
            main_page_screen = scr
            break  # Stop looping once the main page screen is found

    assert (
        "def home(request)" in content
    ), "Missing the method for main page in the generated file."

    # Remove the main page screen from screens
    screens.discard(main_page_screen)

    # Verify CRUD methods for each screen
    for scr in screens:
        for component in scr.view_elements:
            if is_list(component):  # Ensure it's a List component
                for source in component.list_sources:
                    if is_model_element(source):
                        source_name = source.dataSourceClass.name
                        source_name = source_name[0].lower() + source_name[1:]

                        # Check if the Create method exists
                        assert (
                            f"def create_{source_name}(request)" in content
                        ), f"Missing Create method for {scr.name} in the generated file."

                        # Check if the Edit method exists
                        assert (
                            f"def edit_{source_name}(request, {source_name}_id):"
                            in content
                        ), f"Missing Edit method for {scr.name} in the generated file."

                        # Check if the Delete method exists
                        assert (
                            f"def delete_{source_name}(request, {source_name}_id):"
                            in content
                        ), f"Missing Delete method for {scr.name} in the generated file."

    shutil.rmtree(project_folder)


################ Test back-end logic #################
def test_Operations_many_to_many():
    # Define the model to be generated:
    # Class1 definition
    class1_name: Property = Property(name="name", type="int")
    class1: Class = Class(name="Class1", attributes=[class1_name])

    # Class2 definition
    class2_name: Property = Property(name="name", type="int")
    class2: Class = Class(name="Class2", attributes=[class2_name])

    # Class3 definition
    class3_name: Property = Property(name="name", type="int")
    class3: Class = Class(name="Class3", attributes=[class3_name])

    # class1-class2 association definition
    end1: Property = Property(name="end1", type=class1, multiplicity=Multiplicity(1, 1))
    end2: Property = Property(name="end2", type=class2, multiplicity=Multiplicity(0, "*"))
    class1_class2_association: BinaryAssociation = BinaryAssociation(
        name="class1_class2_assoc", ends={end1, end2}
    )

    # class1-class3 association definition
    end3: Property = Property(name="end3", type=class2, multiplicity=Multiplicity(0, "*"))
    end4: Property = Property(name="end4", type=class3, multiplicity=Multiplicity(1, "*"))
    class1_class3_association: BinaryAssociation = BinaryAssociation(
        name="class1_class3_assoc", ends={end3, end4}
    )

    # Domain model definition
    model: DomainModel = DomainModel(
        name="Library_model",
        types={class1, class2, class3},
        associations={class1_class2_association, class1_class3_association}
    )

    # GUI Model:
    # DataSource definition
    datasource1: DataSourceElement = DataSourceElement(name="DataSource", dataSourceClass=class1, fields=[])

    # List definition
    datalist1: DataList = DataList(name="List1", description="", list_sources={datasource1})

    # Screen1 definition
    screen1: Screen = Screen(
        name="Screen1",
        description="",
        x_dpi="",
        y_dpi="",
        screen_size="Small",
        view_elements={datalist1}
    )

    # DataSource definition
    datasource2: DataSourceElement = DataSourceElement(name="DataSource", dataSourceClass=class2, fields=[])

    # List2 definition
    datalist2: DataList = DataList(name="List2", description="", list_sources={datasource2})

    # Screen2 definition
    screen2: Screen = Screen(
        name="Screen2",
        description="",
        x_dpi="",
        y_dpi="",
        screen_size="Small",
        view_elements={datalist2}
    )

    # DataSource definition
    datasource3: DataSourceElement = DataSourceElement(name="DataSource", dataSourceClass=class3, fields=[])

    # List3 definition
    datalist: DataList = DataList(name="List3", description="", list_sources={datasource3})

    # Screen3 definition
    screen3: Screen = Screen(
        name="Screen3",
        description="",
        x_dpi="",
        y_dpi="",
        screen_size="Small",
        view_elements={datalist}
    )

    # Home page Screen definition
    MyHomeScreen: Screen = Screen(
        name="HomeScreen",
        description="",
        x_dpi="",
        y_dpi="",
        screen_size="Small",
        view_elements={},
        is_main_page=True
    )

    # Module definition
    MyModule: Module = Module(
        name="MyModule",
        screens={MyHomeScreen, screen1, screen2, screen3}
    )

    # GUI model definition
    gui_model: GUIModel = GUIModel(
        name="app",
        package="com.example.app",
        versionCode="1",
        versionName="1.0",
        description="This is a comprehensive Django application",
        screenCompatibility=True,
        modules={MyModule}
    )

    # Generate the models file
    models_file = 'my_project/my_app/models.py'

    django = DjangoGenerator(
        model=model,
        project_name="my_project",
        app_name="my_app",
        gui_model=gui_model,
        containerization=False,
    )
    django.generate()

    # Validate project folder creation
    project_folder = f'{django.project_name}'
    assert os.path.exists(project_folder), "The project folder was not created."

    # Validate models file creation
    assert os.path.exists(models_file), "The models file was not created."

    # Read the generated content
    with open(models_file, 'r') as file:
        content = file.read()

    # Validate ManyToMany relationships
    assert "models.ManyToManyField" in content, "Missing ManyToManyField in the generated file."

    # Validate ForeignKey relationships
    assert "models.ForeignKey" in content, "Missing ManyToManyField in the generated file."

    shutil.rmtree(project_folder)
