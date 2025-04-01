import os
from besser.BUML.metamodel.gui import *
from besser.BUML.metamodel.structural import *
from jinja2 import Template, Environment, FileSystemLoader
from besser.generators import GeneratorInterface



##############################
#    SQL_Helper Generator
##############################

class FlutterSQLHelperGenerator(GeneratorInterface):

    """
    FlutterSQLHelperGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the Flutter SQL helper code based on the input B-UML model.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    TYPES = {
        "int": "int?",
        "str": "String",
        "float": "float8",
        "bool": "bool",
        "time": "time",
        "date": "date",
        "datetime": "timestamp",
        "timedelta": "interval",
    }

    def __init__(self, model: DomainModel, output_dir: str = None):
       super().__init__(model, output_dir)


    def generate(self):

        """
        Generates Flutter SQL helper code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but stores the generated code as a file named sql_helper.dart.
        """

        copy_model: DomainModel = self.model

        for cls in copy_model.get_classes():
                 attr_list = list(cls.attributes)
                 cls.attributes = attr_list

        for cls in copy_model.get_classes():
            for atr in cls.attributes:
              print(cls.name + " ::  "+ atr.name)


        file_path = self.build_generation_path(file_name="sql_helper.dart")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('flutterCodeGeneratorSqlHelperFile.py.j2')
        with open(file_path, mode="w") as f:
            generated_code = template.render(BUMLClasses= copy_model.get_classes(), model=copy_model, types=self.TYPES)
            f.write(generated_code)
            print("Code generated in the location: " + file_path)




##############################
#    Main Dart Generator
##############################

class FlutterMainDartGenerator(GeneratorInterface):

    """
    FlutterMainDartGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the main Dart code for a Flutter application based on the input B-UML and GUI models.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        gui_model (GUIModel): An instance of the GUIModel class representing the GUI model.
        main_page (Screen): An instance of the Screen class representing the main page of the Flutter application.
        module (Module): An instance of the Module class representing the module of the Flutter application.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    def __init__(self, model: DomainModel, gui_model: GUIModel,  main_page: Screen, module: Module = None, output_dir: str = None):
        super().__init__(model, output_dir)
        self.gui_model: GUIModel = gui_model
        self.main_page: Screen = main_page
        self.module: Module = module

    @property
    def gui_model(self) -> GUIModel:
        """GUIModel: Get the instance of the GUIModel class representing the GUI model."""
        return self.__gui_model

    @gui_model.setter
    def gui_model(self, gui_model: GUIModel):
        """GUIModel: Set the instance of the GUIModel class representing the GUI model."""
        self.__gui_model = gui_model

    @property
    def main_page(self) -> Screen:
        """Screen: Get the instance of the Screen class representing the main page of the Flutter application."""
        return self.__main_page

    @main_page.setter
    def main_page(self, main_page: Screen):
        """Screen: Set the instance of the Screen class representing the main page of the Flutter application."""
        self.__main_page = main_page

    @property
    def module(self) -> Module:
        """Module: Get the instance of the Module class representing the module of the Flutter application."""
        return self.__module

    @module.setter
    def module(self, module: Module):
        """Module: Set the instance of the Module class representing the module of the Flutter application."""
        self.__module = module

    @staticmethod
    def is_Button(value):
        """Check if the given value is an instance of Button class."""
        return isinstance(value, Button)

    @staticmethod
    def is_List(value):
        """Check if the given value is an instance of DataList class."""
        return isinstance(value, DataList)

    @staticmethod
    def is_ModelElement(value):
        """Check if the given value is an instance of DataSourceElement class."""
        return isinstance(value, DataSourceElement)

    def generate(self):

        """
        Generates the main Dart code for a Flutter application based on the provided B-UML and GUI models and saves it
        to the specified output directory.
        If the output directory was not specified, the code generated will be
        stored in the <current directory>/output folder.

        Returns:
            None, but stores the generated code as a file named main.dart.
        """

        file_path = self.build_generation_path(file_name="main.dart")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('flutterCodeGeneratorMainFile.py.j2')
        env.tests['is_Button'] = self.is_Button
        env.tests['is_List'] = self.is_List
        env.tests['is_ModelElement'] = self.is_ModelElement
        if self.module is None:
          # User did not specify a module, so select the first module from the set of modules
          self.module = next(iter(self.gui_model.modules))

        screens = self.module.screens
        screens.remove(self.main_page)
        for scr in screens:
              print(scr.name + " ::  ")

        with open(file_path, mode="w") as f:
            generated_code = template.render(
                app=self.gui_model,
                screens=screens,
                screen=self.main_page,
                BUMLClasses=self.model.get_classes(),
                model=self.model,
                associations=self.model.associations
            )
            f.write(generated_code)
            print("Code generated in the location: " + file_path)


##############################
#    pubspec Generator
##############################


class FlutterPubspecGenerator(GeneratorInterface):

    """
    FlutterPubspecGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the pubspec.yaml file for a Flutter application based on the input GUI model.

    Args:
        gui_model (GUIModel): An instance of the GUIModel class representing the GUI model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    def __init__(self, gui_model: GUIModel, output_dir: str = None):
        super().__init__(output_dir)
        self.gui_model: GUIModel = gui_model


    @property
    def gui_model(self) -> GUIModel:
        """GUIModel: Get the instance of the GUIModel class representing the GUI model."""
        return self.__gui_model

    @gui_model.setter
    def gui_model(self, gui_model: GUIModel):
        """GUIModel: Set the instance of the GUIModel class representing the GUI model."""
        self.__gui_model = gui_model


    def generate(self):

        """
        Generates the pubspec.yaml file for a Flutter application based on the provided GUI model and saves it
        to the specified output directory. If the output directory was not specified, the code generated will be
        stored in the <current directory>/output folder.

        Returns:
            None, but stores the generated code as a file named pubspec.yaml.
        """

        file_path = self.build_generation_path(file_name="pubspec.yaml")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('flutterCodeGeneratorPubspecFile.py.j2')
        with open(file_path, mode="w") as f:
            generated_code = template.render(
                app=self.gui_model
            )
            f.write(generated_code)
            print("Code generated in the location: " + file_path)



##############################
#    Flutter Generator
##############################

class FlutterGenerator(GeneratorInterface):

    """
        A class that represents a Flutter code generator.

        This class extends the GeneratorInterface, which is an interface or base class for code generators.

        Args:
            model: An object representing the B-UML model.
            gui_model: An object representing the GUI model.
            main_page: An object representing the main page of the Flutter application.
            module: An object representing the module of the Flutter application.
            output_dir: The output directory where the generated code will be saved. (optional)
     """
    def __init__(self, model, gui_model, main_page, module=None, output_dir=None):
        super().__init__(model, output_dir)

        self.gui_model = gui_model
        self.main_page = main_page
        self.module = module


    def generate(self):

        """
        Generates the Flutter code based on the provided models.

        This method creates instances of the FlutterSQLHelperGenerator, FlutterMainDartGenerator,
        and FlutterPubspecGenerator classes. It then calls the generate() method on each of them
        to generate the SQL helper code, main Dart code, and pubspec.yaml file, respectively.

        Returns:
            None
            None, but store the generated code as files main.dart, sql_helper.dart, and pubspec.yaml.
        """


        sql_helper_generator = FlutterSQLHelperGenerator(model=self.model, output_dir=self.output_dir)
        sql_helper_generator.generate()

        main_dart_generator = FlutterMainDartGenerator(model=self.model, gui_model=self.gui_model, main_page=self.main_page, module=self.module, output_dir=self.output_dir)
        main_dart_generator.generate()

        pubspec_generator = FlutterPubspecGenerator(gui_model=self.gui_model, output_dir=self.output_dir)
        pubspec_generator.generate()