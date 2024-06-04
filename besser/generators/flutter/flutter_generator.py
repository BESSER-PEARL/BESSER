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
        dataSourceClass (list[Class]): A list of Class instances representing the data source classes.
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

    def __init__(self, model: DomainModel, dataSourceClass: list[Class], output_dir: str = None):
       super().__init__(model, output_dir)
       self.dataSourceClass: list[Class]= dataSourceClass
       

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
        template = env.get_template('flutterCodeGeneratorSqlHelperFile.jinja')
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
        application (Application): An instance of the Application class representing the GUI model.
        mainPage (Screen): An instance of the Screen class representing the main page of the Flutter application.
        module (Module): An instance of the Module class representing the module of the Flutter application.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    def __init__(self, model: DomainModel, application: Application,  mainPage: Screen, module:Module, output_dir: str = None):
        super().__init__(model, output_dir)
        self.application: Application = application
        self.mainPage: Screen = mainPage
        self.module: Module = module

    @property
    def application(self) -> Application:
        return self.__application

    @application.setter
    def application(self, application: Application):
        self.__application = application

    @property
    def mainPage(self) -> Screen:
        return self.__mainPage

    @mainPage.setter
    def mainPage(self, mainPage: Screen):
        self.__mainPage = mainPage

    @property
    def module(self) -> Module:
        return self.__module

    @module.setter
    def module(self, module: Module):
        self.__module = module
    
    @staticmethod
    def is_Button(value):
        return isinstance(value, Button)

    @staticmethod
    def is_List(value):
        return isinstance(value, DataList)

    @staticmethod
    def is_ModelElement(value):
        return isinstance(value, ModelElement)

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
        template = env.get_template('flutterCodeGeneratorMainFile.jinja')
        env.tests['is_Button'] = self.is_Button
        env.tests['is_List'] = self.is_List
        env.tests['is_ModelElement'] = self.is_ModelElement
        screens = self.module.screens
        screens.remove(self.mainPage)
        for scr in screens:
              print(scr.name + " ::  ")
  
        with open(file_path, mode="w") as f:
            generated_code = template.render(
                app=self.application,
                screens=screens,
                screen=self.mainPage,
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
        application (Application): An instance of the Application class representing the GUI model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    def __init__(self, application: Application, output_dir: str = None):
        super().__init__(output_dir)
        self.application: Application = application


    @property
    def application(self) -> Application:
        return self.__application

    @application.setter
    def application(self, application: Application):
        self.__application = application


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
        template = env.get_template('flutterCodeGeneratorPubspecFile.jinja')
        with open(file_path, mode="w") as f:
            generated_code = template.render(
                app=self.application
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
            application: An object representing the GUI model.
            mainPage: An object representing the main page of the Flutter application.
            module: An object representing the module of the Flutter application.
            dataSourceClass: A list of Class objects representing the data source classes.
            output_dir: The output directory where the generated code will be saved. (optional)
     """
    def __init__(self, model, application, mainPage, module, dataSourceClass, output_dir=None):
        super().__init__(model, output_dir)

        self.application = application
        self.mainPage = mainPage
        self.module = module
        self.dataSourceClass = dataSourceClass


    def generate(self):

        """
        Generates the Flutter code based on the provided models and data source classes.

        This method creates instances of the FlutterSQLHelperGenerator, FlutterMainDartGenerator,
        and FlutterPubspecGenerator classes. It then calls the generate() method on each of them
        to generate the SQL helper code, main Dart code, and pubspec.yaml file, respectively.

        Returns:
            None
            None, but store the generated code as files main.dart, sql_helper.dart, and pubspec.yaml.
        """


        sql_helper_generator = FlutterSQLHelperGenerator(model=self.model, dataSourceClass=self.dataSourceClass, output_dir=self.output_dir)
        sql_helper_generator.generate()

        main_dart_generator = FlutterMainDartGenerator(model=self.model, application=self.application, mainPage=self.mainPage, module=self.module, output_dir=self.output_dir)
        main_dart_generator.generate()

        pubspec_generator = FlutterPubspecGenerator(application=self.application, output_dir=self.output_dir)
        pubspec_generator.generate()