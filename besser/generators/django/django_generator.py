import os
from besser.BUML.metamodel.gui import *
from besser.BUML.metamodel.structural import *
from jinja2 import Template, Environment, FileSystemLoader
from besser.generators import GeneratorInterface

##############################
#    Django Generator
##############################
class DjangoGenerator(GeneratorInterface):
    """
    DjangoGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the Django executable code based on the input B-UML and GUI models for a web application.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        application (Application): An instance of the Application class representing the Django web application.
        main_page (Screen): An instance of the Screen class representing the main page of the web application.
        module (Module, optional): An instance of the Module class representing a module of the web application. Defaults to None.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """
    
    def __init__(self, model: DomainModel, application: Application,  main_page: Screen, module: Module = None, output_dir: str = None):
        super().__init__(model, output_dir)
        self.application: Application = application
        self.main_page: Screen = main_page
        self.module: Module = module
        self.output_dir = output_dir or os.path.join(os.getcwd(), "output", "app_files")

    @property
    def application(self) -> Application:
        """Application: Get the instance of the Application class representing the GUI model."""
        return self.__application

    @application.setter
    def application(self, application: Application):
        """Application: Set the instance of the Application class representing the GUI model."""
        self.__application = application

    @property
    def main_page(self) -> Screen:
        """Screen: Get the instance of the Screen class representing the main page of the Django application."""
        return self.__main_page

    @main_page.setter
    def main_page(self, main_page: Screen):
        """Screen: Set the instance of the Screen class representing the main page of the Django application."""
        self.__main_page = main_page

    @property
    def module(self) -> Module:
        """Module: Get the instance of the Module class representing the module of the Django application."""
        return self.__module

    @module.setter
    def module(self, module: Module):
        """Module: Set the instance of the Module class representing the module of the Django application."""
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
        """Check if the given value is an instance of ModelElement class."""
        return isinstance(value, ModelElement)  


    ## DjangoGeneratorModelsFile:
    def generate_models(self):

        """
        Generates Django models code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but stores the generated code as a file named models.py.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        copy_model: DomainModel = self.model

        for cls in copy_model.get_classes():
                 attr_list = list(cls.attributes)
                 cls.attributes = attr_list
           
        file_path = self.build_generation_path(file_name="models.py")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('djangoCodeGeneratorModelsFile.py.j2')
        with open(file_path, mode="w") as f:
            generated_code = template.render(model=copy_model)
            f.write(generated_code)
            print("Models code generated in the location: " + file_path)


    ## DjangoGeneratorURLsFile:
    def generate_urls(self):

        """
        Generates the Django URLs file for a web application based on the provided B-UML and GUI models and saves it
        to the specified output directory. 
        If the output directory was not specified, the code generated will be
        stored in the <current directory>/output folder.

        Returns:
            None, but stores the generated code as a file named urls.py.
        """

        os.makedirs(self.output_dir, exist_ok=True)

        file_path = self.build_generation_path(file_name="urls.py")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('djangoCodeGeneratorURLsFile.py.j2')
        env.tests['is_Button'] = self.is_Button
        env.tests['is_List'] = self.is_List
        env.tests['is_ModelElement'] = self.is_ModelElement
        if self.module is None:
          # User did not specify a module, so select the first module from the set of modules
          self.module = next(iter(self.application.modules))

        screens = self.module.screens

        if self.main_page in screens:
            screens.remove(self.main_page)
        else:
            print("Main Page not found in the screens list.")
            
        for scr in screens:
              print(scr.name + " ::  ")
  
        with open(file_path, mode="w") as f:
            generated_code = template.render(
                app=self.application,
                screens=screens,
                screen=self.main_page,
                model=self.model,
            )
            f.write(generated_code)
            print("URLs code generated in the location: " + file_path)


    ## DjangoGeneratorFormsFile:
    def generate_forms(self):

        """
        Generates the Django Forms file for a web application based on the provided B-UML and GUI models and saves it
        to the specified output directory. 
        If the output directory was not specified, the code generated will be
        stored in the <current directory>/output folder.

        Returns:
            None, but stores the generated code as a file named forms.py.
        """

        os.makedirs(self.output_dir, exist_ok=True)

        file_path = self.build_generation_path(file_name="forms.py")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('djangoCodeGeneratorFormsFile.py.j2')
        env.tests['is_Button'] = self.is_Button
        env.tests['is_List'] = self.is_List
        env.tests['is_ModelElement'] = self.is_ModelElement
        if self.module is None:
          # User did not specify a module, so select the first module from the set of modules
          self.module = next(iter(self.application.modules))

        screens = self.module.screens

        if self.main_page in screens:
            screens.remove(self.main_page)
        else:
            print("Main Page not found in the screens list.")
            
  
        with open(file_path, mode="w") as f:
            generated_code = template.render(
                app=self.application,
                screens=screens,
                screen=self.main_page,
                model=self.model,
                associations=self.model.associations
            )
            f.write(generated_code)
            print("Forms Code generated in the location: " + file_path)


    ##  Home Page Template Generator:
    def generate_home_page(self):

        """
        Generates the Home Page Template code for a Django application based on the provided GUI model and saves it
        to the specified output directory. 
        If the output directory was not specified, the code generated will be
        stored in the <current directory>/output folder.

        Returns:
            None, but stores the generated code as a file named main.dart.
        """

        os.makedirs(self.output_dir, exist_ok=True)

        # Customize the output directory here
        self.output_dir = os.path.join(os.getcwd(), "output", "app_files", "templates")
        

        file_path = self.build_generation_path(file_name="home.html")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('djangoCodeGeneratorHomePageTemplateFile.py.j2')
        env.tests['is_Button'] = self.is_Button
        env.tests['is_List'] = self.is_List
        env.tests['is_ModelElement'] = self.is_ModelElement
        if self.module is None:
          # User did not specify a module, so select the first module from the set of modules
          self.module = next(iter(self.application.modules))

        screens = self.module.screens

        if self.main_page in screens:
            screens.remove(self.main_page)
        else:
            print("Main Page not found in the screens list.")
            

        with open(file_path, mode="w") as f:
            generated_code = template.render(
                app=self.application,
                screens=screens,
                screen=self.main_page,
            )
            f.write(generated_code)
            print("Code generated in the location: " + file_path)


    ## base Pages Template Generator:
    def generate_base_pages(self):
        """
        Generate HTML files for each screen in the module, using a Jinja template.
        Each HTML file is saved in the output directory, named based on the screen's name.
        """
        
        os.makedirs(self.output_dir, exist_ok=True)
        # Customize the output directory here
        self.output_dir = os.path.join(os.getcwd(), "output", "app_files", "templates")


        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        
        # Load the Jinja template
        template = env.get_template('djangoCodeGeneratorBasePageTemplateFile.py.j2')
        env.tests['is_Button'] = self.is_Button
        env.tests['is_List'] = self.is_List
        env.tests['is_ModelElement'] = self.is_ModelElement

        # Select the first module if none was specified
        if self.module is None:
            self.module = next(iter(self.application.modules))

        screens = self.module.screens

        # Generate HTML files for each screen in screens
        for module in self.application.modules:
            for screen in module.screens:
            # Loop through view elements to find `source_name`
                for component in screen.view_elements:
                    if self.is_List(component):  # Ensure it's a List component
                        for source in component.list_sources:
                            if self.is_ModelElement(source):
                                # Format the file name based on `source.dataSourceClass.name`
                                source_name = source.dataSourceClass.name
                                file_name = f"{source_name[0].lower() + source_name[1:]}.html"
                                file_path = os.path.join(self.output_dir, file_name)
                                
                                # Render the HTML with specific screen data
                                rendered_html = template.render(
                                    app=self.application,
                                    screens=screens,
                                    screen=screen,
                                    source_name=source_name,
                                )

                                # Write to the HTML file
                            with open(file_path, mode="w") as f:
                                f.write(rendered_html)
                                print(f"Generated HTML for {source_name} at {file_path}")

        print("HTML files generated for all components.")
    

    # List Pages Template Generator
    def generate_list_html_pages(self):
        """
        Generate List HTML files for each screen in the module, using a Jinja template.
        Each HTML file is saved in the output directory, named based on the screen's name_list.
        """
        
        os.makedirs(self.output_dir, exist_ok=True)
        # Customize the output directory here
        self.output_dir = os.path.join(os.getcwd(), "output", "app_files", "templates")


        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        
        # Load the Jinja template
        template = env.get_template('djangoCodeGeneratorListPageTemplateFile.py.j2')
        env.tests['is_Button'] = self.is_Button
        env.tests['is_List'] = self.is_List
        env.tests['is_ModelElement'] = self.is_ModelElement

        # Select the first module if none was specified
        if self.module is None:
            self.module = next(iter(self.application.modules))

        screens = self.module.screens

        # Generate HTML files for each screen in screens
        for module in self.application.modules:
            for screen in module.screens:
                # Loop through view elements to find `source_name`
                for component in screen.view_elements:
                    if self.is_List(component):  # Ensure it's a List component
                        for source in component.list_sources:
                            if self.is_ModelElement(source):
                                # Format the file name based on `source.dataSourceClass.name`
                                source_name = source.dataSourceClass.name
                                file_name = f"{source_name[0].lower() + source_name[1:]}_list.html"
                                file_path = os.path.join(self.output_dir, file_name)

                                # Render the HTML with specific screen data
                                rendered_html = template.render(
                                    app=self.application,
                                    model=self.model,
                                    screens=screens,
                                    screen=screen,
                                    source_name=source_name,
                                )

                                # Write to the HTML file
                                with open(file_path, mode="w") as f:
                                    f.write(rendered_html)
                                    print(f"Generated HTML for {source_name} at {file_path}")

        print("HTML files generated for all components.")


    ## Form Pages Template Generator:
    def generate_form_html_pages(self):
        """
        Generate HTML files for each screen in the module, using a Jinja template.
        Each HTML file is saved in the output directory, named based on the screen's name_form.
        """
        
        os.makedirs(self.output_dir, exist_ok=True)
        # Customize the output directory here
        self.output_dir = os.path.join(os.getcwd(), "output", "app_files", "templates")
        
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        
        # Load the Jinja template
        template = env.get_template('djangoCodeGeneratorFormPageTemplateFile.py.j2')
        env.tests['is_Button'] = self.is_Button
        env.tests['is_List'] = self.is_List
        env.tests['is_ModelElement'] = self.is_ModelElement

        # Select the first module if none was specified
        if self.module is None:
            self.module = next(iter(self.application.modules))

        screens = self.module.screens

        # Generate HTML files for each screen in screens
        for module in self.application.modules:
            for screen in module.screens:
                # Loop through view elements to find `source_name`
                for component in screen.view_elements:
                    if self.is_List(component):  # Ensure it's a List component
                        for source in component.list_sources:
                            if self.is_ModelElement(source):
                                # Format the file name based on `source.dataSourceClass.name`
                                source_name = source.dataSourceClass.name
                                file_name = f"{source_name[0].lower() + source_name[1:]}_form.html"
                                file_path = os.path.join(self.output_dir, file_name)

                                # Render the HTML with specific screen data
                                rendered_html = template.render(
                                    app=self.application,
                                    model=self.model,
                                    screens=screens,
                                    screen=screen,
                                    source_name=source_name,
                                )

                                # Write to the HTML file
                                with open(file_path, mode="w") as f:
                                    f.write(rendered_html)
                                    print(f"Generated HTML for {source_name} at {file_path}")

        print("HTML files generated for all components.")


    ##  views Generator:
    def generate_views(self):

        """
        Generates the Django views file for a web application based on the provided B-UML and GUI models and saves it
        to the specified output directory. 
        If the output directory was not specified, the code generated will be
        stored in the <current directory>/output folder.

        Returns:
            None, but stores the generated code as a file named views.py.
        """

        os.makedirs(self.output_dir, exist_ok=True)
        
        file_path = self.build_generation_path(file_name="views.py")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('djangoCodeGeneratorViewsFile.py.j2')
        env.tests['is_Button'] = self.is_Button
        env.tests['is_List'] = self.is_List
        env.tests['is_ModelElement'] = self.is_ModelElement
        if self.module is None:
          # User did not specify a module, so select the first module from the set of modules
          self.module = next(iter(self.application.modules))

        screens = self.module.screens

        if self.main_page in screens:
            screens.remove(self.main_page)
        else:
            print("Main Page not found in the screens list.")

  
        with open(file_path, mode="w") as f:
            generated_code = template.render(
                app=self.application,
                screens=screens,
                screen=self.main_page,
                BUMLClasses=self.model.get_classes(),
                model=self.model,
                associations=self.model.associations
            )
            f.write(generated_code)
            print("Code generated in the location: " + file_path)


    ## project urls file Generator:
    def generate_project_urls(self):

        """
        Generates the Django project URLs file based on the provided GUI model and saves it
        to the specified output directory. 
        If the output directory was not specified, the code generated will be
        stored in the <current directory>/output folder.

        Returns:
            None, but stores the generated code as a file named project_urls.py.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_dir = os.path.join(os.getcwd(), "output", "project_files")

        file_path = self.build_generation_path(file_name="urls.py")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('djangoCodeGeneratorProjectURLsFile.py.j2')
        
  
        with open(file_path, mode="w") as f:
            generated_code = template.render(
                app=self.application
            )
            f.write(generated_code)
            print("Code generated in the location: " + file_path)

    ## project settings file Generator:
    def generate_project_settings(self):

        """
        Generates the Django project settings file based on the provided GUI model and saves it
        to the specified output directory. 
        If the output directory was not specified, the code generated will be
        stored in the <current directory>/output folder.

        Returns:
            None, but stores the generated code as a file named project_settings.py.
        """

        os.makedirs(self.output_dir, exist_ok=True)
        self.output_dir = os.path.join(os.getcwd(), "output", "project_files")
        
        file_path = self.build_generation_path(file_name="settings.py")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('djangoCodeGeneratorProjectSettingsFile.py.j2')
       
  
        with open(file_path, mode="w") as f:
            generated_code = template.render(
                app=self.application
            )
            f.write(generated_code)
            print("Code generated in the location: " + file_path)
       
    
    def generate(self):
        """
           Generates the Django code based on the provided models.
        """
        self.generate_models()
        self.generate_urls()
        self.generate_forms()  
        self.generate_views() 
        self.generate_home_page()
        self.generate_base_pages()
        self.generate_list_html_pages()
        self.generate_form_html_pages()
        self.generate_project_urls()
        self.generate_project_settings()
       
       