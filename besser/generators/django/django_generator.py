import os
from besser.BUML.metamodel.gui import *
from besser.BUML.metamodel.structural import *
from jinja2 import Template, Environment, FileSystemLoader
from besser.generators import GeneratorInterface

import subprocess
import sys
from besser.utilities import sort_by_timestamp


##############################
#    Django Generator
##############################
class DjangoGenerator(GeneratorInterface):
    """
    DjangoGenerator is responsible for generating Django executable code based on input B-UML and GUI models.
    It implements the GeneratorInterface and facilitates the creation of a Django web application structure.

    Args:
        model (DomainModel): The B-UML model representing the application's domain.
        project_name (str): The name of the Django project.
        app_name (str): The name of the Django application.
        application (Application): The application instance containing necessary configurations.
        main_page (Screen): The main page of the web application.
        containerization (bool, optional): Whether to enable containerization support. Defaults to False.
        module (Module, optional): A module representing a specific component of the application. Defaults to None.
        output_dir (str, optional): Directory where generated code will be saved. Defaults to None.
    """

    def __init__(self, model: DomainModel, project_name: str, app_name: str, application: Application, main_page: Screen, containerization: bool = False, module: Module = None, output_dir: str = None):
        super().__init__(model, output_dir)
        self.project_name: str = project_name
        self.app_name: str = app_name
        self.containerization: bool = containerization
        self.application: Application = application
        self.main_page: Screen = main_page
        self.module: Module = module
        # Jinja environment configuration
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.env = Environment(loader=FileSystemLoader(templates_path), trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.do'])

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

        copy_model: DomainModel = self.model

        for cls in copy_model.get_classes():
                 attr_list = list(cls.attributes)
                 cls.attributes = attr_list
  

        file_path = os.path.join(self.project_name, self.app_name, "models.py")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('models.py.j2')
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

        file_path = os.path.join(self.project_name, self.app_name, "urls.py")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('urls.py.j2')
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

        file_path = os.path.join(self.project_name, self.app_name, "forms.py")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('forms.py.j2')
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
        
        file_path = os.path.join(self.project_name, self.app_name, "views.py")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('views.py.j2')
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

        # Customize the output directory here
        self.output_dir = os.path.join(os.getcwd(), self.project_name, self.app_name, "templates")
        
        file_path = self.build_generation_path(file_name="home.html")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('home_page.py.j2')
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

        # Customize the output directory here
        self.output_dir = os.path.join(os.getcwd(), self.project_name, self.app_name, "templates")


        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        
        # Load the Jinja template
        template = env.get_template('basePageFile.py.j2')
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
        self.output_dir = os.path.join(os.getcwd(), self.project_name, self.app_name, "templates")


        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        
        # Load the Jinja template
        template = env.get_template('list_page.py.j2')
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
        self.output_dir = os.path.join(os.getcwd(), self.project_name, self.app_name, "templates")
        
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        
        # Load the Jinja template
        template = env.get_template('form_page.py.j2')
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

        self.output_dir = os.path.join(os.getcwd(), self.project_name, self.project_name)

        file_path = self.build_generation_path(file_name="urls.py")
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('project_urls.py.j2')
        
  
        with open(file_path, mode="w") as f:
            generated_code = template.render(
                app=self.app_name
            )
            f.write(generated_code)
            print("Code generated in the location: " + file_path)


    
    def create_file_from_template(self, template_name, output_name):
        """Create a file from a Jinja2 template."""
        template = self.env.get_template(template_name)
        file_path = os.path.join(self.project_name, output_name)
        with open(file_path, mode="w", newline='\n', encoding='utf-8') as f:
            f.write(template.render(app_name=self.app_name,
                                    project_name=self.project_name,
                                    model=self.model,
                                    sort=sort_by_timestamp))
            
    
    def update_settings(self):
        """Update the configuration in settings.py."""
        settings_file_path = os.path.join(self.project_name, self.project_name, 'settings.py')
        new_database_config = ""
        if self.containerization is True:
            new_database_config = """
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('POSTGRES_NAME'),
        'USER': os.environ.get('POSTGRES_USER'),
        'PASSWORD': os.environ.get('POSTGRES_PASSWORD'),
        'HOST': 'db',
        'PORT': 5432,
    }
}
"""
        jazzmin_settings = f"""
# Admin template configuration
JAZZMIN_SETTINGS = {{
    # title of the window (Will default to current_admin_site.site_title if absent or None)
    "site_title": "{self.project_name} - Admin",

    # Title on the login screen (19 chars max) (defaults to current_admin_site.site_header if absent or None)
    "site_header": "{self.project_name}",

    # Title on the brand (19 chars max) (defaults to current_admin_site.site_header if absent or None)
    "site_brand": "{self.project_name}",

    # Logo to use for your site, must be present in static files, used for brand on top left
    # "site_logo": "img/logo.png",

    # Links to put along the top menu
    "topmenu_links": [

        # Url that gets reversed (Permissions can be added)
        {{"name": "Home",  "url": "admin:index", "permissions": ["auth.view_user"]}},

        # App with dropdown menu to all its models pages (Permissions checked against models)
        {{"app": "{self.app_name}"}},

        # model admin to link to (Permissions checked against model)
        {{"model": "auth.User"}},
    ],
}}
"""
        try:
            with open(settings_file_path, 'r', encoding='utf-8') as file:
                content = file.readlines()

            # Ensure 'import os' is present
            if not any(line.startswith('import os') for line in content):
                for index, line in enumerate(content):
                    if line.strip() and not line.strip().startswith('#'):
                        content.insert(index, 'import os\n')
                        break

            if self.containerization is True:
                # Replace the DATABASES section
                start_index, end_index = None, None
                for index, line in enumerate(content):
                    if 'DATABASES' in line and '=' in line:
                        start_index = index
                    if start_index is not None and line.strip() == '}':
                        end_index = index
                        break

                if start_index is not None and end_index is not None:
                    content = content[:start_index] + [new_database_config] + content[end_index + 2:]

            # Add the app to INSTALLED_APPS
            for index, line in enumerate(content):
                if line.strip().startswith('INSTALLED_APPS') and '=' in line:
                    # Find the start of the list
                    open_bracket_index = index
                    while '[' not in content[open_bracket_index]:
                        open_bracket_index += 1

                    # Find the end of the list
                    close_bracket_index = open_bracket_index
                    while ']' not in content[close_bracket_index]:
                        close_bracket_index += 1

                    # Add the app if not already in the list
                    apps_section = content[open_bracket_index:close_bracket_index + 1]
                    if f"'{self.app_name}'," not in ''.join(apps_section):
                        # Insert the app just before the closing bracket
                        content.insert(close_bracket_index, f"    '{self.app_name}',\n")
                    if f"'{'jazzmin'}'," not in ''.join(apps_section):
                        # Insert the jazzmin app just before the closing bracket
                        content.insert(open_bracket_index + 1, "    'jazzmin',\n")
                    break

            # Add the JAZZMIN_SETTINGS block at the end of the file
            if jazzmin_settings.strip() not in ''.join(content):
                content.append(f"\n{jazzmin_settings}\n")

            # Write the updated settings back to the file
            with open(settings_file_path, 'w', encoding='utf-8') as file:
                file.writelines(content)

        except (IOError, OSError) as e:
            print(f"An I/O error occurred: {e}")
        except ValueError as e:
            print(f"A value error occurred: {e}")
            

    

       
    
    def generate(self, *args):
        """
           Generates the Django code based on the provided models.
        """
         # Build Django project
        subprocess.run(['django-admin', 'startproject', self.project_name], check=True)

        # Create Django app
        subprocess.run([sys.executable, 'manage.py', 'startapp', self.app_name],
                        cwd=self.project_name, check=True)
        
        # Create requirements.txt
        self.create_file_from_template('requirements.txt.j2', 'requirements.txt')

        # Update settings.py
        self.update_settings()

        # Create docker files if containerization is True
        if self.containerization is True:
            self.create_file_from_template('docker_compose.j2', 'docker-compose.yml')
            self.create_file_from_template('dockerfile.j2', 'Dockerfile')
            self.create_file_from_template('entrypoint.sh.j2', 'entrypoint.sh')
        

        self.generate_models()
        self.generate_urls()
        self.generate_forms()
        self.generate_views() 
        self.generate_home_page()
        self.generate_base_pages()
        self.generate_list_html_pages()
        self.generate_form_html_pages()
        self.generate_project_urls()

        
        
       
       