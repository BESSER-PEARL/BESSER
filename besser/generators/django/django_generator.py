"""
This module generates Django code using Jinja2 templates based on BUML models.
"""
import os
import subprocess
import sys
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.gui import GUIModel, Module, Button, DataList, DataSourceElement
from besser.BUML.metamodel.structural import DomainModel, PrimitiveDataType, Enumeration
from besser.generators import GeneratorInterface
from besser.utilities import sort_by_timestamp

##############################
#    Django Generator
##############################
class DjangoGenerator(GeneratorInterface):
    """
    DjangoGenerator is responsible for generating Django executable code based on
    input B-UML and GUI models. It implements the GeneratorInterface and facilitates
    the creation of a Django web application structure.

    Args:
        model (DomainModel): The B-UML model representing the application's domain.
        project_name (str): The name of the Django project.
        app_name (str): The name of the Django application.
        gui_model (GUIModel): The GUI model instance containing necessary configurations.
        main_page (Screen): The main page of the web application.
        containerization (bool, optional): Whether to enable containerization
        support. Defaults to False.
        module (Module, optional): Represents a specific module within the application,
          typically grouping related screens and functionalities.
        output_dir (str, optional): Directory where generated code will be saved. Defaults to None.
    """

    def __init__(self, model: DomainModel, project_name: str, app_name: str,
                 gui_model: GUIModel = None, containerization: bool = False,
                 module: Module = None, output_dir: str = None):
        super().__init__(model, output_dir)
        self.project_name: str = project_name
        self.app_name: str = app_name
        self.containerization: bool = containerization
        self.gui_model: GUIModel = gui_model
        self.module: Module = module
        self.one_to_one = {}
        self.fkeys = {}
        self.many_to_many = {}
        # Jinja environment configuration
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.env = Environment(loader=FileSystemLoader(templates_path), trim_blocks=True,
                               lstrip_blocks=True, extensions=['jinja2.ext.do'])

    @property
    def gui_model(self) -> GUIModel:
        """GUIModel: Get the instance of the GUIModel class representing the GUI model."""
        return self.__gui_model

    @gui_model.setter
    def gui_model(self, gui_model: GUIModel):
        """GUIModel: Set the instance of the GUIModel class representing the GUI model."""
        self.__gui_model = gui_model

    @property
    def module(self) -> Module:
        """Module: Get the instance of the Module class representing
             the module of the Django application."""
        return self.__module

    @module.setter
    def module(self, module: Module):
        """Module: Set the instance of the Module class representing the
               module of the Django application."""
        self.__module = module

    @staticmethod
    def is_button(value):
        """Check if the given value is an instance of Button class."""
        return isinstance(value, Button)

    @staticmethod
    def is_list(value):
        """Check if the given value is an instance of DataList class."""
        return isinstance(value, DataList)

    @staticmethod
    def is_model_element(value):
        """Check if the given value is an instance of DataSourceElement class."""
        return isinstance(value, DataSourceElement)

    @staticmethod
    def is_primitive_data_type(value):
        """Check if the given value is an instance of PrimitiveDataType class."""
        return isinstance(value, PrimitiveDataType)

    @staticmethod
    def is_enumeration(value):
        """Check if the given value is an instance of Enumeration class."""
        return isinstance(value, Enumeration)

    ## DjangoGeneratorModelsFile:
    def generate_models(self):

        """
        Generates Django models code based on the provided B-UML model
        and saves it to the specified output directory.
        If the output directory was not specified, the code generated
        will be stored in the <current directory>/output
        folder.

        Returns:
            None, but stores the generated code as a file named models.py.
        """
        for association in self.model.associations:
            ends = list(association.ends)  # Convert set to list

            # One-to-one
            if ends[0].multiplicity.max == 1 and ends[1].multiplicity.max == 1:
                if ends[1].is_navigable and not ends[0].is_navigable:
                    self.one_to_one[association.name] = ends[0].type.name
                elif not ends[1].is_navigable and ends[0].is_navigable:
                    self.one_to_one[association.name] = ends[1].type.name
                elif ends[1].multiplicity.min == 0:
                    self.one_to_one[association.name] = ends[1].type.name
                else:
                    self.one_to_one[association.name] = ends[0].type.name

            # Foreign Keys
            elif ends[0].multiplicity.max > 1 and ends[1].multiplicity.max <= 1:
                self.fkeys[association.name] = ends[0].type.name

            elif ends[0].multiplicity.max <= 1 and ends[1].multiplicity.max > 1:
                self.fkeys[association.name] = ends[1].type.name

            # Many to many
            elif ends[0].multiplicity.max > 1 and ends[1].multiplicity.max > 1:
                if ends[1].is_navigable and not ends[0].is_navigable:
                    self.many_to_many[association.name] = ends[0].type.name
                elif not ends[1].is_navigable and ends[0].is_navigable:
                    self.many_to_many[association.name] = ends[1].type.name
                elif ends[0].multiplicity.min >= 1:
                    self.many_to_many[association.name] = ends[1].type.name
                else:
                    self.many_to_many[association.name] = ends[0].type.name

        file_path = os.path.join(self.project_name, self.app_name, "models.py")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('models.py.j2')

        env.tests['is_primitive_data_type'] = self.is_primitive_data_type
        with open(file_path, mode="w", encoding="utf-8") as f:
            generated_code = template.render(model=self.model,
                                            sort_by_timestamp=sort_by_timestamp,
                                            one_to_one = self.one_to_one,
                                            many_to_many = self.many_to_many,
                                            fkeys = self.fkeys)
            f.write(generated_code)

    ## DjangoGeneratorURLsFile:
    def generate_urls(self):

        """
        Generates the Django URLs file for a web application based on
        the provided B-UML and GUI models and saves it
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
        env.tests['is_Button'] = self.is_button
        env.tests['is_List'] = self.is_list
        env.tests['is_ModelElement'] = self.is_model_element

        if self.module is None:
            # User did not specify a module, so select the first module from the set of modules
            self.module = next(iter(self.gui_model.modules))

        screens = sort_by_timestamp(self.module.screens)

        # Identify the main screen
        main_page = None
        for scr in screens:
            if scr.is_main_page:
                main_page = scr
                break

        if main_page in screens:
            screens.remove(main_page)
        else:
            print("Warning: No main page found.")

        with open(file_path, mode="w", encoding="utf-8") as f:
            generated_code = template.render(
                app=self.gui_model,
                screens=screens,
                screen=main_page,
                model=self.model,
            )
            f.write(generated_code)


    ## DjangoGeneratorFormsFile:
    def generate_forms(self, one_to_one, many_to_many, fkeys):

        """
        Generates the Django Forms file for a web application based on
        the provided B-UML and GUI models and saves it
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
        env.tests['is_Button'] = self.is_button
        env.tests['is_List'] = self.is_list
        env.tests['is_ModelElement'] = self.is_model_element
        env.tests['is_enumeration'] = self.is_enumeration
        if self.module is None:
            # User did not specify a module, so select the first module from the set of modules
            self.module = next(iter(self.gui_model.modules))

        screens = sort_by_timestamp(self.module.screens)


        # Identify the main screen
        main_page = None
        for scr in screens:
            if scr.is_main_page:
                main_page = scr
                break

        if main_page in screens:
            screens.remove(main_page)
        else:
            print("Warning: No main page found.")
            screens = list(screens)  # Convert set to list

        with open(file_path, mode="w", encoding="utf-8") as f:
            generated_code = template.render(
                app=self.gui_model,
                sort_by_timestamp=sort_by_timestamp,
                screens=screens,
                screen=main_page,
                model=self.model,
                associations=self.model.associations,
                one_to_one = one_to_one,
                many_to_many = many_to_many,
                fkeys = fkeys
            )
            f.write(generated_code)


    ##  views Generator:
    def generate_views(self):

        """
        Generates the Django views file for a web application based on
        the provided B-UML and GUI models and saves it
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
        env.tests['is_Button'] = self.is_button
        env.tests['is_List'] = self.is_list
        env.tests['is_ModelElement'] = self.is_model_element
        if self.module is None:
            # User did not specify a module, so select the first module from the set of modules
            self.module = next(iter(self.gui_model.modules))

        screens = sort_by_timestamp(self.module.screens)

        # Identify the main screen
        main_page = None
        for scr in screens:
            if scr.is_main_page:
                main_page = scr
                break

        if main_page in screens:
            screens.remove(main_page)
        else:
            print("Warning: No main page found.")
            screens = list(screens)  # Convert set to list

        with open(file_path, mode="w", encoding="utf-8") as f:
            generated_code = template.render(
                app=self.gui_model,
                screens=screens,
                screen=main_page,
                BUMLClasses=self.model.get_classes(),
                model=self.model,
                associations=self.model.associations
            )
            f.write(generated_code)


  ##  Home Page Template Generator:
    def generate_home_page(self):

        """
        Generates the Home Page Template code for a Django application based on
        the provided GUI model and saves it
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
        env.tests['is_Button'] = self.is_button
        env.tests['is_List'] = self.is_list
        env.tests['is_ModelElement'] = self.is_model_element
        if self.module is None:
            # User did not specify a module, so select the first module from the set of modules
            self.module = next(iter(self.gui_model.modules))

        screens = sort_by_timestamp(self.module.screens)

        # Identify the main screen
        main_page = None
        for scr in screens:
            if scr.is_main_page:
                main_page = scr
                break

        if main_page in screens:
            screens.remove(main_page)
        else:
            print("Main Page not found in the screens list.")
            screens = list(screens)  # Convert set to list

        with open(file_path, mode="w", encoding="utf-8") as f:
            generated_code = template.render(
                app=self.gui_model,
                screens=screens,
                screen=main_page,
            )
            f.write(generated_code)

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
        env.tests['is_Button'] = self.is_button
        env.tests['is_List'] = self.is_list
        env.tests['is_ModelElement'] = self.is_model_element

        # Select the first module if none was specified
        if self.module is None:
            self.module = next(iter(self.gui_model.modules))

        screens = self.module.screens

        # Generate HTML files for each screen in screens
        for module in self.gui_model.modules:
            for screen in module.screens:
            # Loop through view elements to find `source_name`
                for component in screen.view_elements:
                    if self.is_list(component):  # Ensure it's a List component
                        for source in component.list_sources:
                            if self.is_model_element(source):
                                # Format the file name based on `source.dataSourceClass.name`
                                source_name = source.dataSourceClass.name
                                file_name = f"{source_name[0].lower() + source_name[1:]}.html"
                                file_path = os.path.join(self.output_dir, file_name)
                                # Render the HTML with specific screen data
                                rendered_html = template.render(
                                    app=self.gui_model,
                                    screens=screens,
                                    screen=screen,
                                    source_name=source_name,
                                )


                                # Write to the HTML file
                                with open(file_path, mode="w", encoding="utf-8") as f:
                                    f.write(rendered_html)


    # List Pages Template Generator
    def generate_list_html_pages(self, one_to_one, many_to_many, fkeys):
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
        env.tests['is_Button'] = self.is_button
        env.tests['is_List'] = self.is_list
        env.tests['is_ModelElement'] = self.is_model_element

        # Select the first module if none was specified
        if self.module is None:
            self.module = next(iter(self.gui_model.modules))

        screens = self.module.screens

        # Generate HTML files for each screen in screens
        for module in self.gui_model.modules:
            for screen in module.screens:
                # Loop through view elements to find `source_name`
                for component in screen.view_elements:
                    if self.is_list(component):  # Ensure it's a List component
                        for source in component.list_sources:
                            if self.is_model_element(source):
                                # Format the file name based on `source.dataSourceClass.name`
                                source_name = source.dataSourceClass.name
                                file_name = f"{source_name[0].lower() + source_name[1:]}_list.html"
                                file_path = os.path.join(self.output_dir, file_name)

                                # Render the HTML with specific screen data
                                rendered_html = template.render(
                                    app=self.gui_model,
                                    sort_by_timestamp=sort_by_timestamp,
                                    model=self.model,
                                    screens=screens,
                                    screen=screen,
                                    source_name=source_name,
                                    one_to_one= one_to_one,
                                    many_to_many = many_to_many,
                                    fkeys = fkeys
                                )

                                # Write to the HTML file
                                with open(file_path, mode="w", encoding="utf-8") as f:
                                    f.write(rendered_html)


    ## Form Pages Template Generator:
    def generate_form_html_pages(self, one_to_one, many_to_many, fkeys):


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
        env.tests['is_Button'] = self.is_button
        env.tests['is_List'] = self.is_list
        env.tests['is_ModelElement'] = self.is_model_element

        # Select the first module if none was specified
        if self.module is None:
            self.module = next(iter(self.gui_model.modules))

        screens = self.module.screens

        # Generate HTML files for each screen in screens
        for module in self.gui_model.modules:
            for screen in module.screens:
                # Loop through view elements to find `source_name`
                for component in screen.view_elements:
                    if self.is_list(component):  # Ensure it's a List component
                        for source in component.list_sources:
                            if self.is_model_element(source):
                                # Format the file name based on `source.dataSourceClass.name`
                                source_name = source.dataSourceClass.name
                                file_name = f"{source_name[0].lower() + source_name[1:]}_form.html"
                                file_path = os.path.join(self.output_dir, file_name)

                                # Render the HTML with specific screen data
                                rendered_html = template.render(
                                    app=self.gui_model,
                                    sort_by_timestamp=sort_by_timestamp,
                                    model=self.model,
                                    screens=screens,
                                    screen=screen,
                                    source_name=source_name,
                                    one_to_one = one_to_one,
                                    many_to_many = many_to_many,
                                    fkeys = fkeys
                                )

                                # Write to the HTML file
                                with open(file_path, mode="w", encoding="utf-8") as f:
                                    f.write(rendered_html)


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

        with open(file_path, mode="w", encoding="utf-8") as f:
            generated_code = template.render(app=self.app_name)
            f.write(generated_code)

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
                    content = (
                           content[:start_index]
                           + [new_database_config]
                           + content[end_index + 2:]
                        )

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
        """Generates the Django project, app, and necessary configurations."""

        try:
            # Step 1: Initialize Django project and app
            subprocess.run(['django-admin', 'startproject', self.project_name], check=True)
            subprocess.run([sys.executable, 'manage.py', 'startapp',
                                self.app_name], cwd=self.project_name, check=True)

            # Step 2: Update settings.py
            self.update_settings()

            # Step 3: Create requirements.txt
            self.create_file_from_template('requirements.txt.j2', 'requirements.txt')

            # Step 4: Generate containerization files if enabled
            if self.containerization:
                self.create_file_from_template('docker_compose.j2', 'docker-compose.yml')
                self.create_file_from_template('dockerfile.j2', 'Dockerfile')
                self.create_file_from_template('entrypoint.sh.j2', 'entrypoint.sh')

            # Step 5: Generate models

            self.generate_models()
            self.create_file_from_template('admin.py.j2',
                            os.path.join(self.app_name, "admin.py"))

            # Step 6: Generate either admin panel or GUI-based components
            if self.gui_model:
                self.generate_urls()
                self.generate_forms(one_to_one=self.one_to_one,
                                    many_to_many=self.many_to_many,
                                    fkeys=self.fkeys)
                self.generate_views()
                self.generate_home_page()
                self.generate_base_pages()
                self.generate_list_html_pages(one_to_one=self.one_to_one,
                                              many_to_many=self.many_to_many,
                                              fkeys=self.fkeys)
                self.generate_form_html_pages(one_to_one=self.one_to_one,
                                              many_to_many=self.many_to_many,
                                              fkeys=self.fkeys)
                self.generate_project_urls()

            print("✅ Django project generation completed successfully!")

        except subprocess.CalledProcessError as e:
            print(f"❌ Error during project generation: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
