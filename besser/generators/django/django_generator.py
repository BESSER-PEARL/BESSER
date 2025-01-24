import os
import subprocess
import sys
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.structural import DomainModel
from besser.generators import GeneratorInterface
from besser.utilities import sort_by_timestamp

class DjangoGenerator(GeneratorInterface):
    """
    DjangoGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the Django application code based on the input B-UML model.

    Args:
        model (DomainModel): An instance of the DomainModel class representing the B-UML model.
        project_name (str): The name of the Django project.
        app_name (str): The name of the Django app.
        containerization (bool, optional): Flag indicating if containerization support should be added 
            (False as defaults).
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    def __init__(self, model: DomainModel, project_name: str, app_name: str,
                containerization: bool = False, output_dir: str = None):
        super().__init__(model, output_dir)
        self.project_name: str = project_name
        self.app_name: str = app_name
        self.containerization: bool = containerization
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.env = Environment(loader=FileSystemLoader(templates_path),
                                trim_blocks=True,
                                lstrip_blocks=True,
                                extensions=['jinja2.ext.do'])

    def generate(self, *args):
        """
        Generates Django application code based on the provided B-UML model and saves it to the specified 
        output directory. If the output directory was not specified, the code generated will be stored in 
        the <current directory>/<project_name>.

        Returns:
            None, but store the generated code.
        """

        # Build Django project
        subprocess.run(['django-admin', 'startproject', self.project_name], check=True)
        # Create Django app
        subprocess.run([sys.executable, 'manage.py', 'startapp', self.app_name],
                        cwd=self.project_name, check=True)
        # Update settings.py
        self.update_settings()
        # Create requirements.txt
        self.create_file_from_template('requirements.txt.j2', 'requirements.txt')
        # Create docker files if containerization is True
        if self.containerization is True:
            self.create_file_from_template('docker_compose.j2', 'docker-compose.yml')
            self.create_file_from_template('dockerfile.j2', 'Dockerfile')
            self.create_file_from_template('entrypoint.sh.j2', 'entrypoint.sh')
        # Create models.py
        self.create_file_from_template('models.py.j2', os.path.join(self.app_name, "models.py"))
        # Create admin.py
        self.create_file_from_template('admin.py.j2', os.path.join(self.app_name, "admin.py"))

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
