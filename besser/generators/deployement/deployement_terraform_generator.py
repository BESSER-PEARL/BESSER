import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.lla import PublicCluster, IPRangeType, ServiceType
from besser.generators import GeneratorInterface

class DeploymentGenerator(GeneratorInterface):
    """
    DeploymentGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the deployment models code based on the input B-UML model.
    Args:
        public_cluster (PublicCluster): The cluster object containing deployment information.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    def __init__(self, public_cluster: PublicCluster, output_dir: str = None):
        super().__init__(public_cluster, output_dir)
        self.public_cluster = public_cluster
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        self.env = self.setup_environment()

    def setup_environment(self):
        """
        Sets up the Jinja2 environment and adds custom filters.
        """
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path), trim_blocks=True, lstrip_blocks=True)
        env.filters['class_name'] = self.get_class_name
        env.filters['to_str'] = self.convert_type
        env.filters['set_dict_item'] = self.set_dict_item
        return env

    @staticmethod
    def get_class_name(value):
        """
        Custom filter to get the class name of an object.
        """
        return value.__class__.__name__

    @staticmethod
    def set_dict_item(dictionary, key, value):
        dictionary[key] = f"default-{value}"
        return ''

    @staticmethod
    def convert_type(value):
        """
        Converts the input value to its underlying value if it is an instance of an enum class,
        otherwise converts it to a string.
        Args:
            value: The value to convert, which can be of any type including enum instances.
        Returns:
            The underlying value if an enum instance, or the string representation of the value.
        """
        # Handle enum instances by returning their associated value
        if isinstance(value, (IPRangeType, ServiceType)):
            return value.value
        # Convert other types to their string representation
        return str(value)
    def build_generation_path(self, file_name: str) -> str:
        """
        Constructs the file path for the generated file in the output directory.

        Args:
            file_name (str): The name of the file to be generated.

        Returns:
            str: The full file path for the generated file.
        """
        return os.path.join(self.output_dir, file_name)

    def generate(self):
        """
        Generates deployment models code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but stores the generated code as files with specified names.
        """
        # Dictionary mapping template file names to output file names
        template_to_file_map = {
            'version.py.j2': 'version.tf',
            'variables.py.j2': 'variables.tf',
            'cluster.py.j2': 'cluster.tf',
            'app.py.j2': 'app.tf'
        }

        for template_name, output_file_name in template_to_file_map.items():
            file_path = self.build_generation_path(file_name=output_file_name)
            template = self.env.get_template(template_name)
            with open(file_path, mode="w") as f:
                generated_code = template.render(public_cluster=self.public_cluster)
                f.write(generated_code)
                print(f"Code generated in the location: {file_path}")
