import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.deployment import DeploymentModel, IPRangeType, ServiceType, Protocol
from besser.generators import GeneratorInterface

class DeploymentGenerator(GeneratorInterface):
    """
    DeploymentGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the deployment models code based on the input B-UML model.
    Args:
        deployement_model (DeployementModel): The deployment model containing multiple public clusters..
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """

    def __init__(self, deployment_model: DeploymentModel, output_dir: str = None):
        super().__init__(deployment_model, output_dir)
        self.deployment_model = deployment_model
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
        if isinstance(value, (IPRangeType, ServiceType, Protocol)):
            return value.value
        return str(value)
    
    def build_generation_path(self, file_name: str) -> str:
        return os.path.join(self.output_dir, file_name)

    def generate(self):
        """
        Generates deployment models code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but stores the generated code as files with specified names.
        """
        for public_cluster in self.deployment_model.clusters:
            # Read the configuration file .conf line by line
            with open(public_cluster.config_file, 'r') as file:
                config_lines = file.readlines()
            # Dictionary mapping template file names to output file names
            # Create the "gcp" directory if it doesn't exist
            gcp_dir = os.path.join(self.output_dir, "gcp")
            os.makedirs(gcp_dir, exist_ok=True)

            if public_cluster.provider.value == 'Google':
                gcp_dir = os.path.join(self.output_dir, "gcp")
                os.makedirs(gcp_dir, exist_ok=True)
                template_to_file_map = {
                    'gcp/version.tf.j2': 'gcp/version.tf',
                    'gcp/cluster.tf.j2': 'gcp/cluster.tf',
                    'gcp/app.tf.j2': 'gcp/app.tf',
                    'gcp/api.tf.j2': 'gcp/api.tf'
                }   
            elif public_cluster.provider.value == 'AWS':
                aws_dir = os.path.join(self.output_dir, "aws")
                os.makedirs(aws_dir, exist_ok=True)
                template_to_file_map = {
                    'aws/eks.tf.j2': 'aws/eks.tf',
                    'aws/iam-oidc.tf.j2': 'aws/iam-oidc.tf',
                    'aws/provider.tf.j2': 'aws/provider.tf',
                    'aws/igw.tf.j2': 'aws/igw.tf',
                    'aws/nat.tf.j2': 'aws/nat.tf',
                    'aws/routes.tf.j2': 'aws/routes.tf',
                    'aws/vpc.tf.j2': 'aws/vpc.tf',
                    'aws/nodes.tf.j2': 'aws/nodes.tf',
                    'aws/subnets.tf.j2': 'aws/subnets.tf',

                }

            for template_name, output_file_name in template_to_file_map.items():
                file_path = self.build_generation_path(file_name=output_file_name)
                template = self.env.get_template(template_name)

                with open(file_path, mode="w") as f:
                    generated_code = template.render(public_cluster=public_cluster, config_lines=config_lines)
                    f.write(generated_code)
                    print(f"Code generated in the location: {file_path}")

