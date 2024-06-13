import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.nn import NN, TrainingDataset, TestDataset
from besser.generators import GeneratorInterface

class PytorchGenerator(GeneratorInterface):
    """
    PytorchGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the Pytorch code for neural networks training and evaluation based on the input B-UML model.

    Args:
        model (NN): An instance of the NN Model class representing the B-UML model.
        train_data (TrainingDataset): An instance of the training dataset class representing the dataset used to train the NN model.
        test_data (TestDataset): An instance of the test dataset class representing the dataset used to evaluate the NN model.
        model_type (str): The type of pytorch model generated. It can be either "sequential" or "functional". It defaults to "functional" 
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """
    def __init__(self, model: NN, train_data: TrainingDataset, test_data: TestDataset,
                 output_dir: str = None, model_type: str = "functional"):
        super().__init__(model, output_dir)
        self.train_data: TrainingDataset = train_data
        self.test_data: TestDataset = test_data
        self.model_type: str = model_type

    def generate(self):
        """
        Generates Pytorch code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but store the generated code as a file named pytorch_nn.py 
        """
        file_path = self.build_generation_path(file_name="pytorch_nn.py")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template(f"template_pytorch_{self.model_type}.py.j2")
        with open(file_path, mode="w") as f:
            generated_code = template.render(model=self.model, train_data=self.train_data, test_data=self.test_data)
            f.write(generated_code)
            print("Code generated in the location: " + file_path)