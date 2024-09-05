import os
from jinja2 import Environment, FileSystemLoader
from besser.BUML.metamodel.nn import NN, TrainingDataset, TestDataset
from besser.generators import GeneratorInterface
from besser.generators.tf.utils import get_layer_notation, get_tensorop_notation, \
     add_notation_to_sub_nn
from typing import Dict

class TFGenerator(GeneratorInterface):
    """
    TFGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the TF code for neural networks training and evaluation based on the input B-UML model.

    Args:
        model (NN): An instance of the NN Model class representing the B-UML model.
        train_data (TrainingDataset): An instance of the training dataset class representing the dataset used to train the NN model.
        test_data (TestDataset): An instance of the test dataset class representing the dataset used to evaluate the NN model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """
    def __init__(self, model: NN, train_data: TrainingDataset = None, test_data: TestDataset = None,
                 output_dir: str = None):
        super().__init__(model, output_dir)
        self.train_data: TrainingDataset = train_data
        self.test_data: TestDataset = test_data
        self.modules_details: Dict = self.get_modules_details()

        @property
        def train_data(self) -> TrainingDataset:
            """TrainingDataset: Get the dataset used to train the NN model."""
            return self.__train_data
    
        @train_data.setter
        def train_data(self, train_data: TrainingDataset):
            """TrainingDataset: Set the dataset used to train the NN model."""
            self.__train_data = train_data

        @property
        def test_data(self) -> TestDataset:
            """TestDataset: Get the dataset used for evaluating the NN model."""
            return self.__test_data
    
        @test_data.setter
        def test_data(self, test_data: TestDataset):
            """TestDataset: Set the dataset used for evaluating the NN model."""
            self.__test_data = test_data

        @property
        def modules_details(self) -> Dict:
            """Dict: Get the dict that contains the names of layers or sub_nns or tensorops and their syntax and notation."""
            return self.__modules_details
    
        @modules_details.setter
        def modules_details(self, modules_details: Dict):
            """Dict: Set the dict that contains the names of layers or sub_nns or tensorops and their syntax and notation."""
            raise AttributeError("modules_details attribute is read-only")


    def get_modules_details(self):
        counter_sub_nn = 0
        modules_details = {}
        for module in self.model.modules:
            if module.__class__.__name__ == "NN":
                sub_nn_modules_details = {}
                for sub_nn_layer in module.layers:
                    sub_nn_modules_details = get_layer_notation(sub_nn_layer, sub_nn_modules_details)
                name_sub_nn = module.name + "_" + str(counter_sub_nn) + "_nn" 
                modules_details[name_sub_nn] = sub_nn_modules_details
                counter_sub_nn += 1
                modules_details = add_notation_to_sub_nn(modules_details)
            elif module.__class__.__name__ != "TensorOp":
                modules_details = get_layer_notation(module, modules_details)
            else:
                modules_details = get_tensorop_notation(module, modules_details)
        return modules_details
    
    def build_generation_path(self, file_name:str) -> str:
        if self.output_dir != None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            file_path = os.path.join(self.output_dir, file_name)
        else:
            working_path = os.path.abspath('')
            os.makedirs(os.path.join(working_path, "output"), exist_ok=True)
            file_path = os.path.join(working_path, "output", file_name)
        return file_path

    def generate(self):
        """
        Generates TF code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but store the generated code as a file named tf_nn.py 
        """
        file_path = self.build_generation_path(file_name="tf_nn.py")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template(f"template_tf_functional.py.j2")

        with open(file_path, mode="w") as f:
            generated_code = template.render(model=self.model, train_data=self.train_data, test_data=self.test_data, modules_details=self.modules_details)
            f.write(generated_code)
            print("Code generated in the location: " + file_path)