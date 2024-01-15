import pickle, os
from besser.BUML.metamodel.structural import DomainModel

class ModelSerializer():
    """
    ModelSerializer is a simple class for serializing and deserializing BESSER models and Python objects.

    Attributes:
        None

    Note:
        This class uses the `pickle` module for serialization and deserialization. Ensure that the models/objects you are
        serializing are safe and trusted since `pickle` can execute arbitrary code.
    """

    def __init__(self) -> None:
        pass

    def dump(self, model: any, output_dir: str=None, output_file_name: str = None):
        """Serialize and save a model to a file.

        Args:
            model (any): the B-UML model to be serialized and saved.
            output_dir (str, optional): the directory where the file should be saved. If not provided, the file will be saved in the current working directory.
            output_file_name (str, optional): The name of the output file. If not provided, a default name will be used based on the type of the model:
                For objects of type DomainModel: "{model_name}.buml"
                For other types: "model.pkl"

        Returns:
            None, but store the model as a file.
        """
        file_path = output_file_name
        if output_file_name == None:
            file_path = "model.pkl"
            if type(model) == DomainModel:
                file_path = model.name + ".buml"
        if output_dir != None:
            file_path = os.path.join(output_dir, file_path)
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
            f.close()

    def load(self, model_path= str):
        """Deserialize and load a model from a serialized file using pickle.

        Args:
            model_path (str): the path to the serialized model file.

        Returns:
            model_loaded: the deserialized model object.
        """
        with open(model_path, "rb") as f:
            model_loaded = pickle.load(f)
            f.close()
            return model_loaded