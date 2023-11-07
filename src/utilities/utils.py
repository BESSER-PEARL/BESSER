import pickle, os
from BUML.metamodel.structural.structural import DomainModel

class ModelSerializer():
    def __init__(self) -> None:
        pass

    def dump(self, model: any, output_dir: str=None, output_file_name: str = None):
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
        with open(model_path, "rb") as f:
            model_loaded = pickle.load(f)
            f.close()
            return model_loaded