import pickle, os
from BUML.metamodel.structural.structural import DomainModel

class ModelSerializer():
    def __init__(self) -> None:
        pass

    def dump(self, model: DomainModel, output_dir: str=None):
        file_path = model.name + ".buml"
        if output_dir != None:
            file_path = os.path.join(output_dir, model.name)
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
            f.close()

    def load(self, model_path= str):
        with open(model_path, "rb") as f:
            model_loaded = pickle.load(f)
            f.close()
            return model_loaded