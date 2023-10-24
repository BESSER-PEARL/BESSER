from abc import ABC, abstractmethod
from BUML.metamodel.structural.structural import DomainModel

# Interface for code generators
class GeneratorInterface(ABC):

    def __init__(self, model: DomainModel, output_dir: str = None):
        self.model = model
        self.output_dir = output_dir

    @abstractmethod
    def generate(self, *args):
        pass