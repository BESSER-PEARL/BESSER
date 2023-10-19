from abc import ABC, abstractmethod
from BUML.metamodel.structural.structural import DomainModel

# Interface for code generators
class GeneratorInterface(ABC):
    @abstractmethod
    def generate(self, buml_model: DomainModel, *args):
        pass