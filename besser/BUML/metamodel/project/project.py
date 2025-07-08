from besser.BUML.metamodel.structural import NamedElement, Model, Metadata

class Project(NamedElement):
    """A class representing a project containing multiple models.
    Args:
        name (str): The name of the project.
        models (list[Model], optional): A list of models contained in the project. Defaults to an empty list.
        metadata (Metadata, optional): Metadata associated with the project. Defaults to None.

    Attributes:
        name (str): The name of the project.
        models (list[Model]): A list of models contained in the project.
        metadata (Metadata): Metadata associated with the project.
    """
    def __init__(self, name: str, models: list[Model] = None, owner: str = "", metadata: Metadata = None):
        super().__init__(name)
        self.models = models if models is not None else []
        self.owner = owner
        self.metadata = metadata

    @property
    def models(self) -> list[Model]:
        """list[Model]: Get the list of models."""
        return self.__models

    @models.setter
    def models(self, models: list[Model]):
        """list[Model]: Set the list of models."""
        if not isinstance(models, list):
            raise TypeError("Models must be a list of Model instances.")
        self.__models = models

    @property
    def owner(self) -> str:
        """str: Get the owner of the project."""
        return self.__owner

    @owner.setter
    def owner(self, owner: str):
        """str: Set the owner of the project."""
        self.__owner = owner
