from besser.BUML.metamodel.structural import NamedElement, Property, Type, Association
from enum import Enum

class Component(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str):
        super().__init__(name)

    def __repr__(self) -> str:
        return f'Component({self.name})'

class Microservice(Component):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str):
        super().__init__(name)

    def __repr__(self) -> str:
        return f'Microservice({self.name})'
    
class Application(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, components: set[Component], cpuRequired: int, memoryRequired: int, image_repo: str):
        super().__init__(name)
        self.components: set[Component] = components
        self.cpuRequired: int = cpuRequired
        self.memoryRequired: int = memoryRequired
        self.image_repo: str = image_repo

    @property
    def components(self) -> set[Component]:
        """set[Component]: Get the components."""
        return self.__components

    @components.setter
    def components(self, components: set[Component]):
        """set[Component]: Set the components."""
        self.__components = components

    @property
    def cpuRequired(self) -> int:
        """int: Get the cpu required."""
        return self.__cpuRequired

    @cpuRequired.setter
    def cpuRequired(self, cpuRequired: int):
        """int: Set the cpu required."""
        self.__cpuRequired = cpuRequired

    @property
    def memoryRequired(self) -> int:
        """int: Get the memory required."""
        return self.__memoryRequired

    @memoryRequired.setter
    def memoryRequired(self, memoryRequired: int):
        """int: Set the memory required."""
        self.__memoryRequired = memoryRequired

    @property
    def image_repo(self) -> int:
        """str: Get the image repository."""
        return self.__image_repo

    @image_repo.setter
    def image_repo(self, image_repo: str):
        """str: Set the image repository."""
        self.__image_repo = image_repo

    def __repr__(self) -> str:
        return f'Application({self.name}, {self.components}, {self.cpuRequired}, {self.memoryRequired}, {self.imageRepo})'
    
class Container(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, application: Application, cpuLimit: int, memory_limit: int):
        super().__init__(name)
        self.application: Application = application
        self.cpuLimit: int = cpuLimit
        self.memory_limit: int = memory_limit

    @property
    def application(self) -> Application:
        """Application: Get the application."""
        return self.__application

    @application.setter
    def application(self, application: Application):
        """Application: Set the application."""
        self.__application = application

    @property
    def cpuLimit(self) -> int:
        """int: Get the cpu limit."""
        return self.__cpuLimit

    @cpuLimit.setter
    def cpuLimit(self, cpuLimit: int):
        """int: Set the cpu limit."""
        self.__cpuLimit = cpuLimit

    @property
    def memory_limit(self) -> int:
        """int: Get the memory limit."""
        return self.__memory_limit

    @memory_limit.setter
    def memory_limit(self, memory_limit: int):
        """int: Set the memory limit."""
        self.__memory_limit = memory_limit

    def __repr__(self) -> str:
        return f'Container({self.name}, {self.application}, {self.cpuLimit}, {self.memory_limit})'

class Provider(Enum):
    gcp = "GCP"
    azure = "Azure"
    aws = "AWS"

class Cluster(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str):
        super().__init__(name)

    def __repr__(self) -> str:
        return f'Cluster({self.name})'

class PublicCluster(Cluster):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, provider: Provider):
        super().__init__(name)
        self.provider: Provider = provider

    @property
    def provider(self) -> Provider:
        """Provider: Get the provider."""
        return self.__provider

    @provider.setter
    def provider(self, provider: Provider):
        """Provider: Set the provider."""
        self.__provider = provider

    def __repr__(self) -> str:
        return f'Cluster({self.name}, {self.provider})'

class PrivateCluster(Cluster):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, hypervisor: str):
        super().__init__(name)
        self.hypervisor: str = hypervisor

    @property
    def hypervisor(self) -> str:
        """str: Get the hypervisor."""
        return self.__hypervisor

    @hypervisor.setter
    def hypervisor(self, hypervisor: str):
        """str: Set the hypervisor."""
        self.__hypervisor = hypervisor

    def __repr__(self) -> str:
        return f'Cluster({self.name})'