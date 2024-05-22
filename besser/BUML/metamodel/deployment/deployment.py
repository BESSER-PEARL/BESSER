from besser.BUML.metamodel.structural import NamedElement, DomainModel, Model
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


class Hypervisor(Enum):
    vm_ware = "VMWare"
    hyper_v = "Hyper-V"
    xen_server = "XenServer"
    rhev = "RHEV"
    kvm = "KVM"


class Processor(Enum):
    x64 = "x64"
    x86 = "x84"
    arm = "ARM"


class IPRangeType(Enum):
    subnet = "Subnetwork"
    pod = "Pod"
    service = "Service"


class ServiceType(Enum):
    lb = "LoadBalancer"
    ingress = "Ingress"
    egress = "Egress"


class Provider(Enum):
    google = "Google"
    aws = "AWS"
    azure = "Azure"
    other = "Other"


class Protocol(Enum):
    http = "HTTP"
    https = "HTTPS"
    tcp = "TCP"
    udp = "UPD"
    all = "ALL"


class Resources:
    """
    Args:
    
    Attributes:

    """

    def __init__(self, cpu: int, memory: int):
        self.cpu: int = cpu
        self.memory: int = memory

    @property
    def cpu(self) -> int:
        """str: Get the cpu value."""
        return self.__cpu

    @cpu.setter
    def cpu(self, cpu: int):
        """str: Set the cpu value."""
        self.__cpu = cpu

    @property
    def memory(self) -> int:
        """str: Get the memory value."""
        return self.__memory

    @memory.setter
    def memory(self, memory: int):
        """str: Set the memory value."""
        self.__memory = memory

    def __repr__(self) -> str:
        return f'Resource({self.cpu}, {self.memory})'


class Application(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, image_repo: str, port: int, required_resources: Resources, domain_model: DomainModel):
        super().__init__(name)
        self.image_repo: str = image_repo
        self.port: str = port
        self.required_resources: Resources = required_resources
        self.domain_model: DomainModel = domain_model

    @property
    def image_repo(self) -> str:
        """str: Get the image repository."""
        return self.__image_repo

    @image_repo.setter
    def image_repo(self, image_repo: str):
        """str: Set the image repository."""
        self.__image_repo = image_repo

    @property
    def port(self) -> int:
        """str: Get the application port."""
        return self.__port

    @port.setter
    def port(self, port: int):
        """str: Set the application port."""
        self.__port = port

    @property
    def required_resources(self) -> Resources:
        """Resource: Get the required resources."""
        return self.__required_resources

    @required_resources.setter
    def required_resources(self, required_resources: Resources):
        """Resource: Set the required resources."""
        self.__required_resources = required_resources
    
    @property
    def domain_model(self) -> DomainModel:
        """str: Get the domain model."""
        return self.__domain_model

    @image_repo.setter
    def domain_model(self, domain_model: DomainModel):
        """str: Set the domain model."""
        self.__domain_model = domain_model

    def __repr__(self) -> str:
        return f'Application({self.name}, {self.required_resources}, {self.image_repo}, {self.domain_model})'


class Volume(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, mount_path: str, sub_path: str):
        super().__init__(name)
        self.mount_path: str = mount_path
        self.sub_path: str = sub_path

    @property
    def mount_path(self) -> str:
        """str: Get the mount path."""
        return self.__mount_path

    @mount_path.setter
    def mount_path(self, mount_path: str):
        """str: Set the mount path."""
        self.__mount_path = mount_path

    @property
    def sub_path(self) -> str:
        """str: Get the sub path."""
        return self.__sub_path

    @sub_path.setter
    def sub_path(self, sub_path: str):
        """str: Set the sub path."""
        self.__sub_path = sub_path

    def __repr__(self) -> str:
        return f'Volume({self.name}, {self.mount_path}, {self.sub_path})'


class Container(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, application: Application, resources_limit: Resources = None, volumes: set[Volume] = set()):
        super().__init__(name)
        self.application: Application = application
        self.resources_limit: Resources = resources_limit
        self.volumes: set[Volume] = volumes

    @property
    def application(self) -> Application:
        """Application: Get the application."""
        return self.__application

    @application.setter
    def application(self, application: Application):
        """Application: Set the application."""
        self.__application = application

    @property
    def resources_limit(self) -> Resources:
        """Resource: Get the resources limit."""
        return self.__resources_limit

    @resources_limit.setter
    def resources_limit(self, resources_limit: Resources):
        """Resource: Set the resources limit."""
        self.__resources_limit = resources_limit

    @property
    def volumes(self) -> set[Volume]:
        """set[Volume]: Get the set of volumes."""
        return self.__volumes

    @volumes.setter
    def volumes(self, volumes: set[Volume]):
        """set[Volume]: Set the set of volumes."""
        self.__volumes = volumes

    def __repr__(self) -> str:
        return f'Container({self.name}, {self.application}, {self.resources_limit}, {self.volumes})'
    

class Deployment(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, replicas: int, containers: set[Container]):
        super().__init__(name)
        self.replicas: int = replicas
        self.containers: set[Container] = containers

    @property
    def replicas(self) -> int:
        return self.__replicas

    @replicas.setter
    def replicas(self, replicas: int):
        self.__replicas = replicas

    @property
    def containers(self) -> set[Container]:
        return self.__containers

    @containers.setter
    def containers(self, containers: set[Container]):
        self.__containers = containers
    
    def __repr__(self) -> str:
        return f'Deployment({self.name}, {self.replicas}, {self.containers})'


class Service(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, port: int, target_port: int, type: ServiceType, protocol: Protocol, application: Application = None):
        super().__init__(name)
        self.port: int = port
        self.target_port: int = target_port
        self.type: ServiceType = type
        self.protocol: Protocol = protocol
        self.application: Application = application

    @property
    def port(self) -> int:
        return self.__port

    @port.setter
    def port(self, port: int):
        self.__port = port

    @property
    def target_port(self) -> int:
        return self.__target_port

    @target_port.setter
    def target_port(self, target_port: int):
        self.__target_port = target_port

    @property
    def type(self) -> ServiceType:
        return self.__type

    @type.setter
    def type(self, type: ServiceType):
        self.__type = type

    @property
    def protocol(self) -> Protocol:
        return self.__protocol

    @protocol.setter
    def protocol(self, protocol: Protocol):
        self.__protocol = protocol

    @property
    def application(self) -> Application:
        return self.__application

    @application.setter
    def application(self, application: Application):
        self.__application = application
    
    def __repr__(self) -> str:
        return f'Service({self.name}, {self.port}, {self.target_port}, {self.type}, {self.protocol}, {self.application})'
    
class IPRange(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, cidr_range: str, type: IPRangeType, public: bool):
        super().__init__(name)
        self.cidr_range: str = cidr_range
        self.type: IPRangeType = type
        self.public: bool = public

    @property
    def cidr_range(self) -> str:
        return self.__cidr_range

    @cidr_range.setter
    def cidr_range(self, cidr_range: str):
        self.__cidr_range = cidr_range

    @property
    def type(self) -> IPRangeType:
        return self.__type

    @type.setter
    def type(self, type: IPRangeType):
        self.__type = type

    @property
    def public(self) -> bool:
        return self.__public

    @public.setter
    def public(self, public: bool):
        self.__public = public

    def __repr__(self) -> str:
        return f'IPRange({self.name}, {self.cidr_range}, {self.type}, {self.public})' 


class SecurityGroup(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, rules: set[Service]):
        super().__init__(name)
        self.rules: set[Service] = rules
    
    @property
    def rules(self) -> set[Service]:
        return self.__rules

    @rules.setter
    def rules(self, rules: set[Service]):
        self.__rules = rules

    def __repr__(self) -> str:
        return f'SecurityGroup({self.name}, {self.rules})'


class Network(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, security_groups: set[SecurityGroup] = set()):
        super().__init__(name)
        self.security_groups: set[SecurityGroup] = security_groups

    @property
    def security_groups(self) -> set[SecurityGroup]:
        return self.__security_groups

    @security_groups.setter
    def security_groups(self, security_groups: set[SecurityGroup]):
        self.__security_groups = security_groups

    def __repr__(self) -> str:
        return f'Network({self.name})' 


class Subnetwork(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, ip_ranges: set[IPRange], network: Network):
        super().__init__(name)
        self.ip_ranges : set[IPRange] = ip_ranges
        self.network: Network = network

    @property
    def ip_ranges(self) -> set[IPRange]:
        return self.__ip_ranges

    @ip_ranges.setter
    def ip_ranges(self, ip_ranges: set[IPRange]):
        self.__ip_ranges = ip_ranges

    @property
    def network(self) -> Network:
        return self.__network

    @network.setter
    def network(self, network: Network):
        self.__network = network

    def __repr__(self) -> str:
        return f'Subnetwork({self.name}, {self.ip_ranges}, {self.network})' 


class Zone(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str):
        super().__init__(name)

    def __repr__(self) -> str:
        return f'Zone({self.name})'

class Region(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, zones: set[Zone]):
        super().__init__(name)
        self.zones : set[Zone] = zones

    @property
    def zones(self) -> set[Zone]:
        return self.__zones

    @zones.setter
    def zones(self, zones: set[Zone]):
        self.__zones = zones

    def __repr__(self) -> str:
        return f'Region({self.name}, {self.zones})'


class Node(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, public_ip: str, private_ip, os: str, resources: Resources, storage: int, processor: Processor):
        super().__init__(name)


class EdgeNode(Node):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, public_ip: str, private_ip, os: str, resources: Resources, storage: int, processor: Processor):
        super().__init__(name, public_ip, private_ip, os, resources, storage, processor)


class CloudNode(Node):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, public_ip: str, private_ip, os: str, resources: Resources, storage: int, processor: Processor):
        super().__init__(name, public_ip, private_ip, os, resources, storage, processor)


class Cluster(NamedElement):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, services: set[Service], deployments: set[Deployment], regions: set[Region], net_config: bool = True, nodes: set[Node] = set(),
                 networks: set[Network] = set(), subnets: set[Subnetwork] = set()):
        super().__init__(name)
        self.services: set[Service] = services
        self.deployments: set[Deployment] = deployments
        self.regions: set[Region] = regions
        self.net_config: bool = net_config
        self.nodes: set[Node] = nodes
        self.networks: set[Network] = networks
        self.subnets: set[Subnetwork] = subnets

    @property
    def services(self) -> set[Service]:
        return self.__services

    @services.setter
    def services(self, services: set[Service]):
        self.__services = services

    @property
    def deployments(self) -> set[Deployment]:
        return self.__deployments

    @deployments.setter
    def deployments(self, deployments: set[Deployment]):
        self.__deployments = deployments

    @property
    def regions(self) -> set[Region]:
        return self.__regions

    @regions.setter
    def regions(self, regions: set[Region]):
        self.__regions = regions

    @property
    def net_config(self) -> bool:
        return self.__net_config

    @net_config.setter
    def net_config(self, net_config: bool):
        self.__net_config = net_config

    @property
    def nodes(self) -> set[Node]:
        return self.__nodes

    @nodes.setter
    def nodes(self, nodes: set[Node]):
        self.__nodes = nodes

    @property
    def networks(self) -> set[Network]:
        return self.__networks

    @networks.setter
    def networks(self, networks: set[Network]):
        self.__networks = networks

    @property
    def subnets(self) -> set[Subnetwork]:
        return self.__subnets

    @subnets.setter
    def subnets(self, subnets: set[Subnetwork]):
        self.__subnets = subnets

    def __repr__(self) -> str:
        return f'Cluster({self.name}, {self.services}, {self.deployments}, {self.regions}, {self.net_config}, {self.nodes}, {self.networks}, {self.subnets})'


class PublicCluster(Cluster):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, services: set[Service], deployments: set[Deployment], regions: Region, num_nodes: int, provider: Provider, 
                 config_file: str, networks: set[Network] = set(), subnets: set[Subnetwork] = set(), net_config: bool = True):
        super().__init__(name, services, deployments, regions, networks, subnets, net_config)
        self.config_file: str = config_file
        self.num_nodes: int = num_nodes
        self.provider: Provider = provider

    @property
    def config_file(self) -> str:
        return self.__config_file

    @config_file.setter
    def config_file(self, config_file: str):
        self.__config_file = config_file

    @property
    def num_nodes(self) -> int:
        return self.__num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int):
        self.__num_nodes = num_nodes

    @property
    def provider(self) -> Provider:
        return self.__provider

    @provider.setter
    def provider(self, provider: Provider):
        self.__provider = provider

    def __repr__(self) -> str:
        return f'PublicCluster({self.name}, {self.services},{self.deployments}, {self.regions}, {self.config_file}, {self.provider}, {self.networks}, {self.subnets}, {self.net_config})'

class OnPremises(Cluster):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, services: set[Service], deployments: set[Deployment], regions: Region, nodes: set[Node], 
                 hypervisor: Hypervisor, networks: set[Network], subnets: set[Subnetwork]):
        super().__init__(name, services, deployments, regions, nodes, networks, subnets)
        self.hypervisor: str = hypervisor

    @property
    def hypervisor(self) -> Hypervisor:
        """str: Get the hypervisor."""
        return self.__hypervisor

    @hypervisor.setter
    def hypervisor(self, hypervisor: Hypervisor):
        """str: Set the hypervisor."""
        self.__hypervisor = hypervisor

    def __repr__(self) -> str:
        return f'Cluster({self.name}, {self.services},{self.deployments}, {self.regions}, {self.nodes}, {self.hypervisor}, {self.networks}, {self.subnets})'
    

class DeploymentModel(Model):
    """
    Args:
    
    Attributes:

    """

    def __init__(self, name: str, clusters: set[Cluster]):
        super().__init__(name)
        self.clusters: set[Cluster] = clusters

    @property
    def clusters(self) -> set[Cluster]:
        """str: Get the set of clusters."""
        return self.__clusters

    @clusters.setter
    def clusters(self, clusters: set[Cluster]):
        """str: Set the set of clusters."""
        self.__clusters = clusters

    def __repr__(self) -> str:
        return f'Cluster({self.name}, {self.clusters})'