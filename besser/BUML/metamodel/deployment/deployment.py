from enum import Enum
from besser.BUML.metamodel.structural import NamedElement, DomainModel, Model


class Hypervisor(Enum):
    '''Enumeration to list various types of hypervisors.

    Hypervisors are software, firmware, or hardware that creates and runs virtual machines (VMs). 
    This enumeration lists some of the most common hypervisors used in virtualization technology.

    Attributes:
        vm_ware (str): Represents the VMWare hypervisor.
        hyper_v (str): Represents the Hyper-V hypervisor developed by Microsoft.
        xen_server (str): Represents the XenServer hypervisor.
        rhev (str): Represents the Red Hat Enterprise Virtualization (RHEV) hypervisor.
        kvm (str): Represents the Kernel-based Virtual Machine (KVM) hypervisor.
    '''

    vm_ware = "VMWare"
    hyper_v = "Hyper-V"
    xen_server = "XenServer"
    rhev = "RHEV"
    kvm = "KVM"


class Processor(Enum):
    '''Enumeration to list various types of processors.

    Processors are the central units in computing that execute instructions.

    Attributes:
        x64 (str): Represents the x64 processor architecture.
        x86 (str): Represents the x86 processor architecture.
        arm (str): Represents the ARM processor architecture.
    '''

    x64 = "x64"
    x86 = "x84"
    arm = "ARM"


class IPRangeType(Enum):
    '''Enumeration to list different types of IP ranges.

    IP ranges are used to define segments of IP addresses in networking.

    Attributes:
        subnet (str): Represents a subnetwork IP range.
        pod (str): Represents a pod IP range.
        service (str): Represents a service IP range.
    '''

    subnet = "Subnetwork"
    pod = "Pod"
    service = "Service"


class ServiceType(Enum):
    '''Enumeration to list different types of services in networking.

    Services are abstractions that define a set of pods as a group and access policy for them.

    Attributes:
        lb (str): Represents a LoadBalancer service.
        ingress (str): Represents an Ingress service.
        egress (str): Represents an Egress service.
    '''

    lb = "LoadBalancer"
    ingress = "Ingress"
    egress = "Egress"


class Provider(Enum):
    '''Enumeration to list different cloud service providers.

    Cloud service providers offer various services such as computing power, storage, and networking.

    Attributes:
        google (str): Represents Google Cloud Platform.
        aws (str): Represents Amazon Web Services.
        azure (str): Represents Microsoft Azure.
        other (str): Represents any other cloud service provider.
    '''

    google = "Google"
    aws = "AWS"
    azure = "Azure"
    other = "Other"


class Protocol(Enum):
    '''Enumeration to list various network protocols.

    Network protocols define the rules for data communication over a network.

    Attributes:
        http (str): Represents the HTTP protocol.
        https (str): Represents the HTTPS protocol.
        tcp (str): Represents the TCP protocol.
        udp (str): Represents the UDP protocol.
        all (str): Represents all protocols.
    '''

    http = "HTTP"
    https = "HTTPS"
    tcp = "TCP"
    udp = "UPD"
    all = "ALL"


class Resources:
    """A class to represent the computational resources.

    Args:
        cpu (int): The number of CPU units.
        memory (int): The amount of memory in megabytes.

    Attributes:
        cpu (int): The number of CPU units.
        memory (int): The amount of memory in megabytes.
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
    """A class to represent an application.

    Args:
        name (str): The name of the application.
        image_repo (str): The image repository for the application.
        port (int): The port on which the application runs.
        required_resources (Resources): The resources required by the application.
        domain_model (DomainModel): The domain model of the application.

    Attributes:
        image_repo (str): The image repository for the application.
        port (int): The port on which the application runs.
        required_resources (Resources): The resources required by the application.
        domain_model (DomainModel): The domain model of the application.
    """

    def __init__(self, name: str, image_repo: str, port: int, required_resources: Resources,
            domain_model: DomainModel):
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

    @domain_model.setter
    def domain_model(self, domain_model: DomainModel):
        """str: Set the domain model."""
        self.__domain_model = domain_model

    def __repr__(self) -> str:
        return (
            f'Application({self.name}, {self.required_resources}, {self.image_repo}, '
            f'{self.domain_model})'
        )


class Volume(NamedElement):
    """A class to represent a volume.

    Args:
        name (str): The name of the volume.
        mount_path (str): The mount path of the volume.
        sub_path (str): The sub-path within the volume.

    Attributes:
        mount_path (str): The mount path of the volume.
        sub_path (str): The sub-path within the volume.
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
    """A class to represent a container.

    Args:
        name (str): The name of the container.
        application (Application): The application running in the container.
        resources_limit (Resources, optional): The resource limits for the container.
        volumes (set[Volume], optional): The set of volumes attached to the container.

    Attributes:
        application (Application): The application running in the container.
        resources_limit (Resources): The resource limits for the container.
        volumes (set[Volume]): The set of volumes attached to the container.
    """

    def __init__(self, name: str, application: Application, resources_limit: Resources = None,
            volumes: set[Volume] = None):
        super().__init__(name)
        self.application: Application = application
        self.resources_limit: Resources = resources_limit
        self.volumes: set[Volume] = volumes if volumes is not None else set()

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
    """ A class to represent a deployment.

    Args:
        name (str): The name of the deployment.
        replicas (int): The number of replicas.
        containers (set[Container]): The set of containers in the deployment.

    Attributes:
        replicas (int): The number of replicas.
        containers (set[Container]): The set of containers in the deployment.
    """

    def __init__(self, name: str, replicas: int, containers: set[Container]):
        super().__init__(name)
        self.replicas: int = replicas
        self.containers: set[Container] = containers

    @property
    def replicas(self) -> int:
        """int: Get the number of replicas."""
        return self.__replicas

    @replicas.setter
    def replicas(self, replicas: int):
        """int: Set the number of replicas."""
        self.__replicas = replicas

    @property
    def containers(self) -> set[Container]:
        """set[Container]: Get the set of containers."""
        return self.__containers

    @containers.setter
    def containers(self, containers: set[Container]):
        """set[Container]: Set the set of containers."""
        self.__containers = containers

    def __repr__(self) -> str:
        return f'Deployment({self.name}, {self.replicas}, {self.containers})'


class Service(NamedElement):
    """A class to represent a service.

    Args:
        name (str): The name of the service.
        port (int): The port on which the service is exposed.
        target_port (int): The port on which the application is running.
        type (ServiceType): The type of service (e.g., LoadBalancer, Ingress, Egress).
        protocol (Protocol): The protocol used by the service (e.g., HTTP, HTTPS, TCP, UDP).
        application (Application, optional): The application associated with the service.

    Attributes:
        port (int): The port on which the service is exposed.
        target_port (int): The port on which the application is running.
        type (ServiceType): The type of service.
        protocol (Protocol): The protocol used by the service.
        application (Application): The application associated with the service.
    """

    def __init__(self, name: str, port: int, target_port: int, type: ServiceType,
                 protocol: Protocol, application: Application = None):
        super().__init__(name)
        self.port: int = port
        self.target_port: int = target_port
        self.type: ServiceType = type
        self.protocol: Protocol = protocol
        self.application: Application = application

    @property
    def port(self) -> int:
        """int: Get the port on which the service is exposed."""
        return self.__port

    @port.setter
    def port(self, port: int):
        """int: Set the port on which the service is exposed."""
        self.__port = port

    @property
    def target_port(self) -> int:
        """int: Get the port on which the application is running."""
        return self.__target_port

    @target_port.setter
    def target_port(self, target_port: int):
        """int: Set the port on which the application is running."""
        self.__target_port = target_port

    @property
    def type(self) -> ServiceType:
        """ServiceType: Get the type of service."""
        return self.__type

    @type.setter
    def type(self, type: ServiceType):
        """ServiceType: Set the type of service."""
        self.__type = type

    @property
    def protocol(self) -> Protocol:
        """Protocol: Get the protocol used by the service."""
        return self.__protocol

    @protocol.setter
    def protocol(self, protocol: Protocol):
        """Protocol: Set the protocol used by the service."""
        self.__protocol = protocol

    @property
    def application(self) -> Application:
        """Application: Get the application associated with the service."""
        return self.__application

    @application.setter
    def application(self, application: Application):
        """Application: Set the application associated with the service."""
        self.__application = application

    def __repr__(self) -> str:
        return (
            f'Service({self.name}, {self.port}, {self.target_port}, {self.type}, '
            f'{self.protocol}, {self.application})'
        )

class IPRange(NamedElement):
    """A class to represent an IP range.

    Args:
        name (str): The name of the IP range.
        cidr_range (str): The CIDR range of the IP addresses.
        type (IPRangeType): The type of IP range (e.g., Subnetwork, Pod, Service).
        public (bool): Whether the IP range is public or private.

    Attributes:
        cidr_range (str): The CIDR range of the IP addresses.
        type (IPRangeType): The type of IP range.
        public (bool): Whether the IP range is public or private.
    """

    def __init__(self, name: str, cidr_range: str, type: IPRangeType, public: bool):
        super().__init__(name)
        self.cidr_range: str = cidr_range
        self.type: IPRangeType = type
        self.public: bool = public

    @property
    def cidr_range(self) -> str:
        """str: Get the CIDR range of the IP addresses."""
        return self.__cidr_range

    @cidr_range.setter
    def cidr_range(self, cidr_range: str):
        """str: Set the CIDR range of the IP addresses."""
        self.__cidr_range = cidr_range

    @property
    def type(self) -> IPRangeType:
        """IPRangeType: Get the type of IP range."""
        return self.__type

    @type.setter
    def type(self, type: IPRangeType):
        """IPRangeType: Set the type of IP range."""
        self.__type = type

    @property
    def public(self) -> bool:
        """bool: Get whether the IP range is public or private."""
        return self.__public

    @public.setter
    def public(self, public: bool):
        """bool: Set whether the IP range is public or private."""
        self.__public = public

    def __repr__(self) -> str:
        return f'IPRange({self.name}, {self.cidr_range}, {self.type}, {self.public})'


class SecurityGroup(NamedElement):
    """A class to represent a security group.

    Args:
        name (str): The name of the security group.
        rules (set[Service]): The set of services that define the security rules.

    Attributes:
        rules (set[Service]): The set of services that define the security rules.
    """

    def __init__(self, name: str, rules: set[Service]):
        super().__init__(name)
        self.rules: set[Service] = rules

    @property
    def rules(self) -> set[Service]:
        """set[Service]: Get the set of security rules."""
        return self.__rules

    @rules.setter
    def rules(self, rules: set[Service]):
        """set[Service]: Set the security rules."""
        self.__rules = rules

    def __repr__(self) -> str:
        return f'SecurityGroup({self.name}, {self.rules})'


class Network(NamedElement):
    """A class to represent a network.

    Args:
        name (str): The name of the network.
        security_groups (set[SecurityGroup], optional): The set of security groups associated
        with the network.

    Attributes:
        security_groups (set[SecurityGroup]): The set of security groups associated with the
        network.
    """

    def __init__(self, name: str, security_groups: set[SecurityGroup] = None):
        super().__init__(name)
        self.security_groups: set[SecurityGroup] = security_groups if security_groups is not None else set()

    @property
    def security_groups(self) -> set[SecurityGroup]:
        """set[SecurityGroup]: Get the set of security groups."""
        return self.__security_groups

    @security_groups.setter
    def security_groups(self, security_groups: set[SecurityGroup]):
        """set[SecurityGroup]: Set the security groups."""
        self.__security_groups = security_groups

    def __repr__(self) -> str:
        return f'Network({self.name})'


class Subnetwork(NamedElement):
    """A class to represent a subnetwork.

    Args:
        name (str): The name of the subnetwork.
        ip_ranges (set[IPRange]): The set of IP ranges within the subnetwork.
        network (Network): The network to which the subnetwork belongs.

    Attributes:
        ip_ranges (set[IPRange]): The set of IP ranges within the subnetwork.
        network (Network): The network to which the subnetwork belongs.
    """

    def __init__(self, name: str, ip_ranges: set[IPRange], network: Network):
        super().__init__(name)
        self.ip_ranges : set[IPRange] = ip_ranges
        self.network: Network = network

    @property
    def ip_ranges(self) -> set[IPRange]:
        """set[IPRange]: Get the set of IP ranges."""
        return self.__ip_ranges

    @ip_ranges.setter
    def ip_ranges(self, ip_ranges: set[IPRange]):
        """set[IPRange]: Set the set of IP ranges."""
        self.__ip_ranges = ip_ranges

    @property
    def network(self) -> Network:
        """Network: Get the network."""
        return self.__network

    @network.setter
    def network(self, network: Network):
        """Network: Set the network."""
        self.__network = network

    def __repr__(self) -> str:
        return f'Subnetwork({self.name}, {self.ip_ranges}, {self.network})'


class Zone(NamedElement):
    """A class to represent a zone.

    Args:
        name (str): The name of the zone.

    Attributes:
        name (str): The name of the zone.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def __repr__(self) -> str:
        return f'Zone({self.name})'

class Region(NamedElement):
    """A class to represent a region.

    Args:
        name (str): The name of the region.
        zones (set[Zone]): The set of zones within the region.

    Attributes:
        zones (set[Zone]): The set of zones within the region.
    """

    def __init__(self, name: str, zones: set[Zone]):
        super().__init__(name)
        self.zones : set[Zone] = zones

    @property
    def zones(self) -> set[Zone]:
        """set[Zone]: Get the set of zones."""
        return self.__zones

    @zones.setter
    def zones(self, zones: set[Zone]):
        """set[Zone]: Set the set of zones."""
        self.__zones = zones

    def __repr__(self) -> str:
        return f'Region({self.name}, {self.zones})'


class Node(NamedElement):
    """A class to represent a node.

    Args:
        name (str): The name of the node.
        public_ip (str): The public IP address of the node.
        private_ip (str): The private IP address of the node.
        os (str): The operating system running on the node.
        resources (Resources): The computational resources of the node.
        storage (int): The storage capacity of the node in gigabytes.
        processor (Processor): The processor type of the node.

    Attributes:
        public_ip (str): The public IP address of the node.
        private_ip (str): The private IP address of the node.
        os (str): The operating system running on the node.
        resources (Resources): The computational resources of the node.
        storage (int): The storage capacity of the node in gigabytes.
        processor (Processor): The processor type of the node.
    """

    def __init__(self, name: str, public_ip: str, private_ip, os: str, resources: Resources,
                 storage: int, processor: Processor):
        super().__init__(name)
        self.public_ip: str = public_ip
        self.private_ip: str = private_ip
        self.os: str = os
        self.resources: Resources = resources
        self.storage: int = storage
        self.processor: Processor = processor

    @property
    def public_ip(self) -> str:
        """str: Get the public IP address of the node."""
        return self.__public_ip

    @public_ip.setter
    def public_ip(self, public_ip: str):
        """str: Set the public IP address of the node."""
        self.__public_ip = public_ip

    @property
    def private_ip(self) -> str:
        """str: Get the private IP address of the node."""
        return self.__private_ip

    @private_ip.setter
    def private_ip(self, private_ip: str):
        """str: Set the private IP address of the node."""
        self.__private_ip = private_ip

    @property
    def os(self) -> str:
        """str: Get the operating system running on the node."""
        return self.__os

    @os.setter
    def os(self, os: str):
        """str: Set the operating system running on the node."""
        self.__os = os

    @property
    def resources(self) -> Resources:
        """Resources: Get the computational resources of the node."""
        return self.__resources

    @resources.setter
    def resources(self, resources: Resources):
        """Resources: Set the computational resources of the node."""
        self.__resources = resources

    @property
    def storage(self) -> int:
        """int: Get the storage capacity of the node in gigabytes."""
        return self.__storage

    @storage.setter
    def storage(self, storage: int):
        """int: Set the storage capacity of the node in gigabytes."""
        self.__storage = storage

    @property
    def processor(self) -> Processor:
        """Processor: Get the processor type of the node."""
        return self.__processor

    @processor.setter
    def processor(self, processor: Processor):
        """Processor: Set the processor type of the node."""
        self.__processor = processor


class EdgeNode(Node):
    """A class to represent an edge node.
    """

    def __init__(self, name: str, public_ip: str, private_ip, os: str, resources: Resources,
                 storage: int, processor: Processor):
        super().__init__(name, public_ip, private_ip, os, resources, storage, processor)


class CloudNode(Node):
    """A class to represent a cloud node.
    """

    def __init__(self, name: str, public_ip: str, private_ip, os: str, resources: Resources,
                 storage: int, processor: Processor):
        super().__init__(name, public_ip, private_ip, os, resources, storage, processor)


class Cluster(NamedElement):
    """ A class to represent a cluster.

    Args:
        name (str): The name of the cluster.
        services (set[Service]): The set of services associated with the cluster.
        deployments (set[Deployment]): The set of deployments in the cluster.
        regions (set[Region]): The set of regions where the cluster is deployed.
        net_config (bool, optional): Indicates if network configuration is enabled. Defaults to True.
        nodes (set[Node], optional): The set of nodes in the cluster. Defaults to an empty set.
        networks (set[Network], optional): The set of networks in the cluster. Defaults to an empty set.
        subnets (set[Subnetwork], optional): The set of subnetworks in the cluster. Defaults to an empty set.

    Attributes:
        services (set[Service]): The set of services associated with the cluster.
        deployments (set[Deployment]): The set of deployments in the cluster.
        regions (set[Region]): The set of regions where the cluster is deployed.
        net_config (bool): Indicates if network configuration is enabled.
        nodes (set[Node]): The set of nodes in the cluster.
        networks (set[Network]): The set of networks in the cluster.
        subnets (set[Subnetwork]): The set of subnetworks in the cluster.
    """

    def __init__(self, name: str, services: set[Service], deployments: set[Deployment],
                 regions: set[Region], net_config: bool = True, nodes: set[Node] = None,
                 networks: set[Network] = None, subnets: set[Subnetwork] = None):
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
        """set[Service]: Get the set of services."""
        return self.__services

    @services.setter
    def services(self, services: set[Service]):
        """set[Service]: Set the set of services."""
        self.__services = services

    @property
    def deployments(self) -> set[Deployment]:
        """set[Deployment]: Get the set of deployments."""
        return self.__deployments

    @deployments.setter
    def deployments(self, deployments: set[Deployment]):
        """set[Deployment]: Set the set of deployments."""
        self.__deployments = deployments

    @property
    def regions(self) -> set[Region]:
        """set[Region]: Get the set of regions."""
        return self.__regions

    @regions.setter
    def regions(self, regions: set[Region]):
        """set[Region]: Set the set of regions."""
        self.__regions = regions

    @property
    def net_config(self) -> bool:
        """bool: Get the network configuration status."""
        return self.__net_config

    @net_config.setter
    def net_config(self, net_config: bool):
        """bool: Set the network configuration status."""
        self.__net_config = net_config

    @property
    def nodes(self) -> set[Node]:
        """set[Node]: Get the set of nodes."""
        return self.__nodes

    @nodes.setter
    def nodes(self, nodes: set[Node]):
        """set[Node]: Set the set of nodes."""
        self.__nodes = nodes

    @property
    def networks(self) -> set[Network]:
        """set[Network]: Get the set of networks."""
        return self.__networks

    @networks.setter
    def networks(self, networks: set[Network]):
        """set[Network]: Set the set of networks."""
        self.__networks = networks

    @property
    def subnets(self) -> set[Subnetwork]:
        """set[Subnetwork]: Get the set of subnetworks."""
        return self.__subnets

    @subnets.setter
    def subnets(self, subnets: set[Subnetwork]):
        """set[Subnetwork]: Set the set of subnetworks."""
        self.__subnets = subnets

    def __repr__(self) -> str:
        return (
            f'Cluster({self.name}, {self.services}, {self.deployments}, {self.regions}, '
            f'{self.net_config}, {self.nodes}, {self.networks}, {self.subnets})'
        )


class PublicCluster(Cluster):
    """A class to represent a public cluster.

    Args:
        name (str): The name of the public cluster.
        services (set[Service]): The set of services associated with the public cluster.
        deployments (set[Deployment]): The set of deployments in the public cluster.
        regions (set[Region]): The set of regions where the public cluster is deployed.
        num_nodes (int): The number of nodes in the public cluster.
        provider (Provider): The provider of the public cluster.
        config_file (str): The configuration file for the public cluster.
        networks (set[Network], optional): The set of networks in the public cluster. Defaults to an empty set.
        subnets (set[Subnetwork], optional): The set of subnetworks in the public cluster. Defaults to an empty set.
        net_config (bool, optional): Indicates if network configuration is enabled. Defaults to True.

    Attributes:
        config_file (str): The configuration file for the public cluster.
        num_nodes (int): The number of nodes in the public cluster.
        provider (Provider): The provider of the public cluster.
    """

    def __init__(self, name: str, services: set[Service], deployments: set[Deployment],
                 regions: Region, num_nodes: int, provider: Provider, config_file: str,
                 networks: set[Network] = None, subnets: set[Subnetwork] = None,
                 net_config: bool = True):
        super().__init__(name, services, deployments, regions, net_config, networks, subnets)
        self.config_file: str = config_file
        self.num_nodes: int = num_nodes
        self.provider: Provider = provider

    @property
    def config_file(self) -> str:
        """str: Get the configuration file."""
        return self.__config_file

    @config_file.setter
    def config_file(self, config_file: str):
        """str: Set the configuration file."""
        self.__config_file = config_file

    @property
    def num_nodes(self) -> int:
        """int: Get the number of nodes."""
        return self.__num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int):
        """int: Set the number of nodes."""
        self.__num_nodes = num_nodes

    @property
    def provider(self) -> Provider:
        """Provider: Get the provider."""
        return self.__provider

    @provider.setter
    def provider(self, provider: Provider):
        """Provider: Set the provider."""
        self.__provider = provider

    def __repr__(self) -> str:
        return (
            f'PublicCluster({self.name}, {self.services},{self.deployments}, '
            f'{self.regions}, {self.config_file}, {self.provider}, '
            f'{self.networks}, {self.subnets}, {self.net_config})'
        )

class OnPremises(Cluster):
    """A class to represent an on-premises cluster.

    Args:
        name (str): The name of the on-premises cluster.
        services (set[Service]): The set of services associated with the on-premises cluster.
        deployments (set[Deployment]): The set of deployments in the on-premises cluster.
        regions (set[Region]): The set of regions where the on-premises cluster is deployed.
        nodes (set[Node]): The set of nodes in the on-premises cluster.
        hypervisor (Hypervisor): The hypervisor used in the on-premises cluster.
        networks (set[Network]): The set of networks in the on-premises cluster.
        subnets (set[Subnetwork]): The set of subnetworks in the on-premises cluster.

    Attributes:
        hypervisor (Hypervisor): The hypervisor used in the on-premises cluster.
    """

    def __init__(self, name: str, services: set[Service], deployments: set[Deployment],
                 regions: Region, nodes: set[Node], hypervisor: Hypervisor,
                 networks: set[Network], subnets: set[Subnetwork]):
        super().__init__(name, services, deployments, regions, nodes, networks, subnets)
        self.hypervisor: str = hypervisor

    @property
    def hypervisor(self) -> Hypervisor:
        """Hypervisor: Get the hypervisor."""
        return self.__hypervisor

    @hypervisor.setter
    def hypervisor(self, hypervisor: Hypervisor):
        """Hypervisor: Set the hypervisor."""
        self.__hypervisor = hypervisor

    def __repr__(self) -> str:
        return (
            f'Cluster({self.name}, {self.services},{self.deployments}, {self.regions}, '
            f'{self.nodes}, {self.hypervisor}, {self.networks}, {self.subnets})'
        )


class DeploymentModel(Model):
    """A class to represent a deployment model.

    Args:
        name (str): The name of the deployment model.
        clusters (set[Cluster]): The set of clusters in the deployment model.

    Attributes:
        clusters (set[Cluster]): The set of clusters in the deployment model.
    """

    def __init__(self, name: str, clusters: set[Cluster]):
        super().__init__(name)
        self.clusters: set[Cluster] = clusters

    @property
    def clusters(self) -> set[Cluster]:
        """set[Cluster]: Get the set of clusters."""
        return self.__clusters

    @clusters.setter
    def clusters(self, clusters: set[Cluster]):
        """set[Cluster]: Set the set of clusters."""
        self.__clusters = clusters

    def __repr__(self) -> str:
        return f'Cluster({self.name}, {self.clusters})'
