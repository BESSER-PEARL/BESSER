import pytest
from besser.BUML.metamodel.deployment import *

def test_node_initialization():
    node = Node(name="MyNode", public_ip="192.168.1.1", private_ip="1.2.0.1", os="Linux", storage=100, processor=Processor.x64, resources=Resources(cpu=1000, memory=2000))
    assert node.name == "MyNode"
    assert node.public_ip == "192.168.1.1"
    assert node.resources.cpu == 1000
    assert node.storage == 100
    assert node.private_ip == "1.2.0.1"
    assert node.os == "Linux"
    assert node.processor == Processor.x64
    assert node.resources.memory == 2000

def test_container_initialization():
    app1 = Application(name="App1", image_repo="my_image:latest", port=8000, required_resources=Resources(cpu=100, memory=500), domain_model=None)
    container = Container(name="MyContainer", application=app1, resources_limit=Resources(cpu=100, memory=200))
    assert container.name == "MyContainer"
    assert container.application.name == "App1"
    assert container.application.required_resources.memory == 500
    assert container.resources_limit.cpu == 100
    assert container.resources_limit.memory == 200

def test_resources():
    resources = Resources(cpu=500, memory=1024)
    assert resources.cpu == 500
    assert resources.memory == 1024
    resources.cpu = 1000
    resources.memory = 2048
    assert resources.cpu == 1000
    assert resources.memory == 2048

def test_application_initialization():
    domain_model = DomainModel(name="TestModel")
    app = Application(name="TestApp", image_repo="repo/image:tag", port=8080, required_resources=Resources(cpu=300, memory=1024), domain_model=domain_model)
    assert app.name == "TestApp"
    assert app.image_repo == "repo/image:tag"
    assert app.port == 8080
    assert app.required_resources.cpu == 300
    assert app.required_resources.memory == 1024


def test_volume_initialization():
    volume = Volume(name="TestVolume", mount_path="/mnt/data", sub_path="subdir")
    assert volume.name == "TestVolume"
    assert volume.mount_path == "/mnt/data"
    assert volume.sub_path == "subdir"

def test_service_initialization():
    app = Application(name="TestApp", image_repo="repo/image:tag", port=8080, required_resources=Resources(cpu=300, memory=1024), domain_model=None)
    service = Service(name="TestService", port=80, target_port=8080, type=ServiceType.lb, protocol=Protocol.http, application=app)
    assert service.name == "TestService"
    assert service.port == 80
    assert service.target_port == 8080
    assert service.type == ServiceType.lb
    assert service.protocol == Protocol.http
    assert service.application == app


def test_container_repr():
    app1 = Application(name="App1", image_repo="my_image:latest", port=8000, required_resources=Resources(cpu=100, memory=500), domain_model=None)
    container = Container(name="MyContainer", application=app1, resources_limit=Resources(cpu=100, memory=200))
    expected_repr = (
        "Container(MyContainer, Application(App1, Resource(100, 500), my_image:latest, None), "
        "Resource(100, 200), set())"
    )
    assert repr(container) == expected_repr
