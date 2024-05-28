import pytest
from besser.BUML.metamodel.deployment import *

def test_node_initialization():
    node = Node(name="MyNode", public_ip="192.168.1.1", private_ip="1.2.0.1", os="Linux", storage=100, processor=Processor.x64, resources=Resources(cpu=1000, memory=2000))
    assert node.name == "MyNode"
    assert node.public_ip == "192.168.1.1"
    assert node.resources.cpu == 1000
    assert node.storage == 100
    assert node.private_ip == "1.2.0.1"

def test_container_initialization():
    app1 = Application(name="App1", image_repo="my_image:latest", port=8000, required_resources=Resources(cpu=100, memory=500), domain_model=None)
    container = Container(name="MyContainer", application=app1, resources_limit=Resources(cpu=100, memory=200))
    assert container.name == "MyContainer"
    assert container.application.name == "App1"
    assert container.application.required_resources.memory == 500
    assert container.resources_limit.cpu == 100