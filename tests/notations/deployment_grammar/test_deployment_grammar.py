from antlr4 import *
from besser.BUML.notations.deployment.deploymentLexer import deploymentLexer
from besser.BUML.notations.deployment.deploymentParser import deploymentParser
from besser.BUML.notations.deployment.deploymentListener import deploymentListener
from besser.BUML.notations.deployment import buml_deployment_model
import shutil
from besser.BUML.metamodel.deployment import Provider, Protocol, ServiceType

model = buml_deployment_model(deployment_textfile="deployment.txt")
shutil.rmtree("buml")

def test_simple_Deployment_Grammar():
    od = "deployment.txt"
    input_stream = FileStream(od)
    lexer = deploymentLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = deploymentParser(stream)
    tree = parser.architecture()
    listener = deploymentListener()
    walker = ParseTreeWalker()
    walker.walk(listener, tree)
    assert parser.getNumberOfSyntaxErrors() == 0

def test_public_clusters():
    assert len(model.clusters) == 2
    for cluster in model.clusters:
        assert len(cluster.deployments) == 1
        assert cluster.num_nodes == 3
        assert cluster.net_config == True
        assert (cluster.name == "cluster1" or cluster.name == "cluster2")
        assert (cluster.provider == Provider.google or cluster.provider == Provider.aws)

def test_deployments():
    cluster = next(iter(model.clusters))
    assert len(cluster.deployments) == 1
    deploy = next(iter(cluster.deployments))
    assert deploy.name == "deployment1"
    assert deploy.replicas == 2
    assert len(deploy.containers) == 1

def test_containers():
    cluster = next(iter(model.clusters))
    deploy = next(iter(cluster.deployments))
    container = next(iter(deploy.containers))
    assert container.name == "container1"
    assert container.application.name == "app1"
    assert container.resources_limit.cpu == 500
    assert container.resources_limit.memory == 512

def test_applications():
    cluster = next(iter(model.clusters))
    deploy = next(iter(cluster.deployments))
    container = next(iter(deploy.containers))
    app = container.application
    assert app.name == "app1"
    assert app.image_repo == "image/latest"
    assert app.port == 8000
    assert app.required_resources.cpu == 10
    assert app.required_resources.memory == 100

def test_services():
    cluster = next(iter(model.clusters))
    service = next(iter(cluster.services))
    assert service.name == "service1"
    assert service.port == 80
    assert service.target_port == 8000
    assert service.protocol == Protocol.http
    assert service.type == ServiceType.lb