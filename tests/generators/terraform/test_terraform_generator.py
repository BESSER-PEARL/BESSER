import os
import tempfile
import pytest
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, StringType
)
from besser.BUML.metamodel.deployment import (
    DeploymentModel, PublicCluster, Deployment, Service, Container,
    Application, Resources, Region, Zone, ServiceType, Protocol, Provider
)
from besser.generators.terraform import TerraformGenerator


@pytest.fixture
def deployment_model(tmpdir):
    """Create a minimal deployment model with a GCP public cluster."""
    # Create a simple domain model
    name_prop = Property(name="name", type=StringType)
    app_class = Class(name="App", attributes={name_prop})
    domain_model = DomainModel(name="TestDomain", types={app_class}, associations=set())

    # Create application
    resources = Resources(cpu=1, memory=512)
    app = Application(
        name="test_app",
        image_repo="gcr.io/test/app",
        port=8080,
        required_resources=resources,
        domain_model=domain_model,
    )

    # Create container and deployment
    container = Container(name="app_container", application=app)
    deployment = Deployment(name="app_deployment", replicas=2, containers={container})

    # Create service
    service = Service(
        name="app_service",
        port=80,
        target_port=8080,
        type=ServiceType.lb,
        protocol=Protocol.http,
        application=app,
    )

    # Create region and zone
    zone = Zone(name="us_central1_a")
    region = Region(name="us_central1", zones={zone})

    # Create a config file for the cluster
    config_file = str(tmpdir.join("cluster.conf"))
    with open(config_file, "w") as f:
        f.write("project_id = test_project\n")
        f.write("cluster_name = test_cluster\n")

    # Create public cluster (regions must be a set for templates to iterate)
    cluster = PublicCluster(
        name="test_cluster",
        services={service},
        deployments={deployment},
        regions={region},
        num_nodes=3,
        provider=Provider.google,
        config_file=config_file,
    )

    model = DeploymentModel(name="TestDeployment", clusters={cluster})
    return model


def test_terraform_generator_instantiation(deployment_model, tmpdir):
    """Test that the TerraformGenerator can be instantiated."""
    output_dir = tmpdir.mkdir("output")
    generator = TerraformGenerator(deployment_model=deployment_model, output_dir=str(output_dir))
    assert generator is not None
    assert generator.deployment_model is deployment_model


def test_terraform_generator_generate(deployment_model, tmpdir):
    """Test that generate() runs without errors and produces output."""
    output_dir = tmpdir.mkdir("output")
    generator = TerraformGenerator(
        deployment_model=deployment_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    # Verify output files were created
    generated_files = []
    for root, dirs, files in os.walk(str(output_dir)):
        for f in files:
            generated_files.append(os.path.join(root, f))

    assert len(generated_files) > 0, "TerraformGenerator should produce output files"


def test_terraform_generator_gcp_files(deployment_model, tmpdir):
    """Test that GCP-specific Terraform files are generated."""
    output_dir = tmpdir.mkdir("output")
    generator = TerraformGenerator(
        deployment_model=deployment_model,
        output_dir=str(output_dir),
    )
    generator.generate()

    # Look for .tf files in the output directory tree
    tf_files = []
    for root, dirs, files in os.walk(str(output_dir)):
        for f in files:
            if f.endswith(".tf"):
                tf_files.append(f)

    assert len(tf_files) > 0, "GCP cluster should generate .tf files"


def test_terraform_generator_template_map():
    """Test that get_template_to_file_map returns correct mappings."""
    gcp_map = TerraformGenerator.get_template_to_file_map("Google")
    assert gcp_map is not None
    assert "gcp/cluster.tf.j2" in gcp_map
    assert gcp_map["gcp/cluster.tf.j2"] == "cluster.tf"

    aws_map = TerraformGenerator.get_template_to_file_map("AWS")
    assert aws_map is not None
    assert "aws/eks.tf.j2" in aws_map

    unknown_map = TerraformGenerator.get_template_to_file_map("Unknown")
    assert unknown_map is None
