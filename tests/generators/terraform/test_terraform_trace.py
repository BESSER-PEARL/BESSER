"""TerraformGenerator records which lines of each .tf file come from which deployment element."""

import os

from besser.BUML.metamodel.deployment import (
    Application, Container, Deployment, DeploymentModel, IPRange, IPRangeType, Network, Protocol, Provider,
    PublicCluster, Region, Resources, Service, ServiceType, Subnetwork, Zone,
)
from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.generators.terraform import TerraformGenerator


def _model(conf_dir) -> DeploymentModel:
    """A model whose elements all have ids (the same objects are reused to render it untraced, as
    a set of services has no stable order between two builds)."""
    def ident(element, value):
        element.id = value
        return element

    domain = DomainModel("D", types={Class("App", attributes={Property("name", StringType)})})
    app = Application("test_app", "gcr.io/x/app", 8080, Resources(1, 512), domain)
    container = ident(Container("c1", app, resources_limit=Resources(2, 1024)), "container-1")
    deployment = ident(Deployment("dep1", 2, {container}), "deployment-1")
    service = ident(Service("svc1", 80, 8080, ServiceType.lb, Protocol.http, app), "service-1")
    ingress = ident(Service("svc2", 81, 8081, ServiceType.ingress, Protocol.tcp, app), "ingress-1")
    region = Region("us_central1", {Zone("a")})
    clusters = set()
    for provider, net_config, cluster_id in ((Provider.google, False, "cluster-gcp"), (Provider.aws, True, "cluster-aws")):
        conf = os.path.join(str(conf_dir), f"{provider.name}.conf")
        with open(conf, "w", encoding="utf-8") as f:
            f.write("project_id = p\n")
        cluster = ident(PublicCluster(f"cl_{provider.name}", {service, ingress}, {deployment}, {region}, 3,
                                      provider, conf, net_config=net_config), cluster_id)
        if not net_config:
            network = ident(Network("net1"), "network-1")
            cluster.networks = {network}
            cluster.subnets = {ident(Subnetwork("sub1", {
                IPRange("r1", "10.0.0.0/16", IPRangeType.subnet, True),
                IPRange("r2", "10.1.0.0/16", IPRangeType.pod, False)}, network), "subnet-1")}
        clusters.add(cluster)
    return DeploymentModel("DM", clusters)


def _generate(model, out):
    generator = TerraformGenerator(model, output_dir=str(out))
    generator.generate()
    files = {}
    for root, _, names in os.walk(str(out)):
        for name in names:
            path = os.path.join(root, name)
            with open(path, encoding="utf-8", newline="") as f:
                files[os.path.relpath(path, str(out)).replace(os.sep, "/")] = f.read()
    return generator, files


def _lines(region, text):
    return text.split("\n")[region.start_line - 1:region.end_line]


def test_every_file_names_its_cluster_as_source_with_the_path_written(tmp_path):
    (tmp_path / "conf").mkdir()
    generator, files = _generate(_model(tmp_path / "conf"), tmp_path / "out")
    assert set(generator.traced_files) == set(files)
    for path, trace in generator.traced_files.items():
        expected = "cluster-gcp" if path.startswith("google_") else "cluster-aws"
        assert [s.element.element_id for s in trace.sources] == [expected]
        assert trace.sources[0].element.element_type == "PublicCluster"


def test_gcp_regions_contain_the_resources_of_their_element(tmp_path):
    (tmp_path / "conf").mkdir()
    generator, files = _generate(_model(tmp_path / "conf"), tmp_path / "out")
    cluster_file = next(p for p in files if p.startswith("google_") and p.endswith("cluster.tf"))
    app_file = cluster_file.replace("cluster.tf", "app.tf")
    by_id = {r.element_id: r for r in generator.traced_files[cluster_file].regions}
    network, subnet = by_id["network-1"], by_id["subnet-1"]
    assert _lines(network, files[cluster_file])[0].startswith('resource "google_compute_network"')
    assert _lines(subnet, files[cluster_file])[0].startswith('resource "google_compute_subnetwork"')
    assert 'name = "sub1"' in "\n".join(_lines(subnet, files[cluster_file]))

    app_regions = {r.element_id: r for r in generator.traced_files[app_file].regions}
    deployment, container = app_regions["deployment-1"], app_regions["container-1"]
    assert _lines(deployment, files[app_file])[0].startswith('resource "kubernetes_deployment_v1"')
    assert 'name  = "c1"' in "\n".join(_lines(container, files[app_file]))
    assert deployment.start_line < container.start_line <= container.end_line < deployment.end_line
    service = app_regions["service-1"]
    assert _lines(service, files[app_file])[0].startswith('resource "kubernetes_service_v1"')
    assert 'name = "svc1"' in "\n".join(_lines(service, files[app_file]))


def test_aws_regions_cover_deployment_service_and_ingress(tmp_path):
    (tmp_path / "conf").mkdir()
    generator, files = _generate(_model(tmp_path / "conf"), tmp_path / "out")
    eks = next(p for p in files if p.startswith("aws_") and p.endswith("eks.tf"))
    regions = {(r.element_id, r.rule.split("#")[1]): r for r in generator.traced_files[eks].regions}
    assert _lines(regions[("deployment-1", "deployment")], files[eks])[0].startswith('resource "kubernetes_deployment_v1"')
    assert _lines(regions[("service-1", "service")], files[eks])[0].startswith('resource "kubernetes_service_v1"')
    assert 'name = "svc2"' in "\n".join(_lines(regions[("ingress-1", "service")], files[eks]))
    assert _lines(regions[("ingress-1", "ingress")], files[eks])[0].startswith('resource "kubernetes_ingress_v1"')
    assert ("container-1", "container") not in regions


def test_elements_without_ids_render_untraced_and_the_output_is_unchanged(tmp_path):
    (tmp_path / "conf").mkdir()
    model = _model(tmp_path / "conf")
    traced, traced_files = _generate(model, tmp_path / "traced")
    for cluster in model.clusters:
        for element in (cluster, *cluster.services, *cluster.deployments, *(c for d in cluster.deployments for c in d.containers),
                        *(cluster.networks or ()), *(cluster.subnets or ())):
            element.id = None
    untraced, untraced_files = _generate(model, tmp_path / "untraced")
    assert traced_files == untraced_files
    assert not any("\x02" in text or "\x03" in text for text in traced_files.values())
    assert all(trace.regions == () and trace.sources == () for trace in untraced.traced_files.values())
    assert any(trace.regions for trace in traced.traced_files.values())
