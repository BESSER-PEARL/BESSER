from besser.BUML.metamodel.lla import *
from besser.generators.deployement import DeploymentGenerator

# network
#network1: Network = Network(name="example-network")

# subnetwork
#subnetwork1: Subnetwork = Subnetwork(name="example-subnetwork", network=network1, ip_ranges={IPRange(name="subnet-range", cidr_range="10.0.0.0/16", type=IPRangeType.subnet),
#                                                                                             IPRange(name="services-range", cidr_range="192.168.0.0/24", type=IPRangeType.service),
#                                                                                             IPRange(name="pod-ranges", cidr_range="192.168.1.0/24", type=IPRangeType.pod)})
# application
app: Application = Application(name="hello-app", image_repo="fitash19/myapp", port=8000, required_resources=Resources(cpu=250, memory=250), domain_model=None)

# services
service1: Service = Service(name="example-hello-app-loadbalancer", port=80, target_port=8000, protocol="TCP",type=ServiceType.lb, application=app)

# containers
container1: Container = Container(name="example", application=app, resources_limit=Resources(cpu=0.5, memory=512))

# deployment
deployment1: Deployment = Deployment(name="terraform-example", replicas=1, containers={container1})

# regions
region1: Region = Region(name="us-east1", zones={})

# provider
#gcp: GCP = GCP(project_id="neon-nexus-422908-u9", deletion_protection=False)



cluster: PublicCluster = PublicCluster(name="example-autopilot-cluster",
                                       services={service1},
                                       deployments={deployment1},
                                       regions={region1},
                                       num_nodes=1,
                                       provider=Provider.google,
                                       config_file="buml/config.conf",
                                       net_config=True)

# generate the deployment
generator = DeploymentGenerator(public_cluster=cluster)
generator.generate()