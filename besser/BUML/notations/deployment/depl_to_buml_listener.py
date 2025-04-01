from .deploymentParser import deploymentParser
from .deploymentListener import deploymentListener

class Deployment_BUML_Listener(deploymentListener):
    """
    Listener for parsing deployment-related models and writing the corresponding B-UML code.

    This class extends `deploymentListener` and overrides methods for handling different components
    of a deployment model.
    It writes the parsed deployment architecture into a specified output stream in the B-UML
    model format.

    Attributes:
        output (TextIO): The output stream where the B-UML model code is written.
        __cluster_list (list): A list to keep track of clusters in the deployment model.
    """

    def __init__(self, output):
        self.output = output
        self.__cluster_list: list = list()
        text = "from besser.BUML.metamodel.deployment import *\n\n# Deployment architecture model definition\n\n"
        self.output.write(text)

    def enterApplication(self, ctx: deploymentParser.ApplicationContext):
        name = ctx.ID().getText()
        repo = ctx.STRING(0).getText()
        port = ctx.INT(0).getText()
        cpu = ctx.INT(1).getText()
        memory = ctx.INT(2).getText()
        model = ctx.STRING(1).getText()
        text = name + " : Application = Application(name=\""+ name + "\", image_repo=" + repo + ", port=" + port + \
            ", required_resources=Resources(cpu=" + cpu + ", memory=" + memory + "), domain_model=" + model + ")\n"
        self.output.write(text)

    def enterService(self, ctx: deploymentParser.ServiceContext):
        name = ctx.ID(0).getText()
        port = ctx.INT(0).getText()
        target_port = ctx.INT(1).getText()
        protocol = ctx.protocol().getText().lower()
        type = ctx.service_type().getText()
        text = name + " : Service = Service(name=\""+ name + "\", port=" + port + ", target_port=" + target_port + \
            ", protocol=Protocol." + protocol + ", type=ServiceType." + type
        if (ctx.app is None):
            text += ")\n"
        else:
            text += ", application=" + ctx.ID(1).getText().replace(" ", "") + ")\n"
        self.output.write(text)

    def enterContainer(self, ctx: deploymentParser.ContainerContext):
        name = ctx.ID(0).getText()
        app = ctx.ID(1).getText()
        cpu = ctx.INT(0).getText()
        memory = ctx.INT(1).getText()
        text = name + " : Container = Container(name=\""+ name + "\", application=" + app + ", resources_limit=Resources(cpu=" + \
            cpu + ", memory=" + memory + "))\n"
        self.output.write(text)

    def enterDeployment(self, ctx: deploymentParser.DeploymentContext):
        name = ctx.ID(0).getText()
        replicas = ctx.INT().getText()
        container = ctx.ID(1).getText()
        text = name + " : Deployment = Deployment(name=\"" + name + "\", replicas=" + replicas + ", containers={" + \
            container.replace("\"", "").replace(" ", "")
        if len(ctx.ID()) > 2:
            for i in range(2, len(ctx.ID())):
                text += ", " + ctx.ID(i).getText()
        text += "})\n"
        self.output.write(text)

    def enterRegion(self, ctx: deploymentParser.RegionContext):
        name = ctx.ID_REG(0).getText()
        text = name.replace("-", "_") + " : Region = Region(name=\"" + name + "\", zones={})\n"
        self.output.write(text)

    def enterPublicCluster(self, ctx: deploymentParser.PublicClusterContext):
        text = "\n# Public cluster definition\n"
        name = ctx.ID(0).getText()
        nodes = ctx.INT().getText()
        provider = ctx.provider().getText()
        config_file = ctx.STRING().getText()
        text += name + " : PublicCluster = PublicCluster(name=\"" + name + "\", num_nodes=" + nodes + ", provider=Provider." + provider + \
            ", config_file=" + config_file
        self.output.write(text)
        self.__cluster_list.append(name)
    
    def enterService_list(self, ctx: deploymentParser.Service_listContext):
        service = ctx.ID(0).getText()
        text = ", services={" + service
        if len(ctx.ID()) > 1:
            for i in range(1, len(ctx.ID())):
                text += ", " + ctx.ID(i).getText()
        text += "}"
        self.output.write(text)

    def enterDeployment_list(self, ctx: deploymentParser.Deployment_listContext):
        deployment = ctx.ID(0).getText()
        text = ", deployments={" + deployment
        if len(ctx.ID()) > 1:
            for i in range(1, len(ctx.ID())):
                text += ", " + ctx.ID(i).getText()
        text += "}"
        self.output.write(text)        

    def enterRegion_list(self, ctx: deploymentParser.Region_listContext):
        region = ctx.ID_REG(0).getText().replace("-", "_")
        text = ", regions={" + region
        if len(ctx.ID_REG()) > 1:
            for i in range(1, len(ctx.ID_REG())):
                text += ", " + ctx.ID_REG(i).getText().replace("-", "_")
        text += "}"
        self.output.write(text)

    def exitPublicCluster(self, ctx: deploymentParser.PublicClusterContext):
        text = ")\n"
        self.output.write(text)

    def exitArchitecture(self, ctx: deploymentParser.ArchitectureContext):
        text = "\n# Deployment architecture model definition\n"
        text += "deployment_model : DeploymentModel = DeploymentModel(name=\"deployment_model\", clusters=" + list_to_str(self.__cluster_list) + ")\n"
        self.output.write(text)

def list_to_str(elements:list):
    """
        Method to transform a list of elements to string

        Args:
           elements (list): The list to transform.
    """
    if len(elements) == 0:
        str_list = "set()"
    else:
        str_list = ", ".join(elements)
        str_list = "{" + str_list + "}"
    return str_list
