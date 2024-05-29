Terraform Generator
====================

BESSER offers a code generator designed to facilitate the multi-cloud deployment of your applications. This generator takes 
as input a :doc:`deployment architecture model <../buml_language/model_types/deployment>` specified with B-UML. Within this 
model, you can define the specific characteristics of your cloud infrastructure environment and the containerized applications to be deployed.

To create this deployment model using B-UML, we recomend to use :doc:`the grammar for deployment models <../buml_language/model_building/deployment_grammar>`
provided by BESSER.


Basic Usage Example
-------------------

.. code-block:: python

    from besser.generators.terraform import TerraformGenerator

    terraform_generator = TerraformGenerator(deployment_model=deployment_model)
    terraform_generator.generate()


Although B-UML enables the modeling of public clusters for different cloud providers and on-premises, currently the 
code generator supports the creation of Terraform files for AWS and Google Cloud Platform (GCP) only.

In addition to giving the code generator your B-UML deployment model, you'll also need to provide a configuration 
file specific to your chosen cloud provider. This file includes provider details (sometimes sensitive data), like the 
login credentials necessary for Terraform deployment.

**Configuration file for AWS**

Your configuration file should have the following format:

.. code-block:: python

    access_key = ""  # Enter AWS IAM access key
    secret_key = ""  # Enter AWS IAM secret key

**Configuration file for GCP**

Your configuration file should have the following format:

.. code-block:: python

    project = "neon-nexus-422908"  # Your GCP project

.. note::

    The location of your configuration file must be specified in the input deployment model as a parameter of each public cluster defined.

Generated Folder Structure
--------------------------

The generator will create a folder with the following structure:

<provider_name>_<cluster_name>/

Inside this folder, you will find the necessary Terraform files for deploying your application on the specified cloud provider.

For example, if deploying to AWS with a cluster named "my-cluster", the folder structure would look like:
::

    aws_my-cluster/
    ├── eks.tf
    ├── iam-config.tf
    ├── igw.tf
    ├── nat.tf
    ├── nodes.tf
    ├── provider.tf
    ├── route.tf
    ├── setup.bat
    ├── subnet.tf
    └── vpc.tf

For GCP with a cluster named "my-cluster", the folder structure would be:
::

    gcp_my-cluster/
    ├── api.tf
    ├── app.tf
    ├── cluster.tf
    ├── setup.bat
    └── version.tf

Multiple Clusters and Network Configuration
-------------------------------------------

The :doc:`B-UML deployment model <../buml_language/model_types/deployment>` supports multiple clusters for multiple 
cloud providers. For public clusters, there is an attribute called `net_config`. If `net_config` is set to `True` (which is 
the default), the generator will create the minimum network and subnetwork configurations required for each cloud provider.
This way, you won't have to worry about modeling the networks and subnets as they will be configured automatically.

Launching Your Terraform Code
-----------------------------

To launch your Terraform code, the generator also creates a setup file that initializes and applies the Terraform configuration. 
Here is an example of the generated `setup.bat`:

To run the setup script, use the following command in your terminal:

.. code-block:: bat

    setup.bat

This script will initialize and apply your Terraform configuration, deploying your resources to the specified cloud provider.