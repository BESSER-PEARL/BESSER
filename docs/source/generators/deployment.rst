Deployment Generator
====================

BESSER provides a code generator that creates a deployment for a specific application. This generator produces Terraform files from either the :doc:`BUML language <../buml_language/model_types/deployment>`.

.. note::

    The BUML can be generated from the grammars. See :doc:`Deployment grammars <../buml_language/model_building/deployment_grammar>`.

Basic Usage Example
-------------------

.. code-block:: python

    from besser.generators.deployment import DeploymentGenerator

    deployment_generator = DeploymentGenerator(deployment_model=deployment_model)
    deployment_generator.generate()

The generator currently supports the creation of Terraform files for AWS and Google Cloud Platform (GCP).

Configuration
-------------

To use this generator, you need to have a configuration file for each cloud provider.

### Amazon Web Services (AWS)

Your configuration file should have the following format:

.. code-block:: python

    region = ""  # Your desired AWS region
    access_key = ""  # Enter AWS IAM access key
    secret_key = ""  # Enter AWS IAM secret key

### Google Cloud Platform (GCP)

Your configuration file should have the following format:

.. code-block:: python

    project = "neon-nexus-422908"  # Your GCP project

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

The deployment model supports multiple clusters for multiple cloud providers. For public clusters, there is an attribute called `net_config`. If `net_config` is set to `true` (which is the default), the generator will create the minimum network and subnetwork configurations required for each cloud provider.

Launching Your Terraform Code
-----------------------------

To launch your Terraform code, the generator also creates a setup file that initializes and applies the Terraform configuration. Here is an example of the generated `setup.bat`:


To run the setup script, use the following command in your terminal:

.. code-block:: bat

    setup.bat

This script will initialize and apply your Terraform configuration, deploying your resources to the specified cloud provider.
