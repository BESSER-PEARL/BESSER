Deployment example
======================================================================

This example demonstrates how to use the BESSER :doc:`../generators/backend` and :doc:`../generators/terraform` with the :doc:`../examples/dpp` model to create a backend service, build and upload Docker images,
and deploy the infrastructure using Terraform.

Deployment Model
----------------

First, define your deployment model using the provided :doc:`grammar <../buml_language/model_building/deployment_grammar>`:

.. code-block:: python

    Deployment model{
        applications {
            ->  name: app1,
                image: "image/latest",
                port: 8000,
                cpu_required: 10m, 
                memory_required: 100Mi, 
                domain_model: "dpp_model"
        }
        
        services {
            ->  name: service1,
                port: 80,
                target_port: 8000,
                protocol: HTTP,
                type: lb,
                app_name: app1
        }
        
        containers {
            ->  name: container1,
                app_name: app1,
                cpu_limit: 500m,
                memory_limit: 512Mi
        }

        deployments {
            ->  name: deployment1,
                replicas: 2,
                containers: [container1]
        }

        regions {
            ->  name: us-east1
        }

        clusters {
            ->  public_cluster
                name: cluster1,
                number_of_nodes: 3,
                provider: google,
                config_file: "config_google.conf",
                services: [service1],
                deployments: [deployment1],
                regions: [us-east1],
                net_config: True
            
            ->  public_cluster
                name: cluster2,
                number_of_nodes: 3,
                provider: aws,
                config_file: "config_aws.conf",
                services: [service1],
                deployments: [deployment1],
                regions: [us-east1],
                net_config: True
        }
    }

Step 1: Generate the Backend
----------------------------

Use the Backend Generator to create the backend code for the ``library_model``.

.. code-block:: python

    from besser.generators.backend import BackendGenerator

    backend = BackendGenerator(model=dpp_model, http_methods=['GET', 'POST', 'PUT', 'DELETE'], nested_creations=True, docker_image = True)
    backend.generate()

This will generate the backend code in the ``output_backend`` directory, including the ``main_api.py``, ``sql_alchemy.py``, ``pydantic_classes.py`` files and 
the Dockerfile for building and uploading the Docker image.


Step 2: Generate Terraform Files
--------------------------------

Use the Terraform Generator to create the Terraform configuration files for deploying the backend to AWS or GCP.

**AWS Configuration File**

Create a configuration file named ``config_aws.conf``:

.. code-block:: ini

    region = "us-east-1"
    access_key = "your_aws_access_key"
    secret_key = "your_aws_secret_key"

**GCP Configuration File**

Create a configuration file named ``config_google.conf``:

.. code-block:: ini

    project = "your_gcp_project_id"

**Generate Terraform Files**
First convert the deployment grammar to B-UML model using the following code:

.. code-block:: python

    from besser.BUML.notations.deployment import buml_deployment_model
    # Deployment architecture model
    deployment_model = buml_deployment_model(deployment_textfile="deployment.txt")

Then, use the Terraform Generator to create the Terraform configuration files:

.. code-block:: python

    from besser.generators.terraform import TerraformGenerator

    terraform_generator = TerraformGenerator(deployment_model=deployment_model)
    terraform_generator.generate()

This will create the necessary Terraform files in directories named ``<provider_name>_<cluster_name>/``.

Step 3: Deploy Infrastructure with Terraform
--------------------------------------------

Navigate to the generated directory (e.g., ``aws_cluster2/`` or ``gcp_cluster1/``) and run the setup script to deploy your infrastructure:

.. code-block:: bash

    setup.bat

This script will initialize and apply your Terraform configuration, deploying the resources to the specified cloud provider.

This example demonstrates the complete workflow for using the Backend Generator and Terraform Generator with the DPP model. It covers defining the deployment model, 
generating the backend code, creating and uploading Docker images, and deploying the infrastructure using Terraform.
