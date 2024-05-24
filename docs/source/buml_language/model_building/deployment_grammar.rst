Grammar for deployment architecture model
=========================================

The deployment architecture model can be defined using a grammar, which allows for a textual specification that enhances
the understanding of the model. For example, you can specify a deployment model using the following syntax, which includes
elements such as applications, services, containers, deployments, regions, and clusters.

.. code-block:: console

    Deployment model{
        applications {
            ->  name: app1,
                image: "hhtp://docker-image/latest",
                port: 8000,
                cpu_required: 10m,
                memory_required: 100Mi,
                domain_model: "library_model"
        }
        
        services {
            ->  name: service1, port: 80,
                target_port: 8000,
                protocol: TCP,
                type: lb,
                app_name: app1
        }
        
        containers {
            ->  name: container1,
                app_name: app1,
                cpu_limit: 500 m,
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
                config_file: "file",
                services: [service1],
                deployments: [deployment1],
                regions: [us-east1],
                net_config: True
        }
    }

.. note::
    You can check `the complete grammar here <https://github.com/BESSER-PEARL/BESSER/blob/master/besser/BUML/notations/deployment/deployment.g4>`_

Once the deployment textual model is defined, you can parse it and obtain the B-UML model as follows.

.. code-block:: python

    from besser.BUML.notations.deployment import buml_deployment_model
    from besser.BUML.metamodel.deployment import DeploymentModel

    # deployment.txt contains the textual definition of the deployment architecture
    deployment_model: DeploymentModel = buml_deployment_model(deployment_textfile="deployment.txt")

This ``deployment_model`` is a B-UML model that you can use in BESSER, for example, to generate the `Terraform <https://www.terraform.io/>`_
code to automate the deployment using our :doc:`../../generators/terraform`