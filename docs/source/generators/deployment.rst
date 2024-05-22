Deployment Generator
========================

BESSER provides a code generator that creates a deployment for a specific application. This generator produces Terraform files from either the :doc:`BUML language <../buml_language/model_types/deployment>`
.. note::

    The BUML can be generated from the grammars. See :doc :doc:`Deployment grammars <../buml_language/model_building/deployment_grammar>`.


Here's a basic usage example:

.. code-block:: python
    
    from besser.generators.deployment import DeploymentGenerator
    
    deployment_generator = DeploymentGenerator(deployment_model=deployment_model)
    deployment_generator.generate()

Currently, the generator supports the creation of Terraform files for AWS or Google Cloud Platform.

To use this generator, you need to have a configuration file for each cloud provider.

For Amazon Web Services (AWS), your configuration file should have the following format:
.. code-block:: python

    region = ""  # Your desired AWS region
    access_key = "" # Enter AWS IAM
    secret_key = "" # Enter AWS IAM

For Google Cloud Platform (GCP), your configuration file should have the following format:
.. code-block:: python

    project="neon-nexus-422908" # Your GCP project 


The generator will create a folder with the following structure:
    "provider name _ name of the cluster"