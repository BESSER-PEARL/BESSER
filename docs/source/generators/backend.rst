Backend Generator
=====================

BESSER provides a code generator for backend services, utilizing the FastAPI framework, SQL Alchemy for database interactions,
and Pydantic for data validation. This tool enables developers to seamlessly transform classes and relationships defined in a B-UML model 
into a fully functional backend service.

The BESSER backend generator streamlines the development process by leveraging multiple specialized generators within the BESSER suite to automatically produce:

- **A RESTful API Service**: Utilizes the BESSER REST API Generator to create dynamic, scalable API endpoints.
- **Database Models**: Integrates BESSER's SQL Alchemy Generator to construct efficient ORM-based models for database interactions.
- **Data Validation Models**: Employs BESSER's Pydantic Generator to ensure that data conforms to the defined schemas, enhancing the integrity and security of the backend.


To generate the complete backend for a B-UML model, follow the steps below. The example uses the ``library`` example B-UML model as a reference.

.. code-block:: python
    
    from besser.generators.backend import BackendGenerator
    
    backend = BackendGenerator(model=library_model, http_methods=['GET', 'POST', 'PUT', 'DELETE'], nested_creations = True, docker_image = True)
    backend.generate()

You can customize the code generation patterns by selecting specific HTTP methods such as ``GET``, ``POST``, ``PUT``, and ``DELETE``.
This selection allows for targeted code generation tailored to only the necessary components of the API.
If specified in the B-UML model with read_only attributes, the generator will automatically exclude the ``PUT`` and ``DELETE`` methods.
Additionally, the ``nested_creations`` parameter configures how the API manages entity relationships in requests. If set to True, the API allows both 
the linking of existing entities via identifiers and the nested creation of new entities. If set to False, the API only permits linking existing 
entities using their identifiers. The default setting is False, which restricts the functionality to linking by identifiers. Finally, the 
``docker_image`` parameter is there to assist you in `creating a Docker image <#docker-image-generation>`_ for your application.


Invoke the generate method to produce the backend code.The generated files will be placed in the ``<<current_directory>>/output_backend``.
This method will generate several files:

   + ``main_api.py``: Contains the REST API endpoints.
   + ``sql_alchemy.py``: Includes SQL Alchemy database models.
   + ``pydantic_classes.py``: Consists of Pydantic validation models.
   + ``database.db``: A SqlLite database file.


.. image:: ../img/backend_generator_schema.png
  :width: 1000
  :alt: Backend Generator user schema
  :align: center

The REST API communicates with the database through SQLAlchemy, and Pydantic validates the data before it's sent or after it's received by the REST API.
This setup encapsulates a modern backend architecture where each piece serves a specific role in data handling and processing.
When you run the code generated, a SqlLite database and the OpenAPI specifications will be generated.

We have an example demonstrating how this generator works, which you can find here: :doc:`../examples/backend_example`.
This example showcases the usage of the Backend Generator with our :doc:`../examples/library_example` Example, illustrating its application in generating a fully functional backend from a B-UML model.

Docker Image Generation
-----------------------
The Backend Generator offers the ``docker_image`` boolean parameter, designed to streamline the creation and uploading of Docker images for the generated backend. When 
you set this parameter to *True*, you have two options for creating your image:

1. Automated DockerHub Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Our code generator can create and upload the Docker image to your `DockerHub <https://hub.docker.com>`_ in one step. Provide a configuration file through the ``docker_config_file`` parameter, which 
enables the generator to automatically create and upload the image in your DockerHub accout using the provided configurations.

To create the configuration file, use the following template and save it as a .conf file:

.. code-block:: ini
    
    [DEFAULT]
    docker_username = dockerhub_username
    docker_password = dockerhub_password
    docker_image_name = image_name
    docker_repository = dockerhub_repository
    docker_tag = image_tag
    docker_port = port

2. Custom Dockerfile Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The second option is to generate the Dockerfile with the instructions to create your Docker image and then run it yourself to build the image and upload it to 
your repository. To follow this option, just don't use the ``docker_image_file`` parameter in the code generator. BESSER will then assume that you will handle 
running the Dockerfile and uploading the image to the repository yourself.

The generator will create two files:

- ``Dockerfile``: Contains the necessary instructions to build the Docker image.
- ``create_docker_image.py``: A Python script that automates the process of building and uploading the Docker image.

By providing this script and the Dockerfile, users can build and upload their Docker images by executing the script with their DockerHub credentials.

.. warning::
   
   If you use the generator to generate and load the Docker image in DockerHub, you must make sure you have a Docker engine installed on your computer. 
   For example `Docker desktop <https://www.docker.com/products/docker-desktop>`_.


Generated API Endpoints
-----------------------

The Backend Generator creates a comprehensive REST API with the following endpoint categories for each entity in your B-UML model:

CRUD Operations
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Method
     - Endpoint
     - Description
   * - GET
     - ``/{entity}/``
     - Get all entities (supports ``detailed=true`` for relationships)
   * - GET
     - ``/{entity}/count/``
     - Get total count of entities
   * - GET
     - ``/{entity}/paginated/``
     - Paginated list with ``skip``, ``limit``, ``detailed`` params
   * - GET
     - ``/{entity}/search/``
     - Search by attributes (auto-generated filters)
   * - GET
     - ``/{entity}/{id}/``
     - Get single entity by ID
   * - POST
     - ``/{entity}/``
     - Create single entity
   * - POST
     - ``/{entity}/bulk/``
     - Bulk create multiple entities
   * - PUT
     - ``/{entity}/{id}/``
     - Full update of entity
   * - DELETE
     - ``/{entity}/{id}/``
     - Delete single entity
   * - DELETE
     - ``/{entity}/bulk/``
     - Bulk delete by IDs

Relationship Management (N:M)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For many-to-many relationships, additional endpoints are generated:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Method
     - Endpoint
     - Description
   * - GET
     - ``/{entity}/{id}/{relationship}/``
     - Get all related entities
   * - POST
     - ``/{entity}/{id}/{relationship}/{related_id}/``
     - Add a relationship
   * - DELETE
     - ``/{entity}/{id}/{relationship}/{related_id}/``
     - Remove a relationship


Class Methods Endpoints
^^^^^^^^^^^^^^^^^^^^^^^

When your B-UML model includes **methods** defined on classes, the generator automatically creates API endpoints to execute them.
The generator distinguishes between two types of methods based on the method signature:

**Instance Methods** (methods with ``self``):

These methods operate on a specific entity instance. The endpoint requires an entity ID.

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Method
     - Endpoint
     - Description
   * - POST
     - ``/{entity}/{id}/methods/{method_name}/``
     - Execute method on a specific entity instance

**Class Methods** (methods without ``self``):

These methods operate at the class level, performing operations on the entire collection or class-level logic.

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Method
     - Endpoint
     - Description
   * - POST
     - ``/{entity}/methods/{method_name}/``
     - Execute class-level method

**Supported Parameter Types:**

The following parameter types are supported for method parameters:

- ``str`` - String values
- ``int`` - Integer values
- ``float`` - Floating-point values
- ``bool`` - Boolean values
- ``date`` - Date values
- ``datetime`` - DateTime values
- ``time`` - Time values

**Request Format:**

Parameters are passed as a JSON body:

.. code-block:: json

    {
        "params": {
            "param_name": "value",
            "another_param": 42
        }
    }

**Response Format:**

The method execution returns a structured response:

.. code-block:: json

    {
        "entity_id": 5,
        "method": "method_name",
        "status": "executed",
        "result": "...",
        "output": "captured print statements..."
    }

**Example - Instance Method:**

If your B-UML model has a ``Book`` class with a method ``apply_discount(self, percent: float)``:

.. code-block:: python

    # B-UML method definition
    def apply_discount(self, percent: float):
        self.price = self.price * (1 - percent / 100)

The generator creates:

- Endpoint: ``POST /book/5/methods/apply_discount/``
- Request body: ``{"params": {"percent": 10}}``

**Example - Class Method:**

If your B-UML model has a ``Book`` class with a method ``get_expensive_books(database, min_price: int)``:

.. code-block:: python

    # B-UML method definition (no self parameter)
    def get_expensive_books(database, min_price: int):
        return database.query(Book).filter(Book.price > min_price).all()

The generator creates:

- Endpoint: ``POST /book/methods/get_expensive_books/``
- Request body: ``{"params": {"min_price": 50}}``

.. note::

   The ``database`` parameter is automatically injected by the API framework and should not be passed in the request body.
   Any ``print()`` statements executed during method execution are captured and returned in the ``output`` field of the response.


System Endpoints
^^^^^^^^^^^^^^^^

The generated API also includes system-level endpoints:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Method
     - Endpoint
     - Description
   * - GET
     - ``/``
     - API information (name, version, status)
   * - GET
     - ``/health``
     - Health check endpoint for monitoring
   * - GET
     - ``/statistics``
     - Database statistics (entity counts)
