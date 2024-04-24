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
    
    from besser.generators.backend import Backend_Generator
    
    backend = Backend_Generator(model=library_model, http_methods=['GET', 'POST', 'PUT', 'DELETE'], nested_creations = True)
    backend.generate()

You can customize the code generation patterns by selecting specific HTTP methods such as ``GET``, ``POST``, ``PUT``, and ``DELETE``.
This selection allows for targeted code generation tailored to only the necessary components of the API.
If specified in the B-UML model with read_only attributes, the generator will automatically exclude the ``PUT`` and ``DELETE`` methods.
Additionally, the nested_creations parameter configures how the API manages entity relationships in requests. If set to True, the API allows both 
the linking of existing entities via identifiers and the nested creation of new entities. If set to False, the API only permits linking existing 
entities using their identifiers. The default setting is False, which restricts the functionality to linking by identifiers.


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

        