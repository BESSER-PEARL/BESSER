REST API Generator
==================

BESSER offers a code generator for `REST API <https://www.redhat.com/en/topics/api/what-is-a-rest-api>`_ utilizing the `FastAPI framework <https://fastapi.tiangolo.com/>`_. 
This tool automatically transforms classes and relationships defined in a B-UML model into a RESTful service.


Let's generate the code for the REST model of our :doc:`../examples/library_example` B-UML model example. 
You should create a ``RESTAPIGenerator`` object, provide the B-UML model, and use the ``generate`` method as follows:

.. code-block:: python
    
    from besser.generators.rest_api import RESTAPIGenerator
    
    rest_api = RESTAPIGenerator(model=library_model)
    rest_api.generate()

Upon executing this code, a ``rest_api.py`` file containing the Pydantic models will be generated.  in the ``<<current_directory>>/output`` 
folder and it will look as follows.

.. literalinclude:: ../../../tests/library_test/output/rest_api.py
   :language: python
   :linenos:


When you run the code generated, the OpenAPI specifications will be generated:

.. code-block:: json
    
    {
    "openapi": "3.1.0",
    "info": {
        "title": "FastAPI",
        "version": "0.1.0"
    },
    "paths": {
        "/author/": {
            "get": {
                "tags": [
                    "author"
                ],
                "summary": "Get Author",
                "operationId": "get_author_author__get",
                "responses": {
                    "200": {
                        "description": "Successful Response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "items": {
                                        "$ref": "#/components/schemas/Author"
                                    },
                                    "type": "array",
                                    "title": "Response Get Author Author  Get"
                                }
        ................
        