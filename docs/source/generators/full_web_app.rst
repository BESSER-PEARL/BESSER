Full Web App Generator
======================

The Full Web App Generator in BESSER allows you to automatically create a complete web
application from your structural (class diagram) and GUI models. This generator streamlines
the process of building modern web apps by producing all the backend, frontend, and deployment
files you need, no manual coding required.

Overview
--------

With a single click, the Full Web App Generator produces:

- **Backend**: A FastAPI application with SQLAlchemy models, REST endpoints, and SQLite database integration.
- **Frontend**: A React application implementing your GUI design, with dynamic forms, charts, lists, and more.
- **Deployment**: Dockerfiles for both backend and frontend, plus a Docker Compose file for easy containerized deployment.

How It Works
------------

1. **Design your models**: Create your :doc:`structural model <../buml_language/model_types/structural>` (classes, attributes, relationships)
   and :doc:`GUI model <../buml_language/model_types/gui>`. You can use the :doc:`BESSER Web Modeling Editor <../web_modeling_editor/use_the_wme>`
   for easily designing these models.

2. **Generate the app**: Click "Generate Code" and select **"Full Web App"**.

3. **Download the output**: You will receive a folder containing:

   - ``/backend`` (FastAPI + SQLAlchemy)
   - ``/frontend`` (React)
   - ``docker-compose.yml``, ``backend/Dockerfile``, ``frontend/Dockerfile``

4. **Deploy**: Use Docker Compose to build and run your app locally or in the cloud.


Example
-------

Suppose you have a simple domain model for a library system and a GUI diagram for managing books and authors.
After generating the full web app, your output will look like:

.. code-block:: text

   my_library_app/
   ├── backend/
   │   ├── main_api.py
   │   ├── pydantic_classes.py
   │   ├── sql_alchemy.py
   │   ├── Dockerfile
   │   └── requirements.txt
   ├── frontend/
   │   ├── src/
   │   ├── public/
   │   ├── package.json
   │   ├── Dockerfile
   │   ├── tsconfig.json
   │   ├── .gitignore
   │   └── README.md
   ├── docker-compose.yml

To run your app:

.. code-block:: bash

   docker-compose up --build

This will start both the backend (FastAPI + SQLite) and frontend (React) containers, fully connected and ready to use.

Customization
-------------

- **Database**: The backend uses SQLite by default, but you can easily switch to PostgreSQL or another database by editing the configuration and Docker Compose file.
- **Frontend**: The React app is generated from your GUI diagram, but you can further customize components, styles, and logic.
- **Backend**: Add new endpoints, business logic, or authentication as needed.

.. note::
   The Full Web App Generator is designed to save you time and reduce errors by automating repetitive tasks. You can always customize and extend the generated code to fit your specific requirements.
