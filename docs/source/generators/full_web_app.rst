Full Web App Generator
======================

The Full Web App Generator in BESSER allows you to automatically create a complete web
application from your structural (class diagram) and GUI models. This generator streamlines
the process of building modern web apps by producing all the backend, frontend, and deployment
files you need—no manual coding required.

Overview
--------

With a single generation, the Full Web App Generator produces:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Component
     - Technology
     - Description
   * - **Backend**
     - FastAPI + SQLAlchemy + Pydantic
     - REST API, database models, validation
   * - **Frontend**
     - React + TypeScript
     - Dynamic UI with forms, tables, charts
   * - **Database**
     - SQLite (default)
     - Configurable to PostgreSQL, MySQL, etc.
   * - **Deployment**
     - Docker + Docker Compose
     - Container orchestration ready


Sub-Generators
--------------

The Full Web App Generator internally uses these specialized generators:

**Backend** (see :doc:`backend`)
   Uses: :doc:`pydantic`, :doc:`alchemy`, :doc:`rest_api`

**Frontend** (see :doc:`react`)
   React application with TableComponent, MethodButton, charts

.. toctree::
   :maxdepth: 1
   :caption: Backend Components

   backend
   pydantic
   alchemy
   rest_api

.. toctree::
   :maxdepth: 1
   :caption: Frontend Components

   react


How It Works
------------

1. **Design your models**: Create your :doc:`structural model <../buml_language/model_types/structural>` (classes, attributes, relationships)
   and :doc:`GUI model <../buml_language/model_types/gui>`. You can use the :doc:`BESSER Web Modeling Editor <../web_editor>`
   for easily designing these models.

2. **Generate the app**: Click "Generate Code" and select **"Full Web App"**.

3. **Download the output**: You will receive a folder containing:

   - ``/backend`` (FastAPI + SQLAlchemy)
   - ``/frontend`` (React)
   - ``docker-compose.yml``, ``backend/Dockerfile``, ``frontend/Dockerfile``

4. **Deploy**: Use Docker Compose to build and run your app locally or in the cloud.


Generated Output Structure
--------------------------

.. code-block:: text

   my_app/
   ├── backend/
   │   ├── main_api.py          # REST API endpoints
   │   ├── pydantic_classes.py  # Data validation models
   │   ├── sql_alchemy.py       # Database ORM models
   │   ├── Dockerfile           # Backend container
   │   └── requirements.txt     # Python dependencies
   ├── frontend/
   │   ├── src/
   │   │   ├── components/      # React components
   │   │   ├── contexts/        # React contexts
   │   │   └── pages/           # Page components
   │   ├── package.json
   │   ├── Dockerfile           # Frontend container
   │   └── README.md
   ├── docker-compose.yml       # Container orchestration


Features
--------

Class Methods
^^^^^^^^^^^^^

When your B-UML model includes **methods** defined on classes, the generator automatically creates 
API endpoints to execute them. The frontend provides interactive buttons to call these methods.

**Instance Methods** (methods with ``self``):

Operate on a specific entity instance. The frontend requires selecting a row first.

.. code-block:: python

    # B-UML method definition
    def apply_discount(self, percent: float):
        self.price = self.price * (1 - percent / 100)

- **Backend Endpoint**: ``POST /{entity}/{id}/methods/{method_name}/``
- **Frontend**: MethodButton component with parameter input modal

**Class Methods** (methods without ``self``):

Operate at the class level, performing operations on the entire collection.

.. code-block:: python

    # B-UML method definition (no self parameter)
    def get_expensive_books(database, min_price: int):
        return database.query(Book).filter(Book.price > min_price).all()

- **Backend Endpoint**: ``POST /{entity}/methods/{method_name}/``
- **Frontend**: MethodButton component (no row selection required)

**Supported Parameter Types:**

``str``, ``int``, ``float``, ``bool``, ``date``, ``datetime``, ``time``

See :doc:`backend` for complete method endpoint documentation.


OCL Constraint Validation
^^^^^^^^^^^^^^^^^^^^^^^^^

When you define OCL constraints in your B-UML model, they are automatically:

1. Parsed from the OCL expression using BESSER's ANTLR-based parser
2. Transformed into Pydantic field validators
3. Displayed as error messages in the frontend

**Example constraint:**

.. code-block:: python

    constraint = Constraint(
        name="min_age",
        context=Player,
        expression="context Player inv: self.age > 10",
        language="OCL"
    )

**Supported operators:** ``>``, ``<``, ``>=``, ``<=``, ``=``, ``<>``

**Frontend display:**

When a user submits invalid data, the error message is shown in a red box inside the form modal.

See :doc:`pydantic` for full OCL validation documentation.


Error Handling
^^^^^^^^^^^^^^

The generated web app includes comprehensive error handling:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Error Type
     - HTTP Status
     - Frontend Behavior
   * - Validation Error
     - 422
     - Shows field-level errors in modal, keeps modal open
   * - Server Error
     - 500
     - Shows error message with details, keeps modal open
   * - Network Error
     - N/A
     - Shows "Network error" message

Both **TableComponent** (for CRUD forms) and **MethodButton** (for method execution) display 
errors inline and keep modals open so users can fix and retry.


REST API Endpoints
^^^^^^^^^^^^^^^^^^

The generated backend includes comprehensive REST endpoints:

**CRUD Operations:**

- ``GET /{entity}/`` - List all
- ``GET /{entity}/{id}/`` - Get by ID
- ``POST /{entity}/`` - Create
- ``PUT /{entity}/{id}/`` - Update
- ``DELETE /{entity}/{id}/`` - Delete
- ``GET /{entity}/paginated/`` - Paginated list
- ``GET /{entity}/search/`` - Search by attributes
- ``POST /{entity}/bulk/`` - Bulk create
- ``DELETE /{entity}/bulk/`` - Bulk delete

**Relationship Management (N:M):**

- ``GET /{entity}/{id}/{relationship}/`` - Get related
- ``POST /{entity}/{id}/{relationship}/{related_id}/`` - Add relationship
- ``DELETE /{entity}/{id}/{relationship}/{related_id}/`` - Remove relationship

See :doc:`backend` for complete endpoint documentation.


Running Your App
----------------

With Docker Compose
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd my_app
   docker-compose up --build

This starts:

- Backend at ``http://localhost:8000``
- Frontend at ``http://localhost:3000``

Without Docker
^^^^^^^^^^^^^^

**Backend:**

.. code-block:: bash

   cd backend
   pip install -r requirements.txt
   uvicorn main_api:app --reload

**Frontend:**

.. code-block:: bash

   cd frontend
   npm install
   npm run dev


Customization
-------------

- **Database**: Switch from SQLite to PostgreSQL by editing the connection string
- **Frontend**: Customize components, styles, and logic in the React code
- **Backend**: Add new endpoints, business logic, or authentication
- **Constraints**: Add OCL constraints to enforce business rules

.. note::
   The Full Web App Generator saves time by automating repetitive tasks. 
   You can always customize and extend the generated code.
