React Generator
===============

The React Generator produces a modern React frontend application from your GUI model. 
It is used internally by the :doc:`full_web_app` generator to create the frontend.

Overview
--------

The React Generator creates:

- **Pages**: React components for each screen in your GUI model
- **Components**: Reusable UI components (tables, forms, charts)
- **Contexts**: React contexts for state management
- **Routing**: Application navigation setup

Generated Components
--------------------

TableComponent
^^^^^^^^^^^^^^

Dynamic data tables with:

- Sorting and filtering
- Pagination
- Add/Edit modals with form validation
- Backend validation error display (keeps modal open on error)
- Support for all relationship types (1:1, 1:N, N:M)
- Lookup fields for related entities

MethodButton
^^^^^^^^^^^^

Executes class and instance methods with:

- Parameter input modal for methods with parameters
- Automatic type handling (int, float, bool, string, date, time)
- Detailed error display for 500 errors
- Modal stays open on error for retry
- Automatic table refresh after execution

Chart Components
^^^^^^^^^^^^^^^^

Visualization components including:

- Line charts
- Bar charts
- Pie charts
- Data binding to backend entities

Usage
-----

The React Generator is typically used via the Full Web App Generator:

.. code-block:: python

    from besser.generators.web_app import WebAppGenerator
    
    gen = WebAppGenerator(domain_model, gui_model, output_dir="output")
    gen.generate()

Or standalone:

.. code-block:: python

    from besser.generators.react import ReactGenerator
    
    gen = ReactGenerator(domain_model, gui_model, output_dir="frontend")
    gen.generate()

Generated Structure
-------------------

.. code-block:: text

   frontend/
   ├── src/
   │   ├── components/
   │   │   ├── table/
   │   │   │   └── TableComponent.tsx
   │   │   ├── MethodButton.tsx
   │   │   └── Renderer.tsx
   │   ├── contexts/
   │   │   └── TableContext.tsx
   │   └── pages/
   │       └── Home.tsx
   ├── public/
   ├── package.json
   └── tsconfig.json
