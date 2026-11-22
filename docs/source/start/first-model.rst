Your first model in Python
==========================

**Goal:** define a ``Book`` model, generate ``classes.py``, and create an
instance of the generated class. This tutorial uses no AI calls.

Install BESSER
--------------

Use Python **3.11 or 3.12**, the versions tested by the project. Create an
empty directory for this tutorial and a virtual environment:

.. code-block:: console

   python -m venv .venv

Activate it with the command for your shell:

.. code-block:: powershell

   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1

.. code-block:: bash

   # Linux or macOS
   source .venv/bin/activate

Then install the library:

.. code-block:: console

   python -m pip install besser

To work from an unreleased development build, use the
:doc:`source installation <../installation>` instead.

Define the model and generate code
----------------------------------

Save this as ``first_model.py``:

.. literalinclude:: ../../../examples/documentation/first_model.py
   :language: python

``Property`` defines an attribute, ``Class`` collects the attributes, and
``DomainModel`` collects the classes. ``PythonGenerator`` writes an ordinary
Python implementation of the model.

Run the script:

.. code-block:: console

   python first_model.py

You should now have ``output/classes.py`` containing a ``Book`` class with
``title`` and ``pages`` properties.

Use the generated class
-----------------------

From the same directory, run:

.. code-block:: console

   python -c "from output.classes import Book; book = Book(title='Dune', pages=412); print(book.title, book.pages)"

Expected output:

.. code-block:: text

   Dune 412

Next steps
----------

* :doc:`../guides/choose-generator` to produce a database schema or backend.
* :doc:`../examples/library_example` to add authors and relationships.
* :doc:`../buml_language/model_types/structural` for model properties and rules.
* :doc:`../troubleshooting` if installation or imports fail.
