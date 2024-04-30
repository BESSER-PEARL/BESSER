Installation
=============

Basic Installation
--------------------------------
BESSER works with Python 3.9+. We recommend creating a virtual environment (e.g. `venv <https://docs.python.org/3/tutorial/venv.html>`_, 
`conda <https://docs.conda.io/en/latest/>`_).

The latest stable version of BESSER is available in the Python Package Index (PyPi) and can be installed using

.. code-block:: console

    $ pip install besser

Building From Source
--------------------
To obtain the full code, including examples and tests, you can clone the git repository.

.. code-block:: console

    $ git clone https://github.com/BESSER-PEARL/BESSER.git
    $ cd BESSER

Now, install *build*, then generate and install the *besser* package. Remember to replace ``*.*.*`` by the package version number.

.. code-block:: console

    $ pip install --upgrade build
    $ python -m build
    $ pip install dist/besser-*.*.*-py3-none-any.whl