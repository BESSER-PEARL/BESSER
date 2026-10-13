Understand BESSER
=================

BESSER keeps a model of your system and generates source code from it. You
can work on the same model through the visual editor, textual imports, or
the Python library.

.. toctree::
   :hidden:

   ../buml_language
   ../spec_driven_agent/index

The workflow
------------

.. code-block:: text

   Draw or describe a system
             |
             v
       Review its model
             |
             v
       Choose generation
         /           \
   Fixed template   AI customisation
         \           /
          v         v
      Inspect and run the code

A **model** describes part of the system. A class model describes data and
relationships; a GUI model describes screens and their data bindings; other
model types describe agents, processes, or neural networks.

A **diagram** is a visual representation of a model. An editor **project**
groups related diagrams and their settings. Exporting a project saves its
models; generating code produces an application or source files.

Two forms of generation
-----------------------

**Deterministic generators** apply templates to models. Their output is
repeatable and they do not need a model-provider API key.

**The Spec-Driven Agent** adds LLM-authored changes to satisfy written
requirements, including features or stacks the templates do not cover. It
uses provider calls, has a budget, and may return incomplete output. Its
checks provide evidence about that output rather than a deployment guarantee.

The modeling assistant and the Spec-Driven Agent have different jobs: the
assistant helps you create and edit diagrams; the Spec-Driven Agent generates
and changes application files.

Learn more
----------

* :doc:`../buml_language`
* :doc:`../spec_driven_agent/index`
