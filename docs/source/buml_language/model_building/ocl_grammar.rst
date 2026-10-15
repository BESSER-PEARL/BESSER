Grammar for OCL specification
=============================

We have designed grammar for parsing OCL constraints. The lexer and parser generated using this grammar can parse all the constraints defined in `Royal and Loyal Example project <https://github.com/jcabot/ocl-repository/blob/master/academic/RoyalAndLoyal/RoyalAndLoyal.ocl/>`_.
The corresponding test cases live in ``tests/BUML/notations/ocl/test_parse_ocl.py``.

The grammar for OCL is shown below:

.. literalinclude:: ../../../../besser/BUML/notations/ocl/BOCL.g4
   :language: antlr
   :caption: besser/BUML/notations/ocl/BOCL.g4

Two lexer keywords double as ordinary feature names, so the grammar carries an
explicit fallback for each: ``size`` (``dotSizeNavigation``, see
`BESSER-PEARL/BESSER#198 <https://github.com/BESSER-PEARL/BESSER/issues/198>`_)
and ``date`` (``dotDateNavigation``, plus ``DATE`` as a ``typeRef``). Without
them ``self.size``, ``self.date`` and ``oclIsTypeOf(date)`` — the last of which
every date-typed attribute produces, since BUML's ``DateType`` is named
``date`` — could not be parsed.

The collection operations the grammar accepts after ``->`` include ``size``,
``isEmpty``, ``sum``, ``includes``, ``excludes``, ``union``, ``asSet``,
``intersection``, ``isUnique``, ``symmetricDifference``, ``subSequence`` and
``subOrderedSet``, alongside the ``forAll``/``exists``/``select``/``reject``/
``collect`` iterators. ``allInstances()`` is accepted in both the ``Class::``
and ``Class.`` spellings. Note that ``isUnique`` takes exactly one argument
(``coll->isUnique(body)``) and that B-OCL has **no collection literal** —
there is no ``Set{...}`` or ``Sequence{...}`` production.

To Evaluate the OCL Constraints you can create the test case using the following code:

.. note::

  ``constraint.expression`` holds the constraint as a source-text string (always a ``str``),
  while ``constraint.ast`` (on :class:`OCLConstraint`) exposes the parsed AST as an
  :class:`OCLExpression` for downstream tooling that needs to walk the tree.

.. code-block:: python

    from models.library_object import library_model,object_model
    from bocl.OCLWrapper import OCLWrapper


    def test_1():
        wrapper = OCLWrapper(library_model, object_model)
        constraint=list(library_model.constraints)[0]
        print("Query: " + constraint.expression, end=": ")
        res = None
        try:
            res = wrapper.evaluate(constraint)
        except Exception as error:
                print('\x1b[0;30;41m' + 'Exception Occured! Info:' + str(error) + '\x1b[0m')
                res = None
        assert(res==True)
