BAL Standard Library
===========

In addition to the core language, BAL also provide a set of predefined method in the form of a standard library.
The function part of this library are presented thereafter for each type of the language.

Sequence Standard Library
--------------------------
For the sequence standard library we assume a sequence of type ``T[]`` with elements of type ``T``.


Method ``size``
^^^^^^^^^^^^^^^^^^
**Signature** (expressed in BAL)

.. code-block::

    def size() -> int

**Semantics**:

Return the number of elements present in the sequence.


Method ``is_empty``
^^^^^^^^^^^^^^^^^^
**Signature** (expressed in BAL)

.. code-block::

    def is_empty() -> bool

**Semantics**:

Return true if the sequence has no elements, false otherwise.


Method ``add``
^^^^^^^^^^^^^^^^^^
**Signature** (expressed in BAL)

.. code-block::

    def add(elem:T) -> nothing

**Semantics**:

Add the element ``elem`` at the end of the sequence


Method ``remove``
^^^^^^^^^^^^^^^^^^
**Signature** (expressed in BAL)

.. code-block::

    def remove(elem:T) -> nothing

**Semantics**:

Remove the first instance of ``elem`` in the sequence


Method ``contains``
^^^^^^^^^^^^^^^^^^
**Signature** (expressed in BAL)

.. code-block::

    def contains(elem:T) -> bool

**Semantics**:

Returns true if the element ``elem`` is in the sequence


Method ``filter``
^^^^^^^^^^^^^^^^^^
**Signature** (expressed in BAL)

.. code-block::

    def filter(predicate:[T] -> bool) -> T[]

**Semantics**:

Returns a sequence containing all the elements for which ``predicate(elem)`` returned true.


Method ``forall``
^^^^^^^^^^^^^^^^^^
**Signature** (expressed in BAL)

.. code-block::

    def forall(predicate:[T] -> bool) -> bool

**Semantics**:

Returns true if the predicate function returns true for all the elements of the sequence


Method ``exists``
^^^^^^^^^^^^^^^^^^
**Signature** (expressed in BAL)

.. code-block::

    def exists(predicate:[T] -> bool) -> bool

**Semantics**:

Returns true if the predicate function returns true for at least one element of the sequence


Method ``one``
^^^^^^^^^^^^^^^^^^
**Signature** (expressed in BAL)

.. code-block::

    def one(predicate:[T] -> bool) -> bool

**Semantics**:

Returns true if the predicate function returns true for exactly one element of the sequence


Method ``map``
^^^^^^^^^^^^^^^^^^
**Signature** (expressed in BAL)

.. code-block::

    def map(mapping:[T] -> any) -> any[]

**Semantics**:

Returns a new sequence containing the result of applying the ``mapping`` function to the elements of the sequence.


Method ``is_unique``
^^^^^^^^^^^^^^^^^^
**Signature** (expressed in BAL)

.. code-block::

    def is_unique(mapping:[T] -> any) -> bool

**Semantics**:

Applies the mapping function to the sequence elements and return true is they are all different, false otherwise.


Method ``reduce``
^^^^^^^^^^^^^^^^^^
**Signature** (expressed in BAL)

.. code-block::

    def reduce(reduce:[T,T] -> T, aggregator:T) -> T

**Semantics**:

Apply iteratively the ``reduce`` function to the aggregator and an element of the sequence and save the result in the aggregator.
Then return the aggregator value.

For example:

.. code-block::

    def add(a:int, b:int) -> int { return a + b }

    seq:int[] = int[] {1..3}; // seq = {1, 2, 3}
    val = seq.reduce(add, 0); // val = ((0 + 1) + 2) + 3 = 6

