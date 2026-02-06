Overview of BAL
=========================

Purpose and Scope
-------------------------

The BESSER Action Language (BAL) is a statically typed, model-driven action language designed to express behavioral logic over B-UML models.
It is intended to define method bodies, computations, and control flow in a way that is:

* Closely aligned with a structural model (classes, properties, enumerations)
* Statically analyzable via a rich type system
* Executable, through code generation for the model generation target

BAL targets behavioral modeling within low-code and model-driven engineering workflows with the goal of providing correctness, traceability, and analyzability

Core Abstractions
-------------------------

The language is structured around the following core concepts:

**Statements:** units of execution (assignments, loops, conditionals, returns)

**Expressions:** typed computations producing values

**Typed declarations:** variables, parameters, and functions

**Functions:** the primary executable units, BAL treats functions as first-class citizens

Functions
^^^^^^^^^^^^^^^^^^^^^^^^^

Functions are first-class constructs, strongly typed and support optional parameters through default value.
The syntax to define a function is the following:

.. code-block::

    def function_name(param1: int, param2: int = 0) -> nothing {
        ...
    }

As first-class citizen, function can be referenced by their name without parenthesis:

.. code-block::

    def function_name(param1: int, param2: int = 0) -> nothing {
        def inner() -> nothing {
            ...
        }
        inner(); //call the function
        take_function_param(inner); //pass the function as parameter
    }

Statements and Control Flow
^^^^^^^^^^^^^^^^^^^^^^^^^

BAL supports structured, imperative control flow.
Similarly to most imperative languages, the main building blocks of BAL are statements.

Part of those statements represent control flow, such as condition, loop, or return statements.
Other statements include function definition, presented above, assignment and expressions.

Conditions
""""""""""""""""""""""""

Condition are composed of an expression, that must evaluate to a boolean, and a block of statements that are executed when the condition evaluates to true.
Optionally, the condition can define an else part that is executed when the condition evaluate to false.
The else part can either be a block of statements or an if statement alone, allowing the definition of else-if structure.

The syntax to define condition is the following:
.. code-block::

    if(condition){
        ...
    }
    // or
    if(condition){
        ...
    } else {
        ...
    }
    // or
    if(condition){
        ...
    } else if(condition) {
        ...
    }

Loops
""""""""""""""""""""""""

BAL support three types of loops: while loops, do-while loops, and for loops.

**While loops** are composed of a condition and a block of statements.
When executed, we first evaluate the condition.
If the condition is true we execute the block of statements and go back to the condition evaluation.
If the condition is false the evaluation of the loop stops and execution proceeds with the next statement.
In BAL, while loop are expressed as follows:

.. code-block::

    while(condition){
        ...
    }

**Do-While loops** are also composed of a condition and a block of statements.
But when executed, we first evaluate the block of statements and then the condition.
If the condition is true we go back to the beginning.
If the condition is false the evaluation of the loop stops and execution proceeds with the next statement.
In BAL, do-while loop are expressed as follows:

.. code-block::

    do{
        ...
    } while(condition)

Finally, **For loops** are composed of one or more iterators and a block of statements.
Iterators are in the form "``elem in sequence``" and declare a variable (here "elem") that will take successively all the values contained in a sequence (here "sequence") for each iteration.
If multiple iterators are specified in the for loop, all iterator will progress at the same time, effectively accessing elements at the same index in all sequences.
Furthermore, the number of iteration performed will always be equal to the number of elements of the shortest sequence.
The syntax to define for loops in BAL is the following:

.. code-block::

    for(x in xs, y in ys){
        ...
    }

Return statement
""""""""""""""""""""""""
The final control flow related statement is the return statement.
As the name indicate, this statements allows to express the return value of a function.
When a return statement is executed the function is exited and return the specified value.
This statement can be expressed as follows:

.. code-block::

    return expression;


Assignments
^^^^^^^^^^^^^^^^^^^^^^^^^

Another type of statement available are assignments.
They allow to store value or modify the model elements.
An assignment is composed of an expression that will be assigned and a target that will take the value of the expression.
There is four types of valid target for an assignment: variable declarations, variable references, field accessors and sequence accessors.

.. note::

    In BAL variable declaration can be explicitly typed or implicitly typed. Explicit declarations define the type of the variable, while implicit declarations infer the type of the variable based on its initial value.

These possible assignments take the following forms:

.. code-block::

    var1: int = 0;          // Explicit declaration
    var2 = "string type";   // Implicit declaration
    object.field = 0;       // Field accessor
    sequence[0] = 0;        // Sequence accessor
    var1 = 0;               // Variable reference


Expressions
^^^^^^^^^^^^^^^^^^^^^^^^^

In BAL, expressions describe computations and yield values.
BAL provides a set of standard expressions including among others arithmetic, boolean and comparison expressions.
In addition, datastructures traversal and method calls are also considered expressions.

.. note::
    Expressions can be used as statements (allowing to execute function with side-effect), by adding a semi-colon (;) at the end.

Basic expressions
""""""""""""""""""""""""""
Basic expressions provided by BAL includes:
* Binary expressions
    * **Arithmetic:** +, -, *, /, %
    * **Boolean:** &&, ||
    * **Comparison:** ==, !=, <, >, <=, >=
* Unary expression (prefix notation)
    * **Arithmetic:** -
    * **Boolean:** !

These expressions can be expressed as follows (replace op by the appropriate operator):
.. code-block::

    expression op expression // Binary expression
    op expr                  // Unary expression

Other types of basic expressions include:
* Function calls
* Ternary expressions
* Null-coalescing
* Type casts
* Instanceof checks

These expressions can be expressed as follows:
.. code-block::

    function_name(parameter)
    condition ? true_expression : false_expression
    expression ?? if_null_expression
    (Type) expression
    expression instanceof Type

Object-Oriented Features
""""""""""""""""""""""""""

The language is explicitly object-oriented and model-aware.
This allows model navigation, method execution and instance creation

Model navigation, and more generally datastructures navigation, is done through instance access, field access and sequence access:

.. code-block::

    this            // Object instance
    this.field      // Field access
    sequence[index] // Sequence access

Method execution can be done using a notation similar to field access with additional parentheses for argument passing.
.. code-block::

    this.method(expression1, expression2)

.. note::

    Methods and function are two different concept in BAL, as methods apply on object, they are not considered as first class citizen.

Finally, instance creation is done using the "new" notation (i.e., as Java) specifying the name of the class to create and the parameters of the constructor.

.. code-block::

    new ClassName(expression1, expression2)


These features allow BAL to directly manipulate instances of B-UML model elements.

Type System
-------------------------

BAL is a strongly typed language with a type system revolving around a set of primitive types, sequence types, model derived types and function types

Primitive and Sequence Types
^^^^^^^^^^^^^^^^^^^^^^^^
Primitive types and sequences are directly embedded in the base language.
Primitive types include: int, float, string, bool, nothing and any.
These types can directly instantiated in the form of literals (e.g. ``1``, ``1.0``, ``"text"``, ``true``, ``null``)

.. note::
    The "any" type does not have a literal as it represent the union of all types, any literal is a valid any literal

On the other hand, the sequence type is a parametric type denoting a sequence of a particular type.
For instance, ``int[]`` and ``MyClass[]`` are both sequence types with elements of type ``int`` and ``MyClass`` respectively.
As for primitive types, sequence types can be instantiated using a literal taking expressions for each of its elements.
This can be expressed as follows:

.. code-block::

    int[] { 0, 1, 2, 1 + 2 }

Sequence of type ``int[]`` containing successive values can also be described using the range literal

.. code-block::

    int[] {0..3} // yields the same sequence


Model Types
^^^^^^^^^^^^^^^^^^^^^^^^
In addition to the primitive and sequence types, types can be referenced from the model itself.
Object types reference B-UML Class elements, while Enumeration types reference B-UML Enumerations.
Enumeration values can be instantiated using an enumeration literal denoted as a field access on the enumeration type.
While objects have no literal per se, they can be created using their constructor (using new).

.. code-block::

    obj: ModelClass = new ModelClass()
    enum: ModelEnumeration = ModelEnumeration.literal

.. note::
    When typechecking BAL, the object types follows the subtyping hierarchies expressed in the model.
    This ensure that object can be assigned to variables and passed as parameter even if their supertype was specified.

Function types
^^^^^^^^^^^^^^^^^^^^^^^^
Function types capture the type of defined functions in terms of their parameter types and return types.
In BAL they are expressed as follows:

.. code-block::

    def func(param:int, opt:int = 0) -> nothing { ... }
    f: [int, int?] -> nothing = func

In this example the function ``func`` is of type ``[int, int?] -> nothing``.
Between the square brackets are the types of its parameters, while on the right of the arrow is its return type.

You may have been intrigued by the ``int?`` type. This is an "optional int" type.
Optional types allows to denote parameters with default value, thus not requiring an actual value when the function is called.
This notation allows to better manage function subtyping and flexibility when defining variable types or parameter types for functions.
For instance :

.. code-block::

    def func(param:int, opt:int = 0) -> nothing { ... }
    f: [int, int?] -> nothing = func    // Matching types
    g: [int] -> nothing = func          // Matching types
    h: [int, int] -> nothing = func     // Matching types
    i: [int, string] -> nothing = func  // Typechecking error

Furthermore, BAL also manage function subtyping checking parameters type contravariance and returns type covariance.

.. code-block::

    def func(param:any) -> int { ... }
    f: [int] -> any = func    // Matching types


Future Features
-------------------------

In the next versions, we plan to introduce new concepts such as:
* Exception handling
* Imports
