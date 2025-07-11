Class Diagrams
==============

Class diagrams are the fundamental building blocks of object-oriented modeling. They show the static structure of a system by displaying classes, their attributes, methods, and the relationships between them.

Overview
--------

Class diagrams provide a visual representation of the structure of your system. They are essential for designing and documenting object-oriented systems, showing how classes relate to each other and what responsibilities each class has.

Key Features
------------

* **Class Definition**: Define classes with attributes and methods
* **Relationships**: Show associations, generalizations, and other relationships
* **Visibility**: Specify public, private, and protected members
* **Multiplicity**: Define how many instances can participate in relationships
* **OCL Constraints**: Add formal constraints to your model

Getting Started
---------------

Adding Classes
~~~~~~~~~~~~~~

To add a class to your diagram:

1. Drag and drop a class element from the left panel onto the canvas
2. Classes represent the main entities in your system
3. Each class should have a meaningful name that reflects its purpose

Editing Classes
~~~~~~~~~~~~~~~

To edit a class:

1. Double-click on the class to open the editing popup
2. Modify the class properties:

   * **Name**: Provide a clear, descriptive class name
   * **Attributes**: Define the data the class holds
   * **Methods**: Define the behavior the class provides

**Attribute Format:**

Attributes can be specified using various formats:

* ``+ attribute : type`` - Public attribute with specified type
* ``+ attribute`` - Public attribute with default string type
* ``attribute`` - Public attribute (default visibility)

**Supported Types:**
* Primitive types: ``int``, ``float``, ``str``, ``bool``, ``time``, ``date``, ``datetime``, ``timedelta``, ``any``
* Class types: Any class defined in your model
* Enum types: Custom enumeration types

**Visibility Modifiers:**
* ``+`` - Public (default)
* ``-`` - Private
* ``#`` - Protected

**Method Format:**

Methods can be defined with parameters and return types:

* ``+ notify(sms: str = 'message')`` - Public method with parameter and default value
* ``- findBook(title: str): Book`` - Private method with parameter and return type
* ``validate()`` - Public method without parameters

Creating Associations and Generalizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create relationships between classes:

1. Select the source class with a single click
2. You'll see blue circles indicating connection points
3. Click and hold on one of these points
4. Drag to another class to create the relationship

**Multiplicity Format:**

Define multiplicity using standard UML notation:
* ``1`` - Exactly one
* ``0..1`` - Zero or one
* ``0..*`` - Zero or many
* ``1..*`` - One or many
* ``2..4`` - Between 2 and 4

Editing Associations and Generalizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To edit a relationship:

1. Double-click on the relationship line to open the editing popup
2. You can modify:

   * **Association Type**: Unidirectional, Bidirectional, Composition
   * **Generalization**: Change to inheritance relationship
   * **Name**: Assign a name to the association
   * **End Names**: Set source and target end names
   * **Multiplicity**: Modify multiplicity at both ends

Working with Class Diagrams
---------------------------

Common Operations
~~~~~~~~~~~~~~~~~

**Deleting Classes or Relationships**
  Select the element and press ``Delete`` or ``Backspace``

**Moving Classes**
  Select a class and use arrow keys or drag-and-drop to reposition

**Undo/Redo**
  Use ``Ctrl+Z`` to undo and ``Ctrl+Y`` to redo changes

OCL Constraints
~~~~~~~~~~~~~~~

Object Constraint Language (OCL) constraints can be added to validate your model:

1. Drag and drop the OCL shape onto your canvas
2. Write constraints using the format: ``Context "class name" ...``
3. Link constraints to classes using dotted lines
4. Use the Quality Check button to validate syntax

**Example OCL Constraint:**
```
Context "Person"
inv: self.age >= 0 and self.age <= 120
```

Association Classes
~~~~~~~~~~~~~~~~~~~

Association classes combine an association with a class:

1. Drag and drop a Class shape onto the canvas
2. Link it to an existing association center point using a dotted line
3. Define attributes for the Association Class like a regular class


Code Generation
~~~~~~~~~~~~~~~

Class diagrams can be used to generate code:

* **Python Classes**: Generate Python class definitions
* **Java Classes**: Generate Java class files
* **SQL Schema**: Generate database schemas
* **JSON Schema**: Generate JSON schema definitions
* **Pydantic Classes**: Generate Pydantic models for data validation


Additional Resources
--------------------

For more information about class diagrams and the BESSER Web Modeling Editor:

* `BESSER Documentation <https://besser.readthedocs.io/en/latest/>`_
* `WME GitHub Repository <https://github.com/BESSER-PEARL/BESSER_WME_standalone>`_
* :doc:`../use_the_wme` - General editor usage guide
* :doc:`object_diagram` - Object diagram documentation
* :doc:`statemachine_diagram` - State machine diagram documentation
* :doc:`agent_diagram` - Agent diagram documentation
