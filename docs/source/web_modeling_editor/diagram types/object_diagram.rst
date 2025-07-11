Object Diagrams
===============

Object diagrams represent instances of classes from class diagrams at a particular point in time. They show how objects interact and what values they contain, providing a snapshot of the system's state.

Overview
--------

Object diagrams are essential for validating your class design by showing concrete examples of how objects will interact in your system. They work best when used in conjunction with class diagrams to demonstrate real-world scenarios and data flow.

Key Features
------------

* **Instance Representation**: Show specific instances of classes with actual data values
* **Object Relationships**: Display links between objects that represent association instances
* **State Snapshots**: Capture the system state at a specific moment in time
* **Validation Tool**: Help verify that your class design works with real data

Getting Started
---------------

Adding Objects
~~~~~~~~~~~~~~

To add an object to your diagram:

1. Drag and drop the object element from the left panel onto the canvas
2. Objects represent instances of classes from your class diagram
3. Each object should have a meaningful name that reflects its purpose

Editing Objects
~~~~~~~~~~~~~~~

To edit an object:

1. Double-click on the object to open the editing popup
2. Modify the object properties:

   * **Name**: Follow the format ``objectName : ClassName``
   * **Type**: Specify the class this object instantiates
   * **Attribute Values**: Set values using the format ``attributeName = value``

**Example attribute values:**
  
* ``name = "John"`` (string value)
* ``age = 25`` (numeric value)
* ``isActive = true`` (boolean value)

Creating Object Links
~~~~~~~~~~~~~~~~~~~~~

To create links between objects:

1. Select the source object with a single click
2. You'll see blue circles indicating connection points
3. Click and hold on one of these points
4. Drag to another object to create the link
5. Object links represent instances of associations from the class diagram

Editing Object Links
~~~~~~~~~~~~~~~~~~~~

To modify object links:

1. Double-click on the link to open the editing popup
2. You can modify:

   * Link name
   * Associated values or roles
   * Link properties

Working with Object Diagrams
-----------------------------

Common Operations
~~~~~~~~~~~~~~~~~

**Deleting Objects or Links**
  Select the element and press ``Delete`` or ``Backspace``

**Moving Objects**
  Select an object and use arrow keys or drag-and-drop to reposition

**Undo/Redo**
  Use ``Ctrl+Z`` to undo and ``Ctrl+Y`` to redo changes

Best Practices
~~~~~~~~~~~~~~

1. **Meaningful Names**: Use descriptive names for objects that reflect their role
2. **Realistic Values**: Provide realistic attribute values to make diagrams understandable
3. **Consistent Data**: Ensure attribute values match the expected data types
4. **Complete Information**: Include all relevant attribute values for clarity
5. **Clear Relationships**: Make object relationships clear and purposeful

Validation and Quality
~~~~~~~~~~~~~~~~~~~~~~

Object diagrams help validate your class design by:

* Demonstrating concrete usage scenarios
* Revealing potential issues with class relationships
* Testing data flow between objects
* Verifying that your model works with real data

When creating object diagrams, ensure that:

* All objects have proper type assignments
* Attribute values are appropriate for their data types
* Links correctly represent the associations from your class diagram
* The diagram tells a coherent story about system state

Advanced Features
-----------------

Integration with Class Diagrams
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Object diagrams are tightly integrated with class diagrams in the BESSER Web Modeling Editor:

* Objects must reference classes defined in your class diagram
* Links must correspond to associations in the class diagram
* The editor validates consistency between object and class diagrams

Quality Validation
~~~~~~~~~~~~~~~~~~

The editor provides validation features to ensure your object diagram is well-formed:

* Checks that objects reference valid classes
* Validates that links correspond to proper associations
* Ensures attribute values match expected types
* Identifies missing or inconsistent data

**OCL Constraint Validation**

When you click the **Quality Check** button, the editor also validates any OCL (Object Constraint Language) constraints defined in your associated structural (class) diagram:

* Evaluates OCL constraints against the object instances and their attribute values
* Checks invariants, pre-conditions, and post-conditions defined in the class diagram
* Reports constraint violations with detailed error messages
* Helps ensure that your object diagram represents a valid system state

This integration between object diagrams and structural model constraints is powered by `B-OCL <https://b-ocl-interpreter.readthedocs.io/en/latest/>`_, our OCL interpreter, providing comprehensive validation of both structural and instance-level constraints.


Additional Resources
--------------------

For more information about object diagrams and the BESSER Web Modeling Editor:

* `BESSER Documentation <https://besser.readthedocs.io/en/latest/>`_
* `WME GitHub Repository <https://github.com/BESSER-PEARL/BESSER_WME_standalone>`_
* :doc:`../use_the_wme` - General editor usage guide
* :doc:`class_diagram` - Related class diagram documentation
