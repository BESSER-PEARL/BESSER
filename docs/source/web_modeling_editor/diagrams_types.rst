Diagram Types
=============

The BESSER Web Modeling Editor supports multiple types of diagrams, each designed for specific modeling purposes. This section provides comprehensive documentation for all supported diagram types.

Overview
--------

The Web Modeling Editor allows you to create various types of UML and domain-specific diagrams:

* **Class Diagrams**: Model the static structure of your system
* **Object Diagrams**: Show instances and snapshots of your system
* **State Machine Diagrams**: Model dynamic behavior and state transitions
* **Agent Diagrams**: Design conversational agents and their behaviors

Each diagram type has its own specific elements, notation, and best practices. You can combine multiple diagram types in a single project to create comprehensive models of your system.

Supported Diagram Types
-----------------------

.. toctree::
   :maxdepth: 2
   
   diagram types/class_diagram
   diagram types/object_diagram
   diagram types/statemachine_diagram
   diagram types/agent_diagram

Common Features
------------------

All diagram types in the BESSER Web Modeling Editor share common features:

Navigation and Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Single Click**: Select elements
* **Double Click**: Edit element properties
* **Drag and Drop**: Add elements from the palette or move existing elements
* **Multi-Selection**: Hold Ctrl while clicking to select multiple elements

Editing Operations
~~~~~~~~~~~~~~~~~~~~~~

* **Copy/Paste**: Use Ctrl+C and Ctrl+V
* **Undo/Redo**: Use Ctrl+Z and Ctrl+Y
* **Delete**: Press Delete or Backspace to remove selected elements
* **Properties**: Double-click elements to edit their properties

Visual Customization
~~~~~~~~~~~~~~~~~~~~~~~~

* **Zoom**: Use mouse wheel or zoom controls
* **Pan**: Drag the canvas background to move the view
* **Grid**: Toggle grid display for alignment
* **Snap**: Enable snap-to-grid for precise positioning

Code Generation
~~~~~~~~~~~~~~~~~~~

Most diagram types support code generation:

* **Multiple Languages**: Generate code in Python, Java, SQL, and more
* **Template-Based**: Customizable code generation templates
* **Export Options**: Download generated code as ZIP files
* **Integration**: Generated code follows best practices and conventions

Quality Assurance
~~~~~~~~~~~~~~~~~~~~~~

* **Validation**: Real-time validation of diagram elements
* **Quality Check**: Comprehensive model validation
* **Error Reporting**: Clear error messages and suggestions
* **OCL Support**: Formal constraint validation (where applicable)

Getting Started
------------------

To begin working with diagrams:

1. **Choose Diagram Type**: Select the appropriate diagram type for your modeling needs
2. **Add Elements**: Drag elements from the palette onto the canvas
3. **Configure Properties**: Double-click elements to set their properties
4. **Create Relationships**: Connect elements using the connection tools
5. **Validate Model**: Use the Quality Check feature to ensure correctness
6. **Generate Code**: Export your model as code when ready

Best Practices
-----------------

General Modeling Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Start Simple**: Begin with core elements and add complexity gradually
* **Use Meaningful Names**: Choose descriptive names for all elements
* **Follow Conventions**: Adhere to standard naming conventions for your domain
* **Document Assumptions**: Use comments and constraints to document design decisions
* **Validate Regularly**: Check your model frequently using the validation tools

Layout and Organization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Consistent Spacing**: Maintain consistent spacing between elements
* **Logical Grouping**: Group related elements together
* **Clear Connections**: Ensure relationship lines are clear and unambiguous
* **Readable Labels**: Use readable font sizes and clear labeling

Model Integration
~~~~~~~~~~~~~~~~~~~~~

* **Cross-Diagram Consistency**: Ensure consistency across different diagram types
* **Shared Elements**: Reuse common elements across diagrams where appropriate
* **Incremental Development**: Build models incrementally, adding detail over time
* **Version Control**: Keep track of model versions and changes

Additional Resources
----------------------

For more information about using the BESSER Web Modeling Editor:

* :doc:`../use_the_wme` - General editor usage guide
* :doc:`../deploy_locally` - Local deployment instructions
* `BESSER Documentation <https://besser.readthedocs.io/en/latest/>`_ - Complete BESSER framework documentation
* `WME GitHub Repository <https://github.com/BESSER-PEARL/BESSER_WME_standalone>`_ - Source code and technical details
