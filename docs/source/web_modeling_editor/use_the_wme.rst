Using the Web Modeling Editor
=============================

This guide explains how to use the BESSER Web Modeling Editor effectively.

Accessing the Editor
----------------------

You can access the BESSER Web Modeling Editor in two ways:

1. Public online version: Visit `editor.besser-pearl.org <https://editor.besser-pearl.org>`_ in your web browser
2. Local deployment: Deploy the editor locally by following the instructions in :doc:`./deploy_locally` 

The BESSER web editor provides a graphical dashboard to create two types of B-UML models:

* Class diagram or :doc:`structural model <./../buml_language/model_types/structural>`
* :doc:`State Machine diagram <./../buml_language/model_types/state_machine>`


To help you get started quickly:

1. Click on "File" in the main menu
2. Select "Start from template" 
3. Choose from the available template options

Templates provide ready-made examples that you can modify according to your needs, helping you understand the editor's capabilities and speeding up your modeling process.

Structural Diagram Modeling and Code Generation
------------------------------------------------

1. Select "Class Diagram" from the diagram type options.
2. Use the toolbar to add elements:

   * **Classes**: Create new classes and define their attributes and operations
   * **Relationships**: Create associations, inheritances, and dependencies between classes
   * **Attributes**: Add properties to your classes with specific types
   * **Operations**: Define methods with parameters and return types

3. For relationships, define multiplicity using the following format: 1, 0..1, 0..*, 1..*, 2..4, etc. (Default is 1)
4. For attributes, specify the type as: ``+ attribute : type`` where type can be primitive (int, float, string) or a class/enum type
5. Add OCL constraints to validate the model's semantics
6. Export or Generate code directly from your diagram:

   * Select the BESSER Generator type (Python classes, Backend, etc.)
   * Click "Generate/Download" in the toolbar
   * Review and download the generated code files

State Machine Modeling
----------------------

1. Select "State Machine Diagram" from the diagram type options.
2. Use the toolbar to add state machine elements:

   * **States**: Create states with names and properties
   * **Transitions**: Connect states with events and conditions
   * **Initial/Final States**: Mark the beginning and end of your state machine

3. Export your state machine as a B-UML or JSON file for further use.

Save & Share
------------

The editor provides functionality to save your diagram for later use and share it with collaborators.

To use the collaboration feature:

1. Click the "Save & Share" button in the top toolbar
2. A unique URL will be generated for your current diagram
3. Share this URL with your collaborators
4. Multiple users can edit the diagram simultaneously with changes synchronized in real-time
5. Each user's cursor is visible with their name displayed to avoid conflicts
6. All diagrams created in collaboration mode are automatically stored in the database

General Features
----------------

* Real-time collaboration with automatic saving and synchronization
* Export/Import diagrams as B-UML or JSON files
* Toggle between dark and light mode for comfortable editing