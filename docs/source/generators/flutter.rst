Flutter Generator
=====================

BESSER introduces a code generator for `Flutter <https://flutter.dev/>`_ applications. This generator simplifies database access in Flutter by 
providing a user-friendly interface for CRUD operations (Create, Read, Update, Delete).

The Flutter Code Generator seamlessly integrates with the BESSER framework and its BUML metamodel. By supplying a 
:doc:`structural model <../buml_language/model_types/structural>` and a 
:doc:`GUI model <../buml_language/model_types/gui>`, the code generator automatically generates efficient code 
for database interaction. The generated code adheres to established Flutter best practices and follows the provided models.

With the Flutter Code Generator, you can expedite the development process and effortlessly manage database operations 
in your Flutter application, all while maintaining code quality and consistency.

.. image:: ../img/flutter_generator_schema.png
  :width: 1000
  :alt: Flutter Generator user schema
  :align: center

Features of our Flutter Code Generator include:

- **CRUD Operations**: Effortlessly handle Create, Read, Update, and Delete operations with the generated code. Our code generator automatically generates methods based on provided models, adhering to established Flutter best practices.
- **Seamless Database Interaction**: Interact with the database seamlessly, abstracting away low-level implementation details. Our code generator simplifies database access, enabling you to focus on building your Flutter application.
- **Efficient Data Management**: Optimize performance and resource usage with our generated code. It leverages Flutter's data handling capabilities, ensuring efficient data management and a smooth experience when working with the database.

Getting Started
---------------

To start using the Flutter Code Generator, follow these steps:

1. Create your :doc:`structural model <../buml_language/model_types/structural>` and :doc:`GUI model <../buml_language/model_types/gui>` according to the B-UML metamodel.
2. Run the code generator, providing your structural model and GUI model as input. The generator will analyze the models and generate the required code files for your Flutter application.
3. To generate the complete Flutter code for a B-UML model and a GUI model, follow the steps below. 

.. code-block:: python
    
    from besser.generators.gui import FlutterSQLHelperGenerator, FlutterMainDartGenerator, FlutterPubspecGenerator

    code_gen = FlutterSQLHelperGenerator(model = library_model, dataSourceClass=list[Class])
    code_gen.generate()

    code_gen = FlutterMainDartGenerator(model=library_model, application=MyApp, mainPage=MyHomeScreen, module=MyModule)
    code_gen.generate()

    code_gen = FlutterPubspecGenerator(application=MyApp)
    code_gen.generate()
       

The code generator will produce several files, which will be located in the ``<<current_directory>>/output`` directory. These files include:

   + ``main.dart``: This file serves as the entry point for your Flutter application, providing the initial configuration and structure. It includes the necessary dependencies and imports to utilize Flutter's UI components and other functionalities. With main.dart, you can easily customize the starting point of your app, define its visual style, and import essential packages for building a robust and engaging user interface.
   + ``sql_helper.dart``: This file contains helpful functions for managing a SQLite database within your Flutter application. It facilitates operations such as table creation, data retrieval, and data manipulation. With ``sql_helper.dart``, you can seamlessly interact with a database in your Flutter app, enabling efficient data storage and retrieval operations.
   + ``pubspec.yaml``: This file is crucial for dependency management and project configuration in a Flutter application. It allows you to control dependencies, versioning, and other important details. With ``pubspec.yaml``, you can ensure a smooth development process for your Flutter app by easily managing dependencies and defining project-specific information.
 

By incorporating these generated files into your Flutter project, you'll have a solid foundation for building your application, including the necessary configuration, database management capabilities, and dependency management.

You can follow the provided documentation for :doc:`../examples/gui_example` to understand how to run the generated code.