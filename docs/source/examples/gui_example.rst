GUI example
=================

To provide clarity on the capabilities of the Flutter Code Generator, we have included an example that demonstrates its usage. This example consists of detailed instructions and code snippets, guiding you through the seamless integration of the generated code into your Flutter application. By following this example, you will gain a solid understanding of how to utilize the code generator effectively.

The example showcases a common domain involving libraries, books, and authors. The :doc:`../examples/library_example` presents a UML diagram illustrating the relationships between these entities. 
Additionally, the Python code for specifying the B-UML model, including classes, attributes, and relationships, is provided in the same section.

The Python code to specify the GUI model, including various elements such as lists, buttons, screens, and view components is presented in the following code. Additionally, the FlutterMainDartGenerator,
FlutterSQLHelperGenerator, and FlutterPubspecGenerator
code generators are implemented in this example (lines 95_102). Running this script will generate the output/ folder with the main.dart, sql_helper.dart, and pubspec.yaml files produced by each of the Generators respectively.


.. code-block:: python
   
  ##################################
  #      GUI model definition      #
  ##################################

  #ViewComponent definition
  viewComponent: ViewComponent=ViewComponent(name="my_component", description= "Detailed Information at a Glance")


  #####  Elements for Library Screen   #####
  # Button4
  libraryAddingButton: Button = Button(name="Library Add Button", description="Add a library", label="", buttonType= ButtonType.FloatingActionButton, actionType= ButtonActionType.Add)


  #LibraryDataSource definition
  datasource_library: ModelElement = ModelElement(name="Library Data Source", dataSourceClass=library, fields=[library_name, address])

  #Library List definition
  libraryList: DataList=DataList(name="LibraryList", description="A diverse group of libraries", list_sources={datasource_library})


  # Library directory screen definition
  LibraryListScreen: Screen = Screen(name="LibraryListScreen", description="Explore a collection of libraries",
                          x_dpi="x_dpi", y_dpi="y_dpi", size="SmallScreen", view_elements={libraryAddingButton, libraryList})


  #####  Elements for Author Screen   #####
  # Button5:
  authorAddingButton: Button = Button(name="Author Add Button", description="Add an author", label="", buttonType= ButtonType.FloatingActionButton, actionType=ButtonActionType.Add)

  #Author DataSource definition
  datasource_author: ModelElement = ModelElement(name="Author Data Source", dataSourceClass= author, fields=[author_name, email])

  # Author List definition
  authorList: DataList=DataList(name="AuthorList", description="A diverse group of authors", list_sources={datasource_author})

  # Author directory screen definition
  AuthorListScreen: Screen = Screen(name="AuthorListScreen", description="Explore a collection of authors",
                          x_dpi="x_dpi", y_dpi="y_dpi", size="SmallScreen", view_elements={authorAddingButton, authorList})

  #####  Elements for Book Screen   #####
  # Button6:
  bookAddingButton: Button = Button(name="Book Add Button", description="Add a book", label="", buttonType= ButtonType.FloatingActionButton, actionType=ButtonActionType.Add)

  #Book DataSource definition
  datasource_book: ModelElement = ModelElement(name="Book Data Source", dataSourceClass= book, fields=[title, pages, release])

  # Book List definition
  BookList: DataList=DataList(name="BookList", description="A diverse group of books", list_sources={datasource_book})

  # Book directory screen definition
  BookListScreen: Screen = Screen(name="BookListScreen", description="Explore a collection of books",
                          x_dpi="x_dpi", y_dpi="y_dpi", size="SmallScreen", view_elements={bookAddingButton, BookList})
                          

  #####  Elements for Home page Screen   #####

  # Button1:
  libraryButton: Button = Button(name="Library List Button", description="Explore the libraries", label="Library List", buttonType= ButtonType.RaisedButton, actionType=ButtonActionType.Navigate, targetScreen=LibraryListScreen)


  # Button2:
  authorButton: Button = Button(name="Author List Button", description="Explore the authors", label="Author List", buttonType= ButtonType.RaisedButton, actionType=ButtonActionType.Navigate, targetScreen=AuthorListScreen)


  # Button3:
  bookButton: Button = Button(name="Book List Button", description="Explore the books", label="Book List", buttonType= ButtonType.RaisedButton, actionType=ButtonActionType.Navigate, targetScreen=BookListScreen)


  # Home page Screen definition
  MyHomeScreen: Screen = Screen(name="Book Liibrary Manager", description="Effortlessly manage your books, libraries, and authors, with the ability to view and update their information.",
                          x_dpi="x_dpi", y_dpi="y_dpi", size="SmallScreen", view_elements={libraryButton, authorButton, bookButton})


  # Module definition:
  MyModule: Module = Module(name="module_name", screens={MyHomeScreen, LibraryListScreen, AuthorListScreen, BookListScreen})

  # Application definition:
  MyApp: Application = Application(name="Library Management", package="com.example.librarymanagement", versionCode="1",
                                 versionName="1.0", description="This is a comprehensive Flutter application for managing a library.",
                                 screenCompatibility=True, modules={MyModule})

  code_gen = FlutterSQLHelperGenerator(model=library_model, dataSourceClass=list[Class])
  code_gen.generate()

  code_gen = FlutterMainDartGenerator(model=library_model, application=MyApp, mainPage=MyHomeScreen, module=MyModule)
  code_gen.generate()

  code_gen = FlutterPubspecGenerator(application=MyApp)
  code_gen.generate()


After generating these files, you will need to incorporate them into your Flutter application. 
Please ensure that you create an app with the same name as specified in the GUI model. To do so, follow these steps:

1. Create a new Flutter application with the desired app name.
2. Locate the sql_helper.dart file generated by the Flutter Code Generator.
3. Copy the sql_helper.dart file into the lib folder of your Flutter application.
4.	Locate the existing main.dart file in the lib folder of your Flutter application.
5.	Replace the existing main.dart file with the generated main.dart file from the Flutter Code Generator.
6.	Locate the existing pubspec.yaml file in the root directory of your Flutter application.
7.	Replace the existing pubspec.yaml file with the generated pubspec.yaml file.

After completing these steps, your Flutter application should have the following structure:

.. image:: ../img/app_structure.png
  :width: 300
  :alt: Flutter aap structure
  :align: center


