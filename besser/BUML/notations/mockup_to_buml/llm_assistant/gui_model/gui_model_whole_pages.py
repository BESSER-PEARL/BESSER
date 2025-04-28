
##################################
#      GUI model definition      #
##################################

#ViewComponent definition
viewComponent: ViewComponent=ViewComponent(name="my_component", description= "Detailed Information at a Glance")


#####  Elements for Library Screen   #####

# Button4
libraryAddingButton: Button = Button(name="Library Add Button", description="Add a library", label="", buttonType= ButtonType.FloatingActionButton, actionType= ButtonActionType.Add)


#LibraryDataSource definition
datasource_library: DataSourceElement = DataSourceElement(name="Library Data Source", dataSourceClass=library, fields=[library_name, address])

#Library List definition
libraryList: DataList=DataList(name="LibraryList", description="A diverse group of libraries", list_sources={datasource_library})


# Library directory screen definition
LibraryListScreen: Screen = Screen(name="LibraryListScreen", description="Explore a collection of libraries",
                          x_dpi="x_dpi", y_dpi="y_dpi", screen_size="Small", view_elements={libraryAddingButton, libraryList})


#####  Elements for Author Screen   #####

# Button5:
authorAddingButton: Button = Button(name="Author Add Button", description="Add an author", label="", buttonType= ButtonType.FloatingActionButton, actionType=ButtonActionType.Add)

#Author DataSource definition
datasource_author: DataSourceElement = DataSourceElement(name="Author Data Source", dataSourceClass= author, fields=[author_name, email])

# Author List definition
authorList: DataList=DataList(name="AuthorList", description="A diverse group of authors", list_sources={datasource_author})

# Author directory screen definition
AuthorListScreen: Screen = Screen(name="AuthorListScreen", description="Explore a collection of authors",
                          x_dpi="x_dpi", y_dpi="y_dpi", screen_size="Small", view_elements={authorAddingButton, authorList})


#####  Elements for Book Screen   #####

# Button6:
bookAddingButton: Button = Button(name="Book Add Button", description="Add a book", label="", buttonType= ButtonType.FloatingActionButton, actionType=ButtonActionType.Add)

#Book DataSource definition
datasource_book: DataSourceElement = DataSourceElement(name="Book Data Source", dataSourceClass= book, fields=[title, pages])

# Book List definition
BookList: DataList=DataList(name="BookList", description="A diverse group of books", list_sources={datasource_book})

# Book directory screen definition
BookListScreen: Screen = Screen(name="BookListScreen", description="Explore a collection of books",
                          x_dpi="x_dpi", y_dpi="y_dpi", screen_size="Small", view_elements={bookAddingButton, BookList})


#####  Elements for Home page Screen   #####

# Button1:
libraryButton: Button = Button(name="Library List Button", description="Explore the libraries", label="Library List", buttonType= ButtonType.RaisedButton, actionType=ButtonActionType.Navigate, targetScreen=LibraryListScreen)


# Button2:
authorButton: Button = Button(name="Author List Button", description="Explore the authors", label="Author List", buttonType= ButtonType.RaisedButton, actionType=ButtonActionType.Navigate, targetScreen=AuthorListScreen)


# Button3:
bookButton: Button = Button(name="Book List Button", description="Explore the books", label="Book List", buttonType= ButtonType.RaisedButton, actionType=ButtonActionType.Navigate, targetScreen=BookListScreen)


# Home page Screen definition
MyHomeScreen: Screen = Screen(name="Book Library Manager", description="Effortlessly manage your books, libraries, and authors, with the ability to view and update their information.",
                          x_dpi="x_dpi", y_dpi="y_dpi", screen_size="Small", view_elements={libraryButton, authorButton, bookButton}, is_main_page=True)


# Module definition:
MyModule: Module = Module(name="module_name", screens={MyHomeScreen, LibraryListScreen, AuthorListScreen, BookListScreen})

# GUIModel definition:
MyApp: GUIModel = GUIModel(name="Library Management", package="com.example.librarymanagement", versionCode="1",
                                 versionName="1.0", description="This is a comprehensive Flutter application for managing a library.",
                                 screenCompatibility=True, modules={MyModule})



django = DjangoGenerator(model=library_model, application=MyApp)
django.generate()




