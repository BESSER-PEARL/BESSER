
##################################
#      GUI model definition      #
##################################

#ViewComponent definition
viewComponent: ViewComponent=ViewComponent(name="my_component", description= "Detailed Information at a Glance")


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
MyModule: Module = Module(name="module_name", screens={MyHomeScreen})

# GUIModel definition:
MyApp: GUIModel = GUIModel(name="Library Management", package="com.example.librarymanagement", versionCode="1",
                                 versionName="1.0", description="This is a comprehensive web application for managing a library.",
                                 screenCompatibility=True, modules={MyModule})


