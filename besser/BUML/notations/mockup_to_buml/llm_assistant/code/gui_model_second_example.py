
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


# Module definition:
MyModule: Module = Module(name="module_name", screens={LibraryListScreen})

# GUIModel definition:
MyApp: GUIModel = GUIModel(name="Library Management", package="com.example.librarymanagement", versionCode="1",
                                 versionName="1.0", description="This is a comprehensive web application for managing a library.",
                                 screenCompatibility=True, modules={MyModule})



