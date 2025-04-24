
##################################
#      GUI model definition      #
##################################

#ViewComponent definition
viewComponent: ViewComponent=ViewComponent(name="my_component", description= "Detailed Information at a Glance")






#####  Elements for Book Screen   #####

# Button6:
bookAddingButton: Button = Button(name="Book Add Button", description="Add a book", label="", buttonType= ButtonType.FloatingActionButton, actionType=ButtonActionType.Add)

#Book DataSource definition
datasource_book: ModelElement = ModelElement(name="Book Data Source", dataSourceClass= book, fields=[title, pages])

# Book List definition
BookList: DataList=DataList(name="BookList", description="A diverse group of books", list_sources={datasource_book})

# Book directory screen definition
BookListScreen: Screen = Screen(name="BookListScreen", description="Explore a collection of books",
                          x_dpi="x_dpi", y_dpi="y_dpi", size="SmallScreen", view_elements={bookAddingButton, BookList})
                          

# Module definition:
MyModule: Module = Module(name="module_name", screens={BookListScreen})

# Application definition:
MyApp: Application = Application(name="Library Management", package="com.example.librarymanagement", versionCode="1",
                                 versionName="1.0", description="This is a comprehensive Flutter application for managing a library.",
                                 screenCompatibility=True, modules={MyModule})

