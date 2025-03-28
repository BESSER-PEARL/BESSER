
##################################
#      GUI model definition      #
##################################

#ViewComponent definition
viewComponent: ViewComponent=ViewComponent(name="my_component", description= "Detailed Information at a Glance")


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


# Module definition:
MyModule: Module = Module(name="module_name", screens={AuthorListScreen})

# Application definition:
MyApp: Application = Application(name="Library Management", package="com.example.librarymanagement", versionCode="1",
                                 versionName="1.0", description="This is a comprehensive Flutter application for managing a library.",
                                 screenCompatibility=True, modules={MyModule})







