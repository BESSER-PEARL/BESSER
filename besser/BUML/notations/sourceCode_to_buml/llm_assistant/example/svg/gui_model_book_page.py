
##################################
#      GUI model definition      #
##################################

#ViewComponent definition
viewComponent: ViewComponent=ViewComponent(name="my_component", description= "Detailed Information at a Glance")



# Colors for Buttons
primaryButtonColor = Color(background_color="#0C4B33", text_color="#000000", border_color="#0C4B33")


# size for buttons:
buttonSize: Size = Size(width="120", height="40", padding="10", margin="", font_size="14", icon_size="", unit_size=UnitSize.PIXELS)

# Positioning for floating button
buttonPosition = Position(type=PositionType.Fixed, top="", left="", right="20px",bottom="20px", alignment="", z_index=1000)

buttonStyling = Styling(size=buttonSize, position=buttonPosition, color=primaryButtonColor)

#####  Elements for Book Screen   #####

# Button6:
bookAddingButton: Button = Button(name="Book Add Button", description="Add a book", label="", buttonType= ButtonType.FloatingActionButton, actionType=ButtonActionType.Add, styling=buttonStyling)

#Book DataSource definition
datasource_book: DataSourceElement = DataSourceElement(name="Book Data Source", dataSourceClass= book, fields=[title, pages])

# Book List definitionf
BookList: DataList=DataList(name="BookList", description="A diverse group of books", list_sources={datasource_book}, styling=BookListStyling)

# Book directory screen definition
BookListScreen: Screen = Screen(name="BookListScreen", description="Explore a collection of books",
                          x_dpi="x_dpi", y_dpi="y_dpi", screen_size="Small", view_elements={bookAddingButton, BookList}, layout=ScreenLayout)


# Module definition:
MyModule: Module = Module(name="module_name", screens={BookListScreen})

# Application definition:
gui_model: GUIModel = GUIModel(name="Library Management", package="com.example.librarymanagement", versionCode="1",
                                 versionName="1.0", description="This is a comprehensive web application for managing a library.",
                                 screenCompatibility=True, modules={MyModule})
