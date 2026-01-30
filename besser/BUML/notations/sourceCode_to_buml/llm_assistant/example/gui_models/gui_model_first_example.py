
##################################
#      GUI model definition      #
##################################



####### Styling definitions ################

#### for screens ##############
ScreenLayout = Layout(type=LayoutType.Flex, orientation="vertical", gap="15px", alignment=JustificationType.Center)

#### for buttons on main page #######
ButtonColor = Color(background_color="#007BFF", text_color="#FFFFFF", border_color="#0056b3") # Blue color for Buttons on main page
buttonPosition = Position(type=PositionType.Relative, top="", left="", right="",bottom="", alignment="", z_index=10)
buttonSize: Size = Size(width="", height="40", padding="8px", margin="24", font_size="14", icon_size="", unit_size=UnitSize.PIXELS)
buttonStyling = Styling(size=buttonSize, position=buttonPosition, color=ButtonColor) # for buttons on main page



# ViewComponent definition
viewComponent: ViewComponent = ViewComponent(name="BookListView", description="Display a list of books with actions")



#####  Elements for Home page Screen   #####

# Button1:
libraryButton: Button = Button(name="LibraryListButton", description="Explore the libraries", label="Library List", buttonType= ButtonType.RaisedButton, actionType=ButtonActionType.Navigate, targetScreen=LibraryListScreen, styling=buttonStyling)


# Button2:
authorButton: Button = Button(name="AuthorListButton", description="Explore the authors", label="Author List", buttonType= ButtonType.RaisedButton, actionType=ButtonActionType.Navigate, targetScreen=AuthorListScreen, styling=buttonStyling)


# Button3:
bookButton: Button = Button(name="BookListButton", description="Explore the books", label="Book List", buttonType= ButtonType.RaisedButton, actionType=ButtonActionType.Navigate, targetScreen=BookListScreen, styling=buttonStyling)


# Home page Screen definition
MyHomeScreen: Screen = Screen(name="BookLibraryManager", description="Effortlessly manage your books, libraries, and authors, with the ability to view and update their information.",
                          x_dpi="x_dpi", y_dpi="y_dpi", screen_size="Medium", view_elements={libraryButton, authorButton, bookButton}, is_main_page= True, layout=ScreenLayout)


# Module definition:
MyModule: Module = Module(name="ModuleName", screens={MyHomeScreen, LibraryListScreen, AuthorListScreen, BookListScreen})

# Application definition:
gui_model: GUIModel = GUIModel(name="LibraryManagement", package="com.example.librarymanagement", versionCode="1",
                                 versionName="1.0", description="This is a comprehensive web application for managing a library.",
                                 screenCompatibility=True, modules={MyModule})








