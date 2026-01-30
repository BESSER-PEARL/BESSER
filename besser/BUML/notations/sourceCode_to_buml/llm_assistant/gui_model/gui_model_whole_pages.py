



##################################
#      GUI model definition      #
##################################

#### for adding buttons##########
AddingButtonColor = Color(background_color="#4B0C38", text_color="#FFFFFF", border_color="#0056b3")  # Green Add Book Button
AddingbuttonPosition = Position(type=PositionType.Relative, top="", left="", right="20px", bottom="20px", alignment="", z_index=1000)
buttonSize: Size = Size(width="120", height="40", padding="10", margin="", font_size="14", icon_size="", unit_size=UnitSize.PIXELS)
AddButtonStyling = Styling(size=buttonSize, position=AddingbuttonPosition, color=AddingButtonColor) # for adding buttons on each pages


##### for lists ################
ListColor = Color(background_color="#F8F9FA", text_color="#343A40", border_color="#CED4DA")
ListPosition = Position(type=PositionType.Relative, top="50px", left="20px", right="", bottom="", alignment="center", z_index=0)
listSize: Size = Size(width="100%", height="auto", padding="10px", margin="", font_size="", icon_size="", unit_size="")
ListStyling = Styling(size=listSize, position=ListPosition, color=ListColor)


#### for screens ##############
ScreenLayout = Layout(type=LayoutType.Flex, orientation="vertical", gap="15px", alignment=JustificationType.Center)



#### for edit buttons##########
editButtonColor = Color(background_color="#3B82F6", text_color="#FFFFFF", border_color="#2563EB")  # Blue color for Edit Button
tableActionButtonPosition = Position(type=PositionType.Inline, alignment="right", z_index=0)
EditbuttonStyling = Styling(size=buttonSize, position=tableActionButtonPosition, color=editButtonColor) # for edit buttons on each pages



#### for Delete buttons ##########
deleteButtonColor = Color(background_color="#EF4444", text_color="#FFFFFF", border_color="#DC2626")  # Red color for Delete Button
DeletebuttonStyling = Styling(size=buttonSize, position=tableActionButtonPosition, color=deleteButtonColor) # for delete buttons on each pages




#### for buttons on main page #######
ButtonColor = Color(background_color="#007BFF", text_color="#FFFFFF", border_color="#0056b3") # Blue color for Buttons on main page
buttonPosition = Position(type=PositionType.Relative, top="", left="", right="",bottom="", alignment="", z_index=10)
buttonStyling = Styling(size=buttonSize, position=buttonPosition, color=ButtonColor) # for buttons on main page

#ViewComponent definition
viewComponent: ViewComponent=ViewComponent(name="MyComponent", description= "Detailed Information at a Glance")


#####  Elements for Library Screen   #####

# Button4
libraryAddingButton: Button = Button(name="LibraryAddButton", description="Add a library", label="", buttonType= ButtonType.FloatingActionButton, actionType= ButtonActionType.Add, styling=AddButtonStyling)

# Button for editing a library
libraryEditButton: Button = Button(name="LibraryEditButton", description="Edit a library", label="Edit", buttonType=ButtonType.RaisedButton, actionType=ButtonActionType.Edit, styling=EditbuttonStyling)


# Button for deleing a library
librarydeleteButton: Button = Button(name="libraryDeleteButton", description="Delete a library", label="Delete", buttonType=ButtonType.RaisedButton, actionType=ButtonActionType.Delete, styling=DeletebuttonStyling)



#LibraryDataSource definition
datasource_library: DataSourceElement = DataSourceElement(name="LibraryDataSource", dataSourceClass=library, fields=[library_name, address])

#Library List definition
libraryList: DataList=DataList(name="LibraryList", description="A diverse group of libraries", list_sources={datasource_library}, styling=ListStyling)


# Library directory screen definition
LibraryListScreen: Screen = Screen(name="LibraryListScreen", description="Explore a collection of libraries",
                          x_dpi="x_dpi", y_dpi="y_dpi", screen_size="Medium", view_elements={libraryAddingButton, libraryList, libraryEditButton, librarydeleteButton}, layout=ScreenLayout)


#####  Elements for Author Screen   #####

# Button5:
authorAddingButton: Button = Button(name="AuthorAddButton", description="Add an author", label="", buttonType= ButtonType.FloatingActionButton, actionType=ButtonActionType.Add, styling=AddButtonStyling)


# Button for editing a author
authorEditButton: Button = Button(name="AuthorEditButton", description="Edit a author", label="Edit", buttonType=ButtonType.RaisedButton, actionType=ButtonActionType.Edit, styling=EditbuttonStyling)

# Button for deleting a author
authordeleteButton: Button = Button(name="AuthorDeleteButton", description="Delete a author", label="Delete", buttonType=ButtonType.RaisedButton, actionType=ButtonActionType.Delete, styling=DeletebuttonStyling)


#Author DataSource definition
datasource_author: DataSourceElement = DataSourceElement(name="AuthorDataSource", dataSourceClass= author, fields=[author_name, email])

# Author List definition
authorList: DataList=DataList(name="AuthorList", description="A diverse group of authors", list_sources={datasource_author}, styling=ListStyling)

# Author directory screen definition
AuthorListScreen: Screen = Screen(name="AuthorListScreen", description="Explore a collection of authors",
                          x_dpi="x_dpi", y_dpi="y_dpi", screen_size="Medium", view_elements={authorAddingButton, authorList, authorEditButton, authordeleteButton}, layout=ScreenLayout)


#####  Elements for Book Screen   #####

# Button6:
bookAddingButton: Button = Button(name="BookAddButton", description="Add a book", label="", buttonType= ButtonType.FloatingActionButton, actionType=ButtonActionType.Add, styling=AddButtonStyling)

# Button for editing a book
bookEditButton: Button = Button(name="BookEditButton", description="Edit a book", label="Edit", buttonType=ButtonType.RaisedButton, actionType=ButtonActionType.Edit, styling=EditbuttonStyling)


# Button for deleting a book
bookdeleteButton: Button = Button(name="BookDeleteButton", description="Delete a book", label="Delete", buttonType=ButtonType.RaisedButton, actionType=ButtonActionType.Delete, styling=DeletebuttonStyling)



#Book DataSource definition
datasource_book: DataSourceElement = DataSourceElement(name="BookDataSource", dataSourceClass= book, fields=[title, pages, release])

# Book List definition
BookList: DataList=DataList(name="BookList", description="A diverse group of books", list_sources={datasource_book}, styling=ListStyling)

# Book directory screen definition
BookListScreen: Screen = Screen(name="BookListScreen", description="Explore a collection of books",
                          x_dpi="x_dpi", y_dpi="y_dpi", screen_size="Medium", view_elements={bookAddingButton, BookList, bookEditButton, bookdeleteButton}, layout=ScreenLayout)


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













