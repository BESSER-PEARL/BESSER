
##################################
#      GUI model definition      #
##################################



####### Styling definitions ################


#### for screens ##############
ScreenLayout = Layout(type=LayoutType.Flex, orientation="vertical", gap="15px", alignment=JustificationType.Center)


#### for adding buttons##########
AddingButtonColor = Color(background_color="#0C4B33", text_color="#FFFFFF", border_color="#0A3A28")  # Green Add Book Button
AddingbuttonPosition = Position(type=PositionType.Relative, top="", left="", right="20px", bottom="20px", alignment="", z_index=0)
buttonSize: Size = Size(width="", height="40", padding="8px", margin="24", font_size="14", icon_size="", unit_size=UnitSize.PIXELS)
AddButtonStyling = Styling(size=buttonSize, position=AddingbuttonPosition, color=AddingButtonColor) # for adding buttons on each pages



##### for lists ################
ListColor = Color(background_color="#FFFFFF", text_color="#343A40", border_color="#CED4DA")
ListPosition = Position(type=PositionType.Relative, top="200px", left="20px", right="", bottom="", alignment="", z_index=0)
listSize: Size = Size(width="100%", height="auto", padding="10px", margin="", font_size="", icon_size="", unit_size="")
ListStyling = Styling(size=listSize, position=ListPosition, color=ListColor)




###### for Action buttons #######
tableActionButtonPosition = Position(type=PositionType.Inline, alignment="right", z_index=0)
tableActionButtonSize: Size = Size(width="60", height="30", padding="10", margin="", font_size="14", icon_size="", unit_size=UnitSize.PIXELS)


#### for edit buttons##########
editButtonColor = Color(background_color="#3B82F6", text_color="#FFFFFF", border_color="#2563EB")  # Blue color for Edit Button
EditbuttonStyling = Styling(size=tableActionButtonSize, position=tableActionButtonPosition, color=editButtonColor) # for edit buttons on each pages



#### for Delete buttons ##########
deleteButtonColor = Color(background_color="#EF4444", text_color="#FFFFFF", border_color="#DC2626")  # Red color for Delete Button
DeletebuttonStyling = Styling(size=tableActionButtonSize, position=tableActionButtonPosition, color=deleteButtonColor) # for delete buttons on each pages



#####  Elements for Library Screen   #####

# ViewComponent definition
viewComponent: ViewComponent = ViewComponent(name="LibraryListView", description="Display a list of libraries with actions")


# Adding Button
libraryAddingButton: Button = Button(name="LibraryAddButton", description="Add a library", label="Add a library", buttonType= ButtonType.FloatingActionButton, actionType= ButtonActionType.Add, styling=AddButtonStyling)

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


# Module definition (grouping screens together into a module)
MyModule: Module = Module(name="LibraryManagementModule", screens={LibraryListScreen})

# Application definition (defining the overall application structure)
gui_model: GUIModel = GUIModel(name="LibraryManagementApp", package="com.example.librarymanagement", versionCode="1",
                                versionName="1.0", description="A comprehensive web application for managing libraries.",
                                screenCompatibility=True, modules={MyModule})







