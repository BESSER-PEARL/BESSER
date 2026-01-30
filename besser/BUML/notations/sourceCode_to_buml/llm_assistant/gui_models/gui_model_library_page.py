
##################################
#      GUI model definition      #
##################################

#### for adding button ##########
AddingButtonColor = Color(background_color="#4B0C38", text_color="#FFFFFF", border_color="#0056b3")  # Green Add Book Button
AddingbuttonPosition = Position(type=PositionType.Relative, top="", left="", right="20px", bottom="20px", alignment="", z_index=1000)
buttonSize: Size = Size(width="120", height="40", padding="10", margin="", font_size="14", icon_size="", unit_size=UnitSize.PIXELS)
AddButtonStyling = Styling(size=buttonSize, position=AddingbuttonPosition, color=AddingButtonColor) # for adding buttons on each pages


##### for list ################
ListColor = Color(background_color="#F8F9FA", text_color="#343A40", border_color="#CED4DA")
ListPosition = Position(type=PositionType.Relative, top="50px", left="20px", right="", bottom="", alignment="center", z_index=0)
listSize: Size = Size(width="100%", height="auto", padding="10px", margin="", font_size="", icon_size="", unit_size="")
ListStyling = Styling(size=listSize, position=ListPosition, color=ListColor)


#### for screen ##############
ScreenLayout = Layout(type=LayoutType.Flex, orientation="vertical", gap="15px", alignment=JustificationType.Center)



#### for edit button ##########
editButtonColor = Color(background_color="#3B82F6", text_color="#FFFFFF", border_color="#2563EB")  # Blue color for Edit Button
tableActionButtonPosition = Position(type=PositionType.Inline, alignment="right", z_index=0)
EditbuttonStyling = Styling(size=buttonSize, position=tableActionButtonPosition, color=editButtonColor) # for edit buttons on each pages



#### for Delete button ##########
deleteButtonColor = Color(background_color="#EF4444", text_color="#FFFFFF", border_color="#DC2626")  # Red color for Delete Button
DeletebuttonStyling = Styling(size=buttonSize, position=tableActionButtonPosition, color=deleteButtonColor) # for delete buttons on each pages





 # ViewComponent definition
viewComponent: ViewComponent=ViewComponent(name="MyComponent", description= "Detailed Information at a Glance")


#####  Elements for Library Screen   #####

# Button4
libraryAddingButton: Button = Button(name="LibraryAddButton", description="Add a library", label="", buttonType= ButtonType.FloatingActionButton, actionType= ButtonActionType.Add, styling=AddButtonStyling)

# Button for editing a library
libraryEditButton: Button = Button(name="LibraryEditButton", description="Edit a library", label="Edit", buttonType=ButtonType.RaisedButton, actionType=ButtonActionType.Edit, styling=EditbuttonStyling)


# Button for deleting a library
librarydeleteButton: Button = Button(name="libraryDeleteButton", description="Delete a library", label="Delete", buttonType=ButtonType.RaisedButton, actionType=ButtonActionType.Delete, styling=DeletebuttonStyling)



#LibraryDataSource definition
datasource_library: DataSourceElement = DataSourceElement(name="LibraryDataSource", dataSourceClass=library, fields=[library_name, address])

#Library List definition
libraryList: DataList=DataList(name="LibraryList", description="A diverse group of libraries", list_sources={datasource_library}, styling=ListStyling)


# Library directory screen definition
LibraryListScreen: Screen = Screen(name="LibraryListScreen", description="Explore a collection of libraries",
                          x_dpi="x_dpi", y_dpi="y_dpi", screen_size="Medium", view_elements={libraryAddingButton, libraryList, libraryEditButton, librarydeleteButton}, layout=ScreenLayout)



# Module definition:
MyModule: Module = Module(name="ModuleName", screens={LibraryListScreen})

# Application definition:
gui_model: GUIModel = GUIModel(name="LibraryManagement", package="com.example.librarymanagement", versionCode="1",
                                 versionName="1.0", description="This is a comprehensive web application for managing a library.",
                                 screenCompatibility=True, modules={MyModule})













