from besser.BUML.metamodel.structural import NamedElement, Class, Property, Model
from enum import Enum


# FileSourceType
class FileSourceType(NamedElement):
    """Represents the type of a file source.

    Args:
        name (str): The name of the file source type.
        type (str): The type of the file source, such as 'FileSystem', 'LocalStorage', or 'DatabaseFileSystem'.

    Attributes:
        name (str): The name of the file source type.
        type (str): The type of the file source, such as 'FileSystem', 'LocalStorage', or 'DatabaseFileSystem'.
    """

    def __init__(self, name: str, type: str):
        super().__init__(name)
        self.type: str = type

    @property
    def type(self) -> str:
        """str: Get the type of the file source."""
        return self.__type

    @type.setter
    def type(self, type: str):
        """
        str: Set the type of the file source.

        Raises:
            ValueError: If the type provided is not one of the allowed values: 'FileSystem', 'LocalStorage',
                or 'DatabaseFileSystem'.
        """
        if type not in ['FileSystem', 'LocalStorage', 'DatabaseFileSystem']:
            raise ValueError("Invalid value of type")
            self.__type = type

    def __repr__(self):
        return f'FileSourceType({self.name}, type={self.type})'

# CollectionSourceType
class CollectionSourceType:
    """Represents the type of a collection source.

    Args:
        name (str): The name of the collection source type.
        type (str): The type of the collection source, such as 'List', 'Table', 'Tree', 'Grid', 'Array', or 'Stack'.

    Attributes:
        name (str): The name of the collection source type.
        type (str): The type of the collection source, such as 'List', 'Table', 'Tree', 'Grid', 'Array', or 'Stack'.
    """

    def __init__(self, name: str, type: str):
        self.type: str = type
        self.name: str = name

    @property
    def name(self) -> str:
        """str: Get the name of the collection source type."""
        return self.__name

    @name.setter
    def name(self, name: str):
        """str: Set the name of the collection source type."""
        self.__name = name

    @property
    def type(self) -> str:
        """str: Get the type of the collection source."""
        return self.__type

    @type.setter
    def type(self, type: str):
        """str: Set the type of the collection source.

        Raises:
            ValueError: If the type provided is not one of the allowed values: 'List', 'Table', 'Tree', 'Grid',
                'Array', or 'Stack'.
        """
        if type not in ['List', 'Table', 'Tree', 'Grid', 'Array', 'Stack']:
            raise ValueError("Invalid value of type")
            self.__type = type

    def __repr__(self):
        return f'CollectionSourceType({self.name}, type={self.type})'

class ButtonActionType(Enum):
    """Represents a button action type.
    """
    Add = "Add"
    ShowList = "Show List"
    OpenForm = "Open Form"
    SubmitForm = "Submit Form"
    Cancel = "Cancel"
    Save = "Save"
    Delete = "Delete"
    Confirm = "Confirm"
    Navigate = "Navigate"
    Search = "Search"
    Filter = "Filter"
    Sort = "Sort"
    Send = "Send"
    Share = "Share"
    Settings = "Settings"
    Back = "Back"
    Next = "Next"
    View = "View"
    Select = "Select"
    Login = "Login"
    Logout = "Sign Out"
    Help = "Help"
    About= "About"
    Exit = "Exit"



class ButtonType(Enum):
    """Represents a button type.
    """
    RaisedButton = "Raised Button"
    TextButton = "Text Button"
    OutlinedButton = "Outlined Button"
    IconButton = "Icon Button"
    FloatingActionButton = "FloatingActionButton"
    DropdownButton = "Dropdown Button"
    ToggleButtons = "Toggle Buttons"
    iOSStyleButton = "iOS-style Button"
    CustomizableButton = "Customizable Button"


#DataSource
class DataSource:
    """Represents a data source.

    Args:
        name (str): The name of the data source.

    Attributes:
        name (str): The name of the data source.
    """

    def __init__(self, name: str):
        self.name: str = name

    @property
    def name(self) -> str:
      """str: Get the name of the data source."""
      return self.__name

    @name.setter
    def name(self, name: str):
      """str: Set the name of the data source."""
      self.__name = name

    def __repr__(self):
      return f'DataSource({self.name})'

#ModelElementDataSource
class ModelElement(DataSource):
    """Represents a data source associated with a model element.

    Args:
        name (str): The name of the model element data source.
        dataSourceClass (Class): The class representing the data source.
        fields: set[Property]: The fields representing the attributes of the model element.

    Attributes:
        name (str): The name of the model element data source.
        dataSourceClass (Class): The class representing the data source.
        fields: set[Property]: The fields representing the attributes of the model element.
    """

    def __init__(self, name: str, dataSourceClass: Class, fields: set[Property]):
        super().__init__(name)
        self.dataSourceClass: Class = dataSourceClass
        self.fields: set[Property]= fields

    @property
    def dataSourceClass(self) -> Class:
        """Class: Get the class representing the data source."""
        return self.__dataSourceClass

    @dataSourceClass.setter
    def dataSourceClass(self, dataSourceClass: Class):
        """Class: Set the class representing the data source."""
        self.__dataSourceClass = dataSourceClass

    @property
    def fields(self) -> set[Property]:
        """set[Property]: Get the set of properties (fields) of the model element."""
        return self.__fields

    @fields.setter
    def fields(self, fields: set[Property]):
        """set[Property]: Set the set of properties (fields) of the model element."""
        if fields is not None:
            names = [field.name for field in fields]
            if len(names) != len(set(names)):
                raise ValueError("A model element cannot have two fields with the same name.")
        self.__fields = fields

    def __repr__(self):
      return f'ModelElement({self.name}, {self.dataSourceClass},{self.fields})'

#FileDataSource
class File(DataSource):
    """Represents a data source that is a file.

    Args:
        name (str): The name of the file data source.
        type (FileSourceType): The type of the file data source.

    Attributes:
        name (str): The name of the file data source.
        type (FileSourceType): The type of the file data source.
    """

    def __init__(self, name: str, type:FileSourceType):
        super().__init__(name)
        self.type: FileSourceType = type

    @property
    def type(self) -> FileSourceType:
      """FileSourceType: Get the type of the file data source."""
      return self.__type

    @type.setter
    def type(self, type: FileSourceType):
      """FileSourceType: Set the type of the file data source."""
      self.__type = type

    def __repr__(self):
      return f'File({self.name}, {self.type})'


#CollectionDataSource
class Collection(DataSource):
    """Represents a data source that is a collection.

    Args:
        name (str): The name of the collection data source.
        type (CollectionSourceType): The type of the collection data source.

    Attributes:
        name (str): The name of the collection data source.
        type (CollectionSourceType): The type of the collection data source.
    """
    def __init__(self, name: str, type:CollectionSourceType):
        super().__init__(name)
        self.type: CollectionSourceType = type

    @property
    def type(self) -> CollectionSourceType:
      """CollectionSourceType: Get the type of the collection data source."""
      return self.__type

    @type.setter
    def type(self, type: CollectionSourceType):
      """CollectionSourceType: Set the type of the collection data source."""
      self.__type = type

    def __repr__(self):
      return f'Collection({self.name}, {self.type})'

#ViewElement
class ViewElement(NamedElement):
    """Represents a view element.

    Args:
        name (str): The name of the view element.
        description (str): the description of the view element.

    Attributes:
        name (str): The name of the view element.
        description (str): the description of the view element.
    """

    def __init__(self, name: str, description: str, visibility: str = "public"):
        super().__init__(name, visibility)
        self.description: str = description

    @property
    def description(self) -> str:
        """str: Get the description of the view element."""
        return self.__description

    @description.setter
    def description(self, description: str):
        """str: Set the description of the view element."""
        self.__description = description

    def __repr__(self):
        return f'ViewElement({self.name})'

#ViewComponent
class ViewComponent(ViewElement):
    """Represents a view component.

    Args:
        name (str): The name of the view component.
        description (str): The description of the view component.

    Attributes:
        name (str): The name of the view component.
        description (str): The description of the view component.
    """

    def __init__(self, name: str, description: str):
        super().__init__(name, description)

    def __repr__(self):
        return f'ViewComponent({self.name}, description={self.description})'

#ViewContainer
class ViewContainer(ViewElement):
    """Represents a view container.

    Args:
        name (str): The name of the view container.
        description (str): The description of the view container.

    Attributes:
        name (str): The name of the view container.
        description (str): The description of the view container.
    """

    def __init__(self, name: str, description: str, view_elements: set[ViewElement]):
        super().__init__(name, description)
        self.view_elements: set[ViewElement] = view_elements

    @property
    def view_elements(self) -> set[ViewElement]:
        """set[ViewComponent]: Get the set of view elements on the screen."""
        return self.__view_elements

    @view_elements.setter
    def view_elements(self, view_elements: set[ViewElement]):
       """set[ViewElement]: Set the set of view elements on the screen."""
       if view_elements is not None:
            names = [view_element.name for view_element in view_elements]
            if len(names) != len(set(names)):
                raise ValueError("A screen cannot have two elements with the same name.")
       self.__view_elements = view_elements

    def __repr__(self):
        return f'ViewContainer({self.name}, description={self.description}, view_elements={self.view_elements})'

#Screen
class Screen(ViewContainer):
    """Represents a screen.

    Args:
        name (str): The name of the screen.
        view_elements (set[ViewElement]): The set of view elements on the screen.
        x_dpi (str): The X DPI (dots per inch) of the screen.
        y_dpi (str): The Y DPI (dots per inch) of the screen.
        size (str): The size of the screen.
        is_main_page (bool): wether this screen serves as the main page of the model.

    Attributes:
        name (str): The name of the screen.
        view_elements (set[ViewElement]): The set of view elements on the screen.
        x_dpi (str): The X DPI (dots per inch) of the screen.
        y_dpi (str): The Y DPI (dots per inch) of the screen.
        size (str): The size of the screen.
        is_main_page (bool): wether this screen serves as the main page of the model.
    """

    def __init__(self, name: str, description: str, view_elements: set[ViewElement], x_dpi: str, y_dpi: str, size: str, is_main_page: bool = False):
        super().__init__(name, description, view_elements)
        self.x_dpi: str = x_dpi
        self.y_dpi: str = y_dpi
        self.size: str = size
        self.is_main_page: bool = is_main_page

    @property
    def x_dpi(self) -> str:
      """str: Get the X DPI (dots per inch) of the screen."""
      return self.__x_pdi

    @x_dpi.setter
    def x_dpi(self, x_dpi: str):
      """str: Set the X DPI (dots per inch) of the screen."""
      self.__x_pdi = x_dpi

    @property
    def y_dpi(self) -> str:
      """str: Get the Y DPI (dots per inch) of the screen."""
      return self.__y_pdi

    @y_dpi.setter
    def y_dpi(self, y_dpi: str):
     """str: Set the Y DPI (dots per inch) of the screen."""
     self.__y_pdi = y_dpi

    @property
    def size(self) -> str:
      """str: Get the size of the screen."""
      return self.__size

    @size.setter
    def size(self, size: str):
        """str: Set the size of the screen.

        Raises:
            ValueError: If the size provided is not one of the allowed options: 'Small','Medium', 'Large', 'xLarge'
        """

        if size not in ['Small', 'Medium', 'Large', 'xLarge']:
           raise ValueError("Invalid value of size")
           self.__size = size

    @property
    def is_main_page(self) -> bool:
        """bool: Get whether the screen is main page."""
        return self.__is_main_page

    @is_main_page.setter
    def is_main_page(self, is_main_page: bool):
        """bool: Set whether the screen is main page."""
        self.__is_main_page = is_main_page


    def __repr__(self):
      return f'Screen({self.name}, {self.x_dpi}, {self.y_dpi}, {self.size}, {self.view_elements}, {self.is_main_page})'

#Module
class Module(NamedElement):
    """Represents a module.

    Args:
        name (str): name (str): The name of the module.
        screens (set[Screen]): The set of screens contained in the module.

    Attributes:
        name (str): name (str): The name of the module.
        screens (set[Screen]): The set of screens contained in the module.
    """

    def __init__(self, name: str, screens: set[Screen], visibility: str = "public"):
        super().__init__(name, visibility)
        self.screens: set[Screen] = screens

    @property
    def screens(self) -> set[Screen]:
        """set[Screen]: Get the set of screens contained."""
        return self.__screens

    @screens.setter
    def screens(self, screens: set[Screen]):
       """set[Screen]: Set the set of screens contained."""
       if screens is not None:
            names = [screen.name for screen in screens]
            if len(names) != len(set(names)):
                raise ValueError("A module cannot have two screens with the same name.")
       self.__screens = screens

    def __repr__(self):
        return f'Module({self.name}, {self.screens})'

# DataList is a type of ViewComponent
class DataList(ViewComponent):
    """Represents a list component that encapsulates properties unique to lists, such as list sources.

    Args:
        name (str): The name of the list.
        list_sources (set[DataSource]): The set of data sources associated with the list.

    Attributes:
        name (str): The name of the list.
        list_sources (set[DataSource]): The set of data sources associated with the list.
    """

    def __init__(self, name: str, description: str, list_sources: set[DataSource]):
        super().__init__(name, description)
        self.list_sources: set[DataSource] = list_sources

    @property
    def list_sources(self) -> set[DataSource]:
        """set[DataSource]: Get the set of data sources associated with the list."""
        return self.__list_sources

    @list_sources.setter
    def list_sources(self, list_sources: set[DataSource]):
        """set[DataSource]: Set the set of data sources associated with the list."""
        if list_sources is not None:
            names = [DataSource.name for DataSource in list_sources]
            if len(names) != len(set(names)):
               raise ValueError("A list cannot have two items with the same name.")
        self.__list_sources = list_sources

    def __repr__(self):
     return f'DataList({self.name}, {self.list_sources})'

# Button is a type of ViewComponent
class Button(ViewComponent):
    """Represents a button component and encapsulates specific properties of a button, such as its name and label.

    Args:
        name (str): The name of the button.
        description (str): The description of the button.
        label (str): The label of the button.
        buttonType (ButtonType): The type of the button.
        actionType (ButtonActionType): The action type of the button.
        targetScreen (Screen, optional): The target Screen associated with the button when the actionType is "Navigate".

    Attributes:
        name (str): The name of the button.
        description (str): The description of the button.
        label (str): The label of the button.
        buttonType (ButtonType): The type of the button.
        actionType (ButtonActionType): The action type of the button.
        targetScreen (Screen, optional): The target Screen associated with the button when the actionType is "Navigate"
    """

    def __init__(self, name: str, description: str, label: str, buttonType: ButtonType, actionType: ButtonActionType, targetScreen: Screen = None):
        super().__init__(name, description)
        self.label=label
        self.buttonType: ButtonType = buttonType
        self.actionType: ButtonActionType=actionType
        self.targetScreen: Screen = targetScreen

    @property
    def label(self) -> str:
        """str: Get the label of the button."""
        return self.__label

    @label.setter
    def label(self, label: str):
        """str: Set the label of the button."""
        self.__label = label

    @property
    def buttonType(self) -> ButtonType:
        """str: Get the type of the button."""
        return self.__buttonType

    @buttonType.setter
    def buttonType(self, buttonType: ButtonType):
        """str: Set the type of the button."""
        self.__buttonType = buttonType

    @property
    def actionType(self) -> ButtonActionType:
        """str: Get the action type of the button."""
        return self.__actionType

    @actionType.setter
    def actionType(self, actionType: ButtonActionType):
        """str: Set the action type of the button."""
        self.__actionType = actionType

    @property
    def targetScreen(self) -> Screen:
        """Type: Get the target Screen of the button."""
        return self.__targetScreen

    @targetScreen.setter
    def targetScreen(self, targetScreen: Screen):
        """
        Set the target Screen associated with the button.

        Args:
            targetScreen (Screen): The target Screen to be associated with the button.

        Raises:
            ValueError: If the actionType is 'Navigate' and the target Screen is not an instance of the Screen class.
            ValueError: If the actionType is not 'Navigate' and an target Screen is specified.

        """
        if self.actionType == ButtonActionType.Navigate:
            if targetScreen is None:
                raise ValueError("A target Screen must be specified for the button when the actionType is 'Navigate'.")
            elif not isinstance(targetScreen, Screen):
                raise ValueError("The target Screen must be an instance of the Screen class when the actionType is 'Navigate'.")
        elif targetScreen is not None:
            raise ValueError("A target Screen cannot be specified for the button when the actionType is not 'Navigate'.")
        self.__targetScreen = targetScreen

    def __repr__(self):
     return f'Button({self.name},{self.label}, {self.description}, {self.buttonType}, {self.actionType})'

# Image is a type of ViewComponent
class Image(ViewComponent):
    """Represents an image component and encapsulates the specific properties of a image, such as its name.

    Args:
        name (str): The name of the image.

    Attributes:
        name (str): The name of the image.
    """

    def __init__(self, name: str, description: str):
        super().__init__(name, description)

    def __repr__(self):
     return f'Image({self.name},{self.description})'


# InputField is a type of ViewComponent
class InputField(ViewComponent):
    """Represents an input field component and encapsulates specific properties of an input field, such as its type and validation rules.

     Args:
        name (str): The name of the input field.
        description (str): The description of the input field.
        fieldType (str): The type of the input field.
        validationRules (str): The validation rules for the input field.

    Attributes:
        name (str): The name of the input field.
        description (str): The description of the input field.
        fieldType (str): The type of the input field.
        validationRules (str): The validation rules for the input field.
    """

    def __init__(self, name: str, description: str, fieldType: str, validationRules: str):
        super().__init__(name, description)
        self.fieldType: str= fieldType
        self.validationRules: str = validationRules

    @property
    def fieldType(self) -> str:
      """str: Get the type of the field."""
      return self.__fieldType


    @fieldType.setter
    def fieldType(self, fieldType: str):
        """str: Set the type of the field.

        Raises:
            ValueError: If the type provided is not one of the allowed options: 'Text','Number', 'Email', 'Password', 'Date', 'Time', 'File', 'Color', 'Range', 'URL', 'Tel', and 'Search'
        """

        if fieldType not in ['Text','Number', 'Email', 'Password', 'Date', 'Time', 'File', 'Color', 'Range', 'URL', 'Tel', 'Search']:
           raise ValueError("Invalid value of fieldType")
           self.__fieldType = fieldType

    @property
    def validationRules(self) -> str:
        """str: Get the validation rules of the input field."""
        return self.__validationRules


    @validationRules.setter
    def validationRules(self, str):
        """str: Set the validation rules of the input field."""
        self.__validationRules = validationRules

    def __repr__(self):
     return f'InputField({self.name},{self.description}, {self.fieldType}, {self.validationRules})'

# Form is a type of ViewComponent
class Form(ViewComponent):
    """Represents a form component and encapsulates the specific properties of a form, such as its name.

    Args:
        name (str): The name of the form.
        description (str): The description of the form.
        inputFields (set[InputField]): The set of input fields contained in the form.

    Attributes:
        name (str): The name of the form.
        description (str): The description of the form.
        inputFields (set[InputField]): The set of input fields contained in the form.
    """

    def __init__(self, name: str, description: str, inputFields: set[InputField]):
        super().__init__(name, description)
        self.inputFields: set[InputField] = inputFields

    @property
    def inputFields(self) -> set[InputField]:
        """set[InputField]: Get the set of input Fields contained in the form."""
        return self.__inputFields

    @inputFields.setter
    def inputFields(self, inputFields: set[InputField]):
       """set[InputField]: Set the set of input Fields contained in the form."""
       self.__inputFields = inputFields

    def __repr__(self):
     return f'Form({self.name},{self.description}, {self.inputFields})'

# MenuItem
class MenuItem:
    """Represents an item of a menu.

    Args:
        label (str): The label of the menu item.

    Attributes:
        label (str): The label of the menu item.
    """

    def __init__(self, label: str):
        self.label: str = label

    def __repr__(self):
     return f'MenuItem({self.label})'

# Menu is a type of ViewComponent
class Menu(ViewComponent):
    """Represents a menu component and encapsulates the specific properties of a menu, such as its name.

    Args:
        name (str): The name of the menu.
        description (str): The description of the menu.
        menuItems (set[MenuItem]): The set of menu items contained in the menu.

    Attributes:
        name (str): The name of the menu.
        description (str): The description of the menu.
        menuItems (set[MenuItem]): The set of menu items contained in the menu.
    """

    def __init__(self, name: str, description: str, menuItems: set[MenuItem]):
        super().__init__(name, description)
        self.menuItems: set[MenuItem] = menuItems

    @property
    def menuItems(self) -> set[MenuItem]:
        """set[MenuItem]: Get the set of menuItems."""
        return self.__menuItems

    @menuItems.setter
    def menuItems(self, menuItems: set[MenuItem]):
        """set[MenuItem]: Set the set of menuItems."""
        self.__menuItems = menuItems

    def __repr__(self):
     return f'Menu({self.name},{self.description}, {self.menuItems})'

#GUIModel
class GUIModel(Model):
    """It is a subclass of the NamedElement class and encapsulates the properties and behavior of the GUI part of an application, including its name,
       package, version code, version name, modules, description, and screen compatibility.

    Args:
        name (str): The name of the model.
        package (str): The package of the model.
        versionCode (str): The version code of the model.
        versionName (str): The version name of the model.
        modules (set[Module]): The set of modules contained in the model.
        description (str): The description of the model.
        screenCompatibility (bool): Indicates whether the model has screen compatibility.

    Attributes:
        name (str): The name of the model.
        package (str): The package of the model.
        versionCode (str): The version code of the model.
        versionName (str): The version name of the model.
        modules (set[Module]): The set of modules contained in the model.
        description (str): The description of the model.
        screenCompatibility (bool): Indicates whether the model has screen compatibility.
    """
    def __init__(self, name: str, package: str, versionCode: str, versionName: str, modules: set[Module], description: str, screenCompatibility: bool = False):
        super().__init__(name)
        self.package: str = package
        self.versionCode: str = versionCode
        self.versionName: str = versionName
        self.description: str = description
        self.modules: set[Module] = modules
        self.screenCompatibility: str = screenCompatibility

    @property
    def package(self) -> str:
        """str: Get the package of the model."""
        return self.__package

    @package.setter
    def package(self, package: str):
        """str: Set the package of the model."""
        self.__package = package

    @property
    def versionCode(self) -> str:
        """str: Get the version code of the model."""
        return self.__versionCode

    @versionCode.setter
    def versionCode(self, versionCode: str):
        """str: Set the version code of the model."""
        self.__versionCode = versionCode

    @property
    def versionName(self) -> str:
        """str: Get the version name of the model."""
        return self.__versionName

    @versionName.setter
    def versionName(self, versionName: str):
        """str: Set the version name of the model."""
        self.__versionName = versionName

    @property
    def description(self) -> str:
        """str: Get the description of the model."""
        return self.__description

    @description.setter
    def description(self, description: str):
        """str: Set the description of the model."""
        self.__description = description

    @property
    def screenCompatibility(self) -> bool:
        """bool: Get the screen compatibility of the model."""
        return self.__screenCompatibility

    @screenCompatibility.setter
    def screenCompatibility(self, screenCompatibility: bool):
        """bool: Set the screen compatibility of the model."""
        self.__screenCompatibility = screenCompatibility

    @property
    def modules(self) -> set[Module]:
        """set[Module]: Get the set of modules contained in the model."""
        return self.__modules

    @modules.setter
    def modules(self, modules: set[Module]):
       """set[Module]: Set the set of modules contained in the model."""
       if modules is not None:
            names = [module.name for module in modules]
            if len(names) != len(set(names)):
                raise ValueError("An app cannot have two modules with the same name")
       self.__modules = modules


    def __repr__(self):
        return f'GUIModel({self.name}, {self.package}, {self.versionCode}, {self.versionName},{self.description},{self.screenCompatibility}, {self.modules})'





