from enum import Enum
from typing import Dict, List, Optional

from besser.BUML.metamodel.structural import NamedElement, Class, Property, Model, Element
from besser.BUML.metamodel.gui.style import Styling, Layout
from besser.BUML.metamodel.gui.binding import DataBinding


# FileSourceType
class FileSourceType(Enum):
    """Represents the type of a file source.
    """
    FileSystem = "FileSystem"
    LocalStorage = "LocalStorage"
    DatabaseFileSystem = "DatabaseFileSystem"


# CollectionSourceType
class CollectionSourceType(Enum):
    """Represents the type of a collection source.
    """
    List = "List"
    Table = "Table"
    Tree = "Tree"
    Grid = "Grid"
    Array = "Array"
    Stack = "Stack"


# InputFieldType
class InputFieldType(Enum):
    """Represents the type of a Input Field.
    """
    Text = "Text"
    Number = "Number"
    Email = "Email"
    Password = "Password"
    Date = "Date"
    Time = "Time"
    File = "File"
    Color = "Color"
    Range = "Range"
    URL = "URL"
    Tel = "Tel"
    Search = "Search"


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
    Edit = "Edit"

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
class DataSource(NamedElement):
    """Represents a data source.

    Args:
        name (str): The name of the data source.

    Attributes:
        name (str): The name of the data source.
    """

    def __init__(self, name: str):
        super().__init__(name)


    def __repr__(self):
        return f'DataSource({self.name})'

#DataSourceElement
class DataSourceElement(DataSource):
    """Represents a data source associated with a model element.

    Args:
        name (str): Display name of the data source.
        dataSourceClass (Class | None): Domain class backing the data source.
        fields (set[Property] | None): Subset of attributes included in the source.
        label_field (Property | None): Property used as label when rendering.
        value_field (Property | None): Property used as value when rendering.
        field_names (list[str] | None): Field names preserved when properties are unresolved.
        label_field_name (str | None): Label field name preserved when property is unresolved.
        value_field_name (str | None): Value field name preserved when property is unresolved.
    """

    def __init__(
        self,
        name: str,
        dataSourceClass: Class | None = None,
        fields: set[Property] | None = None,
        label_field: Property | None = None,
        value_field: Property | None = None,
        field_names: Optional[List[str]] = None,
        label_field_name: Optional[str] = None,
        value_field_name: Optional[str] = None,
    ):
        super().__init__(name)
        self.dataSourceClass = dataSourceClass
        self.fields = fields or set()
        self.field_names = field_names or []
        self.label_field = label_field
        self.value_field = value_field
        if label_field_name and not self.label_field_name:
            self.label_field_name = label_field_name
        if value_field_name and not self.value_field_name:
            self.value_field_name = value_field_name

    @property
    def dataSourceClass(self) -> Class | None:
        """Class | None: Get the class representing the data source."""
        return self.__dataSourceClass

    @dataSourceClass.setter
    def dataSourceClass(self, dataSourceClass: Class | None):
        """Assign the backing class for the data source."""
        self.__dataSourceClass = dataSourceClass

    @property
    def domain_concept(self) -> Class | None:
        """Alias to maintain backwards compatibility with older generators."""
        return self.dataSourceClass

    @domain_concept.setter
    def domain_concept(self, value: Class | None):
        self.dataSourceClass = value

    @property
    def fields(self) -> set[Property]:
        """set[Property]: Get the set of properties (fields) of the model element."""
        return self.__fields

    @fields.setter
    def fields(self, fields: set[Property] | None):
        """Set the subset of fields included in the data source."""
        if fields:
            normalized = set(fields)
            names = [field.name for field in normalized]
            if len(names) != len(set(names)):
                raise ValueError("A model element cannot have two fields with the same name.")
            self.__fields = normalized
            self.__field_names = names
        else:
            self.__fields = set()
            self.__field_names = []

    @property
    def field_names(self) -> List[str]:
        """List[str]: Names of fields kept when the domain model is unresolved."""
        return self.__field_names

    @field_names.setter
    def field_names(self, names: Optional[List[str]]):
        self.__field_names = list(names or [])

    @property
    def label_field(self) -> Property | None:
        """Property | None: Get the label field property."""
        return self.__label_field

    @label_field.setter
    def label_field(self, field: Property | None):
        self.__label_field = field
        self.__label_field_name = field.name if field is not None else getattr(self, "__label_field_name", None)

    @property
    def value_field(self) -> Property | None:
        """Property | None: Get the value field property."""
        return self.__value_field

    @value_field.setter
    def value_field(self, field: Property | None):
        self.__value_field = field
        self.__value_field_name = field.name if field is not None else getattr(self, "__value_field_name", None)

    @property
    def label_field_name(self) -> str | None:
        """str | None: Get the stored label field name."""
        return getattr(self, "__label_field_name", None)

    @label_field_name.setter
    def label_field_name(self, name: Optional[str]):
        self.__label_field_name = name

    @property
    def value_field_name(self) -> str | None:
        """str | None: Get the stored value field name."""
        return getattr(self, "__value_field_name", None)

    @value_field_name.setter
    def value_field_name(self, name: Optional[str]):
        self.__value_field_name = name

    def __repr__(self):
        return (
            f'DataSourceElement({self.name}, {self.dataSourceClass}, '
            f'fields={[field.name for field in self.fields]}, '
            f'label={self.label_field_name}, value={self.value_field_name})'
        )

#FileDataSource
class File(DataSource):
    """Represents a data source that is a file.

    Args:
        name (str): The name of the file data source.
        type (FileSourceType): The type of the file data source.

    Attributes:
        name (str): The name of the file data source.
        file_type (FileSourceType): The type of the file data source.
    """

    def __init__(self, name: str, file_type:FileSourceType):
        super().__init__(name)
        self.file_type: FileSourceType = file_type

    @property
    def file_type(self) -> FileSourceType:
        """FileSourceType: Get the type of the file data source."""
        return self.__file_type

    @file_type.setter
    def file_type(self, file_type: FileSourceType):
        """FileSourceType: Set the type of the file data source."""
        self.__file_type = file_type

    def __repr__(self):
        return f'File({self.name}, {self.file_type})'


#CollectionDataSource
class Collection(DataSource):
    """Represents a data source that is a collection.

    Args:
        name (str): The name of the collection data source.
        col_type (CollectionSourceType): The type of the collection data source.

    Attributes:
        name (str): The name of the collection data source.
        col_type (CollectionSourceType): The type of the collection data source.
    """
    def __init__(self, name: str, col_type: CollectionSourceType):
        super().__init__(name)
        self.col_type: CollectionSourceType = col_type

    @property
    def col_type(self) -> CollectionSourceType:
        """CollectionSourceType: Get the type of the collection data source."""
        return self.__col_type

    @col_type.setter
    def col_type(self, col_type: CollectionSourceType):
        """CollectionSourceType: Set the type of the collection data source."""
        self.__col_type = col_type

    def __repr__(self):
        return f'Collection({self.name}, {self.col_type})'

#ViewElement
class ViewElement(NamedElement):
    """
    Represents a view element with optional size, position, and color attributes.

    Args:
        name (str): The name of the view element.
        description (str): A brief description of the view element (optional).
        timestamp (datetime): Object creation datetime (default is current time).
        visibility (str, optional): Visibility scope (default: "public").
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).

    Attributes:
        name (str): The name of the view element.
        description (str): A brief description of the view element (optional).
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        visibility (str): Visibility scope (default: "public").
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).
        owner (ViewContainer | None): The container that owns this view element (if any).
        component_id (str | None): Original GrapesJS component ID for code generation fidelity.
        component_type (str | None): Original GrapesJS component type for code generation fidelity.
        tag_name (str | None): HTML tag name (e.g., "div", "button", "img").
        css_classes (list[str]): List of CSS class names applied to the component.
        custom_attributes (dict): Dictionary of custom HTML attributes.
        display_order (int): Order index from original JSON to preserve component sequence.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        visibility: str = "public",
        timestamp: int = None,
        styling: Styling = None,
    ):
        super().__init__(name, visibility, timestamp)
        self.description: str = description
        self.styling: Styling = styling
        self._owner: "ViewContainer" = None
        # Store original component metadata for code generation fidelity
        self.component_id: str | None = None  # Original GrapesJS component ID
        self.component_type: str | None = None  # Original GrapesJS component type
        self.tag_name: str | None = None  # HTML tag name
        self.css_classes: list[str] = []  # CSS class list
        self.custom_attributes: dict = {}  # Custom HTML attributes
        self.display_order: int = 0  # Preserve original order from JSON

    @property
    def description(self) -> str:
        """str: Get the description of the view element."""
        return self.__description

    @description.setter
    def description(self, description: str):
        """str: Set the description of the view element."""
        self.__description = description

    @property
    def styling(self) -> Styling:
        """Styling: Get the styling of the view element."""
        return self.__styling

    @styling.setter
    def styling(self, styling: Styling):
        """Styling: Set the styling of the view element."""
        self.__styling = styling


    @property
    def owner(self) -> "ViewContainer":
        """ViewContainer : Get the owner of the view element."""
        return self._owner

    @owner.setter
    def owner(self, owner: "ViewContainer"):
        """ViewContainer: Internal method for assigning the owner."""
        self._owner = owner

    def __repr__(self):
        return (
            "ViewElement("
            f"name={self.name}, description={self.description}, visibility={self.visibility}, "
            f"timestamp={self.timestamp}, styling={self.styling}"
            ")"
        )

#ViewComponent
class ViewComponent(ViewElement):
    """
    Represents a view component that extends a generic ViewElement.

    Args:
        name (str): The name of the view component.
        description (str): A brief description of the view component (optional).
        timestamp (datetime): Object creation datetime (default is current time).
        visibility (str, optional): The visibility scope (default: "public").
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).
        data_binding (DataBinding | None, optional): The data binding configuration for the view component (if any).

    Attributes:
        name (str): The name of the view component.
        description (str): A brief description of the view component (optional).
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        visibility (str): The visibility scope of the component.
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).
        data_binding (DataBinding | None): The data binding configuration for the view component (if any).
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        visibility: str = "public",
        timestamp: int = None,
        styling: Styling = None,
        data_binding: DataBinding = None,
    ):
        super().__init__(
            name,
            description=description,
            visibility=visibility,
            timestamp=timestamp,
            styling=styling
        )
        self.data_binding: DataBinding = data_binding

    @property
    def data_binding(self) -> DataBinding:
        """DataBinding: Get the data binding of the view component."""
        return self.__data_binding

    @data_binding.setter
    def data_binding(self, data_binding: DataBinding):
        """DataBinding: Set the data binding of the view component."""
        self.__data_binding = data_binding

    def __repr__(self):
        return (
            f'ViewComponent({self.name}, description={self.description}, visibility={self.visibility}, '
            f'timestamp={self.timestamp}, styling={self.styling}, data_binding={self.data_binding})'
        )

#ViewContainer
class ViewContainer(ViewElement):
    """Represents a view container.

    Args:
        name (str): The name of the view container.
        description (str): The description of the view container.
        timestamp (datetime): Object creation datetime (default is current time).
        layout (Layout | None, optional): The layout settings of the container. Defaults to None.

    Attributes:
        name (str): The name of the view container.
        description (str): The description of the view container.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        layout (Layout | None): The layout settings of the container.
    """

    def __init__(
        self,
        name: str,
        description: str,
        view_elements: set[ViewElement],
        timestamp: int = None,
        layout: Layout = None,
        styling: Styling = None,
    ):
        super().__init__(
            name,
            description=description,
            timestamp=timestamp,
            styling=styling
        )
        self.view_elements: set[ViewElement] = view_elements
        self.layout: Layout | None = layout  # Ensure layout is properly stored

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
            for view_element in view_elements:
                view_element.owner = self
        self.__view_elements = view_elements

    @property
    def layout(self) -> Layout | None:
        """Layout | None: Get the layout settings of the container."""
        return self.__layout

    @layout.setter
    def layout(self, layout: Layout | None):
        """Layout | None: Set the layout settings of the container."""
        self.__layout = layout

    def __repr__(self):
        return (
            f"ViewContainer({self.name}, description={self.description}, timestamp={self.timestamp}, "
            f"view_elements={self.view_elements}, "
            f"layout={self.layout}, styling={self.styling})"
        )

#Screen
class Screen(ViewContainer):
    """Represents a screen.

    Args:
        name (str): The name of the screen.
        description (str): The description of the screen.
        timestamp (datetime): Object creation datetime (default is current time).
        view_elements (set[ViewElement]): The set of view elements on the screen.
        x_dpi (str): The X DPI (dots per inch) of the screen.
        y_dpi (str): The Y DPI (dots per inch) of the screen.
        screen_size (str): The size of the screen.
        is_main_page (bool): Indicates whether this screen is the main page.
        layout (Layout | None, optional): The layout settings (Defaults to None).

    Attributes:
        name (str): The name of the screen.
        description (str): The description of the screen.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        view_elements (set[ViewElement]): The set of view elements on the screen.
        x_dpi (str): The X DPI (dots per inch) of the screen.
        y_dpi (str): The Y DPI (dots per inch) of the screen.
        screen_size (str): The size of the screen.
        is_main_page (bool): wether this screen serves as the main page.
        layout (Layout | None, optional): The layout settings (Defaults to None)
    """

    def __init__(
        self,
        name: str,
        description: str,
        view_elements: set[ViewElement],
        x_dpi: str = "",
        y_dpi: str = "",
        screen_size: str = "Medium",
        timestamp: int = None,
        is_main_page: bool = False,
        layout: Layout = None,
        styling: Styling = None,
        route_path: str | None = None,
    ):
        super().__init__(
            name,
            description,
            view_elements,
            timestamp,
            layout,
            styling=styling
        )
        self.x_dpi: str = x_dpi
        self.y_dpi: str = y_dpi
        self.screen_size: str = screen_size
        self.is_main_page: bool = is_main_page
        self.route_path = route_path or f"/{name}"

    @property
    def x_dpi(self) -> str:
        """str: Get the X DPI (dots per inch) of the screen."""
        return self.__x_dpi

    @x_dpi.setter
    def x_dpi(self, x_dpi: str):
        """str: Set the X DPI (dots per inch) of the screen."""
        self.__x_dpi = x_dpi

    @property
    def y_dpi(self) -> str:
        """str: Get the Y DPI (dots per inch) of the screen."""
        return self.__y_dpi

    @y_dpi.setter
    def y_dpi(self, y_dpi: str):
        """str: Set the Y DPI (dots per inch) of the screen."""
        self.__y_dpi = y_dpi

    @property
    def screen_size(self) -> str:
        """str: Get the size of the screen."""
        return self.__screen_size

    @screen_size.setter
    def screen_size(self, screen_size: str):
        """str: Set the size of the screen.

        Raises:
            ValueError: If the size provided is not one of the allowed options: 'Small','Medium', 'Large', 'xLarge'
        """

        if screen_size not in ['Small', 'Medium', 'Large', 'xLarge']:
            raise ValueError("Invalid value of screen size")

        self.__screen_size = screen_size

    @property
    def route_path(self) -> str:
        """str: Get the route path for the screen."""
        return self.__route_path

    @route_path.setter
    def route_path(self, route_path: str | None):
        """str: Set the route path for the screen."""
        if not route_path:
            route_path = f"/{self.name}"
        self.__route_path = route_path

    @property
    def is_main_page(self) -> bool:
        """bool: Get whether the screen is main page."""
        return self.__is_main_page

    @is_main_page.setter
    def is_main_page(self, is_main_page: bool):
        """bool: Set whether the screen is main page."""
        self.__is_main_page = is_main_page

    def __repr__(self):
        return (
            f"Screen({self.name}, description={self.description}, {self.x_dpi}, {self.y_dpi}, "
            f"{self.screen_size}, route_path={self.route_path}, timestamp={self.timestamp}, "
            f"{self.view_elements}, {self.is_main_page})"
        )

#Module
class Module(NamedElement):
    """Represents a module.

    Args:
        name (str): name (str): The name of the module.
        description (str): The description of the input field.
        screens (set[Screen]): The set of screens contained in the module.

    Attributes:
        name (str): name (str): The name of the module.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        screens (set[Screen]): The set of screens contained in the module.
    """

    def __init__(self, name: str, screens: set[Screen], visibility: str = "public", timestamp: int = None):
        super().__init__(name, visibility, timestamp)
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
        return f'Module({self.name}, {self.screens}, {self.visibility}, timestamp={self.timestamp})'

# DataList is a type of ViewComponent
class DataList(ViewComponent):
    """Represents a list component that encapsulates properties unique to lists, such as list sources.

    Args:
        name (str): The name of the list.
        timestamp (datetime): Object creation datetime (default is current time).
        list_sources (set[DataSource]): The set of data sources associated with the list.
        styling (Styling, optional): The styling configuration, which includes size, position, and color settings (default: None).

    Attributes:
        name (str): The name of the list.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        list_sources (set[DataSource]): The set of data sources associated with the list.
        styling (Styling, optional): The styling configuration, which includes size, position, and color settings (default: None).
    """

    def __init__(self, name: str, description: str, list_sources: set[DataSource], visibility: str = "public", timestamp: int = None, styling: Styling = None):
        super().__init__(name, description, visibility, timestamp, styling=styling)
        self.list_sources: set[DataSource] = list_sources

    @property
    def list_sources(self) -> set[DataSource]:
        """set[DataSource]: Get the set of data sources associated with the list."""
        return self.__list_sources

    @list_sources.setter
    def list_sources(self, list_sources: set[DataSource]):
        """set[DataSource]: Set the set of data sources associated with the list."""
        if list_sources is not None:
            names = [data_source.name for data_source in list_sources]
            if len(names) != len(set(names)):
                raise ValueError("A list cannot have two items with the same name.")
        self.__list_sources = list_sources

    def __repr__(self):
        return f'DataList({self.name}, {self.description}, {self.list_sources}, {self.visibility}, {self.timestamp}, {self.styling})'

# Button is a type of ViewComponent
class Button(ViewComponent):
    """
    Represents a button component and encapsulates specific properties of a button, such as its name and label.
    """

    def __init__(self, name: str, description: str, label: str, buttonType: ButtonType, actionType: ButtonActionType,
                 targetScreen: Screen = None, timestamp: int = None, visibility: str = "public",
                 styling: Styling = None):
        super().__init__(name, description, visibility, timestamp, styling=styling)
        self.label = label
        self.buttonType = buttonType
        self.actionType = actionType
        self.targetScreen = targetScreen

    @property
    def label(self) -> str:
        return self.__label

    @label.setter
    def label(self, label: str):
        self.__label = label

    @property
    def buttonType(self) -> ButtonType:
        return self.__buttonType

    @buttonType.setter
    def buttonType(self, buttonType: ButtonType):
        self.__buttonType = buttonType

    @property
    def actionType(self) -> ButtonActionType:
        return self.__actionType

    @actionType.setter
    def actionType(self, actionType: ButtonActionType):
        self.__actionType = actionType

    @property
    def targetScreen(self) -> Screen:
        return self.__targetScreen

    @targetScreen.setter
    def targetScreen(self, targetScreen: Screen):
        if self.actionType == ButtonActionType.Navigate:
            if not isinstance(targetScreen, Screen):
                print("Error: For 'Navigate' actionType, targetScreen must be a Screen object.")
        self.__targetScreen = targetScreen

    def __repr__(self):
        return (
            f'Button({self.name}, {self.label}, {self.description}, {self.visibility}, '
            f'{self.timestamp}, {self.label}, {self.buttonType}, {self.actionType}, {self.styling})'
        )


class Link(ViewComponent):
    """Represents a hyperlink component."""

    def __init__(
        self,
        name: str,
        description: str,
        label: str = "",
        url: Optional[str] = None,
        target: Optional[str] = None,
        rel: Optional[str] = None,
        visibility: str = "public",
        timestamp: int = None,
        styling: Styling = None,
    ):
        super().__init__(name, description, visibility, timestamp, styling=styling)
        self.label = label
        self.url = url
        self.target = target
        self.rel = rel

    @property
    def label(self) -> str:
        return self.__label

    @label.setter
    def label(self, value: str):
        self.__label = value or ""

    @property
    def url(self) -> Optional[str]:
        return self.__url

    @url.setter
    def url(self, value: Optional[str]):
        self.__url = value

    @property
    def target(self) -> Optional[str]:
        return self.__target

    @target.setter
    def target(self, value: Optional[str]):
        self.__target = value

    @property
    def rel(self) -> Optional[str]:
        return self.__rel

    @rel.setter
    def rel(self, value: Optional[str]):
        self.__rel = value

    def __repr__(self):
        return f'Link({self.name}, label={self.label}, url={self.url}, target={self.target})'


class EmbeddedContent(ViewComponent):
    """Represents embedded content such as iframes or maps."""

    def __init__(
        self,
        name: str,
        description: str,
        source: Optional[str] = None,
        content_type: Optional[str] = None,
        visibility: str = "public",
        timestamp: int = None,
        styling: Styling = None,
        extra_props: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, description, visibility, timestamp, styling=styling)
        self.source = source
        self.content_type = content_type
        self.extra_props = extra_props

    @property
    def source(self) -> Optional[str]:
        return self.__source

    @source.setter
    def source(self, value: Optional[str]):
        self.__source = value

    @property
    def content_type(self) -> Optional[str]:
        return self.__content_type

    @content_type.setter
    def content_type(self, value: Optional[str]):
        self.__content_type = value

    @property
    def extra_props(self) -> Dict[str, str]:
        return self.__extra_props

    @extra_props.setter
    def extra_props(self, props: Optional[Dict[str, str]]):
        self.__extra_props = {str(k): str(v) for k, v in (props or {}).items() if v is not None}

    def __repr__(self):
        return (
            f'EmbeddedContent({self.name}, source={"set" if self.source else None}, '
            f"content_type={self.content_type}, extra_props={bool(self.extra_props)})"
        )


# Image is a type of ViewComponent
class Image(ViewComponent):
    """Represents an image component and encapsulates image-specific properties.

    Args:
        name (str): The name of the image.
        description (str): The description of the image.
        timestamp (datetime): Object creation datetime (default is current time).
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).
        source (str | None): Raw URI/base64 string representing the image source.

    Attributes:
        name (str): The name of the image.
        description (str): The description of the image.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).
        source (str | None): Raw URI/base64 string representing the image source.
    """

    def __init__(
        self,
        name: str,
        description: str,
        timestamp: int = None,
        styling: Styling = None,
        source: str | None = None,
    ):
        super().__init__(
            name,
            description,
            timestamp=timestamp,
            styling=styling
        )
        self.source = source

    @property
    def source(self) -> str | None:
        return self.__source

    @source.setter
    def source(self, value: str | None):
        self.__source = value

    def __repr__(self):
        return f'Image({self.name},{self.description}, {self.timestamp}, {self.styling}, source={"set" if self.source else None})'


# InputField is a type of ViewComponent
class InputField(ViewComponent):
    """Represents an input field component and encapsulates specific properties of an input field, such as its type and validation rules.

     Args:
        name (str): The name of the input field.
        description (str): The description of the input field.
        timestamp (datetime): Object creation datetime (default is current time).
        field_type (InputFieldType): The type of the input field.
        validationRules (str): The validation rules for the input field.
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).

    Attributes:
        name (str): The name of the input field.
        description (str): The description of the input field.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        field_type (InputFieldType): The type of the input field.
        validationRules (str): The validation rules for the input field.
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).
    """

    def __init__(self, name: str, description: str, field_type: InputFieldType, timestamp: int = None, validationRules: str = None, visibility: str = "public", styling: Styling = None):
        super().__init__(name, description, visibility, timestamp, styling=styling)
        self.field_type: InputFieldType = field_type
        self.validationRules: str = validationRules


    @property
    def field_type(self) -> InputFieldType:
        """InputFieldType: Get the type of the input field."""
        return self.__field_type

    @field_type.setter
    def field_type(self, field_type: InputFieldType):
        """InputFieldType: Set the type of the collection data source."""
        self.__field_type = field_type


    @property
    def validationRules(self) -> str:
        """str: Get the validation rules of the input field."""
        return self.__validationRules


    @validationRules.setter
    def validationRules(self, validationRules: str):
        """str: Set the validation rules of the input field."""
        self.__validationRules = validationRules

    def __repr__(self):
        return f'InputField({self.name},{self.description}, {self.field_type}, {self.timestamp}, {self.validationRules}, {self.visibility}, {self.styling})'

# Form is a type of ViewComponent
class Form(ViewComponent):
    """Represents a form component and encapsulates the specific properties of a form, such as its name.

    Args:
        name (str): The name of the form.
        description (str): The description of the form.
        timestamp (datetime): Object creation datetime (default is current time).
        inputFields (set[InputField]): The set of input fields contained in the form.
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).

    Attributes:
        name (str): The name of the form.
        description (str): The description of the form.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        inputFields (set[InputField]): The set of input fields contained in the form.
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).
    """

    def __init__(self, name: str, description: str, inputFields: set[InputField], visibility: str = "public", timestamp: int = None, styling: Styling = None):
        super().__init__(name, description, visibility, timestamp, styling=styling)
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
        return f'Form({self.name},{self.description}, {self.inputFields}, {self.visibility}, {self.timestamp}, {self.styling})'

# MenuItem
class MenuItem(Element):
    """Represents an item of a menu.

    Args:
        label (str): The label of the menu item.
        url (str | None): Destination URL associated with the item.
        target (str | None): Link target behaviour (e.g., "_blank").
        rel (str | None): Relationship attribute for the link.
    """

    def __init__(self, label: str, url: Optional[str] = None, target: Optional[str] = None, rel: Optional[str] = None):
        super().__init__()
        self.label: str = label
        self.url: Optional[str] = url
        self.target: Optional[str] = target
        self.rel: Optional[str] = rel

    def __repr__(self):
        return f'MenuItem(label={self.label}, url={self.url}, target={self.target})'


# Menu is a type of ViewComponent
class Menu(ViewComponent):
    """Represents a menu component and encapsulates the specific properties of a menu, such as its name.

    Args:
        name (str): The name of the menu.
        description (str): The description of the menu.
        timestamp (datetime): Object creation datetime (default is current time).
        menuItems (set[MenuItem]): The set of menu items contained in the menu.
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).

    Attributes:
        name (str): The name of the menu.
        description (str): The description of the menu.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        menuItems (set[MenuItem]): The set of menu items contained in the menu.
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).
    """

    def __init__(self, name: str, description: str, menuItems: set[MenuItem], visibility: str = "public", timestamp: int = None, styling: Styling = None):
        super().__init__(name, description, visibility, timestamp, styling=styling)
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
        return f'Menu({self.name},{self.description}, {self.menuItems}, {self.visibility}, {self.timestamp}, {self.styling})'

#Text
class Text(ViewComponent):
    """Represents a text component.

    Args:
        name (str): The name of the text component.
        content (str): The content of the text component.
        description (str): The description of the text component.
        visibility (str): The visibility of the text component.
        timestamp (int): The timestamp of the text component.
        styling (Styling): The styling of the text component.

    Attributes:
        name (str): The name of the text component.
        content (str): The content of the text component.
        description (str): The description of the text component.
        visibility (str): The visibility of the text component.
        timestamp (int): The timestamp of the text component.
        styling (Styling): The styling of the text component.
    """

    def __init__(self, name: str, content: str, description: str = "", visibility: str = "public",
                 timestamp: int = None, styling: Styling = None):
        super().__init__(name, description, visibility, timestamp, styling=styling)
        self.content = content

    @property
    def content(self) -> str:
        """str: Get the content of the text component."""
        return self.__content

    @content.setter
    def content(self, content: str):
        """str: Set the content of the text component."""
        self.__content = content

    def __repr__(self):
        return (f'Text({self.name}, {self.content}, {self.description}, '
                f'{self.visibility}, {self.timestamp}, {self.styling})')

#GUIModel
class GUIModel(Model):
    """It is a subclass of the NamedElement class and encapsulates the properties and behavior of the GUI part of an application, including its name,
       package, version code, version name, modules, description, and screen compatibility.

    Args:
        name (str): The name of the model.
        timestamp (datetime): Object creation datetime (default is current time).
        package (str): The package of the model.
        versionCode (str): The version code of the model.
        versionName (str): The version name of the model.
        modules (set[Module]): The set of modules contained in the model.
        description (str): The description of the model.
        screenCompatibility (bool): Indicates whether the model has screen compatibility.

    Attributes:
        name (str): The name of the model.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        package (str): The package of the model.
        versionCode (str): The version code of the model.
        versionName (str): The version name of the model.
        modules (set[Module]): The set of modules contained in the model.
        description (str): The description of the model.
        screenCompatibility (bool): Indicates whether the model has screen compatibility.
    """
    def __init__(self, name: str, package: str, versionCode: str, versionName: str, modules: set[Module],
                 description: str, timestamp: int = None, screenCompatibility: bool = False):
        super().__init__(name, timestamp)
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
        return f'GUIModel({self.name}, {self.package}, {self.versionCode}, {self.versionName},{self.description},{self.timestamp}, {self.screenCompatibility}, {self.modules})'
