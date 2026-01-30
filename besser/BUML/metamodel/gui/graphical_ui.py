from enum import Enum
from besser.BUML.metamodel.structural import NamedElement, Class, Property, Model, Element


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
    RaisedButton = "RaisedButton"
    TextButton = "TextButton"
    OutlinedButton = "OutlinedButton"
    IconButton = "IconButton"
    FloatingActionButton = "FloatingActionButton"
    DropdownButton = "DropdownButton"
    ToggleButtons = "ToggleButtons"
    iOSStyleButton = "iOS-styleButton"
    CustomizableButton = "CustomizableButton"


# LayoutType
class LayoutType(Enum):
    """
    Defines different layout types
    """
    Grid = "grid"          # Grid-based layout
    Flex = "flex"          # Flexible layout
    Stack = "stack"        # Stack elements vertically
    Row = "row"            # Elements in a horizontal row
    Column = "column"      # Elements in a vertical column
    Absolute = "absolute"  # Free positioning

# JustificationType
class JustificationType(Enum):
    """
    Defines justification options
    """
    Left = "left"
    Right = "right"
    Center = "center"
    SpaceBetween = "space-between"
    SpaceAround = "space-around"


# UnitSize
class UnitSize(Enum):
    """
    Defines unit types for size measurements
    """
    PIXELS = "px"   # Pixels: Fixed size, measured in pixels
    PERCENT = "%"   # Percentage: Relative size based on the parent container
    EM = "em"       # Element-relative: Relative to the element’s font size
    REM = "rem"     # Root-relative: Relative to the root font size
    VH = "vh"       # Viewport Height: % of the browser window’s height
    VW = "vw"       # Viewport Width: % of the browser window’s width
    AUTO = "auto"   # auto: Let the browser decide based on content


class PositionType(Enum):
    """
    Enumerates the types of positioning for a UI element.
    """
    Static = "static"
    Relative = "relative"
    Absolute = "absolute"
    Fixed= "fixed"
    Sticky= "sticky"
    Inline = "Inline"


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
      return f'DataSourceElement({self.name}, {self.dataSourceClass},{self.fields})'

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


# Size
class Size:
    """Represents the size properties of a UI element.

    Args:
        width (str): The width of the element (e.g., "100px", "80%", "auto").
        height (str): The height of the element.
        padding (str, optional): The padding inside the element.
        margin (str, optional): The margin outside the element.
        font_size (str, optional): The font size for text-based elements.
        icon_size (str, optional): The size of icons inside elements.
        unit_size (UnitSize): The unit of measurement (px, %, em, etc.).

    Attributes:
        width (str): The width of the element.
        height (str): The height of the element.
        padding (str, optional): The padding inside the element.
        margin (str, optional): The margin outside the element.
        font_size (str, optional): The font size for text-based elements.
        icon_size (str, optional): The size of icons inside elements.
        unit_size (UnitSize): The unit of measurement.
    """

    def __init__(self, width: str="auto", height: str="auto", padding: str="0", margin: str="0",
                 font_size: str=None, icon_size:str=None, unit_size: UnitSize=UnitSize.PIXELS):

        self.width: str = width
        self.height: str = height
        self.padding: str = padding
        self.margin: str = margin
        self.font_size: str = font_size
        self.icon_size: str = icon_size
        self.unit_size = unit_size

    @property
    def width(self) -> str:
        """str: Get the width of the element."""
        return self.__width

    @width.setter
    def width(self, width: str):
        """Set the width of the element."""
        self.__width = width

    @property
    def height(self) -> str:
        """str: Get the height of the element."""
        return self.__height

    @height.setter
    def height(self, height: str):
        """Set the height of the element."""
        self.__height = height

    @property
    def padding(self) -> str:
        """str: Get the padding of the element."""
        return self.__padding

    @padding.setter
    def padding(self, padding: str):
        """Set the padding of the element."""
        self.__padding = padding

    @property
    def margin(self) -> str:
        """str: Get the margin of the element."""
        return self.__margin

    @margin.setter
    def margin(self, margin: str):
        """Set the margin of the element."""
        self.__margin = margin

    def __repr__(self):
        return f"Size(width={self.width}, height={self.height}, padding={self.padding}, unit={self.unit_size}, margin={self.margin}\
                  font_size={self.font_size}, icon_size={self.icon_size}, unit_size={self.unit_size}  )"



class Position:
    """
    Represents the positioning properties of a UI element.

    This class allows defining an element's placement in a UI layout, supporting different positioning
    types (static, absolute, etc.), offsets (top, left, right, bottom), alignment, and stacking order (z-index).

    Attributes:
        position (PositionType): The positioning type (e.g., STATIC, ABSOLUTE, RELATIVE).
        top (str, optional): Distance from the top (e.g., '10px', '10%', None).
        left (str, optional): Distance from the left (e.g., '10px', '10%', None).
        right (str, optional): Distance from the right (e.g., '10px', '10%', None).
        bottom (str, optional): Distance from the bottom (e.g., '10px', '10%', None).
        alignment (str, optional): Defines how the element aligns within its container (e.g., "CENTER", "LEFT").
        z_index (int, optional): Determines the stacking order; higher values appear above lower ones.
    """


    def __init__(self, type: PositionType = PositionType.Static, top: str = None, left: str = None,
                 right: str = None, bottom: str = None, alignment: str = None, z_index: int = 0):

        self.type: PositionType = type      # Positioning type (static, relative, etc.)
        self.top: str = top                # Distance from top (px, %, etc.)
        self.left: str = left              # Distance from left
        self.right: str = right            # Distance from right
        self.bottom: str = bottom          # Distance from bottom
        self.alignment: str = alignment    # Alignment (e.g., CENTER, LEFT)
        self.z_index: int = z_index        # Stacking order

    @property
    def type(self) -> PositionType:
        """PositionType: Get the position type."""
        return self.__type

    @type.setter
    def type(self, type: PositionType):
        """PositionType: Set the position type."""
        #if isinstance(position, PositionType):
        self.__type = type
        #else:
         #   raise ValueError("Invalid position type.")

    @property
    def top(self) -> str:
        """str: Get the distance from the top."""
        return self.__top

    @top.setter
    def top(self, top: str):
        """str: Set the distance from the top."""
        self.__top = top

    @property
    def left(self) -> str:
        """str: Get the distance from the left."""
        return self.__left

    @left.setter
    def left(self, left: str):
        """str: Set the distance from the left."""
        self.__left = left

    @property
    def right(self) -> str:
        """str: Get the distance from the right."""
        return self.__right

    @right.setter
    def right(self, right: str):
        """str: Set the distance from the right."""
        self.__right = right

    @property
    def bottom(self) -> str:
        """str: Get the distance from the bottom."""
        return self.__bottom

    @bottom.setter
    def bottom(self, bottom: str):
        """str: Set the distance from the bottom."""
        self.__bottom = bottom

    @property
    def alignment(self) -> str:
        """str: Get the alignment of the element."""
        return self.__alignment

    @alignment.setter
    def alignment(self, alignment: str):
        """str: Set the alignment of the element."""
        self.__alignment = alignment

    @property
    def z_index(self) -> int:
        """int: Get the z-index value."""
        return self.__z_index

    @z_index.setter
    def z_index(self, z_index: int):
        """int: Set the z-index value."""
        if isinstance(z_index, int) and z_index >= 0:
            self.__z_index = z_index
        else:
            raise ValueError("z_index must be a non-negative integer.")

    def __repr__(self):
        return f"Position(type={self.type}, top={self.top}, left={self.left}, " \
               f"right={self.right}, bottom={self.bottom}, alignment={self.alignment}, z_index={self.z_index})"

# Color
class Color:
    """Represents the color properties of a ViewElement.

    Args:
        background_color (str): The background color (e.g., hex code or color name).
        text_color (str): The text color.
        border_color (str): The border color.

    Attributes:
        background_color (str): The background color.
        text_color (str): The text color.
        border_color (str): The border color.
    """

    def __init__(self, background_color: str, text_color: str, border_color: str):
        self.background_color: str = background_color
        self.text_color: str = text_color
        self.border_color: str = border_color


    @property
    def background_color(self) -> str:
        """str: Get the background color."""
        return self.__background_color

    @background_color.setter
    def background_color(self, background_color: str):
        """str: Set the background color."""
        self.__background_color = background_color

    @property
    def text_color(self) -> str:
        """str: Get the text color."""
        return self.__text_color

    @text_color.setter
    def text_color(self, text_color: str):
        """str: Set the text color."""
        self.__text_color = text_color

    @property
    def border_color(self) -> str:
        """str: Get the border color."""
        return self.__border_color

    @border_color.setter
    def border_color(self, border_color: str):
        """str: Set the border color."""
        self.__border_color = border_color


    def __repr__(self):
        return f"Color(background_color={self.background_color}, text_color={self.text_color}, border_color={self.border_color})"


# Styling
class Styling:
    """
    Represents the general styling properties of a UI element.

    Args:
        size (Size, optional): The size settings of the UI element (default: None).
        position (Position, optional): The position settings of the UI element (default: None).
        color (Color, optional): The color settings of the UI element (default: None).

    Attributes:
        size (Size | None): The size settings of the UI element.
        position (Position | None): The position settings of the UI element.
        color (Color | None): The color settings of the UI element.
    """
    def __init__(self, size:Size=None, position: Position=None, color: Color=None):
        self.size: Size=size
        self.position: Position = position
        self.color: Position = color

    @property
    def size(self) -> Size | None:
        """Size | None: Get the size settings of the styling."""
        return self.__size

    @size.setter
    def size(self, size: Size | None):
        """Size | None: Set the size settings of the styling."""
        self.__size = size

    @property
    def position(self) -> Position | None:
        """Position | None: Get the position settings of the styling."""
        return self.__position

    @position.setter
    def position(self, position: Position | None):
        """Size | None: Set the position settings of the styling."""
        self.__position = position

    @property
    def color(self) -> Color | None:
        """Color | None: Get the color settings of the styling."""
        return self.__color

    @color.setter
    def color(self, color: Color | None):
        """Size | None: Set the color settings of the styling."""
        self.__color = color

    def __repr__(self):
        return (f"Styling(size={self.size}, position={self.position}, color={self.color}")


#ViewElement
class ViewElement(NamedElement):
    """
    Represents a view element with optional size, position, and color attributes.

    Args:
        name (str): The name of the view element.
        description (str): A brief description of the view element.
        timestamp (datetime): Object creation datetime (default is current time).
        visibility (str, optional): Visibility scope (default: "public").
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).


    Attributes:
        name (str): The name of the view element.
        description (str): A brief description of the view element.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        visibility (str): Visibility scope (default: "public").
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).

    """

    def __init__(self, name: str, description: str, visibility: str = "public", timestamp: int = None, styling: Styling = None):
        super().__init__(name, visibility, timestamp)
        self.description: str = description
        self.styling: Styling = styling


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


    def __repr__(self):
        return (f"ViewElement(name={self.name}, description={self.description}, visibility={self.visibility}, "
                f"timestamp={self.timestamp}, styling={self.styling})")


# Layout
class Layout:
    """Defines layout properties of a UI container.

    Args:
        layoutType (LayoutType): The type of layout (e.g., Flex, Grid). Defaults to Flex.
        orientation (str): The layout direction ('horizontal' or 'vertical'). Defaults to 'vertical'.
        padding (str): Space inside the container (e.g., "10px", "1em"). Defaults to "10px".
        margin (str): Space outside the container (e.g., "10px", "1em"). Defaults to "10px".
        gap (str): Space between elements (e.g., "5px", "0.5em"). Defaults to "5px".
        alignment (JustificationType): Justification of elements. Defaults to CENTER.
        wrap (bool): Whether elements wrap in a flex/grid container. Defaults to True.

    Attributes:
        layoutType (LayoutType): The type of layout.
        orientation (str): The direction of the layout.
        padding (str): The padding inside the container.
        margin (str): The margin outside the container.
        gap (str): The gap between elements.
        alignment (JustificationType): The alignment of elements within the layout.
        wrap (bool): Whether elements wrap in flex/grid layout.
    """

    def __init__(self, type=LayoutType.Flex, orientation: str = "vertical",
                 padding: str = "10px", margin: str = "10px", gap: str = "5px",
                 alignment: JustificationType = JustificationType.Center, wrap: bool = True):
        self.type: LayoutType = type
        self.orientation: str = orientation
        self.padding: str = padding
        self.margin: str = margin
        self.gap: str = gap
        self.alignment: JustificationType = alignment
        self.wrap: bool = wrap

    @property
    def type(self) -> LayoutType:
        """LayoutType: Get the layout type."""
        return self.__type

    @type.setter
    def type(self, type: LayoutType):
        """LayoutType: Set the layout type."""
        self.__type = type

    @property
    def orientation(self) -> str:
        """str: Get the layout orientation ('horizontal' or 'vertical')."""
        return self.__orientation

    @orientation.setter
    def orientation(self, orientation: str):
        """str: Set the layout orientation ('horizontal' or 'vertical')."""
        if orientation not in ["horizontal", "vertical"]:
            raise ValueError("Orientation must be either 'horizontal' or 'vertical'.")
        self.__orientation = orientation

    @property
    def padding(self) -> str:
        """str: Get the padding value."""
        return self.__padding

    @padding.setter
    def padding(self, padding: str):
        """str: Set the padding value."""
        self.__padding = padding

    @property
    def margin(self) -> str:
        """str: Get the margin value."""
        return self.__margin

    @margin.setter
    def margin(self, margin: str):
        """str: Set the margin value."""
        self.__margin = margin

    @property
    def gap(self) -> str:
        """str: Get the gap value."""
        return self.__gap

    @gap.setter
    def gap(self, gap: str):
        """str: Set the gap value."""
        self.__gap = gap

    @property
    def alignment(self) -> JustificationType:
        """JustificationType: Get the alignment of elements."""
        return self.__alignment

    @alignment.setter
    def alignment(self, alignment: JustificationType):
        """JustificationType: Set the alignment of elements."""
        self.__alignment = alignment

    @property
    def wrap(self) -> bool:
        """bool: Get whether elements wrap in a flex/grid container."""
        return self.__wrap

    @wrap.setter
    def wrap(self, wrap: bool):
        """bool: Set whether elements wrap in a flex/grid container."""
        self.__wrap = wrap

    def __repr__(self):
        return (f'LayoutClass(type={self.layoutType}, orientation={self.orientation}, '
                f'padding={self.padding}, margin={self.margin}, gap={self.gap}, '
                f'alignment={self.alignment}, wrap={self.wrap})')


#ViewComponent
class ViewComponent(ViewElement):
    """
    Represents a view component that extends a generic ViewElement.

    Args:
        name (str): The name of the view component.
        description (str): A brief description of the view component.
        timestamp (datetime): Object creation datetime (default is current time).
        visibility (str, optional): The visibility scope (default: "public").
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).

    Attributes:
        name (str): The name of the view component.
        description (str): A brief description of the view component.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        visibility (str): The visibility scope of the component.
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).
    """

    def __init__(self, name: str, description: str, visibility: str = "public", timestamp: int = None, styling: Styling = None):
        super().__init__(name, description, visibility, timestamp, styling)

    def __repr__(self):
        return f'ViewComponent({self.name}, description={self.description}, visibility={self.visibility}, timestamp={self.timestamp}, styling={self.styling})'

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

    def __init__(self, name: str, description: str, view_elements: set[ViewElement], timestamp: int = None, layout: Layout=None, styling: Styling=None):
        super().__init__(name, description, timestamp, styling)
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
        return f'ViewContainer({self.name}, description={self.description}, timestamp={self.timestamp}, view_elements={self.view_elements}, layout={self.layout}, styling={self.styling})'

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

    def __init__(self, name: str, description: str, view_elements: set[ViewElement], x_dpi: str, y_dpi: str, screen_size: str, timestamp: int = None, is_main_page: bool = False, layout: Layout=None):
        super().__init__(name, description, view_elements, timestamp, layout)
        self.x_dpi: str = x_dpi
        self.y_dpi: str = y_dpi
        self.screen_size: str = screen_size
        self.is_main_page: bool = is_main_page

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
    def is_main_page(self) -> bool:
        """bool: Get whether the screen is main page."""
        return self.__is_main_page

    @is_main_page.setter
    def is_main_page(self, is_main_page: bool):
        """bool: Set whether the screen is main page."""
        self.__is_main_page = is_main_page


    def __repr__(self):
      return f'Screen({self.name}, description={self.description}, {self.x_dpi}, {self.y_dpi}, {self.screen_size}, timestamp={self.timestamp}, {self.view_elements}, {self.is_main_page})'

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
            names = [DataSource.name for DataSource in list_sources]
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

    def __init__(
        self,
        name: str,
        description: str,
        label: str,
        buttonType: ButtonType,
        actionType: ButtonActionType,
        targetScreen: Screen = None,
        timestamp: int = None,
        visibility: str = "public",
        styling: Styling = None
    ):
        self._in_init = True  # 🔒 Skip validation during init
        super().__init__(name, description, visibility, timestamp, styling=styling)
        self.label = label
        self.buttonType = buttonType
        self.actionType = actionType
        #self.__targetScreen = None  # avoid triggering setter logic before init is done
        self.targetScreen = targetScreen
        self._in_init = False  # ✅ Turn validation back on

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
        if not getattr(self, "_in_init", False):  # ✅ Only validate after init
            if self.actionType == ButtonActionType.Navigate:
                if targetScreen is None:
                    raise ValueError("A target Screen must be specified for 'Navigate' action.")
                elif not isinstance(targetScreen, Screen):
                    raise ValueError("The targetScreen must be a Screen instance when actionType is 'Navigate'.")
            elif targetScreen is not None:
                raise ValueError("targetScreen must be None when actionType is not 'Navigate'.")
        self.__targetScreen = targetScreen

    def __repr__(self):
        return (
            f'Button({self.name}, {self.label}, {self.description}, {self.visibility}, '
            f'{self.timestamp}, {self.label}, {self.buttonType}, {self.actionType}, {self.styling})'
        )


# Image is a type of ViewComponent
class Image(ViewComponent):
    """Represents an image component and encapsulates the specific properties of a image, such as its name.

    Args:
        name (str): The name of the image.
        description (str): The description of the input field.
        timestamp (datetime): Object creation datetime (default is current time).
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).

    Attributes:
        name (str): The name of the image.
        description (str): The description of the input field.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).
    """

    def __init__(self, name: str, description: str, timestamp: int = None, styling: Styling =None):
        super().__init__(name, description, timestamp, styling)

    def __repr__(self):
     return f'Image({self.name},{self.description}, {self.timestamp}, {self.styling})'


# InputField is a type of ViewComponent
class InputField(ViewComponent):
    """Represents an input field component and encapsulates specific properties of an input field, such as its type and validation rules.

     Args:
        name (str): The name of the input field.
        description (str): The description of the input field.
        timestamp (datetime): Object creation datetime (default is current time).
        type (str): The type of the input field.
        validationRules (str): The validation rules for the input field.
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).

    Attributes:
        name (str): The name of the input field.
        description (str): The description of the input field.
        timestamp (datetime): Inherited from NamedElement; object creation datetime (default is current time).
        type (str): The type of the input field.
        validationRules (str): The validation rules for the input field.
        styling (Styling, optional): The styling configuration for the view element, which includes size, position, and color settings (default: None).
    """

    def __init__(self, name: str, description: str, type: InputFieldType, timestamp: int = None, validationRules: str = None, visibility: str = "public", styling: Styling = None):
        super().__init__(name, description, visibility, timestamp, styling=styling)
        self.type: InputFieldType= type
        self.validationRules: str = validationRules


    @property
    def type(self) -> InputFieldType:
      """InputFieldType: Get the type of the input field."""
      return self.__type

    @type.setter
    def type(self, type: InputFieldType):
      """InputFieldType: Set the type of the collection data source."""
      self.__type = type


    @property
    def validationRules(self) -> str:
        """str: Get the validation rules of the input field."""
        return self.__validationRules


    @validationRules.setter
    def validationRules(self, validationRules: str):
        """str: Set the validation rules of the input field."""
        self.__validationRules = validationRules

    def __repr__(self):
     return f'InputField({self.name},{self.description}, {self.type}, {self.timestamp}, {self.validationRules}, {self.visibility}, {self.styling})'

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

    Attributes:
        label (str): The label of the menu item.
    """

    def __init__(self, label: str):
        super().__init__()
        self.label: str = label

    def __repr__(self):
     return f'MenuItem({self.label})'


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
    def __init__(self, name: str, package: str, versionCode: str, versionName: str, modules: set[Module], description: str, timestamp: int = None, screenCompatibility: bool = False):
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





