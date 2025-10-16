from enum import Enum

# LayoutType
class LayoutType(Enum):
    """
    Defines different layout types
    """
    GRID = "grid"          # Grid-based layout
    FLEX = "flex"          # Flexible layout
    SRACK = "stack"        # Stack elements vertically
    ROW = "row"            # Elements in a horizontal row
    COLUMN = "column"      # Elements in a vertical column
    ABSOLUTE = "absolute"  # Free positioning

# JustificationType
class JustificationType(Enum):
    """
    Defines justification options
    """
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    SPACE_BETWEEN = "space-between"
    SPACE_AROUND = "space-around"

# Aligment
class Alignment(Enum):
    """
    Defines alignment options
    """
    START = "start"
    END = "end"
    CENTER = "center"
    STRETCH = "stretch"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    INSIDE = "inside"
    OUTSIDE = "outside"

# UnitSize
class UnitSize(Enum):
    """
    Defines unit types for size measurements
    """
    PIXELS = "px"   # Pixels: Fixed size, measured in pixels
    PERCENTAGE = "%"   # Percentage: Relative size based on the parent container
    EM = "em"       # Element-relative: Relative to the element’s font size
    REM = "rem"     # Root-relative: Relative to the root font size
    VH = "vh"       # Viewport Height: % of the browser window’s height
    VW = "vw"       # Viewport Width: % of the browser window’s width
    AUTO = "auto"   # auto: Let the browser decide based on content

class PositionType(Enum):
    """
    Enumerates the types of positioning for a UI element.
    """
    STATIC = "static"
    RELATIVE = "relative"
    ABSOLUTE = "absolute"
    FIXED = "fixed"
    STICKY = "sticky"
    INLINE = "Inline"

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
        return (f"Size(width={self.width}, height={self.height}, padding={self.padding}, "
                f"unit={self.unit_size}, margin={self.margin}, font_size={self.font_size}, "
                f"icon_size={self.icon_size})"
        )

class Position:
    """
    Represents the positioning properties of a UI element.

    This class allows defining an element's placement in a UI layout, supporting different positioning
    types (static, absolute, etc.), offsets (top, left, right, bottom), alignment, and stacking order (z-index).

    Args:
        p_type (PositionType): The positioning type (e.g., STATIC, ABSOLUTE, RELATIVE).
        top (str, optional): Distance from the top (e.g., '10px', '10%', None).
        left (str, optional): Distance from the left (e.g., '10px', '10%', None).
        right (str, optional): Distance from the right (e.g., '10px', '10%', None).
        bottom (str, optional): Distance from the bottom (e.g., '10px', '10%', None).
        alignment (str, optional): Defines how the element aligns within its container (e.g., "CENTER", "LEFT").
        z_index (int, optional): Determines the stacking order; higher values appear above lower ones.
    
    Attributes:
        p_type (PositionType): The positioning type (e.g., STATIC, ABSOLUTE, RELATIVE).
        top (str): Distance from the top (e.g., '10px', '10%', None).
        left (str): Distance from the left (e.g., '10px', '10%', None).
        right (str): Distance from the right (e.g., '10px', '10%', None).
        bottom (str): Distance from the bottom (e.g., '10px', '10%', None).
        alignment (str): Defines how the element aligns within its container (e.g., "CENTER", "LEFT").
        z_index (int): Determines the stacking order; higher values appear above lower ones.
    """

    def __init__(self, p_type: PositionType = PositionType.STATIC, top: str = "auto", left: str = "auto",
                 right: str = "auto", bottom: str = "auto", alignment: Alignment = Alignment.LEFT,
                 z_index: int = 0):

        self.p_type: PositionType = p_type
        self.top: str = top
        self.left: str = left
        self.right: str = right
        self.bottom: str = bottom
        self.alignment: str = alignment
        self.z_index: int = z_index

    @property
    def p_type(self) -> PositionType:
        """PositionType: Get the position type."""
        return self.__p_type

    @p_type.setter
    def p_type(self, p_type: PositionType):
        """PositionType: Set the position type."""
        self.__p_type = p_type

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
        return (f"Position(p_type={self.p_type}, top={self.top}, left={self.left}, right={self.right}, "
                f"bottom={self.bottom}, alignment={self.alignment}, z_index={self.z_index})")

# Color
class Color:
    """Represents the color properties of a ViewElement.

    Args:
        background_color (str): The background color (#FFFFFF as default).
        text_color (str): The text color (#000000 as default).
        border_color (str): The border color (#CCCCCC as default).
        line_color (str): The line color (#000000 as default).
        grid_color (str): The grid color (#CCCCCC as default).
        axis_color (str): The axis color (#CCCCCC as default).
        bar_color (str): The bar color (#CCCCCC as default).
        fill_color (str): The fill color (#CCCCCC as default).
        color_palette (str): The color palette used for charts.

    Attributes:
        background_color (str): The background color.
        text_color (str): The text color.
        border_color (str): The border color.
        line_color (str): The line color.
        grid_color (str): The grid color.
        axis_color (str): The axis color.
        bar_color (str): The bar color.
        fill_color (str): The fill color.
        color_palette (str): The color palette used for charts.
    """

    def __init__(self, background_color: str = "#FFFFFF", text_color: str = "#000000",
                border_color: str = "#CCCCCC", line_color: str = "#000000",
                grid_color: str = "#CCCCCC", axis_color: str = "#CCCCCC",
                bar_color: str = "#CCCCCC", label_color: str = "#CCCCCC",
                fill_color: str = "#CCCCCC", color_palette: str = "default"):
        self.background_color: str = background_color
        self.text_color: str = text_color
        self.border_color: str = border_color
        self.line_color: str = line_color
        self.grid_color: str = grid_color
        self.axis_color: str = axis_color
        self.bar_color: str = bar_color
        self.label_color: str = label_color
        self.fill_color: str = fill_color
        self.color_palette: str = color_palette

    @property
    def background_color(self) -> str:
        """str: Get the background color."""
        return self.__background_color

    @background_color.setter
    def background_color(self, background: str):
        """str: Set the background color."""
        self.__background_color = background

    @property
    def text_color(self) -> str:
        """str: Get the text color."""
        return self.__text_color

    @text_color.setter
    def text_color(self, text: str):
        """str: Set the text color."""
        self.__text_color = text

    @property
    def border_color(self) -> str:
        """str: Get the border color."""
        return self.__border_color

    @border_color.setter
    def border_color(self, border: str):
        """str: Set the border color."""
        self.__border_color = border

    @property
    def line_color(self) -> str:
        """str: Get the line color."""
        return self.__line_color

    @line_color.setter
    def line_color(self, line: str):
        """str: Set the line color."""
        self.__line_color = line

    @property
    def grid_color(self) -> str:
        """str: Get the grid color."""
        return self.__grid_color

    @grid_color.setter
    def grid_color(self, grid: str):
        """str: Set the grid color."""
        self.__grid_color = grid

    @property
    def axis_color(self) -> str:
        """str: Get the axis color."""
        return self.__axis_color

    @axis_color.setter
    def axis_color(self, axis: str):
        """str: Set the axis color."""
        self.__axis_color = axis

    @property
    def bar_color(self) -> str:
        """str: Get the bar color."""
        return self.__bar_color

    @bar_color.setter
    def bar_color(self, bar_color: str):
        """str: Set the bar color."""
        self.__bar_color = bar_color

    @property
    def label_color(self) -> str:
        """str: Get the label color."""
        return self.__label_color

    @label_color.setter
    def label_color(self, label: str):
        """str: Set the label color."""
        self.__label_color = label

    @property
    def fill_color(self) -> str:
        """str: Get the fill color."""
        return self.__fill_color

    @fill_color.setter
    def fill_color(self, fill: str):
        """str: Set the fill color."""
        self.__fill_color = fill

    @property
    def color_palette(self) -> str:
        """Property: Get the color palette of the radial bar chart."""
        return self._color_palette

    @color_palette.setter
    def color_palette(self, value: str):
        """Property: Set the color palette of the radial bar chart."""
        self._color_palette = value

    def __repr__(self):
        return (
            f"Color(background={self.background_color}, text={self.text_color}, "
            f"line={self.line_color}, grid={self.grid_color}, axis={self.axis_color}, "
            f"bar={self.bar_color}, label={self.label_color}, fill={self.fill_color}, "
            f"color_palette={self.color_palette})"
        )

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
        return f"Styling(size={self.size}, position={self.position}, color={self.color})"

# Layout
class Layout:
    """Defines layout properties of a UI container.

    Args:
        layout_type (LayoutType): The type of layout (e.g., Flex, Grid). Defaults to Flex.
        orientation (str): The layout direction ('horizontal' or 'vertical'). Defaults to 'vertical'.
        padding (str): Space inside the container (e.g., "10px", "1em"). Defaults to "10px".
        margin (str): Space outside the container (e.g., "10px", "1em"). Defaults to "10px".
        gap (str): Space between elements (e.g., "5px", "0.5em"). Defaults to "5px".
        alignment (JustificationType): Justification of elements. Defaults to CENTER.
        wrap (bool): Whether elements wrap in a flex/grid container. Defaults to True.

    Attributes:
        layout_type (LayoutType): The type of layout.
        orientation (str): The direction of the layout.
        padding (str): The padding inside the container.
        margin (str): The margin outside the container.
        gap (str): The gap between elements.
        alignment (JustificationType): The alignment of elements within the layout.
        wrap (bool): Whether elements wrap in flex/grid layout.
    """

    def __init__(self, layout_type=LayoutType.FLEX, orientation: str = "vertical",
                 padding: str = "10px", margin: str = "10px", gap: str = "5px",
                 alignment: JustificationType = JustificationType.CENTER, wrap: bool = True):
        self.layout_type: LayoutType = layout_type
        self.orientation: str = orientation
        self.padding: str = padding
        self.margin: str = margin
        self.gap: str = gap
        self.alignment: JustificationType = alignment
        self.wrap: bool = wrap

    @property
    def layout_type(self) -> LayoutType:
        """LayoutType: Get the layout type."""
        return self.__layout_type

    @layout_type.setter
    def layout_type(self, layout_type: LayoutType):
        """LayoutType: Set the layout type."""
        self.__layout_type = layout_type

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
        return (f'LayoutClass(type={self.layout_type}, orientation={self.orientation}, '
                f'padding={self.padding}, margin={self.margin}, gap={self.gap}, '
                f'alignment={self.alignment}, wrap={self.wrap})')
