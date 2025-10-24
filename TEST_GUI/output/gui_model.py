###############
#  GUI MODEL  #
###############

from besser.BUML.metamodel.gui import (
    GUIModel, Module, Screen,
    ViewComponent, ViewContainer,
    Button, ButtonType, ButtonActionType,
    Text, Image, InputField, InputFieldType,
    Form, Menu, MenuItem, DataList,
    DataSource, DataSourceElement,
    Styling, Size, Position, Color, Layout, LayoutType,
    UnitSize, PositionType, Alignment
)
from besser.BUML.metamodel.gui.dashboard import (
    LineChart, BarChart, PieChart, RadarChart, RadialBarChart
)
from besser.BUML.metamodel.gui.events_actions import (
    Event, EventType, Transition, Create, Read, Update, Delete, Parameter
)
from besser.BUML.metamodel.gui.binding import DataBinding

# Module: GUI_Module

# Screen: ifai
ifai = Screen(name="ifai", description="Home", is_main_page=True, screen_size="Medium")
ifai_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ifai_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ifai_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ifai_styling = Styling(size=ifai_styling_size, position=ifai_styling_pos, color=ifai_styling_color)
ifai.styling = ifai_styling
i9j9 = LineChart(name="i9j9", line_width=2, show_grid=True, show_legend=True, show_tooltip=True, curve_type="monotone", animate=True, legend_position="top", grid_color="#e0e0e0", dot_size=5)
# TODO: Set data_binding for i9j9
i9j9_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
i9j9_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i9j9_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i9j9_styling = Styling(size=i9j9_styling_size, position=i9j9_styling_pos, color=i9j9_styling_color)
i9j9.styling = i9j9_styling
itxi = BarChart(name="itxi", bar_width=30, orientation="vertical", show_grid=True, show_legend=True, show_tooltip=True, stacked=False, animate=True, legend_position="top", grid_color="#e0e0e0", bar_gap=4)
itxi_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
itxi_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
itxi_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
itxi_styling = Styling(size=itxi_styling_size, position=itxi_styling_pos, color=itxi_styling_color)
itxi.styling = itxi_styling
ifai.view_elements = {i9j9, itxi}

gui_module = Module(
    name="GUI_Module",
    screens={ifai}
)

# GUI Model
gui_model = GUIModel(
    name="GUI",
    package="ai.factories",
    versionCode="1.0",
    versionName="1.0",
    modules={gui_module},
    description="GUI"
)
