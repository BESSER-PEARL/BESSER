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

# Screen: wrapper
wrapper = Screen(name="wrapper", description="Home", is_main_page=True, screen_size="Medium")
wrapper_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
wrapper_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
wrapper_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
wrapper_styling = Styling(size=wrapper_styling_size, position=wrapper_styling_pos, color=wrapper_styling_color)
wrapper.styling = wrapper_styling
iakik = Image(name="iakik", description="Image component")
ic5o8 = ViewContainer(name="ic5o8", description="footer container")
i91sd = ViewContainer(name="i91sd", description=" component")
component = ViewContainer(name="Component", description=" component")
i3kav = Text(name="i3kav", content="About Us", description="Text element")
i3kav_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i3kav_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i3kav_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i3kav_styling = Styling(size=i3kav_styling_size, position=i3kav_styling_pos, color=i3kav_styling_color)
i3kav.styling = i3kav_styling
imucf = Text(name="imucf", content="Your company description goes here.", description="Text element")
imucf_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
imucf_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
imucf_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.8")
imucf_styling = Styling(size=imucf_styling_size, position=imucf_styling_pos, color=imucf_styling_color)
imucf.styling = imucf_styling
component.view_elements = {i3kav, imucf}
component_2 = ViewContainer(name="Component_2", description=" component")
ijaal = Text(name="ijaal", content="Quick Links", description="Text element")
ijaal_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ijaal_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ijaal_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ijaal_styling = Styling(size=ijaal_styling_size, position=ijaal_styling_pos, color=ijaal_styling_color)
ijaal.styling = ijaal_styling
inqz6 = ViewContainer(name="inqz6", description="ul container")
i7uhn = ViewContainer(name="i7uhn", description="li container")
ik1bx = ViewComponent(name="ik1bx", description="Link element")
ik1bx_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ik1bx_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ik1bx_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
ik1bx_styling = Styling(size=ik1bx_styling_size, position=ik1bx_styling_pos, color=ik1bx_styling_color)
ik1bx.styling = ik1bx_styling
i7uhn.view_elements = {ik1bx}
ipeki = ViewContainer(name="ipeki", description="li container")
i91vb = ViewComponent(name="i91vb", description="Link element")
i91vb_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i91vb_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i91vb_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
i91vb_styling = Styling(size=i91vb_styling_size, position=i91vb_styling_pos, color=i91vb_styling_color)
i91vb.styling = i91vb_styling
ipeki.view_elements = {i91vb}
irrop = ViewContainer(name="irrop", description="li container")
i5851 = ViewComponent(name="i5851", description="Link element")
i5851_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i5851_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i5851_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
i5851_styling = Styling(size=i5851_styling_size, position=i5851_styling_pos, color=i5851_styling_color)
i5851.styling = i5851_styling
irrop.view_elements = {i5851}
inqz6.view_elements = {i7uhn, ipeki, irrop}
component_2.view_elements = {ijaal, inqz6}
component_3 = ViewContainer(name="Component_3", description=" component")
ih4x7 = Text(name="ih4x7", content="Contact", description="Text element")
ih4x7_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ih4x7_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ih4x7_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ih4x7_styling = Styling(size=ih4x7_styling_size, position=ih4x7_styling_pos, color=ih4x7_styling_color)
ih4x7.styling = ih4x7_styling
ihewm = Text(name="ihewm", content="Email: info@example.com\nPhone: (123) 456-7890", description="Text element")
ihewm_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ihewm_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ihewm_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.8")
ihewm_styling = Styling(size=ihewm_styling_size, position=ihewm_styling_pos, color=ihewm_styling_color)
ihewm.styling = ihewm_styling
component_3.view_elements = {ih4x7, ihewm}
i91sd.view_elements = {component, component_2, component_3}
i91sd_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="30px")
i91sd.layout = i91sd_layout
ibe6b = Text(name="ibe6b", content="© 2025 Your Company. All rights reserved.", description="Text element")
ibe6b_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ibe6b_styling_pos = Position(alignment="center", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ibe6b_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.7")
ibe6b_styling = Styling(size=ibe6b_styling_size, position=ibe6b_styling_pos, color=ibe6b_styling_color)
ibe6b.styling = ibe6b_styling
ic5o8.view_elements = {i91sd, ibe6b}
iroxk = Button(name="iroxk", description="Button component", label="Button", buttonType=ButtonType.CustomizableButton, actionType=ButtonActionType.View)
iroxk_event_0_action_0 = Transition(name="navigate_to_page:nvPCYoWq2RVUadd9w", description="Navigate to page:nvPCYoWq2RVUadd9w", target_screen=wrapper_2)
iroxk_event_0 = Event(name="onClick_iroxk", event_type=EventType.OnClick, actions={iroxk_event_0_action_0})
iroxk.events = {iroxk_event_0}
iroxk_styling_size = Size(width="auto", height="auto", padding="12px 24px", margin="0", font_size="16px", unit_size=UnitSize.PIXELS)
iroxk_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iroxk_styling_color = Color(background_color="#3498db", text_color="#ffffff", border_color="#CCCCCC")
iroxk_styling = Styling(size=iroxk_styling_size, position=iroxk_styling_pos, color=iroxk_styling_color)
iroxk.styling = iroxk_styling
iygbn = ViewContainer(name="iygbn", description=" component")
cell = ViewContainer(name="Cell", description=" component")
ijtmy = BarChart(name="ijtmy", bar_width=30, orientation="vertical", show_grid=True, show_legend=True, show_tooltip=True, stacked=False, animate=True, legend_position="top", grid_color="#e0e0e0", bar_gap=4)
ijtmy_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
ijtmy_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ijtmy_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ijtmy_styling = Styling(size=ijtmy_styling_size, position=ijtmy_styling_pos, color=ijtmy_styling_color)
ijtmy.styling = ijtmy_styling
cell.view_elements = {ijtmy}
iygbn.view_elements = {cell}
wrapper.view_elements = {iakik, ic5o8, iroxk, iygbn}


# Screen: wrapper_2
wrapper_2 = Screen(name="wrapper_2", description="about", screen_size="Medium")
wrapper_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
wrapper_2_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
wrapper_2_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
wrapper_2_styling = Styling(size=wrapper_2_styling_size, position=wrapper_2_styling_pos, color=wrapper_2_styling_color)
wrapper_2.styling = wrapper_2_styling
iepd = ViewContainer(name="iepd", description=" component")
h1 = Text(name="h1", content="New Page: about", description="Text element")
h1_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
h1_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
h1_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
h1_styling = Styling(size=h1_styling_size, position=h1_styling_pos, color=h1_styling_color)
h1.styling = h1_styling
iepd.view_elements = {h1}
wrapper_2.view_elements = {iepd}

gui_module = Module(
    name="GUI_Module",
    screens={wrapper, wrapper_2}
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
