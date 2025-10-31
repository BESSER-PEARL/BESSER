####################
# STRUCTURAL MODEL #
####################

from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType,
    AnyType, Constraint, AssociationClass, Metadata
)

# Classes
Sensor = Class(name="Sensor")

# Sensor class attributes and methods
Sensor_date: Property = Property(name="date", type=DateType)
Sensor_value: Property = Property(name="value", type=FloatType)
Sensor.attributes={Sensor_value, Sensor_date}

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={Sensor},
    associations={},
    generalizations={}
)


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

# Screen: icnh
icnh = Screen(name="icnh", description="Home", is_main_page=True, route_path="/home", screen_size="Medium")
icnh_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
icnh_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
icnh_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
icnh_styling = Styling(size=icnh_styling_size, position=icnh_styling_pos, color=icnh_styling_color)
icnh.styling = icnh_styling
igpgl = Text(name="igpgl", content="Card Title", description="Text element")
igpgl_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
igpgl_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
igpgl_styling_color = Color(background_color="#FFFFFF", text_color="#333", border_color="#CCCCCC")
igpgl_styling = Styling(size=igpgl_styling_size, position=igpgl_styling_pos, color=igpgl_styling_color)
igpgl.styling = igpgl_styling
igpgl.display_order = 0
igpgl.component_id = "igpgl"
igpgl.component_type = "text"
igpgl.tag_name = "h3"
igpgl.custom_attributes = {"id": "igpgl"}
i4e15 = Text(name="i4e15", content="This is a card component. Add your content here. Cards are great for organizing information.", description="Text element")
i4e15_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
i4e15_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i4e15_styling_color = Color(background_color="#FFFFFF", text_color="#666", border_color="#CCCCCC")
i4e15_styling = Styling(size=i4e15_styling_size, position=i4e15_styling_pos, color=i4e15_styling_color)
i4e15.styling = i4e15_styling
i4e15.display_order = 1
i4e15.component_id = "i4e15"
i4e15.component_type = "text"
i4e15.tag_name = "p"
i4e15.custom_attributes = {"id": "i4e15"}
imi5f = Button(name="imi5f", description="Button component", label="Learn More", buttonType=ButtonType.CustomizableButton, actionType=ButtonActionType.View)
imi5f_styling_size = Size(width="auto", height="auto", padding="10px 20px", margin="0", unit_size=UnitSize.PIXELS)
imi5f_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
imi5f_styling_color = Color(background_color="#2196f3", text_color="white", border_color="#CCCCCC")
imi5f_styling = Styling(size=imi5f_styling_size, position=imi5f_styling_pos, color=imi5f_styling_color)
imi5f.styling = imi5f_styling
imi5f.display_order = 2
imi5f.component_id = "imi5f"
imi5f.component_type = "button"
imi5f.tag_name = "button"
imi5f.custom_attributes = {"type": "button", "id": "imi5f"}
ilmfj = Link(name="ilmfj", description="Link element", label="", url="#")
ilmfj_styling_size = Size(width="auto", height="auto", padding="12px 24px", margin="0", font_size="16px", unit_size=UnitSize.PIXELS)
ilmfj_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ilmfj_styling_color = Color(background_color="#3498db", text_color="#ffffff", border_color="#CCCCCC")
ilmfj_styling = Styling(size=ilmfj_styling_size, position=ilmfj_styling_pos, color=ilmfj_styling_color)
ilmfj.styling = ilmfj_styling
ilmfj.display_order = 3
ilmfj.component_id = "ilmfj"
ilmfj.component_type = "link-button"
ilmfj.tag_name = "a"
ilmfj.css_classes = ["link-button-component"]
ilmfj.custom_attributes = {"href": "#", "id": "ilmfj"}
i3kgi = Button(name="i3kgi", description="Button component", label="Button", buttonType=ButtonType.CustomizableButton, actionType=ButtonActionType.View)
i3kgi_event_0_action_0 = Transition(name="navigate_to_page:DDahouVru1rTBJNJ9", description="Navigate to page:DDahouVru1rTBJNJ9", target_screen=wrapper)
i3kgi_event_0 = Event(name="onClick_i3kgi", event_type=EventType.OnClick, actions={i3kgi_event_0_action_0})
i3kgi.events = {i3kgi_event_0}
i3kgi_styling_size = Size(width="auto", height="auto", padding="12px 24px", margin="0", font_size="16px", unit_size=UnitSize.PIXELS)
i3kgi_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i3kgi_styling_color = Color(background_color="#3498db", text_color="#ffffff", border_color="#CCCCCC")
i3kgi_styling = Styling(size=i3kgi_styling_size, position=i3kgi_styling_pos, color=i3kgi_styling_color)
i3kgi.styling = i3kgi_styling
i3kgi.display_order = 4
i3kgi.component_id = "i3kgi"
i3kgi.component_type = "action-button"
i3kgi.tag_name = "button"
i3kgi.css_classes = ["action-button-component"]
i3kgi.custom_attributes = {"type": "button", "id": "i3kgi"}
ia1em = ViewContainer(name="ia1em", description=" component", view_elements={igpgl, i4e15, imi5f, ilmfj, i3kgi})
ia1em_styling_size = Size(width="auto", height="auto", padding="30px", margin="20px", unit_size=UnitSize.PIXELS)
ia1em_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ia1em_styling_color = Color(background_color="white", text_color="#000000", border_color="#CCCCCC")
ia1em_styling = Styling(size=ia1em_styling_size, position=ia1em_styling_pos, color=ia1em_styling_color)
ia1em.styling = ia1em_styling
ia1em.display_order = 0
ia1em.component_id = "ia1em"
ia1em.custom_attributes = {"id": "ia1em"}
ig8z7 = Text(name="ig8z7", content="Our Features", description="Text element")
ig8z7_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="36px", unit_size=UnitSize.PIXELS)
ig8z7_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ig8z7_styling_color = Color(background_color="#FFFFFF", text_color="#333", border_color="#CCCCCC")
ig8z7_styling = Styling(size=ig8z7_styling_size, position=ig8z7_styling_pos, color=ig8z7_styling_color)
ig8z7.styling = ig8z7_styling
ig8z7.display_order = 0
ig8z7.component_id = "ig8z7"
ig8z7.component_type = "text"
ig8z7.tag_name = "h2"
ig8z7.custom_attributes = {"id": "ig8z7"}
itv3t = Text(name="itv3t", content="🚀", description="Text element")
itv3t_styling_size = Size(width="80px", height="80px", padding="0", margin="0 auto 20px", font_size="36px", unit_size=UnitSize.PIXELS)
itv3t_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
itv3t_styling_color = Color(background_color="linear-gradient(135deg, #667eea 0%, #764ba2 100%)", text_color="white", border_color="#CCCCCC")
itv3t_styling = Styling(size=itv3t_styling_size, position=itv3t_styling_pos, color=itv3t_styling_color)
itv3t_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", align_items="center", gap="16px")
itv3t_styling.layout = itv3t_styling_layout
itv3t.styling = itv3t_styling
itv3t.display_order = 0
itv3t.component_id = "itv3t"
itv3t.component_type = "text"
itv3t.custom_attributes = {"id": "itv3t"}
iqpvz = Text(name="iqpvz", content="Fast Performance", description="Text element")
iqpvz_styling_size = Size(width="auto", height="auto", padding="0", margin="20px 0 15px", unit_size=UnitSize.PIXELS)
iqpvz_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iqpvz_styling_color = Color(background_color="#FFFFFF", text_color="#333", border_color="#CCCCCC")
iqpvz_styling = Styling(size=iqpvz_styling_size, position=iqpvz_styling_pos, color=iqpvz_styling_color)
iqpvz.styling = iqpvz_styling
iqpvz.display_order = 1
iqpvz.component_id = "iqpvz"
iqpvz.component_type = "text"
iqpvz.tag_name = "h3"
iqpvz.custom_attributes = {"id": "iqpvz"}
ifuu4 = Text(name="ifuu4", content="Lightning-fast loading times and smooth interactions for the best user experience.", description="Text element")
ifuu4_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
ifuu4_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ifuu4_styling_color = Color(background_color="#FFFFFF", text_color="#666", border_color="#CCCCCC")
ifuu4_styling = Styling(size=ifuu4_styling_size, position=ifuu4_styling_pos, color=ifuu4_styling_color)
ifuu4.styling = ifuu4_styling
ifuu4.display_order = 2
ifuu4.component_id = "ifuu4"
ifuu4.component_type = "text"
ifuu4.tag_name = "p"
ifuu4.custom_attributes = {"id": "ifuu4"}
ipe69 = ViewContainer(name="ipe69", description=" component", view_elements={itv3t, iqpvz, ifuu4})
ipe69_styling_size = Size(width="auto", height="auto", padding="30px", margin="0", unit_size=UnitSize.PIXELS)
ipe69_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ipe69_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ipe69_styling = Styling(size=ipe69_styling_size, position=ipe69_styling_pos, color=ipe69_styling_color)
ipe69.styling = ipe69_styling
ipe69.display_order = 1
ipe69.component_id = "ipe69"
ipe69.custom_attributes = {"id": "ipe69"}
iv13n = Text(name="iv13n", content="🔒", description="Text element")
iv13n_styling_size = Size(width="80px", height="80px", padding="0", margin="0 auto 20px", font_size="36px", unit_size=UnitSize.PIXELS)
iv13n_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iv13n_styling_color = Color(background_color="linear-gradient(135deg, #667eea 0%, #764ba2 100%)", text_color="white", border_color="#CCCCCC")
iv13n_styling = Styling(size=iv13n_styling_size, position=iv13n_styling_pos, color=iv13n_styling_color)
iv13n_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", align_items="center", gap="16px")
iv13n_styling.layout = iv13n_styling_layout
iv13n.styling = iv13n_styling
iv13n.display_order = 0
iv13n.component_id = "iv13n"
iv13n.component_type = "text"
iv13n.custom_attributes = {"id": "iv13n"}
ir525 = Text(name="ir525", content="Secure & Safe", description="Text element")
ir525_styling_size = Size(width="auto", height="auto", padding="0", margin="20px 0 15px", unit_size=UnitSize.PIXELS)
ir525_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ir525_styling_color = Color(background_color="#FFFFFF", text_color="#333", border_color="#CCCCCC")
ir525_styling = Styling(size=ir525_styling_size, position=ir525_styling_pos, color=ir525_styling_color)
ir525.styling = ir525_styling
ir525.display_order = 1
ir525.component_id = "ir525"
ir525.component_type = "text"
ir525.tag_name = "h3"
ir525.custom_attributes = {"id": "ir525"}
ie5w5 = Text(name="ie5w5", content="Enterprise-grade security to protect your data and ensure privacy.", description="Text element")
ie5w5_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
ie5w5_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ie5w5_styling_color = Color(background_color="#FFFFFF", text_color="#666", border_color="#CCCCCC")
ie5w5_styling = Styling(size=ie5w5_styling_size, position=ie5w5_styling_pos, color=ie5w5_styling_color)
ie5w5.styling = ie5w5_styling
ie5w5.display_order = 2
ie5w5.component_id = "ie5w5"
ie5w5.component_type = "text"
ie5w5.tag_name = "p"
ie5w5.custom_attributes = {"id": "ie5w5"}
il2xd = ViewContainer(name="il2xd", description=" component", view_elements={iv13n, ir525, ie5w5})
il2xd_styling_size = Size(width="auto", height="auto", padding="30px", margin="0", unit_size=UnitSize.PIXELS)
il2xd_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
il2xd_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
il2xd_styling = Styling(size=il2xd_styling_size, position=il2xd_styling_pos, color=il2xd_styling_color)
il2xd.styling = il2xd_styling
il2xd.display_order = 3
il2xd.component_id = "il2xd"
il2xd.custom_attributes = {"id": "il2xd"}
ij0gr = Text(name="ij0gr", content="📱", description="Text element")
ij0gr_styling_size = Size(width="80px", height="80px", padding="0", margin="0 auto 20px", font_size="36px", unit_size=UnitSize.PIXELS)
ij0gr_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ij0gr_styling_color = Color(background_color="linear-gradient(135deg, #667eea 0%, #764ba2 100%)", text_color="white", border_color="#CCCCCC")
ij0gr_styling = Styling(size=ij0gr_styling_size, position=ij0gr_styling_pos, color=ij0gr_styling_color)
ij0gr_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", align_items="center", gap="16px")
ij0gr_styling.layout = ij0gr_styling_layout
ij0gr.styling = ij0gr_styling
ij0gr.display_order = 0
ij0gr.component_id = "ij0gr"
ij0gr.component_type = "text"
ij0gr.custom_attributes = {"id": "ij0gr"}
i48p8 = Text(name="i48p8", content="Responsive Design", description="Text element")
i48p8_styling_size = Size(width="auto", height="auto", padding="0", margin="20px 0 15px", unit_size=UnitSize.PIXELS)
i48p8_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i48p8_styling_color = Color(background_color="#FFFFFF", text_color="#333", border_color="#CCCCCC")
i48p8_styling = Styling(size=i48p8_styling_size, position=i48p8_styling_pos, color=i48p8_styling_color)
i48p8.styling = i48p8_styling
i48p8.display_order = 1
i48p8.component_id = "i48p8"
i48p8.component_type = "text"
i48p8.tag_name = "h3"
i48p8.custom_attributes = {"id": "i48p8"}
icejf = Text(name="icejf", content="Works perfectly on all devices - desktop, tablet, and mobile.", description="Text element")
icejf_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
icejf_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
icejf_styling_color = Color(background_color="#FFFFFF", text_color="#666", border_color="#CCCCCC")
icejf_styling = Styling(size=icejf_styling_size, position=icejf_styling_pos, color=icejf_styling_color)
icejf.styling = icejf_styling
icejf.display_order = 2
icejf.component_id = "icejf"
icejf.component_type = "text"
icejf.tag_name = "p"
icejf.custom_attributes = {"id": "icejf"}
i1b51 = ViewContainer(name="i1b51", description=" component", view_elements={ij0gr, i48p8, icejf})
i1b51_styling_size = Size(width="auto", height="auto", padding="30px", margin="0", unit_size=UnitSize.PIXELS)
i1b51_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i1b51_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i1b51_styling = Styling(size=i1b51_styling_size, position=i1b51_styling_pos, color=i1b51_styling_color)
i1b51.styling = i1b51_styling
i1b51.display_order = 5
i1b51.component_id = "i1b51"
i1b51.custom_attributes = {"id": "i1b51"}
idikm = ViewContainer(name="idikm", description=" component", view_elements={ipe69, il2xd, i1b51})
idikm_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
idikm_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
idikm_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
idikm_styling = Styling(size=idikm_styling_size, position=idikm_styling_pos, color=idikm_styling_color)
idikm_styling_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(300px, 1fr))", gap="40px")
idikm_styling.layout = idikm_styling_layout
idikm.styling = idikm_styling
idikm_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(300px, 1fr))", gap="40px")
idikm.layout = idikm_layout
idikm.display_order = 1
idikm.component_id = "idikm"
idikm.custom_attributes = {"id": "idikm"}
ippyf = ViewContainer(name="ippyf", description=" component", view_elements={ig8z7, idikm})
ippyf_styling_size = Size(width="auto", height="auto", padding="0", margin="0 auto", unit_size=UnitSize.PIXELS)
ippyf_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ippyf_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ippyf_styling = Styling(size=ippyf_styling_size, position=ippyf_styling_pos, color=ippyf_styling_color)
ippyf.styling = ippyf_styling
ippyf.display_order = 0
ippyf.component_id = "ippyf"
ippyf.custom_attributes = {"id": "ippyf"}
i14z3 = ViewContainer(name="i14z3", description="section container", view_elements={ippyf})
i14z3_styling_size = Size(width="auto", height="auto", padding="60px 20px", margin="0", unit_size=UnitSize.PIXELS)
i14z3_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i14z3_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i14z3_styling = Styling(size=i14z3_styling_size, position=i14z3_styling_pos, color=i14z3_styling_color)
i14z3.styling = i14z3_styling
i14z3.display_order = 1
i14z3.component_id = "i14z3"
i14z3.tag_name = "section"
i14z3.custom_attributes = {"id": "i14z3"}
ik80v = Text(name="ik80v", content="About Us", description="Text element")
ik80v_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ik80v_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ik80v_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ik80v_styling = Styling(size=ik80v_styling_size, position=ik80v_styling_pos, color=ik80v_styling_color)
ik80v.styling = ik80v_styling
ik80v.display_order = 0
ik80v.component_id = "ik80v"
ik80v.component_type = "text"
ik80v.tag_name = "h4"
ik80v.custom_attributes = {"id": "ik80v"}
i3jrs = Text(name="i3jrs", content="Your company description goes here.", description="Text element")
i3jrs_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
i3jrs_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i3jrs_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.8")
i3jrs_styling = Styling(size=i3jrs_styling_size, position=i3jrs_styling_pos, color=i3jrs_styling_color)
i3jrs.styling = i3jrs_styling
i3jrs.display_order = 1
i3jrs.component_id = "i3jrs"
i3jrs.component_type = "text"
i3jrs.tag_name = "p"
i3jrs.custom_attributes = {"id": "i3jrs"}
component = ViewContainer(name="Component", description=" component", view_elements={ik80v, i3jrs})
component_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
component_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
component_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
component_styling = Styling(size=component_styling_size, position=component_styling_pos, color=component_styling_color)
component.styling = component_styling
component.display_order = 0
i3crk = Text(name="i3crk", content="Quick Links", description="Text element")
i3crk_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i3crk_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i3crk_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i3crk_styling = Styling(size=i3crk_styling_size, position=i3crk_styling_pos, color=i3crk_styling_color)
i3crk.styling = i3crk_styling
i3crk.display_order = 0
i3crk.component_id = "i3crk"
i3crk.component_type = "text"
i3crk.tag_name = "h4"
i3crk.custom_attributes = {"id": "i3crk"}
i8brg = Link(name="i8brg", description="Link element", label="Home", url="#")
i8brg_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i8brg_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i8brg_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
i8brg_styling = Styling(size=i8brg_styling_size, position=i8brg_styling_pos, color=i8brg_styling_color)
i8brg.styling = i8brg_styling
i8brg.display_order = 0
i8brg.component_id = "i8brg"
i8brg.component_type = "link"
i8brg.tag_name = "a"
i8brg.custom_attributes = {"href": "#", "id": "i8brg"}
i0s0g = ViewContainer(name="i0s0g", description="li container", view_elements={i8brg})
i0s0g_styling_size = Size(width="auto", height="auto", padding="0", margin="8px 0", unit_size=UnitSize.PIXELS)
i0s0g_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i0s0g_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i0s0g_styling = Styling(size=i0s0g_styling_size, position=i0s0g_styling_pos, color=i0s0g_styling_color)
i0s0g.styling = i0s0g_styling
i0s0g.display_order = 0
i0s0g.component_id = "i0s0g"
i0s0g.tag_name = "li"
i0s0g.custom_attributes = {"id": "i0s0g"}
ipbq5 = Link(name="ipbq5", description="Link element", label="Services", url="#")
ipbq5_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ipbq5_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ipbq5_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
ipbq5_styling = Styling(size=ipbq5_styling_size, position=ipbq5_styling_pos, color=ipbq5_styling_color)
ipbq5.styling = ipbq5_styling
ipbq5.display_order = 0
ipbq5.component_id = "ipbq5"
ipbq5.component_type = "link"
ipbq5.tag_name = "a"
ipbq5.custom_attributes = {"href": "#", "id": "ipbq5"}
i90ha = ViewContainer(name="i90ha", description="li container", view_elements={ipbq5})
i90ha_styling_size = Size(width="auto", height="auto", padding="0", margin="8px 0", unit_size=UnitSize.PIXELS)
i90ha_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i90ha_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i90ha_styling = Styling(size=i90ha_styling_size, position=i90ha_styling_pos, color=i90ha_styling_color)
i90ha.styling = i90ha_styling
i90ha.display_order = 1
i90ha.component_id = "i90ha"
i90ha.tag_name = "li"
i90ha.custom_attributes = {"id": "i90ha"}
i9vgh = Link(name="i9vgh", description="Link element", label="Contact", url="#")
i9vgh_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i9vgh_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i9vgh_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
i9vgh_styling = Styling(size=i9vgh_styling_size, position=i9vgh_styling_pos, color=i9vgh_styling_color)
i9vgh.styling = i9vgh_styling
i9vgh.display_order = 0
i9vgh.component_id = "i9vgh"
i9vgh.component_type = "link"
i9vgh.tag_name = "a"
i9vgh.custom_attributes = {"href": "#", "id": "i9vgh"}
ickk4 = ViewContainer(name="ickk4", description="li container", view_elements={i9vgh})
ickk4_styling_size = Size(width="auto", height="auto", padding="0", margin="8px 0", unit_size=UnitSize.PIXELS)
ickk4_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ickk4_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ickk4_styling = Styling(size=ickk4_styling_size, position=ickk4_styling_pos, color=ickk4_styling_color)
ickk4.styling = ickk4_styling
ickk4.display_order = 2
ickk4.component_id = "ickk4"
ickk4.tag_name = "li"
ickk4.custom_attributes = {"id": "ickk4"}
idxov = ViewContainer(name="idxov", description="ul container", view_elements={i0s0g, i90ha, ickk4})
idxov_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
idxov_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
idxov_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.8")
idxov_styling = Styling(size=idxov_styling_size, position=idxov_styling_pos, color=idxov_styling_color)
idxov.styling = idxov_styling
idxov.display_order = 1
idxov.component_id = "idxov"
idxov.tag_name = "ul"
idxov.custom_attributes = {"id": "idxov"}
component_2 = ViewContainer(name="Component_2", description=" component", view_elements={i3crk, idxov})
component_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
component_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
component_2_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
component_2_styling = Styling(size=component_2_styling_size, position=component_2_styling_pos, color=component_2_styling_color)
component_2.styling = component_2_styling
component_2.display_order = 1
i59cl = Text(name="i59cl", content="Contact", description="Text element")
i59cl_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i59cl_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i59cl_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i59cl_styling = Styling(size=i59cl_styling_size, position=i59cl_styling_pos, color=i59cl_styling_color)
i59cl.styling = i59cl_styling
i59cl.display_order = 0
i59cl.component_id = "i59cl"
i59cl.component_type = "text"
i59cl.tag_name = "h4"
i59cl.custom_attributes = {"id": "i59cl"}
idy1y = Text(name="idy1y", content="Email: info@example.com\nPhone: (123) 456-7890", description="Text element")
idy1y_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
idy1y_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
idy1y_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.8")
idy1y_styling = Styling(size=idy1y_styling_size, position=idy1y_styling_pos, color=idy1y_styling_color)
idy1y.styling = idy1y_styling
idy1y.display_order = 1
idy1y.component_id = "idy1y"
idy1y.component_type = "text"
idy1y.tag_name = "p"
idy1y.custom_attributes = {"id": "idy1y"}
component_3 = ViewContainer(name="Component_3", description=" component", view_elements={i59cl, idy1y})
component_3_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
component_3_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
component_3_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
component_3_styling = Styling(size=component_3_styling_size, position=component_3_styling_pos, color=component_3_styling_color)
component_3.styling = component_3_styling
component_3.display_order = 2
ifnvi = ViewContainer(name="ifnvi", description=" component", view_elements={component, component_2, component_3})
ifnvi_styling_size = Size(width="auto", height="auto", padding="0", margin="0 auto", unit_size=UnitSize.PIXELS)
ifnvi_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ifnvi_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ifnvi_styling = Styling(size=ifnvi_styling_size, position=ifnvi_styling_pos, color=ifnvi_styling_color)
ifnvi_styling_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="30px")
ifnvi_styling.layout = ifnvi_styling_layout
ifnvi.styling = ifnvi_styling
ifnvi_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="30px")
ifnvi.layout = ifnvi_layout
ifnvi.display_order = 0
ifnvi.component_id = "ifnvi"
ifnvi.custom_attributes = {"id": "ifnvi"}
itx91 = Text(name="itx91", content="© 2025 Your Company. All rights reserved.", description="Text element")
itx91_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
itx91_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
itx91_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.7")
itx91_styling = Styling(size=itx91_styling_size, position=itx91_styling_pos, color=itx91_styling_color)
itx91.styling = itx91_styling
itx91.display_order = 1
itx91.component_id = "itx91"
itx91.component_type = "text"
itx91.custom_attributes = {"id": "itx91"}
icnsn = ViewContainer(name="icnsn", description="footer container", view_elements={ifnvi, itx91})
icnsn_styling_size = Size(width="auto", height="auto", padding="40px 20px", margin="0", unit_size=UnitSize.PIXELS)
icnsn_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
icnsn_styling_color = Color(background_color="#2c3e50", text_color="white", border_color="#CCCCCC")
icnsn_styling = Styling(size=icnsn_styling_size, position=icnsn_styling_pos, color=icnsn_styling_color)
icnsn.styling = icnsn_styling
icnsn.display_order = 2
icnsn.component_id = "icnsn"
icnsn.tag_name = "footer"
icnsn.custom_attributes = {"id": "icnsn"}
ijql = Text(name="ijql", content="Statec Hackathon", description="Text element")
ijql_styling_size = Size(width="auto", height="auto", padding="10px", margin="0", unit_size=UnitSize.PIXELS)
ijql_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ijql_styling_color = Color(background_color="#FFFFFF", text_color="#e23131", border_color="#CCCCCC")
ijql_styling = Styling(size=ijql_styling_size, position=ijql_styling_pos, color=ijql_styling_color)
ijql.styling = ijql_styling
ijql.display_order = 3
ijql.component_id = "ijql"
ijql.component_type = "text"
ijql.custom_attributes = {"id": "ijql"}
i1216 = RadarChart(name="i1216", title="Performance Metrics", primary_color="#8884d8", show_grid=True, show_tooltip=True, show_radius_axis=True, show_legend=True, legend_position="top", dot_size=3, grid_type="polygon", stroke_width=2)
i1216_binding = DataBinding(name="Performance MetricsDataBinding")
domain_model_ref = globals().get('domain_model')
i1216_binding_domain = None
if domain_model_ref is not None:
    i1216_binding_domain = domain_model_ref.get_class_by_name("Sensor")
if i1216_binding_domain:
    i1216_binding.domain_concept = i1216_binding_domain
    i1216_binding.label_field = next((attr for attr in i1216_binding_domain.attributes if attr.name == "date"), None)
    i1216_binding.data_field = next((attr for attr in i1216_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Sensor' not resolved; data binding remains partial.
    pass
i1216.data_binding = i1216_binding
i1216_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
i1216_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i1216_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i1216_styling = Styling(size=i1216_styling_size, position=i1216_styling_pos, color=i1216_styling_color)
i1216.styling = i1216_styling
i1216.display_order = 0
i1216.component_id = "i1216"
i1216.component_type = "radar-chart"
i1216.css_classes = ["radar-chart-component", "has-data-binding"]
i1216.custom_attributes = {"chart-color": "#8884d8", "chart-title": "Performance Metrics", "data-source": "8c4ebc1b-0432-43e9-8d84-3e7e3a554479", "label-field": "86560714-c3a9-4d80-a28d-9e6a360f3ae8", "data-field": "9aa4009c-e0e3-45b0-b405-5ea70309905a", "show-grid": True, "show-tooltip": True, "show-radius-axis": True, "id": "i1216"}
cell = ViewContainer(name="Cell", description=" container", view_elements={i1216})
cell_styling_size = Size(width="8%", height="75px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
cell_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
cell_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
cell_styling = Styling(size=cell_styling_size, position=cell_styling_pos, color=cell_styling_color)
cell.styling = cell_styling
cell.display_order = 0
cell.css_classes = ["gjs-cell"]
iuri1 = LineChart(name="iuri1", title="Sales Owerwerver Time", primary_color="#4CAF50", line_width=2, show_grid=True, show_legend=True, show_tooltip=True, curve_type="step", animate=True, legend_position="top", grid_color="#e0e0e0", dot_size=5)
iuri1_binding = DataBinding(name="Sales Owerwerver TimeDataBinding")
domain_model_ref = globals().get('domain_model')
iuri1_binding_domain = None
if domain_model_ref is not None:
    iuri1_binding_domain = domain_model_ref.get_class_by_name("Sensor")
if iuri1_binding_domain:
    iuri1_binding.domain_concept = iuri1_binding_domain
    iuri1_binding.label_field = next((attr for attr in iuri1_binding_domain.attributes if attr.name == "date"), None)
    iuri1_binding.data_field = next((attr for attr in iuri1_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Sensor' not resolved; data binding remains partial.
    pass
iuri1.data_binding = iuri1_binding
iuri1_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
iuri1_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iuri1_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
iuri1_styling = Styling(size=iuri1_styling_size, position=iuri1_styling_pos, color=iuri1_styling_color)
iuri1.styling = iuri1_styling
iuri1.display_order = 0
iuri1.component_id = "iuri1"
iuri1.component_type = "line-chart"
iuri1.css_classes = ["line-chart-component", "has-data-binding"]
iuri1.custom_attributes = {"chart-color": "#4CAF50", "chart-title": "Sales Owerwerver Time", "data-source": "8c4ebc1b-0432-43e9-8d84-3e7e3a554479", "label-field": "86560714-c3a9-4d80-a28d-9e6a360f3ae8", "data-field": "9aa4009c-e0e3-45b0-b405-5ea70309905a", "line-width": 2, "show-grid": True, "show-legend": True, "show-tooltip": True, "curve-type": "step", "animate": True, "id": "iuri1"}
cell_2 = ViewContainer(name="Cell_2", description=" container", view_elements={iuri1})
cell_2_styling_size = Size(width="8%", height="75px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
cell_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
cell_2_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
cell_2_styling = Styling(size=cell_2_styling_size, position=cell_2_styling_pos, color=cell_2_styling_color)
cell_2.styling = cell_2_styling
cell_2.display_order = 1
cell_2.css_classes = ["gjs-cell"]
i4yjj = LineChart(name="i4yjj", title="Sales Ovreewrewer Time", primary_color="#4CAF50", line_width=2, show_grid=True, show_legend=True, show_tooltip=True, curve_type="monotone", animate=True, legend_position="top", grid_color="#e0e0e0", dot_size=5)
i4yjj_binding = DataBinding(name="Sales Ovreewrewer TimeDataBinding")
domain_model_ref = globals().get('domain_model')
i4yjj_binding_domain = None
if domain_model_ref is not None:
    i4yjj_binding_domain = domain_model_ref.get_class_by_name("Sensor")
if i4yjj_binding_domain:
    i4yjj_binding.domain_concept = i4yjj_binding_domain
    i4yjj_binding.label_field = next((attr for attr in i4yjj_binding_domain.attributes if attr.name == "date"), None)
    i4yjj_binding.data_field = next((attr for attr in i4yjj_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Sensor' not resolved; data binding remains partial.
    pass
i4yjj.data_binding = i4yjj_binding
i4yjj_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
i4yjj_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i4yjj_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i4yjj_styling = Styling(size=i4yjj_styling_size, position=i4yjj_styling_pos, color=i4yjj_styling_color)
i4yjj.styling = i4yjj_styling
i4yjj.display_order = 0
i4yjj.component_id = "i4yjj"
i4yjj.component_type = "line-chart"
i4yjj.css_classes = ["line-chart-component", "has-data-binding"]
i4yjj.custom_attributes = {"chart-color": "#4CAF50", "chart-title": "Sales Ovreewrewer Time", "data-source": "8c4ebc1b-0432-43e9-8d84-3e7e3a554479", "label-field": "86560714-c3a9-4d80-a28d-9e6a360f3ae8", "data-field": "9aa4009c-e0e3-45b0-b405-5ea70309905a", "line-width": 2, "show-grid": True, "show-legend": True, "show-tooltip": True, "curve-type": "monotone", "animate": True, "id": "i4yjj"}
cell_3 = ViewContainer(name="Cell_3", description=" container", view_elements={i4yjj})
cell_3_styling_size = Size(width="8%", height="75px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
cell_3_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
cell_3_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
cell_3_styling = Styling(size=cell_3_styling_size, position=cell_3_styling_pos, color=cell_3_styling_color)
cell_3.styling = cell_3_styling
cell_3.display_order = 2
cell_3.css_classes = ["gjs-cell"]
ik6k = ViewContainer(name="ik6k", description=" container", view_elements={cell, cell_2, cell_3})
ik6k_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
ik6k_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ik6k_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ik6k_styling = Styling(size=ik6k_styling_size, position=ik6k_styling_pos, color=ik6k_styling_color)
ik6k.styling = ik6k_styling
ik6k.display_order = 4
ik6k.component_id = "ik6k"
ik6k.css_classes = ["gjs-row"]
ik6k.custom_attributes = {"id": "ik6k"}
ie5m = BarChart(name="ie5m", title="Revenue by Category", primary_color="#3498db", bar_width=30, orientation="vertical", show_grid=True, show_legend=True, show_tooltip=True, stacked=False, animate=True, legend_position="top", grid_color="#e0e0e0", bar_gap=4)
ie5m_binding = DataBinding(name="Revenue by CategoryDataBinding")
domain_model_ref = globals().get('domain_model')
ie5m_binding_domain = None
if domain_model_ref is not None:
    ie5m_binding_domain = domain_model_ref.get_class_by_name("Sensor")
if ie5m_binding_domain:
    ie5m_binding.domain_concept = ie5m_binding_domain
    ie5m_binding.label_field = next((attr for attr in ie5m_binding_domain.attributes if attr.name == "date"), None)
    ie5m_binding.data_field = next((attr for attr in ie5m_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Sensor' not resolved; data binding remains partial.
    pass
ie5m.data_binding = ie5m_binding
ie5m_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
ie5m_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ie5m_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ie5m_styling = Styling(size=ie5m_styling_size, position=ie5m_styling_pos, color=ie5m_styling_color)
ie5m.styling = ie5m_styling
ie5m.display_order = 5
ie5m.component_id = "ie5m"
ie5m.component_type = "bar-chart"
ie5m.css_classes = ["bar-chart-component", "has-data-binding"]
ie5m.custom_attributes = {"chart-color": "#3498db", "chart-title": "Revenue by Category", "data-source": "8c4ebc1b-0432-43e9-8d84-3e7e3a554479", "label-field": "86560714-c3a9-4d80-a28d-9e6a360f3ae8", "data-field": "9aa4009c-e0e3-45b0-b405-5ea70309905a", "bar-width": 30, "orientation": "vertical", "show-grid": True, "show-legend": True, "stacked": False, "id": "ie5m"}
i3ja1 = BarChart(name="i3ja1", title="Revenue bwerwery Category", primary_color="#3498db", bar_width=30, orientation="horizontal", show_grid=True, show_legend=True, show_tooltip=True, stacked=False, animate=True, legend_position="top", grid_color="#e0e0e0", bar_gap=4)
i3ja1_binding = DataBinding(name="Revenue bwerwery CategoryDataBinding")
domain_model_ref = globals().get('domain_model')
i3ja1_binding_domain = None
if domain_model_ref is not None:
    i3ja1_binding_domain = domain_model_ref.get_class_by_name("Sensor")
if i3ja1_binding_domain:
    i3ja1_binding.domain_concept = i3ja1_binding_domain
    i3ja1_binding.label_field = next((attr for attr in i3ja1_binding_domain.attributes if attr.name == "date"), None)
    i3ja1_binding.data_field = next((attr for attr in i3ja1_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Sensor' not resolved; data binding remains partial.
    pass
i3ja1.data_binding = i3ja1_binding
i3ja1_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
i3ja1_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i3ja1_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i3ja1_styling = Styling(size=i3ja1_styling_size, position=i3ja1_styling_pos, color=i3ja1_styling_color)
i3ja1.styling = i3ja1_styling
i3ja1.display_order = 6
i3ja1.component_id = "i3ja1"
i3ja1.component_type = "bar-chart"
i3ja1.css_classes = ["bar-chart-component", "has-data-binding"]
i3ja1.custom_attributes = {"chart-color": "#3498db", "chart-title": "Revenue bwerwery Category", "data-source": "8c4ebc1b-0432-43e9-8d84-3e7e3a554479", "label-field": "86560714-c3a9-4d80-a28d-9e6a360f3ae8", "data-field": "9aa4009c-e0e3-45b0-b405-5ea70309905a", "bar-width": 30, "orientation": "horizontal", "show-grid": True, "show-legend": True, "stacked": False, "id": "i3ja1"}
icnh.view_elements = {ia1em, i14z3, icnsn, ijql, ik6k, ie5m, i3ja1}


# Screen: wrapper
wrapper = Screen(name="wrapper", description="sdssdf", route_path="/sdssdf", screen_size="Medium")
wrapper_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
wrapper_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
wrapper_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
wrapper_styling = Styling(size=wrapper_styling_size, position=wrapper_styling_pos, color=wrapper_styling_color)
wrapper.styling = wrapper_styling
h1 = Text(name="h1", content="New Page: sdssdf", description="Text element")
h1_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
h1_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
h1_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
h1_styling = Styling(size=h1_styling_size, position=h1_styling_pos, color=h1_styling_color)
h1.styling = h1_styling
h1.display_order = 0
h1.component_type = "text"
h1.tag_name = "h1"
ixwdc = ViewContainer(name="ixwdc", description=" component", view_elements={h1})
ixwdc_styling_size = Size(width="auto", height="auto", padding="50px", margin="0", unit_size=UnitSize.PIXELS)
ixwdc_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ixwdc_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ixwdc_styling = Styling(size=ixwdc_styling_size, position=ixwdc_styling_pos, color=ixwdc_styling_color)
ixwdc.styling = ixwdc_styling
ixwdc.display_order = 0
ixwdc.component_id = "ixwdc"
ixwdc.custom_attributes = {"id": "ixwdc"}
wrapper.view_elements = {ixwdc}

gui_module = Module(
    name="GUI_Module",
    screens={icnh, wrapper}
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
