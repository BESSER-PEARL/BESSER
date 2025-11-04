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
Count = Class(name="Count")

# Count class attributes and methods
Count_attribute: Property = Property(name="attribute", type=StringType)
Count_value: Property = Property(name="value", type=IntegerType)
Count.attributes={Count_attribute, Count_value}

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={Count},
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
    Text, Image, Link, InputField, InputFieldType,
    Form, Menu, MenuItem, DataList,
    DataSource, DataSourceElement, EmbeddedContent,
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

# Screen: i9vb
i9vb = Screen(name="i9vb", description="Home", view_elements=set(), is_main_page=True, route_path="/home", screen_size="Medium")
i9vb.component_id = "PGkfAwt8hu7jL4s9"
ifh3 = Text(name="ifh3", content="Logo", description="Text element")
ifh3_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="24px", unit_size=UnitSize.PIXELS)
ifh3_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ifh3_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ifh3_styling = Styling(size=ifh3_styling_size, position=ifh3_styling_pos, color=ifh3_styling_color)
ifh3.styling = ifh3_styling
ifh3.display_order = 0
ifh3.component_id = "ifh3"
ifh3.component_type = "text"
ifh3.custom_attributes = {"id": "ifh3"}
imub = Link(name="imub", description="Link element", label="Home", url="/")
imub_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
imub_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
imub_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
imub_styling = Styling(size=imub_styling_size, position=imub_styling_pos, color=imub_styling_color)
imub.styling = imub_styling
imub.display_order = 0
imub.component_id = "imub"
imub.component_type = "link"
imub.tag_name = "a"
imub.custom_attributes = {"href": "/", "id": "imub"}
ixv3d = Link(name="ixv3d", description="Link element", label="About", url="/about")
ixv3d_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ixv3d_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ixv3d_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
ixv3d_styling = Styling(size=ixv3d_styling_size, position=ixv3d_styling_pos, color=ixv3d_styling_color)
ixv3d.styling = ixv3d_styling
ixv3d.display_order = 1
ixv3d.component_id = "ixv3d"
ixv3d.component_type = "link"
ixv3d.tag_name = "a"
ixv3d.custom_attributes = {"href": "/about", "id": "ixv3d"}
i6pk5 = Link(name="i6pk5", description="Link element", label="Services", url="#")
i6pk5_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i6pk5_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i6pk5_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
i6pk5_styling = Styling(size=i6pk5_styling_size, position=i6pk5_styling_pos, color=i6pk5_styling_color)
i6pk5.styling = i6pk5_styling
i6pk5.display_order = 2
i6pk5.component_id = "i6pk5"
i6pk5.component_type = "link"
i6pk5.tag_name = "a"
i6pk5.custom_attributes = {"href": "#", "id": "i6pk5"}
iz6f6 = Link(name="iz6f6", description="Link element", label="Contact", url="/contact")
iz6f6_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iz6f6_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iz6f6_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
iz6f6_styling = Styling(size=iz6f6_styling_size, position=iz6f6_styling_pos, color=iz6f6_styling_color)
iz6f6.styling = iz6f6_styling
iz6f6.display_order = 3
iz6f6.component_id = "iz6f6"
iz6f6.component_type = "link"
iz6f6.tag_name = "a"
iz6f6.custom_attributes = {"href": "/contact", "id": "iz6f6"}
ijc9 = ViewContainer(name="ijc9", description=" component", view_elements={imub, ixv3d, i6pk5, iz6f6})
ijc9_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ijc9_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ijc9_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ijc9_styling = Styling(size=ijc9_styling_size, position=ijc9_styling_pos, color=ijc9_styling_color)
ijc9_styling_layout = Layout(layout_type=LayoutType.FLEX, gap="30px")
ijc9_styling.layout = ijc9_styling_layout
ijc9.styling = ijc9_styling
ijc9_layout = Layout(layout_type=LayoutType.FLEX, gap="30px")
ijc9.layout = ijc9_layout
ijc9.display_order = 1
ijc9.component_id = "ijc9"
ijc9.custom_attributes = {"id": "ijc9"}
i4sm = ViewContainer(name="i4sm", description="nav container", view_elements={ifh3, ijc9})
i4sm_styling_size = Size(width="auto", height="auto", padding="15px 30px", margin="0", unit_size=UnitSize.PIXELS)
i4sm_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i4sm_styling_color = Color(background_color="#333", text_color="white", border_color="#CCCCCC")
i4sm_styling = Styling(size=i4sm_styling_size, position=i4sm_styling_pos, color=i4sm_styling_color)
i4sm_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="space-between", align_items="center", gap="16px")
i4sm_styling.layout = i4sm_styling_layout
i4sm.styling = i4sm_styling
i4sm_layout = Layout(layout_type=LayoutType.FLEX, justify_content="space-between", align_items="center", gap="16px")
i4sm.layout = i4sm_layout
i4sm.display_order = 0
i4sm.component_id = "i4sm"
i4sm.tag_name = "nav"
i4sm.custom_attributes = {"id": "i4sm"}
i7m71 = LineChart(name="i7m71", title="Line Chart Title", primary_color="#ff0606", line_width=2, show_grid=True, show_legend=True, show_tooltip=True, curve_type="monotone", animate=True, legend_position="top", grid_color="#e0e0e0", dot_size=5)
domain_model_ref = globals().get('domain_model')
i7m71_binding_domain = None
if domain_model_ref is not None:
    i7m71_binding_domain = domain_model_ref.get_class_by_name("Count")
if i7m71_binding_domain:
    i7m71_binding = DataBinding(domain_concept=i7m71_binding_domain)
    i7m71_binding.label_field = next((attr for attr in i7m71_binding_domain.attributes if attr.name == "attribute"), None)
    i7m71_binding.data_field = next((attr for attr in i7m71_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Count' not resolved; data binding skipped.
    i7m71_binding = None
if i7m71_binding:
    i7m71.data_binding = i7m71_binding
i7m71_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
i7m71_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i7m71_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i7m71_styling = Styling(size=i7m71_styling_size, position=i7m71_styling_pos, color=i7m71_styling_color)
i7m71.styling = i7m71_styling
i7m71.display_order = 0
i7m71.component_id = "i7m71"
i7m71.component_type = "line-chart"
i7m71.css_classes = ["line-chart-component", "has-data-binding"]
i7m71.custom_attributes = {"chart-color": "#ff0606", "chart-title": "Line Chart Title", "data-source": "d1dbf7eb-4792-467e-82ce-b36aab94145d", "label-field": "badb769f-ec24-4c30-afdc-79ddae20b58d", "data-field": "0680976e-c20d-4899-ab11-b1e99385548e", "line-width": 2, "show-grid": True, "show-legend": True, "show-tooltip": True, "curve-type": "monotone", "animate": True, "id": "i7m71"}
cell = ViewContainer(name="Cell", description=" container", view_elements={i7m71})
cell_styling_size = Size(width="8%", height="75px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
cell_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
cell_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
cell_styling = Styling(size=cell_styling_size, position=cell_styling_pos, color=cell_styling_color)
cell.styling = cell_styling
cell.display_order = 0
cell.component_id = "container_cell"
cell.css_classes = ["gjs-cell"]
iia8n = BarChart(name="iia8n", title="Bar Chart Title", primary_color="#db34d2", bar_width=30, orientation="vertical", show_grid=True, show_legend=True, show_tooltip=True, stacked=False, animate=True, legend_position="top", grid_color="#e0e0e0", bar_gap=4)
domain_model_ref = globals().get('domain_model')
iia8n_binding_domain = None
if domain_model_ref is not None:
    iia8n_binding_domain = domain_model_ref.get_class_by_name("Count")
if iia8n_binding_domain:
    iia8n_binding = DataBinding(domain_concept=iia8n_binding_domain)
    iia8n_binding.label_field = next((attr for attr in iia8n_binding_domain.attributes if attr.name == "attribute"), None)
    iia8n_binding.data_field = next((attr for attr in iia8n_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Count' not resolved; data binding skipped.
    iia8n_binding = None
if iia8n_binding:
    iia8n.data_binding = iia8n_binding
iia8n_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
iia8n_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iia8n_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
iia8n_styling = Styling(size=iia8n_styling_size, position=iia8n_styling_pos, color=iia8n_styling_color)
iia8n.styling = iia8n_styling
iia8n.display_order = 0
iia8n.component_id = "iia8n"
iia8n.component_type = "bar-chart"
iia8n.css_classes = ["bar-chart-component", "has-data-binding"]
iia8n.custom_attributes = {"chart-color": "#db34d2", "chart-title": "Bar Chart Title", "data-source": "d1dbf7eb-4792-467e-82ce-b36aab94145d", "label-field": "badb769f-ec24-4c30-afdc-79ddae20b58d", "data-field": "0680976e-c20d-4899-ab11-b1e99385548e", "bar-width": 30, "orientation": "vertical", "show-grid": True, "show-legend": True, "stacked": False, "id": "iia8n"}
cell_2 = ViewContainer(name="Cell_2", description=" container", view_elements={iia8n})
cell_2_styling_size = Size(width="8%", height="75px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
cell_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
cell_2_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
cell_2_styling = Styling(size=cell_2_styling_size, position=cell_2_styling_pos, color=cell_2_styling_color)
cell_2.styling = cell_2_styling
cell_2.display_order = 1
cell_2.component_id = "container_cell_2"
cell_2.css_classes = ["gjs-cell"]
i3d3h = RadarChart(name="i3d3h", title="Radar Chart Title", primary_color="#8884d8", show_grid=True, show_tooltip=True, show_radius_axis=True, show_legend=True, legend_position="top", dot_size=3, grid_type="polygon", stroke_width=2)
domain_model_ref = globals().get('domain_model')
i3d3h_binding_domain = None
if domain_model_ref is not None:
    i3d3h_binding_domain = domain_model_ref.get_class_by_name("Count")
if i3d3h_binding_domain:
    i3d3h_binding = DataBinding(domain_concept=i3d3h_binding_domain)
    i3d3h_binding.label_field = next((attr for attr in i3d3h_binding_domain.attributes if attr.name == "attribute"), None)
    i3d3h_binding.data_field = next((attr for attr in i3d3h_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Count' not resolved; data binding skipped.
    i3d3h_binding = None
if i3d3h_binding:
    i3d3h.data_binding = i3d3h_binding
i3d3h_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
i3d3h_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i3d3h_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i3d3h_styling = Styling(size=i3d3h_styling_size, position=i3d3h_styling_pos, color=i3d3h_styling_color)
i3d3h.styling = i3d3h_styling
i3d3h.display_order = 0
i3d3h.component_id = "i3d3h"
i3d3h.component_type = "radar-chart"
i3d3h.css_classes = ["radar-chart-component", "has-data-binding"]
i3d3h.custom_attributes = {"chart-color": "#8884d8", "chart-title": "Radar Chart Title", "data-source": "d1dbf7eb-4792-467e-82ce-b36aab94145d", "label-field": "badb769f-ec24-4c30-afdc-79ddae20b58d", "data-field": "0680976e-c20d-4899-ab11-b1e99385548e", "show-grid": True, "show-tooltip": True, "show-radius-axis": True, "id": "i3d3h"}
cell_3 = ViewContainer(name="Cell_3", description=" container", view_elements={i3d3h})
cell_3_styling_size = Size(width="8%", height="75px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
cell_3_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
cell_3_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
cell_3_styling = Styling(size=cell_3_styling_size, position=cell_3_styling_pos, color=cell_3_styling_color)
cell_3.styling = cell_3_styling
cell_3.display_order = 2
cell_3.component_id = "container_cell_3"
cell_3.css_classes = ["gjs-cell"]
izmfj = ViewContainer(name="izmfj", description=" container", view_elements={cell, cell_2, cell_3})
izmfj_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
izmfj_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
izmfj_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
izmfj_styling = Styling(size=izmfj_styling_size, position=izmfj_styling_pos, color=izmfj_styling_color)
izmfj.styling = izmfj_styling
izmfj.display_order = 1
izmfj.component_id = "izmfj"
izmfj.css_classes = ["gjs-row"]
izmfj.custom_attributes = {"id": "izmfj"}
irv6i = Text(name="irv6i", content="Ready to Get Started?", description="Text element")
irv6i_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="42px", unit_size=UnitSize.PIXELS)
irv6i_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
irv6i_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
irv6i_styling = Styling(size=irv6i_styling_size, position=irv6i_styling_pos, color=irv6i_styling_color)
irv6i.styling = irv6i_styling
irv6i.display_order = 0
irv6i.component_id = "irv6i"
irv6i.component_type = "text"
irv6i.tag_name = "h2"
irv6i.custom_attributes = {"id": "irv6i"}
iznz9 = Text(name="iznz9", content="Join thousands of satisfied customers and take your business to the next level.", description="Text element")
iznz9_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="20px", unit_size=UnitSize.PIXELS)
iznz9_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iznz9_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.95")
iznz9_styling = Styling(size=iznz9_styling_size, position=iznz9_styling_pos, color=iznz9_styling_color)
iznz9.styling = iznz9_styling
iznz9.display_order = 1
iznz9.component_id = "iznz9"
iznz9.component_type = "text"
iznz9.tag_name = "p"
iznz9.custom_attributes = {"id": "iznz9"}
i27yo = Button(name="i27yo", description="Button component", label="Start Free Trial", buttonType=ButtonType.CustomizableButton, actionType=ButtonActionType.View)
i27yo_styling_size = Size(width="auto", height="auto", padding="18px 40px", margin="0", font_size="18px", unit_size=UnitSize.PIXELS)
i27yo_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i27yo_styling_color = Color(background_color="white", text_color="#f5576c", border_color="#CCCCCC")
i27yo_styling = Styling(size=i27yo_styling_size, position=i27yo_styling_pos, color=i27yo_styling_color)
i27yo.styling = i27yo_styling
i27yo.display_order = 0
i27yo.component_id = "i27yo"
i27yo.component_type = "button"
i27yo.tag_name = "button"
i27yo.custom_attributes = {"type": "button", "id": "i27yo"}
ig59s = Button(name="ig59s", description="Button component", label="Learssn More", buttonType=ButtonType.CustomizableButton, actionType=ButtonActionType.View)
ig59s_styling_size = Size(width="auto", height="auto", padding="18px 40px", margin="0", font_size="18px", unit_size=UnitSize.PIXELS)
ig59s_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ig59s_styling_color = Color(background_color="transparent", text_color="white", border_color="#CCCCCC")
ig59s_styling = Styling(size=ig59s_styling_size, position=ig59s_styling_pos, color=ig59s_styling_color)
ig59s.styling = ig59s_styling
ig59s.display_order = 1
ig59s.component_id = "ig59s"
ig59s.component_type = "button"
ig59s.tag_name = "button"
ig59s.custom_attributes = {"type": "button", "id": "ig59s"}
i9fuo = ViewContainer(name="i9fuo", description=" component", view_elements={i27yo, ig59s})
i9fuo_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i9fuo_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i9fuo_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i9fuo_styling = Styling(size=i9fuo_styling_size, position=i9fuo_styling_pos, color=i9fuo_styling_color)
i9fuo_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", flex_wrap="wrap", gap="20px")
i9fuo_styling.layout = i9fuo_styling_layout
i9fuo.styling = i9fuo_styling
i9fuo_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", flex_wrap="wrap", gap="20px")
i9fuo.layout = i9fuo_layout
i9fuo.display_order = 2
i9fuo.component_id = "i9fuo"
i9fuo.custom_attributes = {"id": "i9fuo"}
ibfd5 = ViewContainer(name="ibfd5", description=" component", view_elements={irv6i, iznz9, i9fuo})
ibfd5_styling_size = Size(width="auto", height="auto", padding="0", margin="0 auto", unit_size=UnitSize.PIXELS)
ibfd5_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ibfd5_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ibfd5_styling = Styling(size=ibfd5_styling_size, position=ibfd5_styling_pos, color=ibfd5_styling_color)
ibfd5.styling = ibfd5_styling
ibfd5.display_order = 0
ibfd5.component_id = "ibfd5"
ibfd5.custom_attributes = {"id": "ibfd5"}
imt5g = ViewContainer(name="imt5g", description="section container", view_elements={ibfd5})
imt5g_styling_size = Size(width="auto", height="auto", padding="80px 20px", margin="0", unit_size=UnitSize.PIXELS)
imt5g_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
imt5g_styling_color = Color(background_color="linear-gradient(135deg, #f093fb 0%, #f5576c 100%)", text_color="white", border_color="#CCCCCC")
imt5g_styling = Styling(size=imt5g_styling_size, position=imt5g_styling_pos, color=imt5g_styling_color)
imt5g.styling = imt5g_styling
imt5g.display_order = 2
imt5g.component_id = "imt5g"
imt5g.tag_name = "section"
imt5g.custom_attributes = {"id": "imt5g"}
i9ijl = Text(name="i9ijl", content="Our Features", description="Text element")
i9ijl_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="36px", unit_size=UnitSize.PIXELS)
i9ijl_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i9ijl_styling_color = Color(background_color="#FFFFFF", text_color="#333", border_color="#CCCCCC")
i9ijl_styling = Styling(size=i9ijl_styling_size, position=i9ijl_styling_pos, color=i9ijl_styling_color)
i9ijl.styling = i9ijl_styling
i9ijl.display_order = 0
i9ijl.component_id = "i9ijl"
i9ijl.component_type = "text"
i9ijl.tag_name = "h2"
i9ijl.custom_attributes = {"id": "i9ijl"}
ijti8 = Text(name="ijti8", content="🚀", description="Text element")
ijti8_styling_size = Size(width="80px", height="80px", padding="0", margin="0 auto 20px", font_size="36px", unit_size=UnitSize.PIXELS)
ijti8_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ijti8_styling_color = Color(background_color="linear-gradient(135deg, #667eea 0%, #764ba2 100%)", text_color="white", border_color="#CCCCCC")
ijti8_styling = Styling(size=ijti8_styling_size, position=ijti8_styling_pos, color=ijti8_styling_color)
ijti8_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", align_items="center", gap="16px")
ijti8_styling.layout = ijti8_styling_layout
ijti8.styling = ijti8_styling
ijti8.display_order = 0
ijti8.component_id = "ijti8"
ijti8.component_type = "text"
ijti8.custom_attributes = {"id": "ijti8"}
izgub = Text(name="izgub", content="Fast Performance", description="Text element")
izgub_styling_size = Size(width="auto", height="auto", padding="0", margin="20px 0 15px", unit_size=UnitSize.PIXELS)
izgub_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
izgub_styling_color = Color(background_color="#FFFFFF", text_color="#333", border_color="#CCCCCC")
izgub_styling = Styling(size=izgub_styling_size, position=izgub_styling_pos, color=izgub_styling_color)
izgub.styling = izgub_styling
izgub.display_order = 1
izgub.component_id = "izgub"
izgub.component_type = "text"
izgub.tag_name = "h3"
izgub.custom_attributes = {"id": "izgub"}
iuq7f = Text(name="iuq7f", content="Lightning-fast loading times and smooth interactions for the best user experience.", description="Text element")
iuq7f_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
iuq7f_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iuq7f_styling_color = Color(background_color="#FFFFFF", text_color="#666", border_color="#CCCCCC")
iuq7f_styling = Styling(size=iuq7f_styling_size, position=iuq7f_styling_pos, color=iuq7f_styling_color)
iuq7f.styling = iuq7f_styling
iuq7f.display_order = 2
iuq7f.component_id = "iuq7f"
iuq7f.component_type = "text"
iuq7f.tag_name = "p"
iuq7f.custom_attributes = {"id": "iuq7f"}
i94zd = ViewContainer(name="i94zd", description=" component", view_elements={ijti8, izgub, iuq7f})
i94zd_styling_size = Size(width="auto", height="auto", padding="30px", margin="0", unit_size=UnitSize.PIXELS)
i94zd_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i94zd_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i94zd_styling = Styling(size=i94zd_styling_size, position=i94zd_styling_pos, color=i94zd_styling_color)
i94zd.styling = i94zd_styling
i94zd.display_order = 1
i94zd.component_id = "i94zd"
i94zd.custom_attributes = {"id": "i94zd"}
isprs = Text(name="isprs", content="🔒", description="Text element")
isprs_styling_size = Size(width="80px", height="80px", padding="0", margin="0 auto 20px", font_size="36px", unit_size=UnitSize.PIXELS)
isprs_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
isprs_styling_color = Color(background_color="linear-gradient(135deg, #667eea 0%, #764ba2 100%)", text_color="white", border_color="#CCCCCC")
isprs_styling = Styling(size=isprs_styling_size, position=isprs_styling_pos, color=isprs_styling_color)
isprs_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", align_items="center", gap="16px")
isprs_styling.layout = isprs_styling_layout
isprs.styling = isprs_styling
isprs.display_order = 0
isprs.component_id = "isprs"
isprs.component_type = "text"
isprs.custom_attributes = {"id": "isprs"}
i97zg = Text(name="i97zg", content="Secure & Safe", description="Text element")
i97zg_styling_size = Size(width="auto", height="auto", padding="0", margin="20px 0 15px", unit_size=UnitSize.PIXELS)
i97zg_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i97zg_styling_color = Color(background_color="#FFFFFF", text_color="#333", border_color="#CCCCCC")
i97zg_styling = Styling(size=i97zg_styling_size, position=i97zg_styling_pos, color=i97zg_styling_color)
i97zg.styling = i97zg_styling
i97zg.display_order = 1
i97zg.component_id = "i97zg"
i97zg.component_type = "text"
i97zg.tag_name = "h3"
i97zg.custom_attributes = {"id": "i97zg"}
im3io = Text(name="im3io", content="Enterprise-grade security to protect your data and ensure privacy.", description="Text element")
im3io_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
im3io_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
im3io_styling_color = Color(background_color="#FFFFFF", text_color="#666", border_color="#CCCCCC")
im3io_styling = Styling(size=im3io_styling_size, position=im3io_styling_pos, color=im3io_styling_color)
im3io.styling = im3io_styling
im3io.display_order = 2
im3io.component_id = "im3io"
im3io.component_type = "text"
im3io.tag_name = "p"
im3io.custom_attributes = {"id": "im3io"}
i2pjk = ViewContainer(name="i2pjk", description=" component", view_elements={isprs, i97zg, im3io})
i2pjk_styling_size = Size(width="auto", height="auto", padding="30px", margin="0", unit_size=UnitSize.PIXELS)
i2pjk_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i2pjk_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i2pjk_styling = Styling(size=i2pjk_styling_size, position=i2pjk_styling_pos, color=i2pjk_styling_color)
i2pjk.styling = i2pjk_styling
i2pjk.display_order = 3
i2pjk.component_id = "i2pjk"
i2pjk.custom_attributes = {"id": "i2pjk"}
i3ee7 = Text(name="i3ee7", content="📱", description="Text element")
i3ee7_styling_size = Size(width="80px", height="80px", padding="0", margin="0 auto 20px", font_size="36px", unit_size=UnitSize.PIXELS)
i3ee7_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i3ee7_styling_color = Color(background_color="linear-gradient(135deg, #667eea 0%, #764ba2 100%)", text_color="white", border_color="#CCCCCC")
i3ee7_styling = Styling(size=i3ee7_styling_size, position=i3ee7_styling_pos, color=i3ee7_styling_color)
i3ee7_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", align_items="center", gap="16px")
i3ee7_styling.layout = i3ee7_styling_layout
i3ee7.styling = i3ee7_styling
i3ee7.display_order = 0
i3ee7.component_id = "i3ee7"
i3ee7.component_type = "text"
i3ee7.custom_attributes = {"id": "i3ee7"}
itkgh = Text(name="itkgh", content="Responsive Design", description="Text element")
itkgh_styling_size = Size(width="auto", height="auto", padding="0", margin="20px 0 15px", unit_size=UnitSize.PIXELS)
itkgh_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
itkgh_styling_color = Color(background_color="#FFFFFF", text_color="#333", border_color="#CCCCCC")
itkgh_styling = Styling(size=itkgh_styling_size, position=itkgh_styling_pos, color=itkgh_styling_color)
itkgh.styling = itkgh_styling
itkgh.display_order = 1
itkgh.component_id = "itkgh"
itkgh.component_type = "text"
itkgh.tag_name = "h3"
itkgh.custom_attributes = {"id": "itkgh"}
i5pal = Text(name="i5pal", content="Works perfectly on all devices - desktop, tablet, and mobile.", description="Text element")
i5pal_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
i5pal_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i5pal_styling_color = Color(background_color="#FFFFFF", text_color="#666", border_color="#CCCCCC")
i5pal_styling = Styling(size=i5pal_styling_size, position=i5pal_styling_pos, color=i5pal_styling_color)
i5pal.styling = i5pal_styling
i5pal.display_order = 2
i5pal.component_id = "i5pal"
i5pal.component_type = "text"
i5pal.tag_name = "p"
i5pal.custom_attributes = {"id": "i5pal"}
i9z3l = ViewContainer(name="i9z3l", description=" component", view_elements={i3ee7, itkgh, i5pal})
i9z3l_styling_size = Size(width="auto", height="auto", padding="30px", margin="0", unit_size=UnitSize.PIXELS)
i9z3l_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i9z3l_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i9z3l_styling = Styling(size=i9z3l_styling_size, position=i9z3l_styling_pos, color=i9z3l_styling_color)
i9z3l.styling = i9z3l_styling
i9z3l.display_order = 5
i9z3l.component_id = "i9z3l"
i9z3l.custom_attributes = {"id": "i9z3l"}
iexvq = ViewContainer(name="iexvq", description=" component", view_elements={i94zd, i2pjk, i9z3l})
iexvq_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iexvq_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iexvq_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
iexvq_styling = Styling(size=iexvq_styling_size, position=iexvq_styling_pos, color=iexvq_styling_color)
iexvq_styling_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(300px, 1fr))", gap="40px")
iexvq_styling.layout = iexvq_styling_layout
iexvq.styling = iexvq_styling
iexvq_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(300px, 1fr))", gap="40px")
iexvq.layout = iexvq_layout
iexvq.display_order = 1
iexvq.component_id = "iexvq"
iexvq.custom_attributes = {"id": "iexvq"}
imbe6 = ViewContainer(name="imbe6", description=" component", view_elements={i9ijl, iexvq})
imbe6_styling_size = Size(width="auto", height="auto", padding="0", margin="0 auto", unit_size=UnitSize.PIXELS)
imbe6_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
imbe6_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
imbe6_styling = Styling(size=imbe6_styling_size, position=imbe6_styling_pos, color=imbe6_styling_color)
imbe6.styling = imbe6_styling
imbe6.display_order = 0
imbe6.component_id = "imbe6"
imbe6.custom_attributes = {"id": "imbe6"}
i28pk = Text(name="i28pk", content="About Us", description="Text element")
i28pk_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i28pk_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i28pk_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i28pk_styling = Styling(size=i28pk_styling_size, position=i28pk_styling_pos, color=i28pk_styling_color)
i28pk.styling = i28pk_styling
i28pk.display_order = 0
i28pk.component_id = "i28pk"
i28pk.component_type = "text"
i28pk.tag_name = "h4"
i28pk.custom_attributes = {"id": "i28pk"}
imfbl = Text(name="imfbl", content="Your company description goes here.", description="Text element")
imfbl_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
imfbl_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
imfbl_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.8")
imfbl_styling = Styling(size=imfbl_styling_size, position=imfbl_styling_pos, color=imfbl_styling_color)
imfbl.styling = imfbl_styling
imfbl.display_order = 1
imfbl.component_id = "imfbl"
imfbl.component_type = "text"
imfbl.tag_name = "p"
imfbl.custom_attributes = {"id": "imfbl"}
component = ViewContainer(name="Component", description=" component", view_elements={i28pk, imfbl})
component.display_order = 0
ivkxw = Text(name="ivkxw", content="Quick Links", description="Text element")
ivkxw_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ivkxw_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ivkxw_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ivkxw_styling = Styling(size=ivkxw_styling_size, position=ivkxw_styling_pos, color=ivkxw_styling_color)
ivkxw.styling = ivkxw_styling
ivkxw.display_order = 0
ivkxw.component_id = "ivkxw"
ivkxw.component_type = "text"
ivkxw.tag_name = "h4"
ivkxw.custom_attributes = {"id": "ivkxw"}
iwmx6 = Link(name="iwmx6", description="Link element", label="Home", url="#")
iwmx6_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iwmx6_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iwmx6_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
iwmx6_styling = Styling(size=iwmx6_styling_size, position=iwmx6_styling_pos, color=iwmx6_styling_color)
iwmx6.styling = iwmx6_styling
iwmx6.display_order = 0
iwmx6.component_id = "iwmx6"
iwmx6.component_type = "link"
iwmx6.tag_name = "a"
iwmx6.custom_attributes = {"href": "#", "id": "iwmx6"}
ihzg1 = ViewContainer(name="ihzg1", description="li container", view_elements={iwmx6})
ihzg1_styling_size = Size(width="auto", height="auto", padding="0", margin="8px 0", unit_size=UnitSize.PIXELS)
ihzg1_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ihzg1_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ihzg1_styling = Styling(size=ihzg1_styling_size, position=ihzg1_styling_pos, color=ihzg1_styling_color)
ihzg1.styling = ihzg1_styling
ihzg1.display_order = 0
ihzg1.component_id = "ihzg1"
ihzg1.tag_name = "li"
ihzg1.custom_attributes = {"id": "ihzg1"}
i2nrg = Link(name="i2nrg", description="Link element", label="Services", url="#")
i2nrg_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i2nrg_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i2nrg_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
i2nrg_styling = Styling(size=i2nrg_styling_size, position=i2nrg_styling_pos, color=i2nrg_styling_color)
i2nrg.styling = i2nrg_styling
i2nrg.display_order = 0
i2nrg.component_id = "i2nrg"
i2nrg.component_type = "link"
i2nrg.tag_name = "a"
i2nrg.custom_attributes = {"href": "#", "id": "i2nrg"}
i9bhy = ViewContainer(name="i9bhy", description="li container", view_elements={i2nrg})
i9bhy_styling_size = Size(width="auto", height="auto", padding="0", margin="8px 0", unit_size=UnitSize.PIXELS)
i9bhy_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i9bhy_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i9bhy_styling = Styling(size=i9bhy_styling_size, position=i9bhy_styling_pos, color=i9bhy_styling_color)
i9bhy.styling = i9bhy_styling
i9bhy.display_order = 1
i9bhy.component_id = "i9bhy"
i9bhy.tag_name = "li"
i9bhy.custom_attributes = {"id": "i9bhy"}
isffb = Link(name="isffb", description="Link element", label="Contact", url="#")
isffb_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
isffb_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
isffb_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
isffb_styling = Styling(size=isffb_styling_size, position=isffb_styling_pos, color=isffb_styling_color)
isffb.styling = isffb_styling
isffb.display_order = 0
isffb.component_id = "isffb"
isffb.component_type = "link"
isffb.tag_name = "a"
isffb.custom_attributes = {"href": "#", "id": "isffb"}
ihdhf = ViewContainer(name="ihdhf", description="li container", view_elements={isffb})
ihdhf_styling_size = Size(width="auto", height="auto", padding="0", margin="8px 0", unit_size=UnitSize.PIXELS)
ihdhf_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ihdhf_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ihdhf_styling = Styling(size=ihdhf_styling_size, position=ihdhf_styling_pos, color=ihdhf_styling_color)
ihdhf.styling = ihdhf_styling
ihdhf.display_order = 2
ihdhf.component_id = "ihdhf"
ihdhf.tag_name = "li"
ihdhf.custom_attributes = {"id": "ihdhf"}
iv1e6 = ViewContainer(name="iv1e6", description="ul container", view_elements={ihzg1, i9bhy, ihdhf})
iv1e6_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iv1e6_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iv1e6_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.8")
iv1e6_styling = Styling(size=iv1e6_styling_size, position=iv1e6_styling_pos, color=iv1e6_styling_color)
iv1e6.styling = iv1e6_styling
iv1e6.display_order = 1
iv1e6.component_id = "iv1e6"
iv1e6.tag_name = "ul"
iv1e6.custom_attributes = {"id": "iv1e6"}
component_2 = ViewContainer(name="Component_2", description=" component", view_elements={ivkxw, iv1e6})
component_2.display_order = 1
imt95 = Text(name="imt95", content="Contact", description="Text element")
imt95_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
imt95_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
imt95_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
imt95_styling = Styling(size=imt95_styling_size, position=imt95_styling_pos, color=imt95_styling_color)
imt95.styling = imt95_styling
imt95.display_order = 0
imt95.component_id = "imt95"
imt95.component_type = "text"
imt95.tag_name = "h4"
imt95.custom_attributes = {"id": "imt95"}
iimpc = Text(name="iimpc", content="Email: info@example.com\nPhone: (123) 456-7890", description="Text element")
iimpc_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iimpc_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iimpc_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.8")
iimpc_styling = Styling(size=iimpc_styling_size, position=iimpc_styling_pos, color=iimpc_styling_color)
iimpc.styling = iimpc_styling
iimpc.display_order = 1
iimpc.component_id = "iimpc"
iimpc.component_type = "text"
iimpc.tag_name = "p"
iimpc.custom_attributes = {"id": "iimpc"}
component_3 = ViewContainer(name="Component_3", description=" component", view_elements={imt95, iimpc})
component_3.display_order = 2
i2m8v = ViewContainer(name="i2m8v", description=" component", view_elements={component, component_2, component_3})
i2m8v_styling_size = Size(width="auto", height="auto", padding="0", margin="0 auto", unit_size=UnitSize.PIXELS)
i2m8v_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i2m8v_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i2m8v_styling = Styling(size=i2m8v_styling_size, position=i2m8v_styling_pos, color=i2m8v_styling_color)
i2m8v_styling_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="30px")
i2m8v_styling.layout = i2m8v_styling_layout
i2m8v.styling = i2m8v_styling
i2m8v_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="30px")
i2m8v.layout = i2m8v_layout
i2m8v.display_order = 0
i2m8v.component_id = "i2m8v"
i2m8v.custom_attributes = {"id": "i2m8v"}
ibyqn = Text(name="ibyqn", content="© 2025 Your Company. All rights reserved.", description="Text element")
ibyqn_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ibyqn_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ibyqn_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.7")
ibyqn_styling = Styling(size=ibyqn_styling_size, position=ibyqn_styling_pos, color=ibyqn_styling_color)
ibyqn.styling = ibyqn_styling
ibyqn.display_order = 1
ibyqn.component_id = "ibyqn"
ibyqn.component_type = "text"
ibyqn.custom_attributes = {"id": "ibyqn"}
irj9h = ViewContainer(name="irj9h", description="footer container", view_elements={i2m8v, ibyqn})
irj9h_styling_size = Size(width="auto", height="auto", padding="40px 20px", margin="0", unit_size=UnitSize.PIXELS)
irj9h_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
irj9h_styling_color = Color(background_color="#2c3e50", text_color="white", border_color="#CCCCCC")
irj9h_styling = Styling(size=irj9h_styling_size, position=irj9h_styling_pos, color=irj9h_styling_color)
irj9h.styling = irj9h_styling
irj9h.display_order = 1
irj9h.component_id = "irj9h"
irj9h.tag_name = "footer"
irj9h.custom_attributes = {"id": "irj9h"}
ibuph = ViewContainer(name="ibuph", description="section container", view_elements={imbe6, irj9h})
ibuph_styling_size = Size(width="auto", height="auto", padding="60px 20px", margin="0", unit_size=UnitSize.PIXELS)
ibuph_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ibuph_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ibuph_styling = Styling(size=ibuph_styling_size, position=ibuph_styling_pos, color=ibuph_styling_color)
ibuph.styling = ibuph_styling
ibuph.display_order = 3
ibuph.component_id = "ibuph"
ibuph.tag_name = "section"
ibuph.custom_attributes = {"id": "ibuph"}
i9vb.view_elements = {i4sm, izmfj, imt5g, ibuph}


# Screen: ixlmh
ixlmh = Screen(name="ixlmh", description="about", view_elements=set(), route_path="/about", screen_size="Medium")
ixlmh.component_id = "about"
i9yqn = Text(name="i9yqn", content="Logo", description="Text element")
i9yqn_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="24px", unit_size=UnitSize.PIXELS)
i9yqn_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i9yqn_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i9yqn_styling = Styling(size=i9yqn_styling_size, position=i9yqn_styling_pos, color=i9yqn_styling_color)
i9yqn.styling = i9yqn_styling
i9yqn.display_order = 0
i9yqn.component_id = "i9yqn"
i9yqn.component_type = "text"
i9yqn.custom_attributes = {"id": "i9yqn"}
ieeso = Link(name="ieeso", description="Link element", label="Home", url="#")
ieeso_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ieeso_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ieeso_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
ieeso_styling = Styling(size=ieeso_styling_size, position=ieeso_styling_pos, color=ieeso_styling_color)
ieeso.styling = ieeso_styling
ieeso.display_order = 0
ieeso.component_id = "ieeso"
ieeso.component_type = "link"
ieeso.tag_name = "a"
ieeso.custom_attributes = {"href": "#", "id": "ieeso"}
iu4wsh = Link(name="iu4wsh", description="Link element", label="About", url="#")
iu4wsh_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iu4wsh_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iu4wsh_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
iu4wsh_styling = Styling(size=iu4wsh_styling_size, position=iu4wsh_styling_pos, color=iu4wsh_styling_color)
iu4wsh.styling = iu4wsh_styling
iu4wsh.display_order = 1
iu4wsh.component_id = "iu4wsh"
iu4wsh.component_type = "link"
iu4wsh.tag_name = "a"
iu4wsh.custom_attributes = {"href": "#", "id": "iu4wsh"}
i1t83h = Link(name="i1t83h", description="Link element", label="Services", url="#")
i1t83h_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i1t83h_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i1t83h_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
i1t83h_styling = Styling(size=i1t83h_styling_size, position=i1t83h_styling_pos, color=i1t83h_styling_color)
i1t83h.styling = i1t83h_styling
i1t83h.display_order = 2
i1t83h.component_id = "i1t83h"
i1t83h.component_type = "link"
i1t83h.tag_name = "a"
i1t83h.custom_attributes = {"href": "#", "id": "i1t83h"}
ia0taw = Link(name="ia0taw", description="Link element", label="Contact", url="#")
ia0taw_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ia0taw_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ia0taw_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
ia0taw_styling = Styling(size=ia0taw_styling_size, position=ia0taw_styling_pos, color=ia0taw_styling_color)
ia0taw.styling = ia0taw_styling
ia0taw.display_order = 3
ia0taw.component_id = "ia0taw"
ia0taw.component_type = "link"
ia0taw.tag_name = "a"
ia0taw.custom_attributes = {"href": "#", "id": "ia0taw"}
iuj7h = ViewContainer(name="iuj7h", description=" component", view_elements={ieeso, iu4wsh, i1t83h, ia0taw})
iuj7h_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iuj7h_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iuj7h_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
iuj7h_styling = Styling(size=iuj7h_styling_size, position=iuj7h_styling_pos, color=iuj7h_styling_color)
iuj7h_styling_layout = Layout(layout_type=LayoutType.FLEX, gap="30px")
iuj7h_styling.layout = iuj7h_styling_layout
iuj7h.styling = iuj7h_styling
iuj7h_layout = Layout(layout_type=LayoutType.FLEX, gap="30px")
iuj7h.layout = iuj7h_layout
iuj7h.display_order = 1
iuj7h.component_id = "iuj7h"
iuj7h.custom_attributes = {"id": "iuj7h"}
ifihk = ViewContainer(name="ifihk", description="nav container", view_elements={i9yqn, iuj7h})
ifihk_styling_size = Size(width="auto", height="auto", padding="15px 30px", margin="0", unit_size=UnitSize.PIXELS)
ifihk_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ifihk_styling_color = Color(background_color="#333", text_color="white", border_color="#CCCCCC")
ifihk_styling = Styling(size=ifihk_styling_size, position=ifihk_styling_pos, color=ifihk_styling_color)
ifihk_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="space-between", align_items="center", gap="16px")
ifihk_styling.layout = ifihk_styling_layout
ifihk.styling = ifihk_styling
ifihk_layout = Layout(layout_type=LayoutType.FLEX, justify_content="space-between", align_items="center", gap="16px")
ifihk.layout = ifihk_layout
ifihk.display_order = 0
ifihk.component_id = "ifihk"
ifihk.tag_name = "nav"
ifihk.custom_attributes = {"id": "ifihk"}
iwr92g = Text(name="iwr92g", content="Our Features", description="Text element")
iwr92g_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="36px", unit_size=UnitSize.PIXELS)
iwr92g_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iwr92g_styling_color = Color(background_color="#FFFFFF", text_color="#333", border_color="#CCCCCC")
iwr92g_styling = Styling(size=iwr92g_styling_size, position=iwr92g_styling_pos, color=iwr92g_styling_color)
iwr92g.styling = iwr92g_styling
iwr92g.display_order = 0
iwr92g.component_id = "iwr92g"
iwr92g.component_type = "text"
iwr92g.tag_name = "h2"
iwr92g.custom_attributes = {"id": "iwr92g"}
iex3vq = Text(name="iex3vq", content="🚀", description="Text element")
iex3vq_styling_size = Size(width="80px", height="80px", padding="0", margin="0 auto 20px", font_size="36px", unit_size=UnitSize.PIXELS)
iex3vq_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iex3vq_styling_color = Color(background_color="linear-gradient(135deg, #667eea 0%, #764ba2 100%)", text_color="white", border_color="#CCCCCC")
iex3vq_styling = Styling(size=iex3vq_styling_size, position=iex3vq_styling_pos, color=iex3vq_styling_color)
iex3vq_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", align_items="center", gap="16px")
iex3vq_styling.layout = iex3vq_styling_layout
iex3vq.styling = iex3vq_styling
iex3vq.display_order = 0
iex3vq.component_id = "iex3vq"
iex3vq.component_type = "text"
iex3vq.custom_attributes = {"id": "iex3vq"}
ibcl0g = Text(name="ibcl0g", content="Fast Performance", description="Text element")
ibcl0g_styling_size = Size(width="auto", height="auto", padding="0", margin="20px 0 15px", unit_size=UnitSize.PIXELS)
ibcl0g_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ibcl0g_styling_color = Color(background_color="#FFFFFF", text_color="#333", border_color="#CCCCCC")
ibcl0g_styling = Styling(size=ibcl0g_styling_size, position=ibcl0g_styling_pos, color=ibcl0g_styling_color)
ibcl0g.styling = ibcl0g_styling
ibcl0g.display_order = 1
ibcl0g.component_id = "ibcl0g"
ibcl0g.component_type = "text"
ibcl0g.tag_name = "h3"
ibcl0g.custom_attributes = {"id": "ibcl0g"}
isujzh = Text(name="isujzh", content="Lightning-fast loading times and smooth interactions for the best user experience.", description="Text element")
isujzh_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
isujzh_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
isujzh_styling_color = Color(background_color="#FFFFFF", text_color="#666", border_color="#CCCCCC")
isujzh_styling = Styling(size=isujzh_styling_size, position=isujzh_styling_pos, color=isujzh_styling_color)
isujzh.styling = isujzh_styling
isujzh.display_order = 2
isujzh.component_id = "isujzh"
isujzh.component_type = "text"
isujzh.tag_name = "p"
isujzh.custom_attributes = {"id": "isujzh"}
imb08m = ViewContainer(name="imb08m", description=" component", view_elements={iex3vq, ibcl0g, isujzh})
imb08m_styling_size = Size(width="auto", height="auto", padding="30px", margin="0", unit_size=UnitSize.PIXELS)
imb08m_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
imb08m_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
imb08m_styling = Styling(size=imb08m_styling_size, position=imb08m_styling_pos, color=imb08m_styling_color)
imb08m.styling = imb08m_styling
imb08m.display_order = 1
imb08m.component_id = "imb08m"
imb08m.custom_attributes = {"id": "imb08m"}
iywmg2 = Text(name="iywmg2", content="🔒", description="Text element")
iywmg2_styling_size = Size(width="80px", height="80px", padding="0", margin="0 auto 20px", font_size="36px", unit_size=UnitSize.PIXELS)
iywmg2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iywmg2_styling_color = Color(background_color="linear-gradient(135deg, #667eea 0%, #764ba2 100%)", text_color="white", border_color="#CCCCCC")
iywmg2_styling = Styling(size=iywmg2_styling_size, position=iywmg2_styling_pos, color=iywmg2_styling_color)
iywmg2_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", align_items="center", gap="16px")
iywmg2_styling.layout = iywmg2_styling_layout
iywmg2.styling = iywmg2_styling
iywmg2.display_order = 0
iywmg2.component_id = "iywmg2"
iywmg2.component_type = "text"
iywmg2.custom_attributes = {"id": "iywmg2"}
ik41p4 = Text(name="ik41p4", content="Secure & Safe", description="Text element")
ik41p4_styling_size = Size(width="auto", height="auto", padding="0", margin="20px 0 15px", unit_size=UnitSize.PIXELS)
ik41p4_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ik41p4_styling_color = Color(background_color="#FFFFFF", text_color="#333", border_color="#CCCCCC")
ik41p4_styling = Styling(size=ik41p4_styling_size, position=ik41p4_styling_pos, color=ik41p4_styling_color)
ik41p4.styling = ik41p4_styling
ik41p4.display_order = 1
ik41p4.component_id = "ik41p4"
ik41p4.component_type = "text"
ik41p4.tag_name = "h3"
ik41p4.custom_attributes = {"id": "ik41p4"}
izso6w = Text(name="izso6w", content="Enterprise-grade security to protect your data and ensure privacy.", description="Text element")
izso6w_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
izso6w_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
izso6w_styling_color = Color(background_color="#FFFFFF", text_color="#666", border_color="#CCCCCC")
izso6w_styling = Styling(size=izso6w_styling_size, position=izso6w_styling_pos, color=izso6w_styling_color)
izso6w.styling = izso6w_styling
izso6w.display_order = 2
izso6w.component_id = "izso6w"
izso6w.component_type = "text"
izso6w.tag_name = "p"
izso6w.custom_attributes = {"id": "izso6w"}
i1xfjc = ViewContainer(name="i1xfjc", description=" component", view_elements={iywmg2, ik41p4, izso6w})
i1xfjc_styling_size = Size(width="auto", height="auto", padding="30px", margin="0", unit_size=UnitSize.PIXELS)
i1xfjc_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i1xfjc_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i1xfjc_styling = Styling(size=i1xfjc_styling_size, position=i1xfjc_styling_pos, color=i1xfjc_styling_color)
i1xfjc.styling = i1xfjc_styling
i1xfjc.display_order = 3
i1xfjc.component_id = "i1xfjc"
i1xfjc.custom_attributes = {"id": "i1xfjc"}
ivl41f = Text(name="ivl41f", content="📱", description="Text element")
ivl41f_styling_size = Size(width="80px", height="80px", padding="0", margin="0 auto 20px", font_size="36px", unit_size=UnitSize.PIXELS)
ivl41f_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ivl41f_styling_color = Color(background_color="linear-gradient(135deg, #667eea 0%, #764ba2 100%)", text_color="white", border_color="#CCCCCC")
ivl41f_styling = Styling(size=ivl41f_styling_size, position=ivl41f_styling_pos, color=ivl41f_styling_color)
ivl41f_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", align_items="center", gap="16px")
ivl41f_styling.layout = ivl41f_styling_layout
ivl41f.styling = ivl41f_styling
ivl41f.display_order = 0
ivl41f.component_id = "ivl41f"
ivl41f.component_type = "text"
ivl41f.custom_attributes = {"id": "ivl41f"}
cell_4 = ViewContainer(name="Cell_4", description=" container", view_elements=set())
cell_4_styling_size = Size(width="8%", height="75px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
cell_4_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
cell_4_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
cell_4_styling = Styling(size=cell_4_styling_size, position=cell_4_styling_pos, color=cell_4_styling_color)
cell_4.styling = cell_4_styling
cell_4.display_order = 0
cell_4.component_id = "container_cell_4"
cell_4.css_classes = ["gjs-cell"]
cell_5 = ViewContainer(name="Cell_5", description=" container", view_elements=set())
cell_5_styling_size = Size(width="8%", height="75px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
cell_5_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
cell_5_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
cell_5_styling = Styling(size=cell_5_styling_size, position=cell_5_styling_pos, color=cell_5_styling_color)
cell_5.styling = cell_5_styling
cell_5.display_order = 1
cell_5.component_id = "container_cell_5"
cell_5.css_classes = ["gjs-cell"]
iuws54 = ViewContainer(name="iuws54", description=" container", view_elements={cell_4, cell_5})
iuws54_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
iuws54_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iuws54_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
iuws54_styling = Styling(size=iuws54_styling_size, position=iuws54_styling_pos, color=iuws54_styling_color)
iuws54.styling = iuws54_styling
iuws54.display_order = 1
iuws54.component_id = "iuws54"
iuws54.css_classes = ["gjs-row"]
iuws54.custom_attributes = {"id": "iuws54"}
ikruby = Text(name="ikruby", content="Responsive Design", description="Text element")
ikruby_styling_size = Size(width="auto", height="auto", padding="0", margin="20px 0 15px", unit_size=UnitSize.PIXELS)
ikruby_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ikruby_styling_color = Color(background_color="#FFFFFF", text_color="#333", border_color="#CCCCCC")
ikruby_styling = Styling(size=ikruby_styling_size, position=ikruby_styling_pos, color=ikruby_styling_color)
ikruby.styling = ikruby_styling
ikruby.display_order = 2
ikruby.component_id = "ikruby"
ikruby.component_type = "text"
ikruby.tag_name = "h3"
ikruby.custom_attributes = {"id": "ikruby"}
ixm63e = Text(name="ixm63e", content="Works perfectly on all devices - desktop, tablet, and mobile.", description="Text element")
ixm63e_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
ixm63e_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ixm63e_styling_color = Color(background_color="#FFFFFF", text_color="#666", border_color="#CCCCCC")
ixm63e_styling = Styling(size=ixm63e_styling_size, position=ixm63e_styling_pos, color=ixm63e_styling_color)
ixm63e.styling = ixm63e_styling
ixm63e.display_order = 3
ixm63e.component_id = "ixm63e"
ixm63e.component_type = "text"
ixm63e.tag_name = "p"
ixm63e.custom_attributes = {"id": "ixm63e"}
idixfc = ViewContainer(name="idixfc", description=" component", view_elements={ivl41f, iuws54, ikruby, ixm63e})
idixfc_styling_size = Size(width="auto", height="auto", padding="30px", margin="0", unit_size=UnitSize.PIXELS)
idixfc_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
idixfc_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
idixfc_styling = Styling(size=idixfc_styling_size, position=idixfc_styling_pos, color=idixfc_styling_color)
idixfc.styling = idixfc_styling
idixfc.display_order = 5
idixfc.component_id = "idixfc"
idixfc.custom_attributes = {"id": "idixfc"}
iukzqn = ViewContainer(name="iukzqn", description=" component", view_elements={imb08m, i1xfjc, idixfc})
iukzqn_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iukzqn_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iukzqn_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
iukzqn_styling = Styling(size=iukzqn_styling_size, position=iukzqn_styling_pos, color=iukzqn_styling_color)
iukzqn_styling_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(300px, 1fr))", gap="40px")
iukzqn_styling.layout = iukzqn_styling_layout
iukzqn.styling = iukzqn_styling
iukzqn_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(300px, 1fr))", gap="40px")
iukzqn.layout = iukzqn_layout
iukzqn.display_order = 1
iukzqn.component_id = "iukzqn"
iukzqn.custom_attributes = {"id": "iukzqn"}
ip47kp = ViewContainer(name="ip47kp", description=" component", view_elements={iwr92g, iukzqn})
ip47kp_styling_size = Size(width="auto", height="auto", padding="0", margin="0 auto", unit_size=UnitSize.PIXELS)
ip47kp_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ip47kp_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ip47kp_styling = Styling(size=ip47kp_styling_size, position=ip47kp_styling_pos, color=ip47kp_styling_color)
ip47kp.styling = ip47kp_styling
ip47kp.display_order = 0
ip47kp.component_id = "ip47kp"
ip47kp.custom_attributes = {"id": "ip47kp"}
igf89k = ViewContainer(name="igf89k", description="section container", view_elements={ip47kp})
igf89k_styling_size = Size(width="auto", height="auto", padding="60px 20px", margin="0", unit_size=UnitSize.PIXELS)
igf89k_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
igf89k_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
igf89k_styling = Styling(size=igf89k_styling_size, position=igf89k_styling_pos, color=igf89k_styling_color)
igf89k.styling = igf89k_styling
igf89k.display_order = 1
igf89k.component_id = "igf89k"
igf89k.tag_name = "section"
igf89k.custom_attributes = {"id": "igf89k"}
i3naif = Text(name="i3naif", content="Ready to Get Started?", description="Text element")
i3naif_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="42px", unit_size=UnitSize.PIXELS)
i3naif_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i3naif_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i3naif_styling = Styling(size=i3naif_styling_size, position=i3naif_styling_pos, color=i3naif_styling_color)
i3naif.styling = i3naif_styling
i3naif.display_order = 0
i3naif.component_id = "i3naif"
i3naif.component_type = "text"
i3naif.tag_name = "h2"
i3naif.custom_attributes = {"id": "i3naif"}
i8zt4z = Text(name="i8zt4z", content="Join thousands of satisfied customers and take your business to the next level.", description="Text element")
i8zt4z_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="20px", unit_size=UnitSize.PIXELS)
i8zt4z_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i8zt4z_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.95")
i8zt4z_styling = Styling(size=i8zt4z_styling_size, position=i8zt4z_styling_pos, color=i8zt4z_styling_color)
i8zt4z.styling = i8zt4z_styling
i8zt4z.display_order = 1
i8zt4z.component_id = "i8zt4z"
i8zt4z.component_type = "text"
i8zt4z.tag_name = "p"
i8zt4z.custom_attributes = {"id": "i8zt4z"}
idxhhv = Button(name="idxhhv", description="Button component", label="Start Free Trial", buttonType=ButtonType.CustomizableButton, actionType=ButtonActionType.View)
idxhhv_styling_size = Size(width="auto", height="auto", padding="18px 40px", margin="0", font_size="18px", unit_size=UnitSize.PIXELS)
idxhhv_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
idxhhv_styling_color = Color(background_color="white", text_color="#f5576c", border_color="#CCCCCC")
idxhhv_styling = Styling(size=idxhhv_styling_size, position=idxhhv_styling_pos, color=idxhhv_styling_color)
idxhhv.styling = idxhhv_styling
idxhhv.display_order = 0
idxhhv.component_id = "idxhhv"
idxhhv.component_type = "button"
idxhhv.tag_name = "button"
idxhhv.custom_attributes = {"type": "button", "id": "idxhhv"}
i9x0hm = Button(name="i9x0hm", description="Button component", label="Learn More", buttonType=ButtonType.CustomizableButton, actionType=ButtonActionType.View)
i9x0hm_styling_size = Size(width="auto", height="auto", padding="18px 40px", margin="0", font_size="18px", unit_size=UnitSize.PIXELS)
i9x0hm_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i9x0hm_styling_color = Color(background_color="transparent", text_color="white", border_color="#CCCCCC")
i9x0hm_styling = Styling(size=i9x0hm_styling_size, position=i9x0hm_styling_pos, color=i9x0hm_styling_color)
i9x0hm.styling = i9x0hm_styling
i9x0hm.display_order = 1
i9x0hm.component_id = "i9x0hm"
i9x0hm.component_type = "button"
i9x0hm.tag_name = "button"
i9x0hm.custom_attributes = {"type": "button", "id": "i9x0hm"}
idvesc = ViewContainer(name="idvesc", description=" component", view_elements={idxhhv, i9x0hm})
idvesc_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
idvesc_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
idvesc_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
idvesc_styling = Styling(size=idvesc_styling_size, position=idvesc_styling_pos, color=idvesc_styling_color)
idvesc_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", flex_wrap="wrap", gap="20px")
idvesc_styling.layout = idvesc_styling_layout
idvesc.styling = idvesc_styling
idvesc_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", flex_wrap="wrap", gap="20px")
idvesc.layout = idvesc_layout
idvesc.display_order = 2
idvesc.component_id = "idvesc"
idvesc.custom_attributes = {"id": "idvesc"}
iq8dag = ViewContainer(name="iq8dag", description=" component", view_elements={i3naif, i8zt4z, idvesc})
iq8dag_styling_size = Size(width="auto", height="auto", padding="0", margin="0 auto", unit_size=UnitSize.PIXELS)
iq8dag_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iq8dag_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
iq8dag_styling = Styling(size=iq8dag_styling_size, position=iq8dag_styling_pos, color=iq8dag_styling_color)
iq8dag.styling = iq8dag_styling
iq8dag.display_order = 0
iq8dag.component_id = "iq8dag"
iq8dag.custom_attributes = {"id": "iq8dag"}
iy4v2q = ViewContainer(name="iy4v2q", description="section container", view_elements={iq8dag})
iy4v2q_styling_size = Size(width="auto", height="auto", padding="80px 20px", margin="0", unit_size=UnitSize.PIXELS)
iy4v2q_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iy4v2q_styling_color = Color(background_color="linear-gradient(135deg, #f093fb 0%, #f5576c 100%)", text_color="white", border_color="#CCCCCC")
iy4v2q_styling = Styling(size=iy4v2q_styling_size, position=iy4v2q_styling_pos, color=iy4v2q_styling_color)
iy4v2q.styling = iy4v2q_styling
iy4v2q.display_order = 2
iy4v2q.component_id = "iy4v2q"
iy4v2q.tag_name = "section"
iy4v2q.custom_attributes = {"id": "iy4v2q"}
i6v43o = Text(name="i6v43o", content="1000+", description="Text element")
i6v43o_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="48px", unit_size=UnitSize.PIXELS)
i6v43o_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i6v43o_styling_color = Color(background_color="#FFFFFF", text_color="#3498db", border_color="#CCCCCC")
i6v43o_styling = Styling(size=i6v43o_styling_size, position=i6v43o_styling_pos, color=i6v43o_styling_color)
i6v43o.styling = i6v43o_styling
i6v43o.display_order = 0
i6v43o.component_id = "i6v43o"
i6v43o.component_type = "text"
i6v43o.custom_attributes = {"id": "i6v43o"}
i2jlwi = Text(name="i2jlwi", content="Happy Clients", description="Text element")
i2jlwi_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="18px", unit_size=UnitSize.PIXELS)
i2jlwi_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i2jlwi_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.9")
i2jlwi_styling = Styling(size=i2jlwi_styling_size, position=i2jlwi_styling_pos, color=i2jlwi_styling_color)
i2jlwi.styling = i2jlwi_styling
i2jlwi.display_order = 1
i2jlwi.component_id = "i2jlwi"
i2jlwi.component_type = "text"
i2jlwi.custom_attributes = {"id": "i2jlwi"}
component_4 = ViewContainer(name="Component_4", description=" component", view_elements={i6v43o, i2jlwi})
component_4.display_order = 1
i4th7k = Text(name="i4th7k", content="50+", description="Text element")
i4th7k_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="48px", unit_size=UnitSize.PIXELS)
i4th7k_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i4th7k_styling_color = Color(background_color="#FFFFFF", text_color="#2ecc71", border_color="#CCCCCC")
i4th7k_styling = Styling(size=i4th7k_styling_size, position=i4th7k_styling_pos, color=i4th7k_styling_color)
i4th7k.styling = i4th7k_styling
i4th7k.display_order = 0
i4th7k.component_id = "i4th7k"
i4th7k.component_type = "text"
i4th7k.custom_attributes = {"id": "i4th7k"}
ik4fzf = Text(name="ik4fzf", content="Team Members", description="Text element")
ik4fzf_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="18px", unit_size=UnitSize.PIXELS)
ik4fzf_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ik4fzf_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.9")
ik4fzf_styling = Styling(size=ik4fzf_styling_size, position=ik4fzf_styling_pos, color=ik4fzf_styling_color)
ik4fzf.styling = ik4fzf_styling
ik4fzf.display_order = 1
ik4fzf.component_id = "ik4fzf"
ik4fzf.component_type = "text"
ik4fzf.custom_attributes = {"id": "ik4fzf"}
component_5 = ViewContainer(name="Component_5", description=" component", view_elements={i4th7k, ik4fzf})
component_5.display_order = 3
iebu3w = Text(name="iebu3w", content="99%", description="Text element")
iebu3w_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="48px", unit_size=UnitSize.PIXELS)
iebu3w_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iebu3w_styling_color = Color(background_color="#FFFFFF", text_color="#e74c3c", border_color="#CCCCCC")
iebu3w_styling = Styling(size=iebu3w_styling_size, position=iebu3w_styling_pos, color=iebu3w_styling_color)
iebu3w.styling = iebu3w_styling
iebu3w.display_order = 0
iebu3w.component_id = "iebu3w"
iebu3w.component_type = "text"
iebu3w.custom_attributes = {"id": "iebu3w"}
ime059 = Text(name="ime059", content="Satisfaction Rate", description="Text element")
ime059_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="18px", unit_size=UnitSize.PIXELS)
ime059_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ime059_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.9")
ime059_styling = Styling(size=ime059_styling_size, position=ime059_styling_pos, color=ime059_styling_color)
ime059.styling = ime059_styling
ime059.display_order = 1
ime059.component_id = "ime059"
ime059.component_type = "text"
ime059.custom_attributes = {"id": "ime059"}
component_6 = ViewContainer(name="Component_6", description=" component", view_elements={iebu3w, ime059})
component_6.display_order = 5
ivqqqo = Text(name="ivqqqo", content="24/7", description="Text element")
ivqqqo_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="48px", unit_size=UnitSize.PIXELS)
ivqqqo_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ivqqqo_styling_color = Color(background_color="#FFFFFF", text_color="#f39c12", border_color="#CCCCCC")
ivqqqo_styling = Styling(size=ivqqqo_styling_size, position=ivqqqo_styling_pos, color=ivqqqo_styling_color)
ivqqqo.styling = ivqqqo_styling
ivqqqo.display_order = 0
ivqqqo.component_id = "ivqqqo"
ivqqqo.component_type = "text"
ivqqqo.custom_attributes = {"id": "ivqqqo"}
iohofv = Text(name="iohofv", content="Support Available", description="Text element")
iohofv_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="18px", unit_size=UnitSize.PIXELS)
iohofv_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iohofv_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.9")
iohofv_styling = Styling(size=iohofv_styling_size, position=iohofv_styling_pos, color=iohofv_styling_color)
iohofv.styling = iohofv_styling
iohofv.display_order = 1
iohofv.component_id = "iohofv"
iohofv.component_type = "text"
iohofv.custom_attributes = {"id": "iohofv"}
component_7 = ViewContainer(name="Component_7", description=" component", view_elements={ivqqqo, iohofv})
component_7.display_order = 7
ictm5k = ViewContainer(name="ictm5k", description=" component", view_elements={component_4, component_5, component_6, component_7})
ictm5k_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ictm5k_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ictm5k_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ictm5k_styling = Styling(size=ictm5k_styling_size, position=ictm5k_styling_pos, color=ictm5k_styling_color)
ictm5k_styling_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(200px, 1fr))", gap="40px")
ictm5k_styling.layout = ictm5k_styling_layout
ictm5k.styling = ictm5k_styling
ictm5k_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(200px, 1fr))", gap="40px")
ictm5k.layout = ictm5k_layout
ictm5k.display_order = 0
ictm5k.component_id = "ictm5k"
ictm5k.custom_attributes = {"id": "ictm5k"}
iqkdfk = ViewContainer(name="iqkdfk", description=" component", view_elements={ictm5k})
iqkdfk_styling_size = Size(width="auto", height="auto", padding="0", margin="0 auto", unit_size=UnitSize.PIXELS)
iqkdfk_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iqkdfk_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
iqkdfk_styling = Styling(size=iqkdfk_styling_size, position=iqkdfk_styling_pos, color=iqkdfk_styling_color)
iqkdfk.styling = iqkdfk_styling
iqkdfk.display_order = 0
iqkdfk.component_id = "iqkdfk"
iqkdfk.custom_attributes = {"id": "iqkdfk"}
i4iofq = ViewContainer(name="i4iofq", description="section container", view_elements={iqkdfk})
i4iofq_styling_size = Size(width="auto", height="auto", padding="80px 20px", margin="0", unit_size=UnitSize.PIXELS)
i4iofq_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i4iofq_styling_color = Color(background_color="#2c3e50", text_color="white", border_color="#CCCCCC")
i4iofq_styling = Styling(size=i4iofq_styling_size, position=i4iofq_styling_pos, color=i4iofq_styling_color)
i4iofq.styling = i4iofq_styling
i4iofq.display_order = 3
i4iofq.component_id = "i4iofq"
i4iofq.tag_name = "section"
i4iofq.custom_attributes = {"id": "i4iofq"}
ixlmh.view_elements = {ifihk, igf89k, iy4v2q, i4iofq}

gui_module = Module(
    name="GUI_Module",
    screens={i9vb, ixlmh}
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
