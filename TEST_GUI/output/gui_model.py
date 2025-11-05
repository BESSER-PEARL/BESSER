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
Counter = Class(name="Counter")

# Counter class attributes and methods
Counter_label: Property = Property(name="label", type=StringType)
Counter_value: Property = Property(name="value", type=StringType)
Counter.attributes={Counter_label, Counter_value}

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={Counter},
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

# Screen: wrapper
wrapper = Screen(name="wrapper", description="Home", view_elements=set(), is_main_page=True, route_path="/home", screen_size="Medium")
wrapper.component_id = "yXWXAvEmfeStNpkm"
meta = ViewComponent(name="meta", description="meta component")
meta.display_order = 0
meta.tag_name = "meta"
meta.custom_attributes = {"charset": "utf-8"}
viewport = ViewComponent(name="viewport", description="meta component")
viewport.display_order = 1
viewport.tag_name = "meta"
viewport.custom_attributes = {"name": "viewport", "content": "width=device-width, initial-scale=1"}
title = Text(name="title", content="BESSER Generated Page", description="Text element")
title.display_order = 2
title.component_type = "text"
title.tag_name = "title"
io07 = Text(name="io07", content="BESSER", description="Text element")
io07_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="24px", unit_size=UnitSize.PIXELS)
io07_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
io07_styling_color = Color()
io07_styling = Styling(size=io07_styling_size, position=io07_styling_pos, color=io07_styling_color)
io07.styling = io07_styling
io07.display_order = 0
io07.component_id = "io07"
io07.component_type = "text"
io07.custom_attributes = {"id": "io07"}
ibtww = Link(name="ibtww", description="Link element", label="Home", url="/")
ibtww_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ibtww_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ibtww_styling_color = Color(text_color="white")
ibtww_styling = Styling(size=ibtww_styling_size, position=ibtww_styling_pos, color=ibtww_styling_color)
ibtww.styling = ibtww_styling
ibtww.display_order = 0
ibtww.component_id = "ibtww"
ibtww.component_type = "link"
ibtww.tag_name = "a"
ibtww.custom_attributes = {"href": "/", "id": "ibtww"}
ild7j = Link(name="ild7j", description="Link element", label="About", url="/about")
ild7j_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ild7j_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ild7j_styling_color = Color(text_color="white")
ild7j_styling = Styling(size=ild7j_styling_size, position=ild7j_styling_pos, color=ild7j_styling_color)
ild7j.styling = ild7j_styling
ild7j.display_order = 1
ild7j.component_id = "ild7j"
ild7j.component_type = "link"
ild7j.tag_name = "a"
ild7j.custom_attributes = {"href": "/about", "id": "ild7j"}
inp2m = ViewContainer(name="inp2m", description=" component", view_elements={ibtww, ild7j})
inp2m_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
inp2m_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
inp2m_styling_color = Color()
inp2m_styling = Styling(size=inp2m_styling_size, position=inp2m_styling_pos, color=inp2m_styling_color)
inp2m.styling = inp2m_styling
inp2m.display_order = 1
inp2m.component_id = "inp2m"
inp2m.custom_attributes = {"id": "inp2m"}
i5jh = ViewContainer(name="i5jh", description="nav container", view_elements={io07, inp2m})
i5jh_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i5jh_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i5jh_styling_color = Color(background_color="linear-gradient(135deg, rgb(75, 60, 130) 0%, rgb(90, 61, 145) 100%) !important", text_color="white")
i5jh_styling = Styling(size=i5jh_styling_size, position=i5jh_styling_pos, color=i5jh_styling_color)
i5jh_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="space-between", align_items="center", gap="16px")
i5jh_styling.layout = i5jh_styling_layout
i5jh.styling = i5jh_styling
i5jh_layout = Layout(layout_type=LayoutType.FLEX, justify_content="space-between", align_items="center", gap="16px")
i5jh.layout = i5jh_layout
i5jh.display_order = 3
i5jh.component_id = "i5jh"
i5jh.tag_name = "nav"
i5jh.custom_attributes = {"id": "i5jh"}
i72zk = LineChart(name="i72zk", title="Line Chart Title", primary_color="#4CAF50", line_width=2, show_grid=False, show_legend=False, show_tooltip=False, curve_type="monotone", animate=False, legend_position="top", grid_color="#e0e0e0", dot_size=5)
domain_model_ref = globals().get('domain_model')
i72zk_binding_domain = None
if domain_model_ref is not None:
    i72zk_binding_domain = domain_model_ref.get_class_by_name("Counter")
if i72zk_binding_domain:
    i72zk_binding = DataBinding(domain_concept=i72zk_binding_domain)
    i72zk_binding.label_field = next((attr for attr in i72zk_binding_domain.attributes if attr.name == "label"), None)
    i72zk_binding.data_field = next((attr for attr in i72zk_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Counter' not resolved; data binding skipped.
    i72zk_binding = None
if i72zk_binding:
    i72zk.data_binding = i72zk_binding
i72zk_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
i72zk_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i72zk_styling_color = Color()
i72zk_styling = Styling(size=i72zk_styling_size, position=i72zk_styling_pos, color=i72zk_styling_color)
i72zk.styling = i72zk_styling
i72zk.display_order = 0
i72zk.component_id = "i72zk"
i72zk.component_type = "line-chart"
i72zk.css_classes = ["line-chart-component", "has-data-binding"]
i72zk.custom_attributes = {"chart-color": "#4CAF50", "chart-title": "Line Chart Title", "data-source": "a87eb810-2e0a-492b-8cd3-62251566f648", "label-field": "f63380b5-ccf8-448a-bb82-70d62d61442e", "data-field": "2294f9b0-aa04-4c5d-8851-b333f6b3c2fc", "line-width": "2", "show-grid": "", "show-legend": "", "show-tooltip": "", "curve-type": "monotone", "animate": "", "id": "i72zk"}
ifczf = ViewContainer(name="ifczf", description=" component", view_elements={i72zk})
ifczf_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
ifczf_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ifczf_styling_color = Color()
ifczf_styling = Styling(size=ifczf_styling_size, position=ifczf_styling_pos, color=ifczf_styling_color)
ifczf.styling = ifczf_styling
ifczf.display_order = 0
ifczf.component_id = "ifczf"
ifczf.css_classes = ["gjs-cell"]
ifczf.custom_attributes = {"id": "ifczf"}
i8fz6 = PieChart(name="i8fz6", title="Pie Chart Title", show_legend=False, legend_position=Alignment.BOTTOM, show_labels=False, label_position=Alignment.INSIDE, padding_angle=0, inner_radius=0, outer_radius=80, start_angle=0, end_angle=360)
domain_model_ref = globals().get('domain_model')
i8fz6_binding_domain = None
if domain_model_ref is not None:
    i8fz6_binding_domain = domain_model_ref.get_class_by_name("Counter")
if i8fz6_binding_domain:
    i8fz6_binding = DataBinding(domain_concept=i8fz6_binding_domain)
    i8fz6_binding.label_field = next((attr for attr in i8fz6_binding_domain.attributes if attr.name == "label"), None)
    i8fz6_binding.data_field = next((attr for attr in i8fz6_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Counter' not resolved; data binding skipped.
    i8fz6_binding = None
if i8fz6_binding:
    i8fz6.data_binding = i8fz6_binding
i8fz6_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
i8fz6_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i8fz6_styling_color = Color()
i8fz6_styling = Styling(size=i8fz6_styling_size, position=i8fz6_styling_pos, color=i8fz6_styling_color)
i8fz6.styling = i8fz6_styling
i8fz6.display_order = 0
i8fz6.component_id = "i8fz6"
i8fz6.component_type = "pie-chart"
i8fz6.css_classes = ["pie-chart-component", "has-data-binding"]
i8fz6.custom_attributes = {"chart-title": "Pie Chart Title", "data-source": "a87eb810-2e0a-492b-8cd3-62251566f648", "label-field": "f63380b5-ccf8-448a-bb82-70d62d61442e", "data-field": "2294f9b0-aa04-4c5d-8851-b333f6b3c2fc", "show-legend": "", "legend-position": "bottom", "show-labels": "", "label-position": "inside", "padding-angle": "0", "id": "i8fz6"}
iz1hy = ViewContainer(name="iz1hy", description=" component", view_elements={i8fz6})
iz1hy_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
iz1hy_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iz1hy_styling_color = Color()
iz1hy_styling = Styling(size=iz1hy_styling_size, position=iz1hy_styling_pos, color=iz1hy_styling_color)
iz1hy.styling = iz1hy_styling
iz1hy.display_order = 1
iz1hy.component_id = "iz1hy"
iz1hy.css_classes = ["gjs-cell"]
iz1hy.custom_attributes = {"id": "iz1hy"}
id9ne = RadarChart(name="id9ne", title="Radar Chart Title", primary_color="#8884d8", show_grid=False, show_tooltip=False, show_radius_axis=False, show_legend=True, legend_position="top", dot_size=3, grid_type="polygon", stroke_width=2)
domain_model_ref = globals().get('domain_model')
id9ne_binding_domain = None
if domain_model_ref is not None:
    id9ne_binding_domain = domain_model_ref.get_class_by_name("Counter")
if id9ne_binding_domain:
    id9ne_binding = DataBinding(domain_concept=id9ne_binding_domain)
    id9ne_binding.label_field = next((attr for attr in id9ne_binding_domain.attributes if attr.name == "label"), None)
    id9ne_binding.data_field = next((attr for attr in id9ne_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Counter' not resolved; data binding skipped.
    id9ne_binding = None
if id9ne_binding:
    id9ne.data_binding = id9ne_binding
id9ne_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
id9ne_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
id9ne_styling_color = Color()
id9ne_styling = Styling(size=id9ne_styling_size, position=id9ne_styling_pos, color=id9ne_styling_color)
id9ne.styling = id9ne_styling
id9ne.display_order = 0
id9ne.component_id = "id9ne"
id9ne.component_type = "radar-chart"
id9ne.css_classes = ["radar-chart-component", "has-data-binding"]
id9ne.custom_attributes = {"chart-color": "#8884d8", "chart-title": "Radar Chart Title", "data-source": "a87eb810-2e0a-492b-8cd3-62251566f648", "label-field": "f63380b5-ccf8-448a-bb82-70d62d61442e", "data-field": "2294f9b0-aa04-4c5d-8851-b333f6b3c2fc", "show-grid": "", "show-tooltip": "", "show-radius-axis": "", "id": "id9ne"}
component = ViewContainer(name="Component", description=" component", view_elements={id9ne})
component_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
component_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
component_styling_color = Color()
component_styling = Styling(size=component_styling_size, position=component_styling_pos, color=component_styling_color)
component.styling = component_styling
component.display_order = 2
component.css_classes = ["gjs-cell"]
iiqx6 = ViewContainer(name="iiqx6", description=" component", view_elements={ifczf, iz1hy, component})
iiqx6_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
iiqx6_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iiqx6_styling_color = Color()
iiqx6_styling = Styling(size=iiqx6_styling_size, position=iiqx6_styling_pos, color=iiqx6_styling_color)
iiqx6.styling = iiqx6_styling
iiqx6.display_order = 4
iiqx6.component_id = "iiqx6"
iiqx6.css_classes = ["gjs-row"]
iiqx6.custom_attributes = {"id": "iiqx6"}
iy0gc = BarChart(name="iy0gc", title="Bar Chart Title", primary_color="#3498db", bar_width=30, orientation="vertical", show_grid=False, show_legend=False, show_tooltip=True, stacked=False, animate=True, legend_position="top", grid_color="#e0e0e0", bar_gap=4)
domain_model_ref = globals().get('domain_model')
iy0gc_binding_domain = None
if domain_model_ref is not None:
    iy0gc_binding_domain = domain_model_ref.get_class_by_name("Counter")
if iy0gc_binding_domain:
    iy0gc_binding = DataBinding(domain_concept=iy0gc_binding_domain)
    iy0gc_binding.label_field = next((attr for attr in iy0gc_binding_domain.attributes if attr.name == "label"), None)
    iy0gc_binding.data_field = next((attr for attr in iy0gc_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Counter' not resolved; data binding skipped.
    iy0gc_binding = None
if iy0gc_binding:
    iy0gc.data_binding = iy0gc_binding
iy0gc_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
iy0gc_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iy0gc_styling_color = Color()
iy0gc_styling = Styling(size=iy0gc_styling_size, position=iy0gc_styling_pos, color=iy0gc_styling_color)
iy0gc.styling = iy0gc_styling
iy0gc.display_order = 0
iy0gc.component_id = "iy0gc"
iy0gc.component_type = "bar-chart"
iy0gc.css_classes = ["bar-chart-component", "has-data-binding"]
iy0gc.custom_attributes = {"chart-color": "#3498db", "chart-title": "Bar Chart Title", "data-source": "a87eb810-2e0a-492b-8cd3-62251566f648", "label-field": "f63380b5-ccf8-448a-bb82-70d62d61442e", "data-field": "2294f9b0-aa04-4c5d-8851-b333f6b3c2fc", "bar-width": "30", "orientation": "vertical", "show-grid": "", "show-legend": "", "stacked": False, "id": "iy0gc"}
i2lt6 = ViewContainer(name="i2lt6", description=" component", view_elements={iy0gc})
i2lt6_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
i2lt6_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i2lt6_styling_color = Color()
i2lt6_styling = Styling(size=i2lt6_styling_size, position=i2lt6_styling_pos, color=i2lt6_styling_color)
i2lt6.styling = i2lt6_styling
i2lt6.display_order = 0
i2lt6.component_id = "i2lt6"
i2lt6.css_classes = ["gjs-cell"]
i2lt6.custom_attributes = {"id": "i2lt6"}
ig59l = RadialBarChart(name="ig59l", title="Radial Bar Chart Title", start_angle=90, end_angle=450, inner_radius=30, outer_radius=80, show_legend=True, legend_position="top", show_tooltip=True)
domain_model_ref = globals().get('domain_model')
ig59l_binding_domain = None
if domain_model_ref is not None:
    ig59l_binding_domain = domain_model_ref.get_class_by_name("Counter")
if ig59l_binding_domain:
    ig59l_binding = DataBinding(domain_concept=ig59l_binding_domain)
else:
    # Domain class 'Counter' not resolved; data binding skipped.
    ig59l_binding = None
if ig59l_binding:
    ig59l.data_binding = ig59l_binding
ig59l_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
ig59l_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ig59l_styling_color = Color()
ig59l_styling = Styling(size=ig59l_styling_size, position=ig59l_styling_pos, color=ig59l_styling_color)
ig59l.styling = ig59l_styling
ig59l.display_order = 0
ig59l.component_id = "ig59l"
ig59l.component_type = "radial-bar-chart"
ig59l.css_classes = ["radial-bar-chart-component", "has-data-binding"]
ig59l.custom_attributes = {"chart-title": "Radial Bar Chart Title", "data-source": "a87eb810-2e0a-492b-8cd3-62251566f648", "label-field": "f63380b5-ccf8-448a-bb82-70d62d61442e", "data-field": "2294f9b0-aa04-4c5d-8851-b333f6b3c2fc", "start-angle": "90", "end-angle": "450", "id": "ig59l"}
igud3 = ViewContainer(name="igud3", description=" component", view_elements={ig59l})
igud3_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
igud3_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
igud3_styling_color = Color()
igud3_styling = Styling(size=igud3_styling_size, position=igud3_styling_pos, color=igud3_styling_color)
igud3.styling = igud3_styling
igud3.display_order = 1
igud3.component_id = "igud3"
igud3.css_classes = ["gjs-cell"]
igud3.custom_attributes = {"id": "igud3"}
i4f21 = LineChart(name="i4f21", title="Line Chart Title", primary_color="#4CAF50", line_width=2, show_grid=False, show_legend=False, show_tooltip=False, curve_type="monotone", animate=False, legend_position="top", grid_color="#e0e0e0", dot_size=5)
domain_model_ref = globals().get('domain_model')
i4f21_binding_domain = None
if domain_model_ref is not None:
    i4f21_binding_domain = domain_model_ref.get_class_by_name("Counter")
if i4f21_binding_domain:
    i4f21_binding = DataBinding(domain_concept=i4f21_binding_domain)
    i4f21_binding.label_field = next((attr for attr in i4f21_binding_domain.attributes if attr.name == "label"), None)
    i4f21_binding.data_field = next((attr for attr in i4f21_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Counter' not resolved; data binding skipped.
    i4f21_binding = None
if i4f21_binding:
    i4f21.data_binding = i4f21_binding
i4f21_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
i4f21_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i4f21_styling_color = Color()
i4f21_styling = Styling(size=i4f21_styling_size, position=i4f21_styling_pos, color=i4f21_styling_color)
i4f21.styling = i4f21_styling
i4f21.display_order = 0
i4f21.component_id = "i4f21"
i4f21.component_type = "line-chart"
i4f21.css_classes = ["line-chart-component", "has-data-binding"]
i4f21.custom_attributes = {"chart-color": "#4CAF50", "chart-title": "Line Chart Title", "data-source": "a87eb810-2e0a-492b-8cd3-62251566f648", "label-field": "f63380b5-ccf8-448a-bb82-70d62d61442e", "data-field": "2294f9b0-aa04-4c5d-8851-b333f6b3c2fc", "line-width": "2", "show-grid": "", "show-legend": "", "show-tooltip": "", "curve-type": "monotone", "animate": "", "id": "i4f21"}
component_2 = ViewContainer(name="Component_2", description=" component", view_elements={i4f21})
component_2_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
component_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
component_2_styling_color = Color()
component_2_styling = Styling(size=component_2_styling_size, position=component_2_styling_pos, color=component_2_styling_color)
component_2.styling = component_2_styling
component_2.display_order = 2
component_2.css_classes = ["gjs-cell"]
is6fi = ViewContainer(name="is6fi", description=" component", view_elements={i2lt6, igud3, component_2})
is6fi_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
is6fi_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
is6fi_styling_color = Color()
is6fi_styling = Styling(size=is6fi_styling_size, position=is6fi_styling_pos, color=is6fi_styling_color)
is6fi.styling = is6fi_styling
is6fi.display_order = 5
is6fi.component_id = "is6fi"
is6fi.css_classes = ["gjs-row"]
is6fi.custom_attributes = {"id": "is6fi"}
i3p8x = Text(name="i3p8x", content="About BESSER", description="Text element")
i3p8x_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i3p8x_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i3p8x_styling_color = Color()
i3p8x_styling = Styling(size=i3p8x_styling_size, position=i3p8x_styling_pos, color=i3p8x_styling_color)
i3p8x.styling = i3p8x_styling
i3p8x.display_order = 0
i3p8x.component_id = "i3p8x"
i3p8x.component_type = "text"
i3p8x.tag_name = "h4"
i3p8x.custom_attributes = {"id": "i3p8x"}
i9x6g = Text(name="i9x6g", content="BESSER is a low-code platform for building smarter software faster. Empower your development with our dashboard generator and modeling tools.", description="Text element")
i9x6g_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
i9x6g_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i9x6g_styling_color = Color(opacity="0.8")
i9x6g_styling = Styling(size=i9x6g_styling_size, position=i9x6g_styling_pos, color=i9x6g_styling_color)
i9x6g.styling = i9x6g_styling
i9x6g.display_order = 1
i9x6g.component_id = "i9x6g"
i9x6g.component_type = "text"
i9x6g.tag_name = "p"
i9x6g.custom_attributes = {"id": "i9x6g"}
component_3 = ViewContainer(name="Component_3", description=" component", view_elements={i3p8x, i9x6g})
component_3.display_order = 0
iybmm = Text(name="iybmm", content="Quick Links", description="Text element")
iybmm_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iybmm_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iybmm_styling_color = Color()
iybmm_styling = Styling(size=iybmm_styling_size, position=iybmm_styling_pos, color=iybmm_styling_color)
iybmm.styling = iybmm_styling
iybmm.display_order = 0
iybmm.component_id = "iybmm"
iybmm.component_type = "text"
iybmm.tag_name = "h4"
iybmm.custom_attributes = {"id": "iybmm"}
i0wq2 = Link(name="i0wq2", description="Link element", label="Dashboard Generator", url="#")
i0wq2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i0wq2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i0wq2_styling_color = Color(text_color="white")
i0wq2_styling = Styling(size=i0wq2_styling_size, position=i0wq2_styling_pos, color=i0wq2_styling_color)
i0wq2.styling = i0wq2_styling
i0wq2.display_order = 0
i0wq2.component_id = "i0wq2"
i0wq2.component_type = "link"
i0wq2.tag_name = "a"
i0wq2.custom_attributes = {"href": "#", "id": "i0wq2"}
ichrl = ViewContainer(name="ichrl", description="li container", view_elements={i0wq2})
ichrl_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ichrl_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ichrl_styling_color = Color()
ichrl_styling = Styling(size=ichrl_styling_size, position=ichrl_styling_pos, color=ichrl_styling_color)
ichrl.styling = ichrl_styling
ichrl.display_order = 0
ichrl.component_id = "ichrl"
ichrl.tag_name = "li"
ichrl.custom_attributes = {"id": "ichrl"}
i2f1j = Link(name="i2f1j", description="Link element", label="API Reference", url="#")
i2f1j_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i2f1j_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i2f1j_styling_color = Color(text_color="white")
i2f1j_styling = Styling(size=i2f1j_styling_size, position=i2f1j_styling_pos, color=i2f1j_styling_color)
i2f1j.styling = i2f1j_styling
i2f1j.display_order = 0
i2f1j.component_id = "i2f1j"
i2f1j.component_type = "link"
i2f1j.tag_name = "a"
i2f1j.custom_attributes = {"href": "#", "id": "i2f1j"}
ixtnt = ViewContainer(name="ixtnt", description="li container", view_elements={i2f1j})
ixtnt_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ixtnt_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ixtnt_styling_color = Color()
ixtnt_styling = Styling(size=ixtnt_styling_size, position=ixtnt_styling_pos, color=ixtnt_styling_color)
ixtnt.styling = ixtnt_styling
ixtnt.display_order = 1
ixtnt.component_id = "ixtnt"
ixtnt.tag_name = "li"
ixtnt.custom_attributes = {"id": "ixtnt"}
inu9a = Link(name="inu9a", description="Link element", label="Support", url="#")
inu9a_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
inu9a_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
inu9a_styling_color = Color(text_color="white")
inu9a_styling = Styling(size=inu9a_styling_size, position=inu9a_styling_pos, color=inu9a_styling_color)
inu9a.styling = inu9a_styling
inu9a.display_order = 0
inu9a.component_id = "inu9a"
inu9a.component_type = "link"
inu9a.tag_name = "a"
inu9a.custom_attributes = {"href": "#", "id": "inu9a"}
irxsn = ViewContainer(name="irxsn", description="li container", view_elements={inu9a})
irxsn_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
irxsn_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
irxsn_styling_color = Color()
irxsn_styling = Styling(size=irxsn_styling_size, position=irxsn_styling_pos, color=irxsn_styling_color)
irxsn.styling = irxsn_styling
irxsn.display_order = 2
irxsn.component_id = "irxsn"
irxsn.tag_name = "li"
irxsn.custom_attributes = {"id": "irxsn"}
im0xi = ViewContainer(name="im0xi", description="ul container", view_elements={ichrl, ixtnt, irxsn})
im0xi_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
im0xi_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
im0xi_styling_color = Color(opacity="0.8")
im0xi_styling = Styling(size=im0xi_styling_size, position=im0xi_styling_pos, color=im0xi_styling_color)
im0xi.styling = im0xi_styling
im0xi.display_order = 1
im0xi.component_id = "im0xi"
im0xi.tag_name = "ul"
im0xi.custom_attributes = {"id": "im0xi"}
component_4 = ViewContainer(name="Component_4", description=" component", view_elements={iybmm, im0xi})
component_4.display_order = 1
igd5a = Text(name="igd5a", content="Contact", description="Text element")
igd5a_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
igd5a_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
igd5a_styling_color = Color()
igd5a_styling = Styling(size=igd5a_styling_size, position=igd5a_styling_pos, color=igd5a_styling_color)
igd5a.styling = igd5a_styling
igd5a.display_order = 0
igd5a.component_id = "igd5a"
igd5a.component_type = "text"
igd5a.tag_name = "h4"
igd5a.custom_attributes = {"id": "igd5a"}
itmu2 = Text(name="itmu2", content="Email: info@besser-pearl.org\nPhone: (123) 456-7890", description="Text element")
itmu2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
itmu2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
itmu2_styling_color = Color(opacity="0.8")
itmu2_styling = Styling(size=itmu2_styling_size, position=itmu2_styling_pos, color=itmu2_styling_color)
itmu2.styling = itmu2_styling
itmu2.display_order = 1
itmu2.component_id = "itmu2"
itmu2.component_type = "text"
itmu2.tag_name = "p"
itmu2.custom_attributes = {"id": "itmu2"}
component_5 = ViewContainer(name="Component_5", description=" component", view_elements={igd5a, itmu2})
component_5.display_order = 2
i8uhg = ViewContainer(name="i8uhg", description=" component", view_elements={component_3, component_4, component_5})
i8uhg_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i8uhg_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i8uhg_styling_color = Color()
i8uhg_styling = Styling(size=i8uhg_styling_size, position=i8uhg_styling_pos, color=i8uhg_styling_color)
i8uhg_styling_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="16px")
i8uhg_styling.layout = i8uhg_styling_layout
i8uhg.styling = i8uhg_styling
i8uhg_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="16px")
i8uhg.layout = i8uhg_layout
i8uhg.display_order = 0
i8uhg.component_id = "i8uhg"
i8uhg.custom_attributes = {"id": "i8uhg"}
iph4d = Text(name="iph4d", content="© 2025 BESSER. All rights reserved.", description="Text element")
iph4d_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iph4d_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iph4d_styling_color = Color(opacity="0.7")
iph4d_styling = Styling(size=iph4d_styling_size, position=iph4d_styling_pos, color=iph4d_styling_color)
iph4d.styling = iph4d_styling
iph4d.display_order = 1
iph4d.component_id = "iph4d"
iph4d.component_type = "text"
iph4d.custom_attributes = {"id": "iph4d"}
ictst = ViewContainer(name="ictst", description="footer container", view_elements={i8uhg, iph4d})
ictst_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ictst_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ictst_styling_color = Color(background_color="linear-gradient(135deg, rgb(75, 60, 130) 0%, rgb(90, 61, 145) 100%) !important", text_color="white")
ictst_styling = Styling(size=ictst_styling_size, position=ictst_styling_pos, color=ictst_styling_color)
ictst.styling = ictst_styling
ictst.display_order = 6
ictst.component_id = "ictst"
ictst.tag_name = "footer"
ictst.custom_attributes = {"id": "ictst"}
wrapper.view_elements = {meta, viewport, title, i5jh, iiqx6, is6fi, ictst}


# Screen: wrapper_2
wrapper_2 = Screen(name="wrapper_2", description="About", view_elements=set(), route_path="/about", screen_size="Medium")
wrapper_2.component_id = "r853I1MU7LvrIvAK"
meta_2 = ViewComponent(name="meta_2", description="meta component")
meta_2.display_order = 0
meta_2.tag_name = "meta"
meta_2.custom_attributes = {"charset": "utf-8"}
viewport_2 = ViewComponent(name="viewport_2", description="meta component")
viewport_2.display_order = 1
viewport_2.tag_name = "meta"
viewport_2.custom_attributes = {"name": "viewport", "content": "width=device-width, initial-scale=1"}
meta_3 = ViewComponent(name="meta_3", description="meta component")
meta_3.display_order = 2
meta_3.tag_name = "meta"
meta_3.custom_attributes = {"charset": "utf-8"}
viewport_3 = ViewComponent(name="viewport_3", description="meta component")
viewport_3.display_order = 3
viewport_3.tag_name = "meta"
viewport_3.custom_attributes = {"name": "viewport", "content": "width=device-width, initial-scale=1"}
io07_2 = Text(name="io07_2", content="BESSER", description="Text element")
io07_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="24px", unit_size=UnitSize.PIXELS)
io07_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
io07_2_styling_color = Color()
io07_2_styling = Styling(size=io07_2_styling_size, position=io07_2_styling_pos, color=io07_2_styling_color)
io07_2.styling = io07_2_styling
io07_2.display_order = 0
io07_2.component_id = "io07-2"
io07_2.component_type = "text"
io07_2.custom_attributes = {"id": "io07-2"}
ibtww_2 = Link(name="ibtww_2", description="Link element", label="Home", url="/")
ibtww_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ibtww_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ibtww_2_styling_color = Color(text_color="white")
ibtww_2_styling = Styling(size=ibtww_2_styling_size, position=ibtww_2_styling_pos, color=ibtww_2_styling_color)
ibtww_2.styling = ibtww_2_styling
ibtww_2.display_order = 0
ibtww_2.component_id = "ibtww-2"
ibtww_2.component_type = "link"
ibtww_2.tag_name = "a"
ibtww_2.custom_attributes = {"href": "/", "id": "ibtww-2"}
ild7j_2 = Link(name="ild7j_2", description="Link element", label="About", url="/about")
ild7j_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ild7j_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ild7j_2_styling_color = Color(text_color="white")
ild7j_2_styling = Styling(size=ild7j_2_styling_size, position=ild7j_2_styling_pos, color=ild7j_2_styling_color)
ild7j_2.styling = ild7j_2_styling
ild7j_2.display_order = 1
ild7j_2.component_id = "ild7j-2"
ild7j_2.component_type = "link"
ild7j_2.tag_name = "a"
ild7j_2.custom_attributes = {"href": "/about", "id": "ild7j-2"}
inp2m_2 = ViewContainer(name="inp2m_2", description=" component", view_elements={ibtww_2, ild7j_2})
inp2m_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
inp2m_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
inp2m_2_styling_color = Color()
inp2m_2_styling = Styling(size=inp2m_2_styling_size, position=inp2m_2_styling_pos, color=inp2m_2_styling_color)
inp2m_2.styling = inp2m_2_styling
inp2m_2.display_order = 1
inp2m_2.component_id = "inp2m-2"
inp2m_2.custom_attributes = {"id": "inp2m-2"}
i5jh_2 = ViewContainer(name="i5jh_2", description="nav container", view_elements={io07_2, inp2m_2})
i5jh_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i5jh_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i5jh_2_styling_color = Color(background_color="linear-gradient(135deg, rgb(75, 60, 130) 0%, rgb(90, 61, 145) 100%) !important", text_color="white")
i5jh_2_styling = Styling(size=i5jh_2_styling_size, position=i5jh_2_styling_pos, color=i5jh_2_styling_color)
i5jh_2_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="space-between", align_items="center", gap="16px")
i5jh_2_styling.layout = i5jh_2_styling_layout
i5jh_2.styling = i5jh_2_styling
i5jh_2_layout = Layout(layout_type=LayoutType.FLEX, justify_content="space-between", align_items="center", gap="16px")
i5jh_2.layout = i5jh_2_layout
i5jh_2.display_order = 4
i5jh_2.component_id = "i5jh-2"
i5jh_2.tag_name = "nav"
i5jh_2.custom_attributes = {"id": "i5jh-2"}
iarech = Text(name="iarech", content="Build Smarter Dashboards Faster with BESSER", description="Text element")
iarech_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="42px", unit_size=UnitSize.PIXELS)
iarech_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iarech_styling_color = Color()
iarech_styling = Styling(size=iarech_styling_size, position=iarech_styling_pos, color=iarech_styling_color)
iarech.styling = iarech_styling
iarech.display_order = 0
iarech.component_id = "iarech"
iarech.component_type = "text"
iarech.tag_name = "h2"
iarech.custom_attributes = {"id": "iarech"}
iznkht = Text(name="iznkht", content="Create interactive dashboards effortlessly and streamline your workflow with BESSER's low-code platform.", description="Text element")
iznkht_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="20px", unit_size=UnitSize.PIXELS)
iznkht_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iznkht_styling_color = Color(opacity="0.95")
iznkht_styling = Styling(size=iznkht_styling_size, position=iznkht_styling_pos, color=iznkht_styling_color)
iznkht.styling = iznkht_styling
iznkht.display_order = 1
iznkht.component_id = "iznkht"
iznkht.component_type = "text"
iznkht.tag_name = "p"
iznkht.custom_attributes = {"id": "iznkht"}
i6qufd = Button(name="i6qufd", description="Button component", label="Try Dashboard Generator", buttonType=ButtonType.CustomizableButton, actionType=ButtonActionType.Send)
i6qufd_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="18px", unit_size=UnitSize.PIXELS)
i6qufd_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i6qufd_styling_color = Color(background_color="none", text_color="rgb(75, 60, 130)")
i6qufd_styling = Styling(size=i6qufd_styling_size, position=i6qufd_styling_pos, color=i6qufd_styling_color)
i6qufd.styling = i6qufd_styling
i6qufd.display_order = 0
i6qufd.component_id = "i6qufd"
i6qufd.component_type = "text"
i6qufd.tag_name = "button"
i6qufd.custom_attributes = {"id": "i6qufd"}
ih7y3i = Button(name="ih7y3i", description="Button component", label="Explore Features", buttonType=ButtonType.CustomizableButton, actionType=ButtonActionType.Send)
ih7y3i_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="18px", unit_size=UnitSize.PIXELS)
ih7y3i_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ih7y3i_styling_color = Color(background_color="none", text_color="white")
ih7y3i_styling = Styling(size=ih7y3i_styling_size, position=ih7y3i_styling_pos, color=ih7y3i_styling_color)
ih7y3i.styling = ih7y3i_styling
ih7y3i.display_order = 1
ih7y3i.component_id = "ih7y3i"
ih7y3i.component_type = "text"
ih7y3i.tag_name = "button"
ih7y3i.custom_attributes = {"id": "ih7y3i"}
i45e9h = ViewContainer(name="i45e9h", description=" component", view_elements={i6qufd, ih7y3i})
i45e9h_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i45e9h_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i45e9h_styling_color = Color()
i45e9h_styling = Styling(size=i45e9h_styling_size, position=i45e9h_styling_pos, color=i45e9h_styling_color)
i45e9h_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", flex_wrap="wrap", gap="16px")
i45e9h_styling.layout = i45e9h_styling_layout
i45e9h.styling = i45e9h_styling
i45e9h_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", flex_wrap="wrap", gap="16px")
i45e9h.layout = i45e9h_layout
i45e9h.display_order = 2
i45e9h.component_id = "i45e9h"
i45e9h.custom_attributes = {"id": "i45e9h"}
izn0d4 = ViewContainer(name="izn0d4", description=" component", view_elements={iarech, iznkht, i45e9h})
izn0d4_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
izn0d4_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
izn0d4_styling_color = Color()
izn0d4_styling = Styling(size=izn0d4_styling_size, position=izn0d4_styling_pos, color=izn0d4_styling_color)
izn0d4.styling = izn0d4_styling
izn0d4.display_order = 0
izn0d4.component_id = "izn0d4"
izn0d4.custom_attributes = {"id": "izn0d4"}
ihhucw = ViewContainer(name="ihhucw", description="section container", view_elements={izn0d4})
ihhucw_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ihhucw_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ihhucw_styling_color = Color(background_color="linear-gradient(135deg, rgb(75, 60, 130) 0%, rgb(90, 61, 145) 100%) !important", text_color="white")
ihhucw_styling = Styling(size=ihhucw_styling_size, position=ihhucw_styling_pos, color=ihhucw_styling_color)
ihhucw.styling = ihhucw_styling
ihhucw.display_order = 5
ihhucw.component_id = "ihhucw"
ihhucw.tag_name = "section"
ihhucw.custom_attributes = {"id": "ihhucw"}
ixhdf8 = Text(name="ixhdf8", content="Our Features", description="Text element")
ixhdf8_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="36px", unit_size=UnitSize.PIXELS)
ixhdf8_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ixhdf8_styling_color = Color(text_color="rgb(51, 51, 51)")
ixhdf8_styling = Styling(size=ixhdf8_styling_size, position=ixhdf8_styling_pos, color=ixhdf8_styling_color)
ixhdf8.styling = ixhdf8_styling
ixhdf8.display_order = 0
ixhdf8.component_id = "ixhdf8"
ixhdf8.component_type = "text"
ixhdf8.tag_name = "h2"
ixhdf8.custom_attributes = {"id": "ixhdf8"}
iwznrk = Text(name="iwznrk", content="🚀", description="Text element")
iwznrk_styling_size = Size(width="80px", height="80px", padding="0", margin="0", font_size="36px", unit_size=UnitSize.PIXELS)
iwznrk_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iwznrk_styling_color = Color(background_color="linear-gradient(135deg, rgb(102, 126, 234) 0%, rgb(118, 75, 162) 100%)", text_color="white")
iwznrk_styling = Styling(size=iwznrk_styling_size, position=iwznrk_styling_pos, color=iwznrk_styling_color)
iwznrk_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", align_items="center", gap="16px")
iwznrk_styling.layout = iwznrk_styling_layout
iwznrk.styling = iwznrk_styling
iwznrk.display_order = 0
iwznrk.component_id = "iwznrk"
iwznrk.component_type = "text"
iwznrk.custom_attributes = {"id": "iwznrk"}
it8x1g = Text(name="it8x1g", content="Fast Performance", description="Text element")
it8x1g_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
it8x1g_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
it8x1g_styling_color = Color(text_color="rgb(51, 51, 51)")
it8x1g_styling = Styling(size=it8x1g_styling_size, position=it8x1g_styling_pos, color=it8x1g_styling_color)
it8x1g.styling = it8x1g_styling
it8x1g.display_order = 1
it8x1g.component_id = "it8x1g"
it8x1g.component_type = "text"
it8x1g.tag_name = "h3"
it8x1g.custom_attributes = {"id": "it8x1g"}
iccbeg = Text(name="iccbeg", content="Lightning-fast loading times and smooth interactions for the best user experience.", description="Text element")
iccbeg_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
iccbeg_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iccbeg_styling_color = Color(text_color="rgb(102, 102, 102)")
iccbeg_styling = Styling(size=iccbeg_styling_size, position=iccbeg_styling_pos, color=iccbeg_styling_color)
iccbeg.styling = iccbeg_styling
iccbeg.display_order = 2
iccbeg.component_id = "iccbeg"
iccbeg.component_type = "text"
iccbeg.tag_name = "p"
iccbeg.custom_attributes = {"id": "iccbeg"}
ilfh9s = ViewContainer(name="ilfh9s", description=" component", view_elements={iwznrk, it8x1g, iccbeg})
ilfh9s_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ilfh9s_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ilfh9s_styling_color = Color()
ilfh9s_styling = Styling(size=ilfh9s_styling_size, position=ilfh9s_styling_pos, color=ilfh9s_styling_color)
ilfh9s.styling = ilfh9s_styling
ilfh9s.display_order = 1
ilfh9s.component_id = "ilfh9s"
ilfh9s.custom_attributes = {"id": "ilfh9s"}
i7pg38 = Text(name="i7pg38", content="🔒", description="Text element")
i7pg38_styling_size = Size(width="80px", height="80px", padding="0", margin="0", font_size="36px", unit_size=UnitSize.PIXELS)
i7pg38_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i7pg38_styling_color = Color(background_color="linear-gradient(135deg, rgb(102, 126, 234) 0%, rgb(118, 75, 162) 100%)", text_color="white")
i7pg38_styling = Styling(size=i7pg38_styling_size, position=i7pg38_styling_pos, color=i7pg38_styling_color)
i7pg38_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", align_items="center", gap="16px")
i7pg38_styling.layout = i7pg38_styling_layout
i7pg38.styling = i7pg38_styling
i7pg38.display_order = 0
i7pg38.component_id = "i7pg38"
i7pg38.component_type = "text"
i7pg38.custom_attributes = {"id": "i7pg38"}
it7eyx = Text(name="it7eyx", content="Secure & Safe", description="Text element")
it7eyx_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
it7eyx_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
it7eyx_styling_color = Color(text_color="rgb(51, 51, 51)")
it7eyx_styling = Styling(size=it7eyx_styling_size, position=it7eyx_styling_pos, color=it7eyx_styling_color)
it7eyx.styling = it7eyx_styling
it7eyx.display_order = 1
it7eyx.component_id = "it7eyx"
it7eyx.component_type = "text"
it7eyx.tag_name = "h3"
it7eyx.custom_attributes = {"id": "it7eyx"}
izcgig = Text(name="izcgig", content="Enterprise-grade security to protect your data and ensure privacy.", description="Text element")
izcgig_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
izcgig_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
izcgig_styling_color = Color(text_color="rgb(102, 102, 102)")
izcgig_styling = Styling(size=izcgig_styling_size, position=izcgig_styling_pos, color=izcgig_styling_color)
izcgig.styling = izcgig_styling
izcgig.display_order = 2
izcgig.component_id = "izcgig"
izcgig.component_type = "text"
izcgig.tag_name = "p"
izcgig.custom_attributes = {"id": "izcgig"}
ixjufm = ViewContainer(name="ixjufm", description=" component", view_elements={i7pg38, it7eyx, izcgig})
ixjufm_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ixjufm_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ixjufm_styling_color = Color()
ixjufm_styling = Styling(size=ixjufm_styling_size, position=ixjufm_styling_pos, color=ixjufm_styling_color)
ixjufm.styling = ixjufm_styling
ixjufm.display_order = 3
ixjufm.component_id = "ixjufm"
ixjufm.custom_attributes = {"id": "ixjufm"}
ih9198 = Text(name="ih9198", content="📱", description="Text element")
ih9198_styling_size = Size(width="80px", height="80px", padding="0", margin="0", font_size="36px", unit_size=UnitSize.PIXELS)
ih9198_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ih9198_styling_color = Color(background_color="linear-gradient(135deg, rgb(102, 126, 234) 0%, rgb(118, 75, 162) 100%)", text_color="white")
ih9198_styling = Styling(size=ih9198_styling_size, position=ih9198_styling_pos, color=ih9198_styling_color)
ih9198_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="center", align_items="center", gap="16px")
ih9198_styling.layout = ih9198_styling_layout
ih9198.styling = ih9198_styling
ih9198.display_order = 0
ih9198.component_id = "ih9198"
ih9198.component_type = "text"
ih9198.custom_attributes = {"id": "ih9198"}
ii2qto = Text(name="ii2qto", content="Responsive Design", description="Text element")
ii2qto_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ii2qto_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ii2qto_styling_color = Color(text_color="rgb(51, 51, 51)")
ii2qto_styling = Styling(size=ii2qto_styling_size, position=ii2qto_styling_pos, color=ii2qto_styling_color)
ii2qto.styling = ii2qto_styling
ii2qto.display_order = 1
ii2qto.component_id = "ii2qto"
ii2qto.component_type = "text"
ii2qto.tag_name = "h3"
ii2qto.custom_attributes = {"id": "ii2qto"}
itzqqe = Text(name="itzqqe", content="Works perfectly on all devices - desktop, tablet, and mobile.", description="Text element")
itzqqe_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
itzqqe_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
itzqqe_styling_color = Color(text_color="rgb(102, 102, 102)")
itzqqe_styling = Styling(size=itzqqe_styling_size, position=itzqqe_styling_pos, color=itzqqe_styling_color)
itzqqe.styling = itzqqe_styling
itzqqe.display_order = 2
itzqqe.component_id = "itzqqe"
itzqqe.component_type = "text"
itzqqe.tag_name = "p"
itzqqe.custom_attributes = {"id": "itzqqe"}
ie4pxe = ViewContainer(name="ie4pxe", description=" component", view_elements={ih9198, ii2qto, itzqqe})
ie4pxe_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ie4pxe_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ie4pxe_styling_color = Color()
ie4pxe_styling = Styling(size=ie4pxe_styling_size, position=ie4pxe_styling_pos, color=ie4pxe_styling_color)
ie4pxe.styling = ie4pxe_styling
ie4pxe.display_order = 5
ie4pxe.component_id = "ie4pxe"
ie4pxe.custom_attributes = {"id": "ie4pxe"}
ifzwsk = ViewContainer(name="ifzwsk", description=" component", view_elements={ilfh9s, ixjufm, ie4pxe})
ifzwsk_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ifzwsk_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ifzwsk_styling_color = Color()
ifzwsk_styling = Styling(size=ifzwsk_styling_size, position=ifzwsk_styling_pos, color=ifzwsk_styling_color)
ifzwsk_styling_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(300px, 1fr))", gap="16px")
ifzwsk_styling.layout = ifzwsk_styling_layout
ifzwsk.styling = ifzwsk_styling
ifzwsk_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(300px, 1fr))", gap="16px")
ifzwsk.layout = ifzwsk_layout
ifzwsk.display_order = 1
ifzwsk.component_id = "ifzwsk"
ifzwsk.custom_attributes = {"id": "ifzwsk"}
imjpfb = ViewContainer(name="imjpfb", description=" component", view_elements={ixhdf8, ifzwsk})
imjpfb_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
imjpfb_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
imjpfb_styling_color = Color()
imjpfb_styling = Styling(size=imjpfb_styling_size, position=imjpfb_styling_pos, color=imjpfb_styling_color)
imjpfb.styling = imjpfb_styling
imjpfb.display_order = 0
imjpfb.component_id = "imjpfb"
imjpfb.custom_attributes = {"id": "imjpfb"}
i94ykd = ViewContainer(name="i94ykd", description="section container", view_elements={imjpfb})
i94ykd_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i94ykd_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i94ykd_styling_color = Color()
i94ykd_styling = Styling(size=i94ykd_styling_size, position=i94ykd_styling_pos, color=i94ykd_styling_color)
i94ykd.styling = i94ykd_styling
i94ykd.display_order = 6
i94ykd.component_id = "i94ykd"
i94ykd.tag_name = "section"
i94ykd.custom_attributes = {"id": "i94ykd"}
i3p8x_2 = Text(name="i3p8x_2", content="About BESSER", description="Text element")
i3p8x_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i3p8x_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i3p8x_2_styling_color = Color()
i3p8x_2_styling = Styling(size=i3p8x_2_styling_size, position=i3p8x_2_styling_pos, color=i3p8x_2_styling_color)
i3p8x_2.styling = i3p8x_2_styling
i3p8x_2.display_order = 0
i3p8x_2.component_id = "i3p8x-2"
i3p8x_2.component_type = "text"
i3p8x_2.tag_name = "h4"
i3p8x_2.custom_attributes = {"id": "i3p8x-2"}
i9x6g_2 = Text(name="i9x6g_2", content="BESSER is a low-code platform for building smarter software faster. Empower your development with our dashboard generator and modeling tools.", description="Text element")
i9x6g_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
i9x6g_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i9x6g_2_styling_color = Color(opacity="0.8")
i9x6g_2_styling = Styling(size=i9x6g_2_styling_size, position=i9x6g_2_styling_pos, color=i9x6g_2_styling_color)
i9x6g_2.styling = i9x6g_2_styling
i9x6g_2.display_order = 1
i9x6g_2.component_id = "i9x6g-2"
i9x6g_2.component_type = "text"
i9x6g_2.tag_name = "p"
i9x6g_2.custom_attributes = {"id": "i9x6g-2"}
component_6 = ViewContainer(name="Component_6", description=" component", view_elements={i3p8x_2, i9x6g_2})
component_6.display_order = 0
iybmm_2 = Text(name="iybmm_2", content="Quick Links", description="Text element")
iybmm_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iybmm_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iybmm_2_styling_color = Color()
iybmm_2_styling = Styling(size=iybmm_2_styling_size, position=iybmm_2_styling_pos, color=iybmm_2_styling_color)
iybmm_2.styling = iybmm_2_styling
iybmm_2.display_order = 0
iybmm_2.component_id = "iybmm-2"
iybmm_2.component_type = "text"
iybmm_2.tag_name = "h4"
iybmm_2.custom_attributes = {"id": "iybmm-2"}
i0wq2_2 = Link(name="i0wq2_2", description="Link element", label="Dashboard Generator", url="#")
i0wq2_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i0wq2_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i0wq2_2_styling_color = Color(text_color="white")
i0wq2_2_styling = Styling(size=i0wq2_2_styling_size, position=i0wq2_2_styling_pos, color=i0wq2_2_styling_color)
i0wq2_2.styling = i0wq2_2_styling
i0wq2_2.display_order = 0
i0wq2_2.component_id = "i0wq2-2"
i0wq2_2.component_type = "link"
i0wq2_2.tag_name = "a"
i0wq2_2.custom_attributes = {"href": "#", "id": "i0wq2-2"}
ichrl_2 = ViewContainer(name="ichrl_2", description="li container", view_elements={i0wq2_2})
ichrl_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ichrl_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ichrl_2_styling_color = Color()
ichrl_2_styling = Styling(size=ichrl_2_styling_size, position=ichrl_2_styling_pos, color=ichrl_2_styling_color)
ichrl_2.styling = ichrl_2_styling
ichrl_2.display_order = 0
ichrl_2.component_id = "ichrl-2"
ichrl_2.tag_name = "li"
ichrl_2.custom_attributes = {"id": "ichrl-2"}
i2f1j_2 = Link(name="i2f1j_2", description="Link element", label="API Reference", url="#")
i2f1j_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i2f1j_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i2f1j_2_styling_color = Color(text_color="white")
i2f1j_2_styling = Styling(size=i2f1j_2_styling_size, position=i2f1j_2_styling_pos, color=i2f1j_2_styling_color)
i2f1j_2.styling = i2f1j_2_styling
i2f1j_2.display_order = 0
i2f1j_2.component_id = "i2f1j-2"
i2f1j_2.component_type = "link"
i2f1j_2.tag_name = "a"
i2f1j_2.custom_attributes = {"href": "#", "id": "i2f1j-2"}
ixtnt_2 = ViewContainer(name="ixtnt_2", description="li container", view_elements={i2f1j_2})
ixtnt_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ixtnt_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ixtnt_2_styling_color = Color()
ixtnt_2_styling = Styling(size=ixtnt_2_styling_size, position=ixtnt_2_styling_pos, color=ixtnt_2_styling_color)
ixtnt_2.styling = ixtnt_2_styling
ixtnt_2.display_order = 1
ixtnt_2.component_id = "ixtnt-2"
ixtnt_2.tag_name = "li"
ixtnt_2.custom_attributes = {"id": "ixtnt-2"}
inu9a_2 = Link(name="inu9a_2", description="Link element", label="Support", url="#")
inu9a_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
inu9a_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
inu9a_2_styling_color = Color(text_color="white")
inu9a_2_styling = Styling(size=inu9a_2_styling_size, position=inu9a_2_styling_pos, color=inu9a_2_styling_color)
inu9a_2.styling = inu9a_2_styling
inu9a_2.display_order = 0
inu9a_2.component_id = "inu9a-2"
inu9a_2.component_type = "link"
inu9a_2.tag_name = "a"
inu9a_2.custom_attributes = {"href": "#", "id": "inu9a-2"}
irxsn_2 = ViewContainer(name="irxsn_2", description="li container", view_elements={inu9a_2})
irxsn_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
irxsn_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
irxsn_2_styling_color = Color()
irxsn_2_styling = Styling(size=irxsn_2_styling_size, position=irxsn_2_styling_pos, color=irxsn_2_styling_color)
irxsn_2.styling = irxsn_2_styling
irxsn_2.display_order = 2
irxsn_2.component_id = "irxsn-2"
irxsn_2.tag_name = "li"
irxsn_2.custom_attributes = {"id": "irxsn-2"}
im0xi_2 = ViewContainer(name="im0xi_2", description="ul container", view_elements={ichrl_2, ixtnt_2, irxsn_2})
im0xi_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
im0xi_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
im0xi_2_styling_color = Color(opacity="0.8")
im0xi_2_styling = Styling(size=im0xi_2_styling_size, position=im0xi_2_styling_pos, color=im0xi_2_styling_color)
im0xi_2.styling = im0xi_2_styling
im0xi_2.display_order = 1
im0xi_2.component_id = "im0xi-2"
im0xi_2.tag_name = "ul"
im0xi_2.custom_attributes = {"id": "im0xi-2"}
component_7 = ViewContainer(name="Component_7", description=" component", view_elements={iybmm_2, im0xi_2})
component_7.display_order = 1
igd5a_2 = Text(name="igd5a_2", content="Contact", description="Text element")
igd5a_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
igd5a_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
igd5a_2_styling_color = Color()
igd5a_2_styling = Styling(size=igd5a_2_styling_size, position=igd5a_2_styling_pos, color=igd5a_2_styling_color)
igd5a_2.styling = igd5a_2_styling
igd5a_2.display_order = 0
igd5a_2.component_id = "igd5a-2"
igd5a_2.component_type = "text"
igd5a_2.tag_name = "h4"
igd5a_2.custom_attributes = {"id": "igd5a-2"}
itmu2_2 = Text(name="itmu2_2", content="Email: info@besser-pearl.org\nPhone: (123) 456-7890", description="Text element")
itmu2_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
itmu2_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
itmu2_2_styling_color = Color(opacity="0.8")
itmu2_2_styling = Styling(size=itmu2_2_styling_size, position=itmu2_2_styling_pos, color=itmu2_2_styling_color)
itmu2_2.styling = itmu2_2_styling
itmu2_2.display_order = 1
itmu2_2.component_id = "itmu2-2"
itmu2_2.component_type = "text"
itmu2_2.tag_name = "p"
itmu2_2.custom_attributes = {"id": "itmu2-2"}
iberth = ViewContainer(name="iberth", description=" component", view_elements={igd5a_2, itmu2_2})
iberth.display_order = 2
iberth.component_id = "iberth"
iberth.custom_attributes = {"id": "iberth"}
i8uhg_2 = ViewContainer(name="i8uhg_2", description=" component", view_elements={component_6, component_7, iberth})
i8uhg_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i8uhg_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i8uhg_2_styling_color = Color()
i8uhg_2_styling = Styling(size=i8uhg_2_styling_size, position=i8uhg_2_styling_pos, color=i8uhg_2_styling_color)
i8uhg_2_styling_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="16px")
i8uhg_2_styling.layout = i8uhg_2_styling_layout
i8uhg_2.styling = i8uhg_2_styling
i8uhg_2_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="16px")
i8uhg_2.layout = i8uhg_2_layout
i8uhg_2.display_order = 0
i8uhg_2.component_id = "i8uhg-2"
i8uhg_2.custom_attributes = {"id": "i8uhg-2"}
iph4d_2 = Text(name="iph4d_2", content="© 2025 BESSER. All rights reserved.", description="Text element")
iph4d_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iph4d_2_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iph4d_2_styling_color = Color(opacity="0.7")
iph4d_2_styling = Styling(size=iph4d_2_styling_size, position=iph4d_2_styling_pos, color=iph4d_2_styling_color)
iph4d_2.styling = iph4d_2_styling
iph4d_2.display_order = 1
iph4d_2.component_id = "iph4d-2"
iph4d_2.component_type = "text"
iph4d_2.custom_attributes = {"id": "iph4d-2"}
ictst_2 = ViewContainer(name="ictst_2", description="footer container", view_elements={i8uhg_2, iph4d_2})
ictst_2_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ictst_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ictst_2_styling_color = Color(background_color="linear-gradient(135deg, rgb(75, 60, 130) 0%, rgb(90, 61, 145) 100%) !important", text_color="white")
ictst_2_styling = Styling(size=ictst_2_styling_size, position=ictst_2_styling_pos, color=ictst_2_styling_color)
ictst_2.styling = ictst_2_styling
ictst_2.display_order = 7
ictst_2.component_id = "ictst-2"
ictst_2.tag_name = "footer"
ictst_2.custom_attributes = {"id": "ictst-2"}
iiqx6_2 = ViewComponent(name="iiqx6_2", description=" component")
iiqx6_2_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
iiqx6_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iiqx6_2_styling_color = Color()
iiqx6_2_styling = Styling(size=iiqx6_2_styling_size, position=iiqx6_2_styling_pos, color=iiqx6_2_styling_color)
iiqx6_2.styling = iiqx6_2_styling
iiqx6_2.display_order = 8
iiqx6_2.component_id = "iiqx6-2"
iiqx6_2.css_classes = ["gjs-row"]
iiqx6_2.custom_attributes = {"id": "iiqx6-2"}
is6fi_2 = ViewComponent(name="is6fi_2", description=" component")
is6fi_2_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
is6fi_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
is6fi_2_styling_color = Color()
is6fi_2_styling = Styling(size=is6fi_2_styling_size, position=is6fi_2_styling_pos, color=is6fi_2_styling_color)
is6fi_2.styling = is6fi_2_styling
is6fi_2.display_order = 9
is6fi_2.component_id = "is6fi-2"
is6fi_2.css_classes = ["gjs-row"]
is6fi_2.custom_attributes = {"id": "is6fi-2"}
wrapper_2.view_elements = {meta_2, viewport_2, meta_3, viewport_3, i5jh_2, ihhucw, i94ykd, ictst_2, iiqx6_2, is6fi_2}

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
