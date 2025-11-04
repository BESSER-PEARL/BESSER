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
Count_name: Property = Property(name="name", type=StringType)
Count_value: Property = Property(name="value", type=IntegerType)
Count.attributes={Count_value, Count_name}

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

# Screen: wrapper
wrapper = Screen(name="wrapper", description="Home", view_elements=set(), is_main_page=True, route_path="/home", screen_size="Medium")
wrapper.component_id = "EyJE4gjKEKUGlgx7"
meta = ViewComponent(name="meta", description="meta component")
meta.display_order = 0
meta.tag_name = "meta"
meta.custom_attributes = {"charset": "utf-8"}
viewport = ViewComponent(name="viewport", description="meta component")
viewport.display_order = 1
viewport.tag_name = "meta"
viewport.custom_attributes = {"name": "viewport", "content": "width=device-width, initial-scale=1"}
ijeql = Image(name="ijeql", description="Image component", source="data:image/png;base64,UklGR")
ijeql_styling_size = Size(width="170px", height="39px", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ijeql_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ijeql_styling_color = Color(background_color="#FFFFFF", text_color="black", border_color="#CCCCCC")
ijeql_styling = Styling(size=ijeql_styling_size, position=ijeql_styling_pos, color=ijeql_styling_color)
ijeql.styling = ijeql_styling
ijeql.display_order = 0
ijeql.component_id = "ijeql"
ijeql.component_type = "image"
ijeql.tag_name = "img"
ijeql.custom_attributes = {"id": "ijeql", "src": "data:image/png;base64,UklGR"}
izofz = Link(name="izofz", description="Link element", label="Home", url="#")
izofz_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
izofz_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
izofz_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
izofz_styling = Styling(size=izofz_styling_size, position=izofz_styling_pos, color=izofz_styling_color)
izofz.styling = izofz_styling
izofz.display_order = 0
izofz.component_id = "izofz"
izofz.component_type = "link"
izofz.tag_name = "a"
izofz.custom_attributes = {"href": "#", "id": "izofz"}
iij6i = Link(name="iij6i", description="Link element", label="About", url="about")
iij6i_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iij6i_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iij6i_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
iij6i_styling = Styling(size=iij6i_styling_size, position=iij6i_styling_pos, color=iij6i_styling_color)
iij6i.styling = iij6i_styling
iij6i.display_order = 1
iij6i.component_id = "iij6i"
iij6i.component_type = "link"
iij6i.tag_name = "a"
iij6i.custom_attributes = {"href": "about", "id": "iij6i", "target": False}
i5owi = Link(name="i5owi", description="Link element", label="Services", url="#")
i5owi_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i5owi_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i5owi_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
i5owi_styling = Styling(size=i5owi_styling_size, position=i5owi_styling_pos, color=i5owi_styling_color)
i5owi.styling = i5owi_styling
i5owi.display_order = 2
i5owi.component_id = "i5owi"
i5owi.component_type = "link"
i5owi.tag_name = "a"
i5owi.custom_attributes = {"href": "#", "id": "i5owi"}
ih5tf = Link(name="ih5tf", description="Link element", label="Contact", url="#")
ih5tf_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ih5tf_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ih5tf_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
ih5tf_styling = Styling(size=ih5tf_styling_size, position=ih5tf_styling_pos, color=ih5tf_styling_color)
ih5tf.styling = ih5tf_styling
ih5tf.display_order = 3
ih5tf.component_id = "ih5tf"
ih5tf.component_type = "link"
ih5tf.tag_name = "a"
ih5tf.custom_attributes = {"href": "#", "id": "ih5tf"}
igykc = ViewContainer(name="igykc", description=" component", view_elements={izofz, iij6i, i5owi, ih5tf})
igykc_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
igykc_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
igykc_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
igykc_styling = Styling(size=igykc_styling_size, position=igykc_styling_pos, color=igykc_styling_color)
igykc.styling = igykc_styling
igykc.display_order = 1
igykc.component_id = "igykc"
igykc.custom_attributes = {"id": "igykc"}
i6mjq = ViewContainer(name="i6mjq", description="nav container", view_elements={ijeql, igykc})
i6mjq_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i6mjq_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i6mjq_styling_color = Color(background_color="rgb(202, 202, 202)", text_color="white", border_color="#CCCCCC")
i6mjq_styling = Styling(size=i6mjq_styling_size, position=i6mjq_styling_pos, color=i6mjq_styling_color)
i6mjq_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="space-between", align_items="center", gap="16px")
i6mjq_styling.layout = i6mjq_styling_layout
i6mjq.styling = i6mjq_styling
i6mjq_layout = Layout(layout_type=LayoutType.FLEX, justify_content="space-between", align_items="center", gap="16px")
i6mjq.layout = i6mjq_layout
i6mjq.display_order = 0
i6mjq.component_id = "i6mjq"
i6mjq.tag_name = "nav"
i6mjq.custom_attributes = {"id": "i6mjq"}
component = ViewContainer(name="Component", description=" component", view_elements={i6mjq})
component_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
component_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
component_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
component_styling = Styling(size=component_styling_size, position=component_styling_pos, color=component_styling_color)
component.styling = component_styling
component.display_order = 0
component.css_classes = ["gjs-cell"]
ikdk3 = ViewContainer(name="ikdk3", description=" component", view_elements={component})
ikdk3_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
ikdk3_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ikdk3_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ikdk3_styling = Styling(size=ikdk3_styling_size, position=ikdk3_styling_pos, color=ikdk3_styling_color)
ikdk3.styling = ikdk3_styling
ikdk3.display_order = 2
ikdk3.component_id = "ikdk3"
ikdk3.css_classes = ["gjs-row"]
ikdk3.custom_attributes = {"id": "ikdk3"}
ixia = LineChart(name="ixia", title="Sales Over Time", primary_color="#af4c4c", line_width=2, show_grid=False, show_legend=False, show_tooltip=False, curve_type="monotone", animate=False, legend_position="top", grid_color="#e0e0e0", dot_size=5)
domain_model_ref = globals().get('domain_model')
ixia_binding_domain = None
if domain_model_ref is not None:
    ixia_binding_domain = domain_model_ref.get_class_by_name("Count")
if ixia_binding_domain:
    ixia_binding = DataBinding(domain_concept=ixia_binding_domain)
    ixia_binding.label_field = next((attr for attr in ixia_binding_domain.attributes if attr.name == "name"), None)
    ixia_binding.data_field = next((attr for attr in ixia_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Count' not resolved; data binding skipped.
    ixia_binding = None
if ixia_binding:
    ixia.data_binding = ixia_binding
ixia_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
ixia_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ixia_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ixia_styling = Styling(size=ixia_styling_size, position=ixia_styling_pos, color=ixia_styling_color)
ixia.styling = ixia_styling
ixia.display_order = 0
ixia.component_id = "ixia"
ixia.component_type = "line-chart"
ixia.css_classes = ["line-chart-component", "has-data-binding"]
ixia.custom_attributes = {"chart-color": "#af4c4c", "chart-title": "Sales Over Time", "data-source": "7cb2a7d3-e224-4b6d-8134-5bf644d8bd31", "label-field": "49061bdb-b617-41ad-b926-dfdc568c2501", "data-field": "ff634794-9146-4be9-aa8c-9e54e4595208", "line-width": "2", "show-grid": "", "show-legend": "", "show-tooltip": "", "curve-type": "monotone", "animate": "", "id": "ixia"}
component_2 = ViewContainer(name="Component_2", description=" component", view_elements={ixia})
component_2_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
component_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
component_2_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
component_2_styling = Styling(size=component_2_styling_size, position=component_2_styling_pos, color=component_2_styling_color)
component_2.styling = component_2_styling
component_2.display_order = 0
component_2.css_classes = ["gjs-cell"]
isok = BarChart(name="isok", title="Revenue by Category", primary_color="#3498db", bar_width=30, orientation="vertical", show_grid=False, show_legend=False, show_tooltip=True, stacked=False, animate=True, legend_position="top", grid_color="#e0e0e0", bar_gap=4)
domain_model_ref = globals().get('domain_model')
isok_binding_domain = None
if domain_model_ref is not None:
    isok_binding_domain = domain_model_ref.get_class_by_name("Count")
if isok_binding_domain:
    isok_binding = DataBinding(domain_concept=isok_binding_domain)
    isok_binding.label_field = next((attr for attr in isok_binding_domain.attributes if attr.name == "name"), None)
    isok_binding.data_field = next((attr for attr in isok_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Count' not resolved; data binding skipped.
    isok_binding = None
if isok_binding:
    isok.data_binding = isok_binding
isok_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
isok_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
isok_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
isok_styling = Styling(size=isok_styling_size, position=isok_styling_pos, color=isok_styling_color)
isok.styling = isok_styling
isok.display_order = 0
isok.component_id = "isok"
isok.component_type = "bar-chart"
isok.css_classes = ["bar-chart-component", "has-data-binding"]
isok.custom_attributes = {"chart-color": "#3498db", "chart-title": "Revenue by Category", "data-source": "7cb2a7d3-e224-4b6d-8134-5bf644d8bd31", "label-field": "49061bdb-b617-41ad-b926-dfdc568c2501", "data-field": "ff634794-9146-4be9-aa8c-9e54e4595208", "bar-width": "30", "orientation": "vertical", "show-grid": "", "show-legend": "", "stacked": False, "id": "isok"}
component_3 = ViewContainer(name="Component_3", description=" component", view_elements={isok})
component_3_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
component_3_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
component_3_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
component_3_styling = Styling(size=component_3_styling_size, position=component_3_styling_pos, color=component_3_styling_color)
component_3.styling = component_3_styling
component_3.display_order = 1
component_3.css_classes = ["gjs-cell"]
iqd8w = RadarChart(name="iqd8w", title="Performance Metrics", primary_color="#8884d8", show_grid=False, show_tooltip=False, show_radius_axis=False, show_legend=True, legend_position="top", dot_size=3, grid_type="polygon", stroke_width=2)
domain_model_ref = globals().get('domain_model')
iqd8w_binding_domain = None
if domain_model_ref is not None:
    iqd8w_binding_domain = domain_model_ref.get_class_by_name("Count")
if iqd8w_binding_domain:
    iqd8w_binding = DataBinding(domain_concept=iqd8w_binding_domain)
    iqd8w_binding.label_field = next((attr for attr in iqd8w_binding_domain.attributes if attr.name == "name"), None)
    iqd8w_binding.data_field = next((attr for attr in iqd8w_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Count' not resolved; data binding skipped.
    iqd8w_binding = None
if iqd8w_binding:
    iqd8w.data_binding = iqd8w_binding
iqd8w_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
iqd8w_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iqd8w_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
iqd8w_styling = Styling(size=iqd8w_styling_size, position=iqd8w_styling_pos, color=iqd8w_styling_color)
iqd8w.styling = iqd8w_styling
iqd8w.display_order = 0
iqd8w.component_id = "iqd8w"
iqd8w.component_type = "radar-chart"
iqd8w.css_classes = ["radar-chart-component", "has-data-binding"]
iqd8w.custom_attributes = {"chart-color": "#8884d8", "chart-title": "Performance Metrics", "data-source": "7cb2a7d3-e224-4b6d-8134-5bf644d8bd31", "label-field": "49061bdb-b617-41ad-b926-dfdc568c2501", "data-field": "ff634794-9146-4be9-aa8c-9e54e4595208", "show-grid": "", "show-tooltip": "", "show-radius-axis": "", "id": "iqd8w"}
component_4 = ViewContainer(name="Component_4", description=" component", view_elements={iqd8w})
component_4_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
component_4_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
component_4_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
component_4_styling = Styling(size=component_4_styling_size, position=component_4_styling_pos, color=component_4_styling_color)
component_4.styling = component_4_styling
component_4.display_order = 2
component_4.css_classes = ["gjs-cell"]
ile3 = ViewContainer(name="ile3", description=" component", view_elements={component_2, component_3, component_4})
ile3_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
ile3_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ile3_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ile3_styling = Styling(size=ile3_styling_size, position=ile3_styling_pos, color=ile3_styling_color)
ile3.styling = ile3_styling
ile3.display_order = 3
ile3.component_id = "ile3"
ile3.css_classes = ["gjs-row"]
ile3.custom_attributes = {"id": "ile3"}
ifhxo = LineChart(name="ifhxo", title="Sales Over Time", primary_color="#4CAF50", line_width=2, show_grid=False, show_legend=False, show_tooltip=False, curve_type="monotone", animate=False, legend_position="top", grid_color="#e0e0e0", dot_size=5)
domain_model_ref = globals().get('domain_model')
ifhxo_binding_domain = None
if domain_model_ref is not None:
    ifhxo_binding_domain = domain_model_ref.get_class_by_name("Count")
if ifhxo_binding_domain:
    ifhxo_binding = DataBinding(domain_concept=ifhxo_binding_domain)
    ifhxo_binding.label_field = next((attr for attr in ifhxo_binding_domain.attributes if attr.name == "name"), None)
    ifhxo_binding.data_field = next((attr for attr in ifhxo_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Count' not resolved; data binding skipped.
    ifhxo_binding = None
if ifhxo_binding:
    ifhxo.data_binding = ifhxo_binding
ifhxo_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
ifhxo_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ifhxo_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ifhxo_styling = Styling(size=ifhxo_styling_size, position=ifhxo_styling_pos, color=ifhxo_styling_color)
ifhxo.styling = ifhxo_styling
ifhxo.display_order = 4
ifhxo.component_id = "ifhxo"
ifhxo.component_type = "line-chart"
ifhxo.css_classes = ["line-chart-component", "has-data-binding"]
ifhxo.custom_attributes = {"chart-color": "#4CAF50", "chart-title": "Sales Over Time", "data-source": "7cb2a7d3-e224-4b6d-8134-5bf644d8bd31", "label-field": "49061bdb-b617-41ad-b926-dfdc568c2501", "data-field": "ff634794-9146-4be9-aa8c-9e54e4595208", "line-width": "2", "show-grid": "", "show-legend": "", "show-tooltip": "", "curve-type": "monotone", "animate": "", "id": "ifhxo"}
iuhu4 = Text(name="iuhu4", content="Quick Links", description="Text element")
iuhu4_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iuhu4_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iuhu4_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
iuhu4_styling = Styling(size=iuhu4_styling_size, position=iuhu4_styling_pos, color=iuhu4_styling_color)
iuhu4.styling = iuhu4_styling
iuhu4.display_order = 5
iuhu4.component_id = "iuhu4"
iuhu4.component_type = "text"
iuhu4.tag_name = "h4"
iuhu4.custom_attributes = {"id": "iuhu4"}
imwa5 = Text(name="imwa5", content="About Us", description="Text element")
imwa5_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
imwa5_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
imwa5_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
imwa5_styling = Styling(size=imwa5_styling_size, position=imwa5_styling_pos, color=imwa5_styling_color)
imwa5.styling = imwa5_styling
imwa5.display_order = 0
imwa5.component_id = "imwa5"
imwa5.component_type = "text"
imwa5.tag_name = "h4"
imwa5.custom_attributes = {"id": "imwa5"}
ib3zn = Text(name="ib3zn", content="Your company description goes here.", description="Text element")
ib3zn_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
ib3zn_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ib3zn_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.8")
ib3zn_styling = Styling(size=ib3zn_styling_size, position=ib3zn_styling_pos, color=ib3zn_styling_color)
ib3zn.styling = ib3zn_styling
ib3zn.display_order = 1
ib3zn.component_id = "ib3zn"
ib3zn.component_type = "text"
ib3zn.tag_name = "p"
ib3zn.custom_attributes = {"id": "ib3zn"}
component_5 = ViewContainer(name="Component_5", description=" component", view_elements={imwa5, ib3zn})
component_5.display_order = 0
i1zaf = Link(name="i1zaf", description="Link element", label="Home", url="#")
i1zaf_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i1zaf_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i1zaf_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
i1zaf_styling = Styling(size=i1zaf_styling_size, position=i1zaf_styling_pos, color=i1zaf_styling_color)
i1zaf.styling = i1zaf_styling
i1zaf.display_order = 0
i1zaf.component_id = "i1zaf"
i1zaf.component_type = "link"
i1zaf.tag_name = "a"
i1zaf.custom_attributes = {"href": "#", "id": "i1zaf"}
i58kt = ViewContainer(name="i58kt", description="li container", view_elements={i1zaf})
i58kt_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i58kt_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i58kt_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i58kt_styling = Styling(size=i58kt_styling_size, position=i58kt_styling_pos, color=i58kt_styling_color)
i58kt.styling = i58kt_styling
i58kt.display_order = 0
i58kt.component_id = "i58kt"
i58kt.tag_name = "li"
i58kt.custom_attributes = {"id": "i58kt"}
iuw1f = Link(name="iuw1f", description="Link element", label="Services", url="#")
iuw1f_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iuw1f_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iuw1f_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
iuw1f_styling = Styling(size=iuw1f_styling_size, position=iuw1f_styling_pos, color=iuw1f_styling_color)
iuw1f.styling = iuw1f_styling
iuw1f.display_order = 0
iuw1f.component_id = "iuw1f"
iuw1f.component_type = "link"
iuw1f.tag_name = "a"
iuw1f.custom_attributes = {"href": "#", "id": "iuw1f"}
i2hhl = ViewContainer(name="i2hhl", description="li container", view_elements={iuw1f})
i2hhl_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i2hhl_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i2hhl_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i2hhl_styling = Styling(size=i2hhl_styling_size, position=i2hhl_styling_pos, color=i2hhl_styling_color)
i2hhl.styling = i2hhl_styling
i2hhl.display_order = 1
i2hhl.component_id = "i2hhl"
i2hhl.tag_name = "li"
i2hhl.custom_attributes = {"id": "i2hhl"}
idqrg = Link(name="idqrg", description="Link element", label="Contact", url="#")
idqrg_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
idqrg_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
idqrg_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
idqrg_styling = Styling(size=idqrg_styling_size, position=idqrg_styling_pos, color=idqrg_styling_color)
idqrg.styling = idqrg_styling
idqrg.display_order = 0
idqrg.component_id = "idqrg"
idqrg.component_type = "link"
idqrg.tag_name = "a"
idqrg.custom_attributes = {"href": "#", "id": "idqrg"}
iz9b1 = ViewContainer(name="iz9b1", description="li container", view_elements={idqrg})
iz9b1_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iz9b1_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iz9b1_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
iz9b1_styling = Styling(size=iz9b1_styling_size, position=iz9b1_styling_pos, color=iz9b1_styling_color)
iz9b1.styling = iz9b1_styling
iz9b1.display_order = 2
iz9b1.component_id = "iz9b1"
iz9b1.tag_name = "li"
iz9b1.custom_attributes = {"id": "iz9b1"}
isw3k = ViewContainer(name="isw3k", description="ul container", view_elements={i58kt, i2hhl, iz9b1})
isw3k_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
isw3k_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
isw3k_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.8")
isw3k_styling = Styling(size=isw3k_styling_size, position=isw3k_styling_pos, color=isw3k_styling_color)
isw3k.styling = isw3k_styling
isw3k.display_order = 0
isw3k.component_id = "isw3k"
isw3k.tag_name = "ul"
isw3k.custom_attributes = {"id": "isw3k"}
component_6 = ViewContainer(name="Component_6", description=" component", view_elements={isw3k})
component_6.display_order = 1
i0cm5 = Text(name="i0cm5", content="Contact", description="Text element")
i0cm5_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i0cm5_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i0cm5_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i0cm5_styling = Styling(size=i0cm5_styling_size, position=i0cm5_styling_pos, color=i0cm5_styling_color)
i0cm5.styling = i0cm5_styling
i0cm5.display_order = 0
i0cm5.component_id = "i0cm5"
i0cm5.component_type = "text"
i0cm5.tag_name = "h4"
i0cm5.custom_attributes = {"id": "i0cm5"}
ibh5f = Text(name="ibh5f", content="Email: info@example.com\nPhone: (123) 456-7890", description="Text element")
ibh5f_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ibh5f_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ibh5f_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.8")
ibh5f_styling = Styling(size=ibh5f_styling_size, position=ibh5f_styling_pos, color=ibh5f_styling_color)
ibh5f.styling = ibh5f_styling
ibh5f.display_order = 1
ibh5f.component_id = "ibh5f"
ibh5f.component_type = "text"
ibh5f.tag_name = "p"
ibh5f.custom_attributes = {"id": "ibh5f"}
component_7 = ViewContainer(name="Component_7", description=" component", view_elements={i0cm5, ibh5f})
component_7.display_order = 2
irlm1 = ViewContainer(name="irlm1", description=" component", view_elements={component_5, component_6, component_7})
irlm1_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
irlm1_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
irlm1_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
irlm1_styling = Styling(size=irlm1_styling_size, position=irlm1_styling_pos, color=irlm1_styling_color)
irlm1_styling_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="16px")
irlm1_styling.layout = irlm1_styling_layout
irlm1.styling = irlm1_styling
irlm1_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="16px")
irlm1.layout = irlm1_layout
irlm1.display_order = 0
irlm1.component_id = "irlm1"
irlm1.custom_attributes = {"id": "irlm1"}
ilczb = Text(name="ilczb", content="© 2025 Your Company. All rights reserved.", description="Text element")
ilczb_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ilczb_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ilczb_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC", opacity="0.7")
ilczb_styling = Styling(size=ilczb_styling_size, position=ilczb_styling_pos, color=ilczb_styling_color)
ilczb.styling = ilczb_styling
ilczb.display_order = 1
ilczb.component_id = "ilczb"
ilczb.component_type = "text"
ilczb.custom_attributes = {"id": "ilczb"}
i9hfm = ViewContainer(name="i9hfm", description="footer container", view_elements={irlm1, ilczb})
i9hfm_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i9hfm_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i9hfm_styling_color = Color(background_color="rgb(44, 62, 80)", text_color="white", border_color="#CCCCCC")
i9hfm_styling = Styling(size=i9hfm_styling_size, position=i9hfm_styling_pos, color=i9hfm_styling_color)
i9hfm.styling = i9hfm_styling
i9hfm.display_order = 6
i9hfm.component_id = "i9hfm"
i9hfm.tag_name = "footer"
i9hfm.custom_attributes = {"id": "i9hfm"}
wrapper.view_elements = {meta, viewport, ikdk3, ile3, ifhxo, iuhu4, i9hfm}


# Screen: wrapper_2
wrapper_2 = Screen(name="wrapper_2", description="About", view_elements=set(), route_path="/about", screen_size="Medium")
wrapper_2.component_id = "K65Ks4cuXQO4Aij06"
ip9iy = Text(name="ip9iy", content="Logo", description="Text element")
ip9iy_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="24px", unit_size=UnitSize.PIXELS)
ip9iy_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ip9iy_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ip9iy_styling = Styling(size=ip9iy_styling_size, position=ip9iy_styling_pos, color=ip9iy_styling_color)
ip9iy.styling = ip9iy_styling
ip9iy.display_order = 0
ip9iy.component_id = "ip9iy"
ip9iy.component_type = "text"
ip9iy.custom_attributes = {"id": "ip9iy"}
ieq84h = Link(name="ieq84h", description="Link element", label="Home", url="#")
ieq84h_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ieq84h_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ieq84h_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
ieq84h_styling = Styling(size=ieq84h_styling_size, position=ieq84h_styling_pos, color=ieq84h_styling_color)
ieq84h.styling = ieq84h_styling
ieq84h.display_order = 0
ieq84h.component_id = "ieq84h"
ieq84h.component_type = "link"
ieq84h.tag_name = "a"
ieq84h.custom_attributes = {"href": "#", "id": "ieq84h"}
ixkwwg = Link(name="ixkwwg", description="Link element", label="About", url="#")
ixkwwg_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ixkwwg_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ixkwwg_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
ixkwwg_styling = Styling(size=ixkwwg_styling_size, position=ixkwwg_styling_pos, color=ixkwwg_styling_color)
ixkwwg.styling = ixkwwg_styling
ixkwwg.display_order = 1
ixkwwg.component_id = "ixkwwg"
ixkwwg.component_type = "link"
ixkwwg.tag_name = "a"
ixkwwg.custom_attributes = {"href": "#", "id": "ixkwwg"}
iuol0l = Link(name="iuol0l", description="Link element", label="Services", url="/")
iuol0l_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iuol0l_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iuol0l_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
iuol0l_styling = Styling(size=iuol0l_styling_size, position=iuol0l_styling_pos, color=iuol0l_styling_color)
iuol0l.styling = iuol0l_styling
iuol0l.display_order = 2
iuol0l.component_id = "iuol0l"
iuol0l.component_type = "link"
iuol0l.tag_name = "a"
iuol0l.custom_attributes = {"href": "/", "id": "iuol0l"}
is4p1s = Link(name="is4p1s", description="Link element", label="Contact", url="#")
is4p1s_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
is4p1s_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
is4p1s_styling_color = Color(background_color="#FFFFFF", text_color="white", border_color="#CCCCCC")
is4p1s_styling = Styling(size=is4p1s_styling_size, position=is4p1s_styling_pos, color=is4p1s_styling_color)
is4p1s.styling = is4p1s_styling
is4p1s.display_order = 3
is4p1s.component_id = "is4p1s"
is4p1s.component_type = "link"
is4p1s.tag_name = "a"
is4p1s.custom_attributes = {"href": "#", "id": "is4p1s"}
igl9k = ViewContainer(name="igl9k", description=" component", view_elements={ieq84h, ixkwwg, iuol0l, is4p1s})
igl9k_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
igl9k_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
igl9k_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
igl9k_styling = Styling(size=igl9k_styling_size, position=igl9k_styling_pos, color=igl9k_styling_color)
igl9k_styling_layout = Layout(layout_type=LayoutType.FLEX, gap="30px")
igl9k_styling.layout = igl9k_styling_layout
igl9k.styling = igl9k_styling
igl9k_layout = Layout(layout_type=LayoutType.FLEX, gap="30px")
igl9k.layout = igl9k_layout
igl9k.display_order = 1
igl9k.component_id = "igl9k"
igl9k.custom_attributes = {"id": "igl9k"}
iogy7 = ViewContainer(name="iogy7", description="nav container", view_elements={ip9iy, igl9k})
iogy7_styling_size = Size(width="auto", height="auto", padding="15px 30px", margin="0", unit_size=UnitSize.PIXELS)
iogy7_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iogy7_styling_color = Color(background_color="#333", text_color="white", border_color="#CCCCCC")
iogy7_styling = Styling(size=iogy7_styling_size, position=iogy7_styling_pos, color=iogy7_styling_color)
iogy7_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="space-between", align_items="center", gap="16px")
iogy7_styling.layout = iogy7_styling_layout
iogy7.styling = iogy7_styling
iogy7_layout = Layout(layout_type=LayoutType.FLEX, justify_content="space-between", align_items="center", gap="16px")
iogy7.layout = iogy7_layout
iogy7.display_order = 0
iogy7.component_id = "iogy7"
iogy7.tag_name = "nav"
iogy7.custom_attributes = {"id": "iogy7"}
ingmp = Text(name="ingmp", content="About Us", description="Text element")
ingmp_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="48px", unit_size=UnitSize.PIXELS)
ingmp_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ingmp_styling_color = Color(background_color="#FFFFFF", text_color="#2c3e50", border_color="#CCCCCC")
ingmp_styling = Styling(size=ingmp_styling_size, position=ingmp_styling_pos, color=ingmp_styling_color)
ingmp.styling = ingmp_styling
ingmp.display_order = 0
ingmp.component_id = "ingmp"
ingmp.component_type = "text"
ingmp.tag_name = "h1"
ingmp.custom_attributes = {"style": "color:#2c3e50;font-size:48px;margin-bottom:20px;", "id": "ingmp"}
iyypp = Text(name="iyypp", content="This is the About page. You can edit this content and add your own components.", description="Text element")
iyypp_styling_size = Size(width="auto", height="auto", padding="0", margin="0 auto 40px", font_size="20px", unit_size=UnitSize.PIXELS)
iyypp_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iyypp_styling_color = Color(background_color="#FFFFFF", text_color="#34495e", border_color="#CCCCCC")
iyypp_styling = Styling(size=iyypp_styling_size, position=iyypp_styling_pos, color=iyypp_styling_color)
iyypp.styling = iyypp_styling
iyypp.display_order = 1
iyypp.component_id = "iyypp"
iyypp.component_type = "text"
iyypp.tag_name = "p"
iyypp.custom_attributes = {"style": "color:#34495e;font-size:20px;max-width:800px;margin:0 auto 40px;", "id": "iyypp"}
i3zwt = Text(name="i3zwt", content="Mission", description="Text element")
i3zwt_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i3zwt_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i3zwt_styling_color = Color(background_color="#FFFFFF", text_color="#3498db", border_color="#CCCCCC")
i3zwt_styling = Styling(size=i3zwt_styling_size, position=i3zwt_styling_pos, color=i3zwt_styling_color)
i3zwt.styling = i3zwt_styling
i3zwt.display_order = 0
i3zwt.component_id = "i3zwt"
i3zwt.component_type = "text"
i3zwt.tag_name = "h3"
i3zwt.custom_attributes = {"style": "color:#3498db;margin-bottom:15px;", "id": "i3zwt"}
irehi = Text(name="irehi", content="Our mission is to empower creators", description="Text element")
irehi_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
irehi_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
irehi_styling_color = Color(background_color="#FFFFFF", text_color="#555", border_color="#CCCCCC")
irehi_styling = Styling(size=irehi_styling_size, position=irehi_styling_pos, color=irehi_styling_color)
irehi.styling = irehi_styling
irehi.display_order = 1
irehi.component_id = "irehi"
irehi.component_type = "text"
irehi.tag_name = "p"
irehi.custom_attributes = {"style": "color:#555;", "id": "irehi"}
ivt8q = ViewContainer(name="ivt8q", description=" component", view_elements={i3zwt, irehi})
ivt8q_styling_size = Size(width="auto", height="auto", padding="30px", margin="0", unit_size=UnitSize.PIXELS)
ivt8q_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ivt8q_styling_color = Color(background_color="white", text_color="#000000", border_color="#CCCCCC")
ivt8q_styling = Styling(size=ivt8q_styling_size, position=ivt8q_styling_pos, color=ivt8q_styling_color)
ivt8q.styling = ivt8q_styling
ivt8q.display_order = 0
ivt8q.component_id = "ivt8q"
ivt8q.custom_attributes = {"style": "padding:30px;background:white;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,0.1);", "id": "ivt8q"}
ifv18 = Text(name="ifv18", content="Vision", description="Text element")
ifv18_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ifv18_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ifv18_styling_color = Color(background_color="#FFFFFF", text_color="#e74c3c", border_color="#CCCCCC")
ifv18_styling = Styling(size=ifv18_styling_size, position=ifv18_styling_pos, color=ifv18_styling_color)
ifv18.styling = ifv18_styling
ifv18.display_order = 0
ifv18.component_id = "ifv18"
ifv18.component_type = "text"
ifv18.tag_name = "h3"
ifv18.custom_attributes = {"style": "color:#e74c3c;margin-bottom:15px;", "id": "ifv18"}
i507r = Text(name="i507r", content="Building the future of web design", description="Text element")
i507r_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i507r_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i507r_styling_color = Color(background_color="#FFFFFF", text_color="#555", border_color="#CCCCCC")
i507r_styling = Styling(size=i507r_styling_size, position=i507r_styling_pos, color=i507r_styling_color)
i507r.styling = i507r_styling
i507r.display_order = 1
i507r.component_id = "i507r"
i507r.component_type = "text"
i507r.tag_name = "p"
i507r.custom_attributes = {"style": "color:#555;", "id": "i507r"}
iruch = ViewContainer(name="iruch", description=" component", view_elements={ifv18, i507r})
iruch_styling_size = Size(width="auto", height="auto", padding="30px", margin="0", unit_size=UnitSize.PIXELS)
iruch_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iruch_styling_color = Color(background_color="white", text_color="#000000", border_color="#CCCCCC")
iruch_styling = Styling(size=iruch_styling_size, position=iruch_styling_pos, color=iruch_styling_color)
iruch.styling = iruch_styling
iruch.display_order = 1
iruch.component_id = "iruch"
iruch.custom_attributes = {"style": "padding:30px;background:white;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,0.1);", "id": "iruch"}
i6p9g = Text(name="i6p9g", content="Values", description="Text element")
i6p9g_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i6p9g_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i6p9g_styling_color = Color(background_color="#FFFFFF", text_color="#2ecc71", border_color="#CCCCCC")
i6p9g_styling = Styling(size=i6p9g_styling_size, position=i6p9g_styling_pos, color=i6p9g_styling_color)
i6p9g.styling = i6p9g_styling
i6p9g.display_order = 0
i6p9g.component_id = "i6p9g"
i6p9g.component_type = "text"
i6p9g.tag_name = "h3"
i6p9g.custom_attributes = {"style": "color:#2ecc71;margin-bottom:15px;", "id": "i6p9g"}
ixsdb = Text(name="ixsdb", content="Innovation, quality, and user experience", description="Text element")
ixsdb_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ixsdb_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ixsdb_styling_color = Color(background_color="#FFFFFF", text_color="#555", border_color="#CCCCCC")
ixsdb_styling = Styling(size=ixsdb_styling_size, position=ixsdb_styling_pos, color=ixsdb_styling_color)
ixsdb.styling = ixsdb_styling
ixsdb.display_order = 1
ixsdb.component_id = "ixsdb"
ixsdb.component_type = "text"
ixsdb.tag_name = "p"
ixsdb.custom_attributes = {"style": "color:#555;", "id": "ixsdb"}
iqsia = ViewContainer(name="iqsia", description=" component", view_elements={i6p9g, ixsdb})
iqsia_styling_size = Size(width="auto", height="auto", padding="30px", margin="0", unit_size=UnitSize.PIXELS)
iqsia_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iqsia_styling_color = Color(background_color="white", text_color="#000000", border_color="#CCCCCC")
iqsia_styling = Styling(size=iqsia_styling_size, position=iqsia_styling_pos, color=iqsia_styling_color)
iqsia.styling = iqsia_styling
iqsia.display_order = 2
iqsia.component_id = "iqsia"
iqsia.custom_attributes = {"style": "padding:30px;background:white;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,0.1);", "id": "iqsia"}
i8esc = ViewContainer(name="i8esc", description=" component", view_elements={ivt8q, iruch, iqsia})
i8esc_styling_size = Size(width="auto", height="auto", padding="0", margin="0 auto", unit_size=UnitSize.PIXELS)
i8esc_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i8esc_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i8esc_styling = Styling(size=i8esc_styling_size, position=i8esc_styling_pos, color=i8esc_styling_color)
i8esc_styling_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="30px")
i8esc_styling.layout = i8esc_styling_layout
i8esc.styling = i8esc_styling
i8esc_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="30px")
i8esc.layout = i8esc_layout
i8esc.display_order = 2
i8esc.component_id = "i8esc"
i8esc.custom_attributes = {"style": "display:grid;grid-template-columns:repeat(auto-fit, minmax(250px, 1fr));gap:30px;max-width:1200px;margin:0 auto;", "id": "i8esc"}
ih61q = ViewContainer(name="ih61q", description=" component", view_elements={ingmp, iyypp, i8esc})
ih61q_styling_size = Size(width="auto", height="100vh", padding="80px 40px", margin="0", unit_size=UnitSize.PIXELS)
ih61q_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ih61q_styling_color = Color(background_color="#ecf0f1", text_color="#000000", border_color="#CCCCCC")
ih61q_styling = Styling(size=ih61q_styling_size, position=ih61q_styling_pos, color=ih61q_styling_color)
ih61q.styling = ih61q_styling
ih61q.display_order = 1
ih61q.component_id = "ih61q"
ih61q.custom_attributes = {"style": "padding:80px 40px;text-align:center;font-family:Arial, sans-serif;background:#ecf0f1;min-height:100vh;", "id": "ih61q"}
wrapper_2.view_elements = {iogy7, ih61q}


# Screen: wrapper_3
wrapper_3 = Screen(name="wrapper_3", description="Contact", view_elements=set(), route_path="/contact", screen_size="Medium")
wrapper_3.component_id = "Ps3wnlsWV7RKBN3QUf"
h1 = Text(name="h1", content="Contact Us", description="Text element")
h1_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="48px", unit_size=UnitSize.PIXELS)
h1_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
h1_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
h1_styling = Styling(size=h1_styling_size, position=h1_styling_pos, color=h1_styling_color)
h1.styling = h1_styling
h1.display_order = 0
h1.component_type = "text"
h1.tag_name = "h1"
h1.custom_attributes = {"style": "font-size:48px;margin-bottom:20px;"}
p = Text(name="p", content="Get in touch with our team", description="Text element")
p_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="20px", unit_size=UnitSize.PIXELS)
p_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
p_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
p_styling = Styling(size=p_styling_size, position=p_styling_pos, color=p_styling_color)
p.styling = p_styling
p.display_order = 1
p.component_type = "text"
p.tag_name = "p"
p.custom_attributes = {"style": "font-size:20px;margin-bottom:40px;"}
p_2 = Text(name="p_2", content="📧 Email: hello@example.com", description="Text element")
p_2_styling_size = Size(width="auto", height="auto", padding="0", margin="20px 0", font_size="18px", unit_size=UnitSize.PIXELS)
p_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
p_2_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
p_2_styling = Styling(size=p_2_styling_size, position=p_2_styling_pos, color=p_2_styling_color)
p_2.styling = p_2_styling
p_2.display_order = 0
p_2.component_type = "text"
p_2.tag_name = "p"
p_2.custom_attributes = {"style": "margin:20px 0;font-size:18px;"}
p_3 = Text(name="p_3", content="📱 Phone: (555) 123-4567", description="Text element")
p_3_styling_size = Size(width="auto", height="auto", padding="0", margin="20px 0", font_size="18px", unit_size=UnitSize.PIXELS)
p_3_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
p_3_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
p_3_styling = Styling(size=p_3_styling_size, position=p_3_styling_pos, color=p_3_styling_color)
p_3.styling = p_3_styling
p_3.display_order = 1
p_3.component_type = "text"
p_3.tag_name = "p"
p_3.custom_attributes = {"style": "margin:20px 0;font-size:18px;"}
p_4 = Text(name="p_4", content="📍 Location: New York, NY", description="Text element")
p_4_styling_size = Size(width="auto", height="auto", padding="0", margin="20px 0", font_size="18px", unit_size=UnitSize.PIXELS)
p_4_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
p_4_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
p_4_styling = Styling(size=p_4_styling_size, position=p_4_styling_pos, color=p_4_styling_color)
p_4.styling = p_4_styling
p_4.display_order = 2
p_4.component_type = "text"
p_4.tag_name = "p"
p_4.custom_attributes = {"style": "margin:20px 0;font-size:18px;"}
component_9 = ViewContainer(name="Component_9", description=" component", view_elements={p_2, p_3, p_4})
component_9_styling_size = Size(width="auto", height="auto", padding="40px", margin="0 auto", unit_size=UnitSize.PIXELS)
component_9_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
component_9_styling_color = Color(background_color="rgba(255,255,255,0.1)", text_color="#000000", border_color="#CCCCCC")
component_9_styling = Styling(size=component_9_styling_size, position=component_9_styling_pos, color=component_9_styling_color)
component_9.styling = component_9_styling
component_9.display_order = 2
component_9.custom_attributes = {"style": "background:rgba(255,255,255,0.1);padding:40px;border-radius:12px;max-width:600px;margin:0 auto;backdrop-filter:blur(10px);"}
component_8 = ViewContainer(name="Component_8", description=" component", view_elements={h1, p, component_9})
component_8_styling_size = Size(width="auto", height="100vh", padding="80px 40px", margin="0", unit_size=UnitSize.PIXELS)
component_8_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
component_8_styling_color = Color(background_color="#34495e", text_color="white", border_color="#CCCCCC")
component_8_styling = Styling(size=component_8_styling_size, position=component_8_styling_pos, color=component_8_styling_color)
component_8.styling = component_8_styling
component_8.display_order = 0
component_8.custom_attributes = {"style": "padding:80px 40px;text-align:center;font-family:Arial, sans-serif;background:#34495e;color:white;min-height:100vh;"}
wrapper_3.view_elements = {component_8}

gui_module = Module(
    name="GUI_Module",
    screens={wrapper, wrapper_2, wrapper_3}
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
