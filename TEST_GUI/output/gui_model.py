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
ie5m.component_id = "ie5m"
ie5m.component_type = "bar-chart"
ie5m.css_classes = ["bar-chart-component"]
ie5m.custom_attributes = {"chart-color": "#3498db", "chart-title": "Revenue by Category", "data-source": "8c4ebc1b-0432-43e9-8d84-3e7e3a554479", "label-field": "86560714-c3a9-4d80-a28d-9e6a360f3ae8", "data-field": "9aa4009c-e0e3-45b0-b405-5ea70309905a", "bar-width": 30, "orientation": "vertical", "show-grid": True, "show-legend": True, "stacked": False, "id": "ie5m"}
ijql = Text(name="ijql", content="Statec Hackathon", description="Text element")
ijql_styling_size = Size(width="auto", height="auto", padding="10px", margin="0", unit_size=UnitSize.PIXELS)
ijql_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ijql_styling_color = Color(background_color="#FFFFFF", text_color="#e23131", border_color="#CCCCCC")
ijql_styling = Styling(size=ijql_styling_size, position=ijql_styling_pos, color=ijql_styling_color)
ijql.styling = ijql_styling
ijql.component_id = "ijql"
ijql.component_type = "text"
ijql.custom_attributes = {"id": "ijql"}
iuri1 = LineChart(name="iuri1", title="Sales Over Time", primary_color="#4CAF50", line_width=2, show_grid=True, show_legend=True, show_tooltip=True, curve_type="monotone", animate=True, legend_position="top", grid_color="#e0e0e0", dot_size=5)
iuri1_binding = DataBinding(name="Sales Over TimeDataBinding")
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
iuri1.component_id = "iuri1"
iuri1.component_type = "line-chart"
iuri1.css_classes = ["line-chart-component"]
iuri1.custom_attributes = {"chart-color": "#4CAF50", "chart-title": "Sales Over Time", "data-source": "8c4ebc1b-0432-43e9-8d84-3e7e3a554479", "label-field": "86560714-c3a9-4d80-a28d-9e6a360f3ae8", "data-field": "9aa4009c-e0e3-45b0-b405-5ea70309905a", "line-width": 2, "show-grid": True, "show-legend": True, "show-tooltip": True, "curve-type": "monotone", "animate": True, "id": "iuri1"}
cell = ViewContainer(name="Cell", description=" container", view_elements={iuri1})
cell_styling_size = Size(width="8%", height="75px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
cell_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
cell_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
cell_styling = Styling(size=cell_styling_size, position=cell_styling_pos, color=cell_styling_color)
cell.styling = cell_styling
cell.css_classes = ["gjs-cell"]
i3ja1 = BarChart(name="i3ja1", title="Revenue by Category", primary_color="#3498db", bar_width=30, orientation="vertical", show_grid=True, show_legend=True, show_tooltip=True, stacked=False, animate=True, legend_position="top", grid_color="#e0e0e0", bar_gap=4)
i3ja1_binding = DataBinding(name="Revenue by CategoryDataBinding")
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
i3ja1.component_id = "i3ja1"
i3ja1.component_type = "bar-chart"
i3ja1.css_classes = ["bar-chart-component"]
i3ja1.custom_attributes = {"chart-color": "#3498db", "chart-title": "Revenue by Category", "data-source": "8c4ebc1b-0432-43e9-8d84-3e7e3a554479", "label-field": "86560714-c3a9-4d80-a28d-9e6a360f3ae8", "data-field": "9aa4009c-e0e3-45b0-b405-5ea70309905a", "bar-width": 30, "orientation": "vertical", "show-grid": True, "show-legend": True, "stacked": False, "id": "i3ja1"}
cell_2 = ViewContainer(name="Cell_2", description=" container", view_elements={i3ja1})
cell_2_styling_size = Size(width="8%", height="75px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
cell_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
cell_2_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
cell_2_styling = Styling(size=cell_2_styling_size, position=cell_2_styling_pos, color=cell_2_styling_color)
cell_2.styling = cell_2_styling
cell_2.css_classes = ["gjs-cell"]
i4yjj = LineChart(name="i4yjj", title="Sales Over Time", primary_color="#4CAF50", line_width=2, show_grid=True, show_legend=True, show_tooltip=True, curve_type="monotone", animate=True, legend_position="top", grid_color="#e0e0e0", dot_size=5)
i4yjj_binding = DataBinding(name="Sales Over TimeDataBinding")
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
i4yjj.component_id = "i4yjj"
i4yjj.component_type = "line-chart"
i4yjj.css_classes = ["line-chart-component"]
i4yjj.custom_attributes = {"chart-color": "#4CAF50", "chart-title": "Sales Over Time", "data-source": "8c4ebc1b-0432-43e9-8d84-3e7e3a554479", "label-field": "86560714-c3a9-4d80-a28d-9e6a360f3ae8", "data-field": "9aa4009c-e0e3-45b0-b405-5ea70309905a", "line-width": 2, "show-grid": True, "show-legend": True, "show-tooltip": True, "curve-type": "monotone", "animate": True, "id": "i4yjj"}
cell_3 = ViewContainer(name="Cell_3", description=" container", view_elements={i4yjj})
cell_3_styling_size = Size(width="8%", height="75px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
cell_3_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
cell_3_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
cell_3_styling = Styling(size=cell_3_styling_size, position=cell_3_styling_pos, color=cell_3_styling_color)
cell_3.styling = cell_3_styling
cell_3.css_classes = ["gjs-cell"]
ik6k = ViewContainer(name="ik6k", description=" container", view_elements={cell, cell_2, cell_3})
ik6k_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
ik6k_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ik6k_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ik6k_styling = Styling(size=ik6k_styling_size, position=ik6k_styling_pos, color=ik6k_styling_color)
ik6k.styling = ik6k_styling
ik6k.component_id = "ik6k"
ik6k.css_classes = ["gjs-row"]
ik6k.custom_attributes = {"id": "ik6k"}
icnh.view_elements = {ie5m, ijql, ik6k}

gui_module = Module(
    name="GUI_Module",
    screens={icnh}
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
