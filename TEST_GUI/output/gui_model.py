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
Counter_value: Property = Property(name="value", type=IntegerType)
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

# Screen: i2bh
i2bh = Screen(name="i2bh", description="Home", view_elements=set(), is_main_page=True, route_path="/home", screen_size="Medium")
i2bh.component_id = "7gx6FKvT6NwQwvkM"
i77yl = Text(name="i77yl", content="BESSER", description="Text element")
i77yl_styling_size = Size(width="auto", height="auto", padding="0", margin="0", font_size="24px", unit_size=UnitSize.PIXELS)
i77yl_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i77yl_styling_color = Color()
i77yl_styling = Styling(size=i77yl_styling_size, position=i77yl_styling_pos, color=i77yl_styling_color)
i77yl.styling = i77yl_styling
i77yl.display_order = 0
i77yl.component_id = "i77yl"
i77yl.component_type = "text"
i77yl.custom_attributes = {"id": "i77yl"}
iab76 = Link(name="iab76", description="Link element", label="Home", url="/")
iab76_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iab76_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iab76_styling_color = Color(text_color="white")
iab76_styling = Styling(size=iab76_styling_size, position=iab76_styling_pos, color=iab76_styling_color)
iab76.styling = iab76_styling
iab76.display_order = 0
iab76.component_id = "iab76"
iab76.component_type = "link"
iab76.tag_name = "a"
iab76.custom_attributes = {"href": "/", "id": "iab76"}
ilzt5 = Link(name="ilzt5", description="Link element", label="About", url="/about")
ilzt5_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ilzt5_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ilzt5_styling_color = Color(text_color="white")
ilzt5_styling = Styling(size=ilzt5_styling_size, position=ilzt5_styling_pos, color=ilzt5_styling_color)
ilzt5.styling = ilzt5_styling
ilzt5.display_order = 1
ilzt5.component_id = "ilzt5"
ilzt5.component_type = "link"
ilzt5.tag_name = "a"
ilzt5.custom_attributes = {"href": "/about", "id": "ilzt5"}
i4vcj = ViewContainer(name="i4vcj", description=" component", view_elements={iab76, ilzt5})
i4vcj_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
i4vcj_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i4vcj_styling_color = Color()
i4vcj_styling = Styling(size=i4vcj_styling_size, position=i4vcj_styling_pos, color=i4vcj_styling_color)
i4vcj_styling_layout = Layout(layout_type=LayoutType.FLEX, gap="30px")
i4vcj_styling.layout = i4vcj_styling_layout
i4vcj.styling = i4vcj_styling
i4vcj_layout = Layout(layout_type=LayoutType.FLEX, gap="30px")
i4vcj.layout = i4vcj_layout
i4vcj.display_order = 1
i4vcj.component_id = "i4vcj"
i4vcj.custom_attributes = {"id": "i4vcj"}
irenj = ViewContainer(name="irenj", description="nav container", view_elements={i77yl, i4vcj})
irenj_styling_size = Size(width="auto", height="auto", padding="15px 30px", margin="0", unit_size=UnitSize.PIXELS)
irenj_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
irenj_styling_color = Color(background_color="linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%) !important", text_color="white")
irenj_styling = Styling(size=irenj_styling_size, position=irenj_styling_pos, color=irenj_styling_color)
irenj_styling_layout = Layout(layout_type=LayoutType.FLEX, justify_content="space-between", align_items="center", gap="16px")
irenj_styling.layout = irenj_styling_layout
irenj.styling = irenj_styling
irenj_layout = Layout(layout_type=LayoutType.FLEX, justify_content="space-between", align_items="center", gap="16px")
irenj.layout = irenj_layout
irenj.display_order = 0
irenj.component_id = "irenj"
irenj.tag_name = "nav"
irenj.custom_attributes = {"id": "irenj"}
i1cekk = LineChart(name="i1cekk", title="Line Chart", primary_color="#4CAF50", line_width=2, show_grid=True, show_legend=True, show_tooltip=True, curve_type="monotone", animate=True, legend_position="top", grid_color="#e0e0e0", dot_size=5)
domain_model_ref = globals().get('domain_model')
i1cekk_binding_domain = None
if domain_model_ref is not None:
    i1cekk_binding_domain = domain_model_ref.get_class_by_name("Counter")
if i1cekk_binding_domain:
    i1cekk_binding = DataBinding(domain_concept=i1cekk_binding_domain)
    i1cekk_binding.label_field = next((attr for attr in i1cekk_binding_domain.attributes if attr.name == "label"), None)
    i1cekk_binding.data_field = next((attr for attr in i1cekk_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Counter' not resolved; data binding skipped.
    i1cekk_binding = None
if i1cekk_binding:
    i1cekk.data_binding = i1cekk_binding
i1cekk_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
i1cekk_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i1cekk_styling_color = Color()
i1cekk_styling = Styling(size=i1cekk_styling_size, position=i1cekk_styling_pos, color=i1cekk_styling_color)
i1cekk.styling = i1cekk_styling
i1cekk.display_order = 0
i1cekk.component_id = "i1cekk"
i1cekk.component_type = "line-chart"
i1cekk.css_classes = ["line-chart-component", "has-data-binding"]
i1cekk.custom_attributes = {"chart-color": "#4CAF50", "chart-title": "Line Chart", "data-source": "27243302-e707-43dd-8ac8-25c137307452", "label-field": "58afa26d-dae6-410e-8ca3-31f559a6980e", "data-field": "0c365500-e2b6-47d1-9763-70e404bc01af", "line-width": 2, "show-grid": True, "show-legend": True, "show-tooltip": True, "curve-type": "monotone", "animate": True, "id": "i1cekk"}
cell = ViewContainer(name="Cell", description=" container", view_elements={i1cekk})
cell_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
cell_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
cell_styling_color = Color()
cell_styling = Styling(size=cell_styling_size, position=cell_styling_pos, color=cell_styling_color)
cell.styling = cell_styling
cell.display_order = 0
cell.component_id = "container_cell"
cell.css_classes = ["gjs-cell"]
ig71rx = BarChart(name="ig71rx", title="Bar Chart Title", primary_color="#3498db", bar_width=30, orientation="vertical", show_grid=True, show_legend=True, show_tooltip=True, stacked=False, animate=True, legend_position="top", grid_color="#e0e0e0", bar_gap=4)
domain_model_ref = globals().get('domain_model')
ig71rx_binding_domain = None
if domain_model_ref is not None:
    ig71rx_binding_domain = domain_model_ref.get_class_by_name("Counter")
if ig71rx_binding_domain:
    ig71rx_binding = DataBinding(domain_concept=ig71rx_binding_domain)
    ig71rx_binding.label_field = next((attr for attr in ig71rx_binding_domain.attributes if attr.name == "label"), None)
    ig71rx_binding.data_field = next((attr for attr in ig71rx_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Counter' not resolved; data binding skipped.
    ig71rx_binding = None
if ig71rx_binding:
    ig71rx.data_binding = ig71rx_binding
ig71rx_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
ig71rx_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ig71rx_styling_color = Color()
ig71rx_styling = Styling(size=ig71rx_styling_size, position=ig71rx_styling_pos, color=ig71rx_styling_color)
ig71rx.styling = ig71rx_styling
ig71rx.display_order = 0
ig71rx.component_id = "ig71rx"
ig71rx.component_type = "bar-chart"
ig71rx.css_classes = ["bar-chart-component", "has-data-binding"]
ig71rx.custom_attributes = {"chart-color": "#3498db", "chart-title": "Bar Chart Title", "data-source": "27243302-e707-43dd-8ac8-25c137307452", "label-field": "58afa26d-dae6-410e-8ca3-31f559a6980e", "data-field": "0c365500-e2b6-47d1-9763-70e404bc01af", "bar-width": 30, "orientation": "vertical", "show-grid": True, "show-legend": True, "stacked": False, "id": "ig71rx"}
iiyfi = ViewContainer(name="iiyfi", description=" container", view_elements={ig71rx})
iiyfi_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
iiyfi_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iiyfi_styling_color = Color()
iiyfi_styling = Styling(size=iiyfi_styling_size, position=iiyfi_styling_pos, color=iiyfi_styling_color)
iiyfi.styling = iiyfi_styling
iiyfi.display_order = 1
iiyfi.component_id = "iiyfi"
iiyfi.css_classes = ["gjs-cell"]
iiyfi.custom_attributes = {"id": "iiyfi"}
irekdw = RadarChart(name="irekdw", title="Radar Chart", primary_color="#8884d8", show_grid=True, show_tooltip=True, show_radius_axis=True, show_legend=True, legend_position="top", dot_size=3, grid_type="polygon", stroke_width=2)
domain_model_ref = globals().get('domain_model')
irekdw_binding_domain = None
if domain_model_ref is not None:
    irekdw_binding_domain = domain_model_ref.get_class_by_name("Counter")
if irekdw_binding_domain:
    irekdw_binding = DataBinding(domain_concept=irekdw_binding_domain)
    irekdw_binding.label_field = next((attr for attr in irekdw_binding_domain.attributes if attr.name == "label"), None)
    irekdw_binding.data_field = next((attr for attr in irekdw_binding_domain.attributes if attr.name == "value"), None)
else:
    # Domain class 'Counter' not resolved; data binding skipped.
    irekdw_binding = None
if irekdw_binding:
    irekdw.data_binding = irekdw_binding
irekdw_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
irekdw_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
irekdw_styling_color = Color()
irekdw_styling = Styling(size=irekdw_styling_size, position=irekdw_styling_pos, color=irekdw_styling_color)
irekdw.styling = irekdw_styling
irekdw.display_order = 0
irekdw.component_id = "irekdw"
irekdw.component_type = "radar-chart"
irekdw.css_classes = ["radar-chart-component", "has-data-binding"]
irekdw.custom_attributes = {"chart-color": "#8884d8", "chart-title": "Radar Chart", "data-source": "27243302-e707-43dd-8ac8-25c137307452", "label-field": "58afa26d-dae6-410e-8ca3-31f559a6980e", "data-field": "0c365500-e2b6-47d1-9763-70e404bc01af", "show-grid": True, "show-tooltip": True, "show-radius-axis": True, "id": "irekdw"}
cell_2 = ViewContainer(name="Cell_2", description=" container", view_elements={irekdw})
cell_2_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
cell_2_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
cell_2_styling_color = Color()
cell_2_styling = Styling(size=cell_2_styling_size, position=cell_2_styling_pos, color=cell_2_styling_color)
cell_2.styling = cell_2_styling
cell_2.display_order = 2
cell_2.component_id = "container_cell_2"
cell_2.css_classes = ["gjs-cell"]
iofcf = ViewContainer(name="iofcf", description=" container", view_elements={cell, iiyfi, cell_2})
iofcf_styling_size = Size(width="100%", height="auto", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
iofcf_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iofcf_styling_color = Color()
iofcf_styling = Styling(size=iofcf_styling_size, position=iofcf_styling_pos, color=iofcf_styling_color)
iofcf.styling = iofcf_styling
iofcf.display_order = 1
iofcf.component_id = "iofcf"
iofcf.css_classes = ["gjs-row"]
iofcf.custom_attributes = {"id": "iofcf"}
if1fqg = Text(name="if1fqg", content="About BESSER", description="Text element")
if1fqg_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
if1fqg_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
if1fqg_styling_color = Color()
if1fqg_styling = Styling(size=if1fqg_styling_size, position=if1fqg_styling_pos, color=if1fqg_styling_color)
if1fqg.styling = if1fqg_styling
if1fqg.display_order = 0
if1fqg.component_id = "if1fqg"
if1fqg.component_type = "text"
if1fqg.tag_name = "h4"
if1fqg.custom_attributes = {"id": "if1fqg"}
ijr0hh = Text(name="ijr0hh", content="BESSER is a low-code platform for building smarter software faster. Empower your development with our dashboard generator and modeling tools.", description="Text element")
ijr0hh_styling_size = Size(width="auto", height="auto", padding="0", margin="0", line_height="1.6", unit_size=UnitSize.PIXELS)
ijr0hh_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ijr0hh_styling_color = Color(opacity="0.8")
ijr0hh_styling = Styling(size=ijr0hh_styling_size, position=ijr0hh_styling_pos, color=ijr0hh_styling_color)
ijr0hh.styling = ijr0hh_styling
ijr0hh.display_order = 1
ijr0hh.component_id = "ijr0hh"
ijr0hh.component_type = "text"
ijr0hh.tag_name = "p"
ijr0hh.custom_attributes = {"id": "ijr0hh"}
component = ViewContainer(name="Component", description=" component", view_elements={if1fqg, ijr0hh})
component.display_order = 0
iq3sz9 = Text(name="iq3sz9", content="Quick Links", description="Text element")
iq3sz9_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
iq3sz9_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iq3sz9_styling_color = Color()
iq3sz9_styling = Styling(size=iq3sz9_styling_size, position=iq3sz9_styling_pos, color=iq3sz9_styling_color)
iq3sz9.styling = iq3sz9_styling
iq3sz9.display_order = 0
iq3sz9.component_id = "iq3sz9"
iq3sz9.component_type = "text"
iq3sz9.tag_name = "h4"
iq3sz9.custom_attributes = {"id": "iq3sz9"}
id2pjg = Link(name="id2pjg", description="Link element", label="About", url="/about")
id2pjg_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
id2pjg_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
id2pjg_styling_color = Color(text_color="white")
id2pjg_styling = Styling(size=id2pjg_styling_size, position=id2pjg_styling_pos, color=id2pjg_styling_color)
id2pjg.styling = id2pjg_styling
id2pjg.display_order = 0
id2pjg.component_id = "id2pjg"
id2pjg.component_type = "link"
id2pjg.tag_name = "a"
id2pjg.custom_attributes = {"href": "/about", "id": "id2pjg"}
iyeciq = ViewContainer(name="iyeciq", description="li container", view_elements={id2pjg})
iyeciq_styling_size = Size(width="auto", height="auto", padding="0", margin="8px 0", unit_size=UnitSize.PIXELS)
iyeciq_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iyeciq_styling_color = Color()
iyeciq_styling = Styling(size=iyeciq_styling_size, position=iyeciq_styling_pos, color=iyeciq_styling_color)
iyeciq.styling = iyeciq_styling
iyeciq.display_order = 0
iyeciq.component_id = "iyeciq"
iyeciq.tag_name = "li"
iyeciq.custom_attributes = {"id": "iyeciq"}
icvtnv = ViewContainer(name="icvtnv", description="ul container", view_elements={iyeciq})
icvtnv_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
icvtnv_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
icvtnv_styling_color = Color(opacity="0.8")
icvtnv_styling = Styling(size=icvtnv_styling_size, position=icvtnv_styling_pos, color=icvtnv_styling_color)
icvtnv.styling = icvtnv_styling
icvtnv.display_order = 1
icvtnv.component_id = "icvtnv"
icvtnv.tag_name = "ul"
icvtnv.custom_attributes = {"id": "icvtnv"}
component_2 = ViewContainer(name="Component_2", description=" component", view_elements={iq3sz9, icvtnv})
component_2.display_order = 1
inqfyo = Text(name="inqfyo", content="Contact", description="Text element")
inqfyo_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
inqfyo_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
inqfyo_styling_color = Color()
inqfyo_styling = Styling(size=inqfyo_styling_size, position=inqfyo_styling_pos, color=inqfyo_styling_color)
inqfyo.styling = inqfyo_styling
inqfyo.display_order = 0
inqfyo.component_id = "inqfyo"
inqfyo.component_type = "text"
inqfyo.tag_name = "h4"
inqfyo.custom_attributes = {"id": "inqfyo"}
ivv8oa = Text(name="ivv8oa", content="Email: info@besser-pearl.org\nPhone: (123) 456-7890", description="Text element")
ivv8oa_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ivv8oa_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ivv8oa_styling_color = Color(opacity="0.8")
ivv8oa_styling = Styling(size=ivv8oa_styling_size, position=ivv8oa_styling_pos, color=ivv8oa_styling_color)
ivv8oa.styling = ivv8oa_styling
ivv8oa.display_order = 1
ivv8oa.component_id = "ivv8oa"
ivv8oa.component_type = "text"
ivv8oa.tag_name = "p"
ivv8oa.custom_attributes = {"id": "ivv8oa"}
component_3 = ViewContainer(name="Component_3", description=" component", view_elements={inqfyo, ivv8oa})
component_3.display_order = 2
ikqajz = ViewContainer(name="ikqajz", description=" component", view_elements={component, component_2, component_3})
ikqajz_styling_size = Size(width="auto", height="auto", padding="0", margin="0 auto", unit_size=UnitSize.PIXELS)
ikqajz_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ikqajz_styling_color = Color()
ikqajz_styling = Styling(size=ikqajz_styling_size, position=ikqajz_styling_pos, color=ikqajz_styling_color)
ikqajz_styling_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="30px")
ikqajz_styling.layout = ikqajz_styling_layout
ikqajz.styling = ikqajz_styling
ikqajz_layout = Layout(layout_type=LayoutType.GRID, grid_template_columns="repeat(auto-fit, minmax(250px, 1fr))", gap="30px")
ikqajz.layout = ikqajz_layout
ikqajz.display_order = 0
ikqajz.component_id = "ikqajz"
ikqajz.custom_attributes = {"id": "ikqajz"}
icjzp3 = Text(name="icjzp3", content="© 2025 BESSER. All rights reserved.", description="Text element")
icjzp3_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
icjzp3_styling_pos = Position(alignment=Alignment.CENTER, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
icjzp3_styling_color = Color(opacity="0.7")
icjzp3_styling = Styling(size=icjzp3_styling_size, position=icjzp3_styling_pos, color=icjzp3_styling_color)
icjzp3.styling = icjzp3_styling
icjzp3.display_order = 1
icjzp3.component_id = "icjzp3"
icjzp3.component_type = "text"
icjzp3.custom_attributes = {"id": "icjzp3"}
iv0tsf = ViewContainer(name="iv0tsf", description="footer container", view_elements={ikqajz, icjzp3})
iv0tsf_styling_size = Size(width="auto", height="auto", padding="40px 20px", margin="0", unit_size=UnitSize.PIXELS)
iv0tsf_styling_pos = Position(alignment=Alignment.LEFT, top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
iv0tsf_styling_color = Color(background_color="linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%) !important", text_color="white")
iv0tsf_styling = Styling(size=iv0tsf_styling_size, position=iv0tsf_styling_pos, color=iv0tsf_styling_color)
iv0tsf.styling = iv0tsf_styling
iv0tsf.display_order = 2
iv0tsf.component_id = "iv0tsf"
iv0tsf.tag_name = "footer"
iv0tsf.custom_attributes = {"id": "iv0tsf"}
i2bh.view_elements = {irenj, iofcf, iv0tsf}


# Screen: wrapper
wrapper = Screen(name="wrapper", description="About", view_elements=set(), route_path="/about", screen_size="Medium")
wrapper.component_id = "HWrxLpPdBSm60elV"
meta = ViewComponent(name="meta", description="meta component")
meta.display_order = 0
meta.tag_name = "meta"
meta.custom_attributes = {"charset": "utf-8"}
viewport = ViewComponent(name="viewport", description="meta component")
viewport.display_order = 1
viewport.tag_name = "meta"
viewport.custom_attributes = {"name": "viewport", "content": "width=device-width, initial-scale=1"}
meta_2 = ViewComponent(name="meta_2", description="meta component")
meta_2.display_order = 2
meta_2.tag_name = "meta"
meta_2.custom_attributes = {"charset": "utf-8"}
viewport_2 = ViewComponent(name="viewport_2", description="meta component")
viewport_2.display_order = 3
viewport_2.tag_name = "meta"
viewport_2.custom_attributes = {"name": "viewport", "content": "width=device-width, initial-scale=1"}
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
ii2qto = Text(name="ii2qto", content="Resp Design", description="Text element")
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
component_4 = ViewContainer(name="Component_4", description=" component", view_elements={i3p8x_2, i9x6g_2})
component_4.display_order = 0
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
component_5 = ViewContainer(name="Component_5", description=" component", view_elements={iybmm_2, im0xi_2})
component_5.display_order = 1
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
i8uhg_2 = ViewContainer(name="i8uhg_2", description=" component", view_elements={component_4, component_5, iberth})
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
wrapper.view_elements = {meta, viewport, meta_2, viewport_2, i5jh_2, ihhucw, i94ykd, ictst_2, iiqx6_2, is6fi_2}

gui_module = Module(
    name="GUI_Module",
    screens={i2bh, wrapper}
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
