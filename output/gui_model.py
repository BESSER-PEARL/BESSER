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
Author = Class(name="Author")
Book = Class(name="Book")
Library = Class(name="Library")
Class_ = Class(name="Class")

# Author class attributes and methods
Author_name: Property = Property(name="name", type=StringType)
Author_email: Property = Property(name="email", type=StringType)
Author.attributes={Author_email, Author_name}

# Book class attributes and methods
Book_pages: Property = Property(name="pages", type=IntegerType)
Book_title: Property = Property(name="title", type=StringType)
Book_release: Property = Property(name="release", type=DateType)
Book.attributes={Book_title, Book_release, Book_pages}

# Library class attributes and methods
Library_name: Property = Property(name="name", type=StringType)
Library_address: Property = Property(name="address", type=StringType)
Library.attributes={Library_address, Library_name}

# Class class attributes and methods
Class__attribute: Property = Property(name="attribute", type=StringType)
Class__m_method: Method = Method(name="method", parameters={})
Class_.attributes={Class__attribute}
Class_.methods={Class__m_method}

# Relationships
Author_Book: BinaryAssociation = BinaryAssociation(
    name="Author_Book",
    ends={
        Property(name="writtenBy", type=Author, multiplicity=Multiplicity(1, 9999)),
        Property(name="publishes", type=Book, multiplicity=Multiplicity(0, 9999))
    }
)
Library_Book: BinaryAssociation = BinaryAssociation(
    name="Library_Book",
    ends={
        Property(name="locatedIn", type=Library, multiplicity=Multiplicity(1, 1)),
        Property(name="has", type=Book, multiplicity=Multiplicity(0, 9999))
    }
)

# Domain Model
domain_model = DomainModel(
    name="Class_Diagram",
    types={Author, Book, Library, Class_},
    associations={Author_Book, Library_Book},
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

# Screen: ifai
ifai = Screen(name="ifai", description="Home", is_main_page=True, screen_size="Medium")
ifai_styling_size = Size(width="auto", height="auto", padding="0", margin="0", unit_size=UnitSize.PIXELS)
ifai_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ifai_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ifai_styling = Styling(size=ifai_styling_size, position=ifai_styling_pos, color=ifai_styling_color)
ifai.styling = ifai_styling
i4i5 = Image(name="i4i5", description="Image component")
i9j9 = LineChart(name="i9j9", line_width=2, show_grid=True, show_legend=True, show_tooltip=True, curve_type="monotone", animate=True, legend_position="top", grid_color="#e0e0e0", dot_size=5)
i9j9_binding = DataBinding(name="Sales Over TimeDataBinding")
domain_model_ref = globals().get('domain_model')
i9j9_binding_domain = None
if domain_model_ref is not None:
    i9j9_binding_domain = domain_model_ref.get_class_by_name("Author")
if i9j9_binding_domain:
    i9j9_binding.domain_concept = i9j9_binding_domain
    i9j9_binding.label_field = next((attr for attr in i9j9_binding_domain.attributes if attr.name == "name"), None)
    i9j9_binding.data_field = next((attr for attr in i9j9_binding_domain.attributes if attr.name == "email"), None)
else:
    # Domain class 'Author' not resolved; data binding remains partial.
    pass
i9j9.data_binding = i9j9_binding
i9j9_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
i9j9_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
i9j9_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
i9j9_styling = Styling(size=i9j9_styling_size, position=i9j9_styling_pos, color=i9j9_styling_color)
i9j9.styling = i9j9_styling
itxi = BarChart(name="itxi", bar_width=30, orientation="vertical", show_grid=True, show_legend=True, show_tooltip=True, stacked=False, animate=True, legend_position="top", grid_color="#e0e0e0", bar_gap=4)
itxi_binding = DataBinding(name="Revenue by CategoryDataBinding")
domain_model_ref = globals().get('domain_model')
itxi_binding_domain = None
if domain_model_ref is not None:
    itxi_binding_domain = domain_model_ref.get_class_by_name("Library")
if itxi_binding_domain:
    itxi_binding.domain_concept = itxi_binding_domain
    itxi_binding.label_field = next((attr for attr in itxi_binding_domain.attributes if attr.name == "name"), None)
    itxi_binding.data_field = next((attr for attr in itxi_binding_domain.attributes if attr.name == "address"), None)
else:
    # Domain class 'Library' not resolved; data binding remains partial.
    pass
itxi.data_binding = itxi_binding
itxi_styling_size = Size(width="100%", height="400px", padding="0", margin="0", unit_size=UnitSize.PERCENTAGE)
itxi_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
itxi_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
itxi_styling = Styling(size=itxi_styling_size, position=itxi_styling_pos, color=itxi_styling_color)
itxi.styling = itxi_styling
ived = Text(name="ived", content="dasdasdasdasdasdasdInsert your text 55", description="Text element")
ived_styling_size = Size(width="auto", height="auto", padding="10px", margin="0", unit_size=UnitSize.PIXELS)
ived_styling_pos = Position(alignment="Alignment.LEFT", top="auto", left="auto", right="auto", bottom="auto", z_index=0, p_type=PositionType.STATIC)
ived_styling_color = Color(background_color="#FFFFFF", text_color="#000000", border_color="#CCCCCC")
ived_styling = Styling(size=ived_styling_size, position=ived_styling_pos, color=ived_styling_color)
ived.styling = ived_styling
ifai.view_elements = {i4i5, i9j9, itxi, ived}

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
