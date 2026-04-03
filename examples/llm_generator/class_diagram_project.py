####################
# STRUCTURAL MODEL #
####################

from besser.BUML.metamodel.structural import (
    Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, DomainModel,
    Enumeration, EnumerationLiteral, Multiplicity,
    StringType, IntegerType, FloatType, BooleanType,
    TimeType, DateType, DateTimeType, TimeDeltaType,
    AnyType, Constraint, AssociationClass, Metadata, MethodImplementationType
)

# Enumerations
InteractionType: Enumeration = Enumeration(
    name="InteractionType",
    literals={
            EnumerationLiteral(name="EMAIL"),
			EnumerationLiteral(name="CALL"),
			EnumerationLiteral(name="NOTE"),
			EnumerationLiteral(name="MEETING")
    }
)

LeadScoreLevel: Enumeration = Enumeration(
    name="LeadScoreLevel",
    literals={
            EnumerationLiteral(name="COLD"),
			EnumerationLiteral(name="WARM"),
			EnumerationLiteral(name="HOT")
    }
)

CompanySize: Enumeration = Enumeration(
    name="CompanySize",
    literals={
            EnumerationLiteral(name="STARTUP"),
			EnumerationLiteral(name="ENTERPRISE"),
			EnumerationLiteral(name="LARGE"),
			EnumerationLiteral(name="MEDIUM"),
			EnumerationLiteral(name="SMALL")
    }
)

OpportunityStage: Enumeration = Enumeration(
    name="OpportunityStage",
    literals={
            EnumerationLiteral(name="PROPOSAL"),
			EnumerationLiteral(name="QUALIFICATION"),
			EnumerationLiteral(name="CLOSED_LOST"),
			EnumerationLiteral(name="PROSPECTING"),
			EnumerationLiteral(name="CLOSED_WON"),
			EnumerationLiteral(name="NEGOTIATION")
    }
)

UserRole: Enumeration = Enumeration(
    name="UserRole",
    literals={
            EnumerationLiteral(name="SALES_MANAGER"),
			EnumerationLiteral(name="ADMIN"),
			EnumerationLiteral(name="SALES_REP")
    }
)

InteractionDirection: Enumeration = Enumeration(
    name="InteractionDirection",
    literals={
            EnumerationLiteral(name="INBOUND"),
			EnumerationLiteral(name="OUTBOUND")
    }
)

Industry: Enumeration = Enumeration(
    name="Industry",
    literals={
            EnumerationLiteral(name="MANUFACTURING"),
			EnumerationLiteral(name="HEALTHCARE"),
			EnumerationLiteral(name="FINANCE"),
			EnumerationLiteral(name="OTHER"),
			EnumerationLiteral(name="TECHNOLOGY"),
			EnumerationLiteral(name="SERVICES"),
			EnumerationLiteral(name="RETAIL")
    }
)

# Classes
Tag = Class(name="Tag")
Interaction = Class(name="Interaction")
Opportunity = Class(name="Opportunity")
Contact = Class(name="Contact")
Company = Class(name="Company")
User = Class(name="User")
ScoreHistory = Class(name="ScoreHistory")
EnrichmentLog = Class(name="EnrichmentLog")
GeneratedEmail = Class(name="GeneratedEmail")
EmailTemplate = Class(name="EmailTemplate")
Task = Class(name="Task")

# Tag class attributes and methods
Tag_name: Property = Property(name="name", type=StringType)
Tag_color: Property = Property(name="color", type=StringType, default_value="#3B82F6")
Tag_id: Property = Property(name="id", type=IntegerType)
Tag.attributes={Tag_color, Tag_id, Tag_name}

# Interaction class attributes and methods
Interaction_id: Property = Property(name="id", type=IntegerType)
Interaction_direction: Property = Property(name="direction", type=InteractionDirection, is_optional=True)
Interaction_created_at: Property = Property(name="created_at", type=DateTimeType)
Interaction_subject: Property = Property(name="subject", type=StringType, is_optional=True)
Interaction_type: Property = Property(name="type", type=InteractionType)
Interaction_content: Property = Property(name="content", type=StringType)
Interaction_occurred_at: Property = Property(name="occurred_at", type=DateTimeType)
Interaction.attributes={Interaction_content, Interaction_created_at, Interaction_direction, Interaction_id, Interaction_occurred_at, Interaction_subject, Interaction_type}

# Opportunity class attributes and methods
Opportunity_expected_close_date: Property = Property(name="expected_close_date", type=DateType, is_optional=True)
Opportunity_id: Property = Property(name="id", type=IntegerType)
Opportunity_probability: Property = Property(name="probability", type=IntegerType, is_optional=True)
Opportunity_value: Property = Property(name="value", type=FloatType, is_optional=True)
Opportunity_closed_at: Property = Property(name="closed_at", type=DateTimeType, is_optional=True)
Opportunity_stage: Property = Property(name="stage", type=OpportunityStage, default_value="PROSPECTING")
Opportunity_updated_at: Property = Property(name="updated_at", type=DateTimeType)
Opportunity_description: Property = Property(name="description", type=StringType, is_optional=True)
Opportunity_created_at: Property = Property(name="created_at", type=DateTimeType)
Opportunity_title: Property = Property(name="title", type=StringType)
Opportunity.attributes={Opportunity_closed_at, Opportunity_created_at, Opportunity_description, Opportunity_expected_close_date, Opportunity_id, Opportunity_probability, Opportunity_stage, Opportunity_title, Opportunity_updated_at, Opportunity_value}

# Contact class attributes and methods
Contact_updated_at: Property = Property(name="updated_at", type=DateTimeType)
Contact_phone: Property = Property(name="phone", type=StringType, is_optional=True)
Contact_lead_score_level: Property = Property(name="lead_score_level", type=LeadScoreLevel, default_value="COLD")
Contact_email: Property = Property(name="email", type=StringType, is_optional=True)
Contact_lead_score: Property = Property(name="lead_score", type=IntegerType, default_value=0)
Contact_last_name: Property = Property(name="last_name", type=StringType)
Contact_profile_picture_url: Property = Property(name="profile_picture_url", type=StringType, is_optional=True)
Contact_first_name: Property = Property(name="first_name", type=StringType)
Contact_created_at: Property = Property(name="created_at", type=DateTimeType)
Contact_linkedin_url: Property = Property(name="linkedin_url", type=StringType, is_optional=True)
Contact_id: Property = Property(name="id", type=IntegerType)
Contact_notes: Property = Property(name="notes", type=StringType, is_optional=True)
Contact_job_title: Property = Property(name="job_title", type=StringType, is_optional=True)
Contact_is_enriched: Property = Property(name="is_enriched", type=BooleanType, default_value=False)
Contact.attributes={Contact_created_at, Contact_email, Contact_first_name, Contact_id, Contact_is_enriched, Contact_job_title, Contact_last_name, Contact_lead_score, Contact_lead_score_level, Contact_linkedin_url, Contact_notes, Contact_phone, Contact_profile_picture_url, Contact_updated_at}

# Company class attributes and methods
Company_website: Property = Property(name="website", type=StringType, is_optional=True)
Company_phone: Property = Property(name="phone", type=StringType, is_optional=True)
Company_name: Property = Property(name="name", type=StringType)
Company_created_at: Property = Property(name="created_at", type=DateTimeType)
Company_linkedin_url: Property = Property(name="linkedin_url", type=StringType, is_optional=True)
Company_description: Property = Property(name="description", type=StringType, is_optional=True)
Company_updated_at: Property = Property(name="updated_at", type=DateTimeType)
Company_id: Property = Property(name="id", type=IntegerType)
Company_country: Property = Property(name="country", type=StringType, is_optional=True)
Company_size: Property = Property(name="size", type=CompanySize, is_optional=True)
Company_city: Property = Property(name="city", type=StringType, is_optional=True)
Company_industry: Property = Property(name="industry", type=Industry)
Company_address: Property = Property(name="address", type=StringType, is_optional=True)
Company.attributes={Company_address, Company_city, Company_country, Company_created_at, Company_description, Company_id, Company_industry, Company_linkedin_url, Company_name, Company_phone, Company_size, Company_updated_at, Company_website}

# User class attributes and methods
User_last_name: Property = Property(name="last_name", type=StringType)
User_first_name: Property = Property(name="first_name", type=StringType)
User_password_hash: Property = Property(name="password_hash", type=StringType)
User_last_login: Property = Property(name="last_login", type=DateTimeType, is_optional=True)
User_email: Property = Property(name="email", type=StringType)
User_is_active: Property = Property(name="is_active", type=BooleanType, default_value=True)
User_created_at: Property = Property(name="created_at", type=DateTimeType)
User_id: Property = Property(name="id", type=IntegerType)
User_role: Property = Property(name="role", type=UserRole)
User.attributes={User_created_at, User_email, User_first_name, User_id, User_is_active, User_last_login, User_last_name, User_password_hash, User_role}

# ScoreHistory class attributes and methods
ScoreHistory_id: Property = Property(name="id", type=IntegerType)
ScoreHistory_calculated_at: Property = Property(name="calculated_at", type=DateTimeType)
ScoreHistory_reason: Property = Property(name="reason", type=StringType)
ScoreHistory_new_score: Property = Property(name="new_score", type=IntegerType)
ScoreHistory_old_score: Property = Property(name="old_score", type=IntegerType)
ScoreHistory.attributes={ScoreHistory_calculated_at, ScoreHistory_id, ScoreHistory_new_score, ScoreHistory_old_score, ScoreHistory_reason}

# EnrichmentLog class attributes and methods
EnrichmentLog_id: Property = Property(name="id", type=IntegerType)
EnrichmentLog_error_message: Property = Property(name="error_message", type=StringType, is_optional=True)
EnrichmentLog_is_successful: Property = Property(name="is_successful", type=BooleanType)
EnrichmentLog_linkedin_url: Property = Property(name="linkedin_url", type=StringType)
EnrichmentLog_enriched_at: Property = Property(name="enriched_at", type=DateTimeType)
EnrichmentLog.attributes={EnrichmentLog_enriched_at, EnrichmentLog_error_message, EnrichmentLog_id, EnrichmentLog_is_successful, EnrichmentLog_linkedin_url}

# GeneratedEmail class attributes and methods
GeneratedEmail_body: Property = Property(name="body", type=StringType)
GeneratedEmail_subject: Property = Property(name="subject", type=StringType)
GeneratedEmail_id: Property = Property(name="id", type=IntegerType)
GeneratedEmail_created_at: Property = Property(name="created_at", type=DateTimeType)
GeneratedEmail_sent_at: Property = Property(name="sent_at", type=DateTimeType, is_optional=True)
GeneratedEmail_is_sent: Property = Property(name="is_sent", type=BooleanType, default_value=False)
GeneratedEmail.attributes={GeneratedEmail_body, GeneratedEmail_created_at, GeneratedEmail_id, GeneratedEmail_is_sent, GeneratedEmail_sent_at, GeneratedEmail_subject}

# EmailTemplate class attributes and methods
EmailTemplate_created_at: Property = Property(name="created_at", type=DateTimeType)
EmailTemplate_category: Property = Property(name="category", type=StringType)
EmailTemplate_body_template: Property = Property(name="body_template", type=StringType)
EmailTemplate_subject_template: Property = Property(name="subject_template", type=StringType)
EmailTemplate_id: Property = Property(name="id", type=IntegerType)
EmailTemplate_name: Property = Property(name="name", type=StringType)
EmailTemplate.attributes={EmailTemplate_body_template, EmailTemplate_category, EmailTemplate_created_at, EmailTemplate_id, EmailTemplate_name, EmailTemplate_subject_template}

# Task class attributes and methods
Task_description: Property = Property(name="description", type=StringType, is_optional=True)
Task_due_date: Property = Property(name="due_date", type=DateTimeType, is_optional=True)
Task_created_at: Property = Property(name="created_at", type=DateTimeType)
Task_title: Property = Property(name="title", type=StringType)
Task_id: Property = Property(name="id", type=IntegerType)
Task_completed_at: Property = Property(name="completed_at", type=DateTimeType, is_optional=True)
Task_is_completed: Property = Property(name="is_completed", type=BooleanType, default_value=False)
Task.attributes={Task_completed_at, Task_created_at, Task_description, Task_due_date, Task_id, Task_is_completed, Task_title}

# Relationships
template_user: BinaryAssociation = BinaryAssociation(
    name="template_user",
    ends={
        Property(name="email_templates", type=EmailTemplate, multiplicity=Multiplicity(0, 9999)),
        Property(name="created_by", type=User, multiplicity=Multiplicity(1, 1))
    }
)
contact_company: BinaryAssociation = BinaryAssociation(
    name="contact_company",
    ends={
        Property(name="contacts", type=Contact, multiplicity=Multiplicity(0, 9999)),
        Property(name="company", type=Company, multiplicity=Multiplicity(0, 1))
    }
)
task_opportunity: BinaryAssociation = BinaryAssociation(
    name="task_opportunity",
    ends={
        Property(name="tasks", type=Task, multiplicity=Multiplicity(0, 9999)),
        Property(name="opportunity", type=Opportunity, multiplicity=Multiplicity(0, 1))
    }
)
email_user: BinaryAssociation = BinaryAssociation(
    name="email_user",
    ends={
        Property(name="generated_emails", type=GeneratedEmail, multiplicity=Multiplicity(0, 9999)),
        Property(name="created_by", type=User, multiplicity=Multiplicity(1, 1))
    }
)
contact_created_by: BinaryAssociation = BinaryAssociation(
    name="contact_created_by",
    ends={
        Property(name="created_by", type=User, multiplicity=Multiplicity(1, 1)),
        Property(name="created_contacts", type=Contact, multiplicity=Multiplicity(0, 9999))
    }
)
interaction_user: BinaryAssociation = BinaryAssociation(
    name="interaction_user",
    ends={
        Property(name="interactions", type=Interaction, multiplicity=Multiplicity(0, 9999)),
        Property(name="performed_by", type=User, multiplicity=Multiplicity(1, 1))
    }
)
opportunity_owner: BinaryAssociation = BinaryAssociation(
    name="opportunity_owner",
    ends={
        Property(name="owned_opportunities", type=Opportunity, multiplicity=Multiplicity(0, 9999)),
        Property(name="owner", type=User, multiplicity=Multiplicity(1, 1))
    }
)
task_user: BinaryAssociation = BinaryAssociation(
    name="task_user",
    ends={
        Property(name="tasks", type=Task, multiplicity=Multiplicity(0, 9999)),
        Property(name="assigned_to", type=User, multiplicity=Multiplicity(1, 1))
    }
)
contact_tag: BinaryAssociation = BinaryAssociation(
    name="contact_tag",
    ends={
        Property(name="tagged_contacts", type=Contact, multiplicity=Multiplicity(0, 9999)),
        Property(name="tags", type=Tag, multiplicity=Multiplicity(0, 9999))
    }
)
enrichment_contact: BinaryAssociation = BinaryAssociation(
    name="enrichment_contact",
    ends={
        Property(name="contact", type=Contact, multiplicity=Multiplicity(1, 1)),
        Property(name="enrichment_logs", type=EnrichmentLog, multiplicity=Multiplicity(0, 9999))
    }
)
email_template_link: BinaryAssociation = BinaryAssociation(
    name="email_template_link",
    ends={
        Property(name="template", type=EmailTemplate, multiplicity=Multiplicity(0, 1)),
        Property(name="generated_emails", type=GeneratedEmail, multiplicity=Multiplicity(0, 9999))
    }
)
opportunity_contact: BinaryAssociation = BinaryAssociation(
    name="opportunity_contact",
    ends={
        Property(name="contacts", type=Contact, multiplicity=Multiplicity(1, 9999)),
        Property(name="opportunities", type=Opportunity, multiplicity=Multiplicity(0, 9999))
    }
)
task_contact: BinaryAssociation = BinaryAssociation(
    name="task_contact",
    ends={
        Property(name="contact", type=Contact, multiplicity=Multiplicity(0, 1)),
        Property(name="tasks", type=Task, multiplicity=Multiplicity(0, 9999))
    }
)
email_contact: BinaryAssociation = BinaryAssociation(
    name="email_contact",
    ends={
        Property(name="contact", type=Contact, multiplicity=Multiplicity(1, 1)),
        Property(name="generated_emails", type=GeneratedEmail, multiplicity=Multiplicity(0, 9999))
    }
)
score_contact: BinaryAssociation = BinaryAssociation(
    name="score_contact",
    ends={
        Property(name="score_history", type=ScoreHistory, multiplicity=Multiplicity(0, 9999)),
        Property(name="contact", type=Contact, multiplicity=Multiplicity(1, 1), is_composite=True)
    }
)
company_created_by: BinaryAssociation = BinaryAssociation(
    name="company_created_by",
    ends={
        Property(name="created_companies", type=Company, multiplicity=Multiplicity(0, 9999)),
        Property(name="created_by", type=User, multiplicity=Multiplicity(1, 1))
    }
)
opportunity_company: BinaryAssociation = BinaryAssociation(
    name="opportunity_company",
    ends={
        Property(name="company", type=Company, multiplicity=Multiplicity(0, 1)),
        Property(name="opportunities", type=Opportunity, multiplicity=Multiplicity(0, 9999))
    }
)
interaction_contact: BinaryAssociation = BinaryAssociation(
    name="interaction_contact",
    ends={
        Property(name="interactions", type=Interaction, multiplicity=Multiplicity(0, 9999)),
        Property(name="contact", type=Contact, multiplicity=Multiplicity(1, 1), is_composite=True)
    }
)
company_tag: BinaryAssociation = BinaryAssociation(
    name="company_tag",
    ends={
        Property(name="tagged_companies", type=Company, multiplicity=Multiplicity(0, 9999)),
        Property(name="tags", type=Tag, multiplicity=Multiplicity(0, 9999))
    }
)

# Domain Model
user_model = DomainModel(
    name="NexaCRM",
    types={Tag, Interaction, Opportunity, Contact, Company, User, ScoreHistory, EnrichmentLog, GeneratedEmail, EmailTemplate, Task, InteractionType, LeadScoreLevel, CompanySize, OpportunityStage, UserRole, InteractionDirection, Industry},
    associations={template_user, contact_company, task_opportunity, email_user, contact_created_by, interaction_user, opportunity_owner, task_user, contact_tag, enrichment_contact, email_template_link, opportunity_contact, task_contact, email_contact, score_contact, company_created_by, opportunity_company, interaction_contact, company_tag},
    generalizations={},
    metadata=None
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
    LineChart, BarChart, PieChart, RadarChart, RadialBarChart, Table, AgentComponent,
    Column, FieldColumn, LookupColumn, ExpressionColumn, MetricCard, Series
)
from besser.BUML.metamodel.gui.events_actions import (
    Event, EventType, Transition, Create, Read, Update, Delete, Parameter
)
from besser.BUML.metamodel.gui.binding import DataBinding

# Module: GUI_Module

# Screen: wrapper
wrapper = Screen(name="wrapper", description="Tag", view_elements=set(), is_main_page=True, route_path="/tag", screen_size="Medium")
wrapper.component_id = "page-tag-0"
io4zi = Text(
    name="io4zi",
    content="BESSER",
    description="Text element",
    styling=Styling(size=Size(font_size="24px", font_weight="bold", margin_top="0", margin_bottom="30px"), color=Color(color_palette="default")),
    component_id="io4zi",
    tag_name="h2",
    display_order=0,
    custom_attributes={"id": "io4zi"}
)
ify51 = Link(
    name="ify51",
    description="Link element",
    label="Tag",
    url="/tag",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="rgba(255,255,255,0.2)", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ify51",
    tag_name="a",
    display_order=0,
    custom_attributes={"href": "/tag", "id": "ify51"}
)
i7ha8 = Link(
    name="i7ha8",
    description="Link element",
    label="Interaction",
    url="/interaction",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i7ha8",
    tag_name="a",
    display_order=1,
    custom_attributes={"href": "/interaction", "id": "i7ha8"}
)
iqq6f = Link(
    name="iqq6f",
    description="Link element",
    label="Opportunity",
    url="/opportunity",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iqq6f",
    tag_name="a",
    display_order=2,
    custom_attributes={"href": "/opportunity", "id": "iqq6f"}
)
i3cx4 = Link(
    name="i3cx4",
    description="Link element",
    label="Contact",
    url="/contact",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i3cx4",
    tag_name="a",
    display_order=3,
    custom_attributes={"href": "/contact", "id": "i3cx4"}
)
igt5f = Link(
    name="igt5f",
    description="Link element",
    label="Company",
    url="/company",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="igt5f",
    tag_name="a",
    display_order=4,
    custom_attributes={"href": "/company", "id": "igt5f"}
)
ix3fq = Link(
    name="ix3fq",
    description="Link element",
    label="User",
    url="/user",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ix3fq",
    tag_name="a",
    display_order=5,
    custom_attributes={"href": "/user", "id": "ix3fq"}
)
i8myi = Link(
    name="i8myi",
    description="Link element",
    label="ScoreHistory",
    url="/scorehistory",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i8myi",
    tag_name="a",
    display_order=6,
    custom_attributes={"href": "/scorehistory", "id": "i8myi"}
)
ikyib = Link(
    name="ikyib",
    description="Link element",
    label="EnrichmentLog",
    url="/enrichmentlog",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ikyib",
    tag_name="a",
    display_order=7,
    custom_attributes={"href": "/enrichmentlog", "id": "ikyib"}
)
irp8u = Link(
    name="irp8u",
    description="Link element",
    label="GeneratedEmail",
    url="/generatedemail",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="irp8u",
    tag_name="a",
    display_order=8,
    custom_attributes={"href": "/generatedemail", "id": "irp8u"}
)
iyosb = Link(
    name="iyosb",
    description="Link element",
    label="EmailTemplate",
    url="/emailtemplate",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iyosb",
    tag_name="a",
    display_order=9,
    custom_attributes={"href": "/emailtemplate", "id": "iyosb"}
)
idpwj = Link(
    name="idpwj",
    description="Link element",
    label="Task",
    url="/task",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="idpwj",
    tag_name="a",
    display_order=10,
    custom_attributes={"href": "/task", "id": "idpwj"}
)
iiasi = ViewContainer(
    name="iiasi",
    description=" component",
    view_elements={ify51, i7ha8, iqq6f, i3cx4, igt5f, ix3fq, i8myi, ikyib, irp8u, iyosb, idpwj},
    styling=Styling(position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")),
    component_id="iiasi",
    display_order=1,
    custom_attributes={"id": "iiasi"}
)
iiasi_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")
iiasi.layout = iiasi_layout
ikg6r = Text(
    name="ikg6r",
    content="© 2026 BESSER. All rights reserved.",
    description="Text element",
    styling=Styling(size=Size(font_size="11px", padding_top="20px", margin_top="auto"), position=Position(alignment=Alignment.CENTER), color=Color(opacity="0.8", color_palette="default", border_top="1px solid rgba(255,255,255,0.2)")),
    component_id="ikg6r",
    display_order=2,
    custom_attributes={"id": "ikg6r"}
)
iwmic = ViewContainer(
    name="iwmic",
    description="nav container",
    view_elements={io4zi, iiasi, ikg6r},
    styling=Styling(size=Size(width="250px", padding="20px", unit_size=UnitSize.PIXELS), position=Position(display="flex", overflow_y="auto"), color=Color(background_color="linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", text_color="white", color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column")),
    component_id="iwmic",
    tag_name="nav",
    display_order=0,
    custom_attributes={"id": "iwmic"}
)
iwmic_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column")
iwmic.layout = iwmic_layout
i1x7q = Text(
    name="i1x7q",
    content="Tag",
    description="Text element",
    styling=Styling(size=Size(font_size="32px", margin_top="0", margin_bottom="10px"), color=Color(text_color="#333", color_palette="default")),
    component_id="i1x7q",
    tag_name="h1",
    display_order=0,
    custom_attributes={"id": "i1x7q"}
)
ib1n2 = Text(
    name="ib1n2",
    content="Manage Tag data",
    description="Text element",
    styling=Styling(size=Size(margin_bottom="30px"), color=Color(text_color="#666", color_palette="default")),
    component_id="ib1n2",
    tag_name="p",
    display_order=1,
    custom_attributes={"id": "ib1n2"}
)
table_tag_0_col_0 = FieldColumn(label="Name", field=Tag_name)
table_tag_0_col_1 = FieldColumn(label="Color", field=Tag_color)
table_tag_0_col_2 = FieldColumn(label="Id", field=Tag_id)
table_tag_0_col_3_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "tagged_contacts")
table_tag_0_col_3 = LookupColumn(label="Tagged Contacts", path=table_tag_0_col_3_path, field=Contact_updated_at)
table_tag_0_col_4_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "tagged_companies")
table_tag_0_col_4 = LookupColumn(label="Tagged Companies", path=table_tag_0_col_4_path, field=Company_website)
table_tag_0 = Table(
    name="table_tag_0",
    title="Tag List",
    primary_color="#2c3e50",
    show_header=True,
    striped_rows=False,
    show_pagination=True,
    rows_per_page=5,
    action_buttons=True,
    columns=[table_tag_0_col_0, table_tag_0_col_1, table_tag_0_col_2, table_tag_0_col_3, table_tag_0_col_4],
    styling=Styling(size=Size(width="100%", min_height="400px", unit_size=UnitSize.PERCENTAGE), color=Color(color_palette="default", primary_color="#2c3e50")),
    component_id="table-tag-0",
    display_order=2,
    custom_attributes={"chart-color": "#2c3e50", "chart-title": "Tag List", "data-source": "ff2a298c-cd09-447f-b868-72d4aa1d34dc", "show-header": "true", "striped-rows": "false", "show-pagination": "true", "rows-per-page": "5", "action-buttons": "true", "columns": [{'field': 'name', 'label': 'Name', 'columnType': 'field', '_expanded': False}, {'field': 'color', 'label': 'Color', 'columnType': 'field', '_expanded': False}, {'field': 'id', 'label': 'Id', 'columnType': 'field', '_expanded': False}, {'field': 'tagged_contacts', 'label': 'Tagged Contacts', 'columnType': 'lookup', 'lookupEntity': 'aaddf56f-d3cb-4eab-8223-f08b85c10f1f', 'lookupField': 'updated_at', '_expanded': False}, {'field': 'tagged_companies', 'label': 'Tagged Companies', 'columnType': 'lookup', 'lookupEntity': '9b3c4c9c-64b0-4937-98e5-bdddb729b882', 'lookupField': 'website', '_expanded': False}], "id": "table-tag-0", "filter": ""}
)
domain_model_ref = globals().get('domain_model') or next((v for k, v in globals().items() if k.startswith('domain_model') and hasattr(v, 'get_class_by_name')), None)
table_tag_0_binding_domain = None
if domain_model_ref is not None:
    table_tag_0_binding_domain = domain_model_ref.get_class_by_name("Tag")
if table_tag_0_binding_domain:
    table_tag_0_binding = DataBinding(domain_concept=table_tag_0_binding_domain, name="TagDataBinding")
else:
    # Domain class 'Tag' not resolved; data binding skipped.
    table_tag_0_binding = None
if table_tag_0_binding:
    table_tag_0.data_binding = table_tag_0_binding
ifmqf = ViewContainer(
    name="ifmqf",
    description="main container",
    view_elements={i1x7q, ib1n2, table_tag_0},
    styling=Styling(size=Size(padding="40px"), position=Position(overflow_y="auto"), color=Color(background_color="#f5f5f5", color_palette="default"), layout=Layout(flex="1")),
    component_id="ifmqf",
    tag_name="main",
    display_order=1,
    custom_attributes={"id": "ifmqf"}
)
ifmqf_layout = Layout(flex="1")
ifmqf.layout = ifmqf_layout
ixonz = ViewContainer(
    name="ixonz",
    description=" component",
    view_elements={iwmic, ifmqf},
    styling=Styling(size=Size(height="100vh", font_family="Arial, sans-serif"), position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX)),
    component_id="ixonz",
    display_order=0,
    custom_attributes={"id": "ixonz"}
)
ixonz_layout = Layout(layout_type=LayoutType.FLEX)
ixonz.layout = ixonz_layout
wrapper.view_elements = {ixonz}


# Screen: wrapper_10
wrapper_10 = Screen(name="wrapper_10", description="EmailTemplate", view_elements=set(), route_path="/emailtemplate", screen_size="Medium")
wrapper_10.component_id = "page-emailtemplate-9"
ic32b3 = Text(
    name="ic32b3",
    content="BESSER",
    description="Text element",
    styling=Styling(size=Size(font_size="24px", font_weight="bold", margin_top="0", margin_bottom="30px"), color=Color(color_palette="default")),
    component_id="ic32b3",
    tag_name="h2",
    display_order=0,
    custom_attributes={"id": "ic32b3"}
)
ikc5l5 = Link(
    name="ikc5l5",
    description="Link element",
    label="Tag",
    url="/tag",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ikc5l5",
    tag_name="a",
    display_order=0,
    custom_attributes={"href": "/tag", "id": "ikc5l5"}
)
iih2vg = Link(
    name="iih2vg",
    description="Link element",
    label="Interaction",
    url="/interaction",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iih2vg",
    tag_name="a",
    display_order=1,
    custom_attributes={"href": "/interaction", "id": "iih2vg"}
)
icxqzj = Link(
    name="icxqzj",
    description="Link element",
    label="Opportunity",
    url="/opportunity",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="icxqzj",
    tag_name="a",
    display_order=2,
    custom_attributes={"href": "/opportunity", "id": "icxqzj"}
)
ivs92f = Link(
    name="ivs92f",
    description="Link element",
    label="Contact",
    url="/contact",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ivs92f",
    tag_name="a",
    display_order=3,
    custom_attributes={"href": "/contact", "id": "ivs92f"}
)
ihqu0j = Link(
    name="ihqu0j",
    description="Link element",
    label="Company",
    url="/company",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ihqu0j",
    tag_name="a",
    display_order=4,
    custom_attributes={"href": "/company", "id": "ihqu0j"}
)
ixy4bt = Link(
    name="ixy4bt",
    description="Link element",
    label="User",
    url="/user",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ixy4bt",
    tag_name="a",
    display_order=5,
    custom_attributes={"href": "/user", "id": "ixy4bt"}
)
iracdb = Link(
    name="iracdb",
    description="Link element",
    label="ScoreHistory",
    url="/scorehistory",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iracdb",
    tag_name="a",
    display_order=6,
    custom_attributes={"href": "/scorehistory", "id": "iracdb"}
)
ihku89 = Link(
    name="ihku89",
    description="Link element",
    label="EnrichmentLog",
    url="/enrichmentlog",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ihku89",
    tag_name="a",
    display_order=7,
    custom_attributes={"href": "/enrichmentlog", "id": "ihku89"}
)
igy7s4 = Link(
    name="igy7s4",
    description="Link element",
    label="GeneratedEmail",
    url="/generatedemail",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="igy7s4",
    tag_name="a",
    display_order=8,
    custom_attributes={"href": "/generatedemail", "id": "igy7s4"}
)
itco9s = Link(
    name="itco9s",
    description="Link element",
    label="EmailTemplate",
    url="/emailtemplate",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="rgba(255,255,255,0.2)", text_color="white", color_palette="default", border_radius="4px")),
    component_id="itco9s",
    tag_name="a",
    display_order=9,
    custom_attributes={"href": "/emailtemplate", "id": "itco9s"}
)
itfpj6 = Link(
    name="itfpj6",
    description="Link element",
    label="Task",
    url="/task",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="itfpj6",
    tag_name="a",
    display_order=10,
    custom_attributes={"href": "/task", "id": "itfpj6"}
)
inh6ve = ViewContainer(
    name="inh6ve",
    description=" component",
    view_elements={ikc5l5, iih2vg, icxqzj, ivs92f, ihqu0j, ixy4bt, iracdb, ihku89, igy7s4, itco9s, itfpj6},
    styling=Styling(position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")),
    component_id="inh6ve",
    display_order=1,
    custom_attributes={"id": "inh6ve"}
)
inh6ve_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")
inh6ve.layout = inh6ve_layout
ijn4li = Text(
    name="ijn4li",
    content="© 2026 BESSER. All rights reserved.",
    description="Text element",
    styling=Styling(size=Size(font_size="11px", padding_top="20px", margin_top="auto"), position=Position(alignment=Alignment.CENTER), color=Color(opacity="0.8", color_palette="default", border_top="1px solid rgba(255,255,255,0.2)")),
    component_id="ijn4li",
    display_order=2,
    custom_attributes={"id": "ijn4li"}
)
im41yv = ViewContainer(
    name="im41yv",
    description="nav container",
    view_elements={ic32b3, inh6ve, ijn4li},
    styling=Styling(size=Size(width="250px", padding="20px", unit_size=UnitSize.PIXELS), position=Position(display="flex", overflow_y="auto"), color=Color(background_color="linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", text_color="white", color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column")),
    component_id="im41yv",
    tag_name="nav",
    display_order=0,
    custom_attributes={"id": "im41yv"}
)
im41yv_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column")
im41yv.layout = im41yv_layout
ixqzqs = Text(
    name="ixqzqs",
    content="EmailTemplate",
    description="Text element",
    styling=Styling(size=Size(font_size="32px", margin_top="0", margin_bottom="10px"), color=Color(text_color="#333", color_palette="default")),
    component_id="ixqzqs",
    tag_name="h1",
    display_order=0,
    custom_attributes={"id": "ixqzqs"}
)
igkppr = Text(
    name="igkppr",
    content="Manage EmailTemplate data",
    description="Text element",
    styling=Styling(size=Size(margin_bottom="30px"), color=Color(text_color="#666", color_palette="default")),
    component_id="igkppr",
    tag_name="p",
    display_order=1,
    custom_attributes={"id": "igkppr"}
)
table_emailtemplate_9_col_0 = FieldColumn(label="Created At", field=EmailTemplate_created_at)
table_emailtemplate_9_col_1 = FieldColumn(label="Category", field=EmailTemplate_category)
table_emailtemplate_9_col_2 = FieldColumn(label="Body Template", field=EmailTemplate_body_template)
table_emailtemplate_9_col_3 = FieldColumn(label="Subject Template", field=EmailTemplate_subject_template)
table_emailtemplate_9_col_4 = FieldColumn(label="Id", field=EmailTemplate_id)
table_emailtemplate_9_col_5 = FieldColumn(label="Name", field=EmailTemplate_name)
table_emailtemplate_9_col_6_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "created_by")
table_emailtemplate_9_col_6 = LookupColumn(label="Created By", path=table_emailtemplate_9_col_6_path, field=User_last_name)
table_emailtemplate_9_col_7_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "generated_emails")
table_emailtemplate_9_col_7 = LookupColumn(label="Generated Emails", path=table_emailtemplate_9_col_7_path, field=GeneratedEmail_created_at)
table_emailtemplate_9 = Table(
    name="table_emailtemplate_9",
    title="EmailTemplate List",
    primary_color="#2c3e50",
    show_header=True,
    striped_rows=False,
    show_pagination=True,
    rows_per_page=5,
    action_buttons=True,
    columns=[table_emailtemplate_9_col_0, table_emailtemplate_9_col_1, table_emailtemplate_9_col_2, table_emailtemplate_9_col_3, table_emailtemplate_9_col_4, table_emailtemplate_9_col_5, table_emailtemplate_9_col_6, table_emailtemplate_9_col_7],
    styling=Styling(size=Size(width="100%", min_height="400px", unit_size=UnitSize.PERCENTAGE), color=Color(color_palette="default", primary_color="#2c3e50")),
    component_id="table-emailtemplate-9",
    display_order=2,
    custom_attributes={"chart-color": "#2c3e50", "chart-title": "EmailTemplate List", "data-source": "2d584aa6-7ca2-4d21-8510-57f70750bfb4", "show-header": "true", "striped-rows": "false", "show-pagination": "true", "rows-per-page": "5", "action-buttons": "true", "columns": [{'field': 'created_at', 'label': 'Created At', 'columnType': 'field', '_expanded': False}, {'field': 'category', 'label': 'Category', 'columnType': 'field', '_expanded': False}, {'field': 'body_template', 'label': 'Body Template', 'columnType': 'field', '_expanded': False}, {'field': 'subject_template', 'label': 'Subject Template', 'columnType': 'field', '_expanded': False}, {'field': 'id', 'label': 'Id', 'columnType': 'field', '_expanded': False}, {'field': 'name', 'label': 'Name', 'columnType': 'field', '_expanded': False}, {'field': 'created_by', 'label': 'Created By', 'columnType': 'lookup', 'lookupEntity': '278327b9-3f33-4fcc-aa98-c144c0933a65', 'lookupField': 'last_name', '_expanded': False}, {'field': 'generated_emails', 'label': 'Generated Emails', 'columnType': 'lookup', 'lookupEntity': '72d64804-891c-438b-bfed-6ecee0bfd085', 'lookupField': 'created_at', '_expanded': False}], "id": "table-emailtemplate-9", "filter": ""}
)
domain_model_ref = globals().get('domain_model') or next((v for k, v in globals().items() if k.startswith('domain_model') and hasattr(v, 'get_class_by_name')), None)
table_emailtemplate_9_binding_domain = None
if domain_model_ref is not None:
    table_emailtemplate_9_binding_domain = domain_model_ref.get_class_by_name("EmailTemplate")
if table_emailtemplate_9_binding_domain:
    table_emailtemplate_9_binding = DataBinding(domain_concept=table_emailtemplate_9_binding_domain, name="EmailTemplateDataBinding")
else:
    # Domain class 'EmailTemplate' not resolved; data binding skipped.
    table_emailtemplate_9_binding = None
if table_emailtemplate_9_binding:
    table_emailtemplate_9.data_binding = table_emailtemplate_9_binding
iz2c5t = ViewContainer(
    name="iz2c5t",
    description="main container",
    view_elements={ixqzqs, igkppr, table_emailtemplate_9},
    styling=Styling(size=Size(padding="40px"), position=Position(overflow_y="auto"), color=Color(background_color="#f5f5f5", color_palette="default"), layout=Layout(flex="1")),
    component_id="iz2c5t",
    tag_name="main",
    display_order=1,
    custom_attributes={"id": "iz2c5t"}
)
iz2c5t_layout = Layout(flex="1")
iz2c5t.layout = iz2c5t_layout
i54b02 = ViewContainer(
    name="i54b02",
    description=" component",
    view_elements={im41yv, iz2c5t},
    styling=Styling(size=Size(height="100vh", font_family="Arial, sans-serif"), position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX)),
    component_id="i54b02",
    display_order=0,
    custom_attributes={"id": "i54b02"}
)
i54b02_layout = Layout(layout_type=LayoutType.FLEX)
i54b02.layout = i54b02_layout
wrapper_10.view_elements = {i54b02}


# Screen: wrapper_11
wrapper_11 = Screen(name="wrapper_11", description="Task", view_elements=set(), route_path="/task", screen_size="Medium")
wrapper_11.component_id = "page-task-10"
ipvl06 = Text(
    name="ipvl06",
    content="BESSER",
    description="Text element",
    styling=Styling(size=Size(font_size="24px", font_weight="bold", margin_top="0", margin_bottom="30px"), color=Color(color_palette="default")),
    component_id="ipvl06",
    tag_name="h2",
    display_order=0,
    custom_attributes={"id": "ipvl06"}
)
iu9d9f = Link(
    name="iu9d9f",
    description="Link element",
    label="Tag",
    url="/tag",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iu9d9f",
    tag_name="a",
    display_order=0,
    custom_attributes={"href": "/tag", "id": "iu9d9f"}
)
ivk5sl = Link(
    name="ivk5sl",
    description="Link element",
    label="Interaction",
    url="/interaction",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ivk5sl",
    tag_name="a",
    display_order=1,
    custom_attributes={"href": "/interaction", "id": "ivk5sl"}
)
idp1rz = Link(
    name="idp1rz",
    description="Link element",
    label="Opportunity",
    url="/opportunity",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="idp1rz",
    tag_name="a",
    display_order=2,
    custom_attributes={"href": "/opportunity", "id": "idp1rz"}
)
ifxw2k = Link(
    name="ifxw2k",
    description="Link element",
    label="Contact",
    url="/contact",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ifxw2k",
    tag_name="a",
    display_order=3,
    custom_attributes={"href": "/contact", "id": "ifxw2k"}
)
il4bmp = Link(
    name="il4bmp",
    description="Link element",
    label="Company",
    url="/company",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="il4bmp",
    tag_name="a",
    display_order=4,
    custom_attributes={"href": "/company", "id": "il4bmp"}
)
ilfx85 = Link(
    name="ilfx85",
    description="Link element",
    label="User",
    url="/user",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ilfx85",
    tag_name="a",
    display_order=5,
    custom_attributes={"href": "/user", "id": "ilfx85"}
)
ily41t = Link(
    name="ily41t",
    description="Link element",
    label="ScoreHistory",
    url="/scorehistory",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ily41t",
    tag_name="a",
    display_order=6,
    custom_attributes={"href": "/scorehistory", "id": "ily41t"}
)
ijlf89 = Link(
    name="ijlf89",
    description="Link element",
    label="EnrichmentLog",
    url="/enrichmentlog",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ijlf89",
    tag_name="a",
    display_order=7,
    custom_attributes={"href": "/enrichmentlog", "id": "ijlf89"}
)
iks0u8 = Link(
    name="iks0u8",
    description="Link element",
    label="GeneratedEmail",
    url="/generatedemail",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iks0u8",
    tag_name="a",
    display_order=8,
    custom_attributes={"href": "/generatedemail", "id": "iks0u8"}
)
ikqfnw = Link(
    name="ikqfnw",
    description="Link element",
    label="EmailTemplate",
    url="/emailtemplate",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ikqfnw",
    tag_name="a",
    display_order=9,
    custom_attributes={"href": "/emailtemplate", "id": "ikqfnw"}
)
in7kf9 = Link(
    name="in7kf9",
    description="Link element",
    label="Task",
    url="/task",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="rgba(255,255,255,0.2)", text_color="white", color_palette="default", border_radius="4px")),
    component_id="in7kf9",
    tag_name="a",
    display_order=10,
    custom_attributes={"href": "/task", "id": "in7kf9"}
)
ii42wf = ViewContainer(
    name="ii42wf",
    description=" component",
    view_elements={iu9d9f, ivk5sl, idp1rz, ifxw2k, il4bmp, ilfx85, ily41t, ijlf89, iks0u8, ikqfnw, in7kf9},
    styling=Styling(position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")),
    component_id="ii42wf",
    display_order=1,
    custom_attributes={"id": "ii42wf"}
)
ii42wf_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")
ii42wf.layout = ii42wf_layout
ib6s1l = Text(
    name="ib6s1l",
    content="© 2026 BESSER. All rights reserved.",
    description="Text element",
    styling=Styling(size=Size(font_size="11px", padding_top="20px", margin_top="auto"), position=Position(alignment=Alignment.CENTER), color=Color(opacity="0.8", color_palette="default", border_top="1px solid rgba(255,255,255,0.2)")),
    component_id="ib6s1l",
    display_order=2,
    custom_attributes={"id": "ib6s1l"}
)
iqmvue = ViewContainer(
    name="iqmvue",
    description="nav container",
    view_elements={ipvl06, ii42wf, ib6s1l},
    styling=Styling(size=Size(width="250px", padding="20px", unit_size=UnitSize.PIXELS), position=Position(display="flex", overflow_y="auto"), color=Color(background_color="linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", text_color="white", color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column")),
    component_id="iqmvue",
    tag_name="nav",
    display_order=0,
    custom_attributes={"id": "iqmvue"}
)
iqmvue_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column")
iqmvue.layout = iqmvue_layout
i3y94z = Text(
    name="i3y94z",
    content="Task",
    description="Text element",
    styling=Styling(size=Size(font_size="32px", margin_top="0", margin_bottom="10px"), color=Color(text_color="#333", color_palette="default")),
    component_id="i3y94z",
    tag_name="h1",
    display_order=0,
    custom_attributes={"id": "i3y94z"}
)
ia6eeb = Text(
    name="ia6eeb",
    content="Manage Task data",
    description="Text element",
    styling=Styling(size=Size(margin_bottom="30px"), color=Color(text_color="#666", color_palette="default")),
    component_id="ia6eeb",
    tag_name="p",
    display_order=1,
    custom_attributes={"id": "ia6eeb"}
)
table_task_10_col_0 = FieldColumn(label="Description", field=Task_description)
table_task_10_col_1 = FieldColumn(label="Due Date", field=Task_due_date)
table_task_10_col_2 = FieldColumn(label="Created At", field=Task_created_at)
table_task_10_col_3 = FieldColumn(label="Title", field=Task_title)
table_task_10_col_4 = FieldColumn(label="Id", field=Task_id)
table_task_10_col_5 = FieldColumn(label="Completed At", field=Task_completed_at)
table_task_10_col_6 = FieldColumn(label="Is Completed", field=Task_is_completed)
table_task_10_col_7_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "opportunity")
table_task_10_col_7 = LookupColumn(label="Opportunity", path=table_task_10_col_7_path, field=Opportunity_expected_close_date)
table_task_10_col_8_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "contact")
table_task_10_col_8 = LookupColumn(label="Contact", path=table_task_10_col_8_path, field=Contact_updated_at)
table_task_10_col_9_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "assigned_to")
table_task_10_col_9 = LookupColumn(label="Assigned To", path=table_task_10_col_9_path, field=User_last_name)
table_task_10 = Table(
    name="table_task_10",
    title="Task List",
    primary_color="#2c3e50",
    show_header=True,
    striped_rows=False,
    show_pagination=True,
    rows_per_page=5,
    action_buttons=True,
    columns=[table_task_10_col_0, table_task_10_col_1, table_task_10_col_2, table_task_10_col_3, table_task_10_col_4, table_task_10_col_5, table_task_10_col_6, table_task_10_col_7, table_task_10_col_8, table_task_10_col_9],
    styling=Styling(size=Size(width="100%", min_height="400px", unit_size=UnitSize.PERCENTAGE), color=Color(color_palette="default", primary_color="#2c3e50")),
    component_id="table-task-10",
    display_order=2,
    css_classes=["has-data-binding"],
    custom_attributes={"chart-color": "#2c3e50", "chart-title": "Task List", "data-source": "fb02ddde-ec4d-4181-bc34-7cabad68eca1", "show-header": "true", "striped-rows": "false", "show-pagination": "true", "rows-per-page": "5", "action-buttons": "true", "columns": [{'field': 'description', 'label': 'Description', 'columnType': 'field', '_expanded': False}, {'field': 'due_date', 'label': 'Due Date', 'columnType': 'field', '_expanded': False}, {'field': 'created_at', 'label': 'Created At', 'columnType': 'field', '_expanded': False}, {'field': 'title', 'label': 'Title', 'columnType': 'field', '_expanded': False}, {'field': 'id', 'label': 'Id', 'columnType': 'field', '_expanded': False}, {'field': 'completed_at', 'label': 'Completed At', 'columnType': 'field', '_expanded': False}, {'field': 'is_completed', 'label': 'Is Completed', 'columnType': 'field', '_expanded': False}, {'field': 'opportunity', 'label': 'Opportunity', 'columnType': 'lookup', 'lookupEntity': '00c036a4-454e-4449-9a9b-6b3f03969d49', 'lookupField': 'expected_close_date', '_expanded': False}, {'field': 'contact', 'label': 'Contact', 'columnType': 'lookup', 'lookupEntity': 'aaddf56f-d3cb-4eab-8223-f08b85c10f1f', 'lookupField': 'updated_at', '_expanded': False}, {'field': 'assigned_to', 'label': 'Assigned To', 'columnType': 'lookup', 'lookupEntity': '278327b9-3f33-4fcc-aa98-c144c0933a65', 'lookupField': 'last_name', '_expanded': False}], "id": "table-task-10", "filter": ""}
)
domain_model_ref = globals().get('domain_model') or next((v for k, v in globals().items() if k.startswith('domain_model') and hasattr(v, 'get_class_by_name')), None)
table_task_10_binding_domain = None
if domain_model_ref is not None:
    table_task_10_binding_domain = domain_model_ref.get_class_by_name("Task")
if table_task_10_binding_domain:
    table_task_10_binding = DataBinding(domain_concept=table_task_10_binding_domain, name="TaskDataBinding")
else:
    # Domain class 'Task' not resolved; data binding skipped.
    table_task_10_binding = None
if table_task_10_binding:
    table_task_10.data_binding = table_task_10_binding
ix9l2b = ViewContainer(
    name="ix9l2b",
    description="main container",
    view_elements={i3y94z, ia6eeb, table_task_10},
    styling=Styling(size=Size(padding="40px"), position=Position(overflow_y="auto"), color=Color(background_color="#f5f5f5", color_palette="default"), layout=Layout(flex="1")),
    component_id="ix9l2b",
    tag_name="main",
    display_order=1,
    custom_attributes={"id": "ix9l2b"}
)
ix9l2b_layout = Layout(flex="1")
ix9l2b.layout = ix9l2b_layout
itj953 = ViewContainer(
    name="itj953",
    description=" component",
    view_elements={iqmvue, ix9l2b},
    styling=Styling(size=Size(height="100vh", font_family="Arial, sans-serif"), position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX)),
    component_id="itj953",
    display_order=0,
    custom_attributes={"id": "itj953"}
)
itj953_layout = Layout(layout_type=LayoutType.FLEX)
itj953.layout = itj953_layout
wrapper_11.view_elements = {itj953}


# Screen: wrapper_2
wrapper_2 = Screen(name="wrapper_2", description="Interaction", view_elements=set(), route_path="/interaction", screen_size="Medium")
wrapper_2.component_id = "page-interaction-1"
iiptx = Text(
    name="iiptx",
    content="BESSER",
    description="Text element",
    styling=Styling(size=Size(font_size="24px", font_weight="bold", margin_top="0", margin_bottom="30px"), color=Color(color_palette="default")),
    component_id="iiptx",
    tag_name="h2",
    display_order=0,
    custom_attributes={"id": "iiptx"}
)
i97b4 = Link(
    name="i97b4",
    description="Link element",
    label="Tag",
    url="/tag",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i97b4",
    tag_name="a",
    display_order=0,
    custom_attributes={"href": "/tag", "id": "i97b4"}
)
i8hbn = Link(
    name="i8hbn",
    description="Link element",
    label="Interaction",
    url="/interaction",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="rgba(255,255,255,0.2)", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i8hbn",
    tag_name="a",
    display_order=1,
    custom_attributes={"href": "/interaction", "id": "i8hbn"}
)
if56q = Link(
    name="if56q",
    description="Link element",
    label="Opportunity",
    url="/opportunity",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="if56q",
    tag_name="a",
    display_order=2,
    custom_attributes={"href": "/opportunity", "id": "if56q"}
)
iw2q9 = Link(
    name="iw2q9",
    description="Link element",
    label="Contact",
    url="/contact",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iw2q9",
    tag_name="a",
    display_order=3,
    custom_attributes={"href": "/contact", "id": "iw2q9"}
)
ik7kj = Link(
    name="ik7kj",
    description="Link element",
    label="Company",
    url="/company",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ik7kj",
    tag_name="a",
    display_order=4,
    custom_attributes={"href": "/company", "id": "ik7kj"}
)
ijiaa = Link(
    name="ijiaa",
    description="Link element",
    label="User",
    url="/user",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ijiaa",
    tag_name="a",
    display_order=5,
    custom_attributes={"href": "/user", "id": "ijiaa"}
)
i81mu = Link(
    name="i81mu",
    description="Link element",
    label="ScoreHistory",
    url="/scorehistory",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i81mu",
    tag_name="a",
    display_order=6,
    custom_attributes={"href": "/scorehistory", "id": "i81mu"}
)
irjin = Link(
    name="irjin",
    description="Link element",
    label="EnrichmentLog",
    url="/enrichmentlog",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="irjin",
    tag_name="a",
    display_order=7,
    custom_attributes={"href": "/enrichmentlog", "id": "irjin"}
)
i96c5w = Link(
    name="i96c5w",
    description="Link element",
    label="GeneratedEmail",
    url="/generatedemail",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i96c5w",
    tag_name="a",
    display_order=8,
    custom_attributes={"href": "/generatedemail", "id": "i96c5w"}
)
i63tfv = Link(
    name="i63tfv",
    description="Link element",
    label="EmailTemplate",
    url="/emailtemplate",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i63tfv",
    tag_name="a",
    display_order=9,
    custom_attributes={"href": "/emailtemplate", "id": "i63tfv"}
)
izdxjk = Link(
    name="izdxjk",
    description="Link element",
    label="Task",
    url="/task",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="izdxjk",
    tag_name="a",
    display_order=10,
    custom_attributes={"href": "/task", "id": "izdxjk"}
)
iel0k = ViewContainer(
    name="iel0k",
    description=" component",
    view_elements={i97b4, i8hbn, if56q, iw2q9, ik7kj, ijiaa, i81mu, irjin, i96c5w, i63tfv, izdxjk},
    styling=Styling(position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")),
    component_id="iel0k",
    display_order=1,
    custom_attributes={"id": "iel0k"}
)
iel0k_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")
iel0k.layout = iel0k_layout
iq3e8n = Text(
    name="iq3e8n",
    content="© 2026 BESSER. All rights reserved.",
    description="Text element",
    styling=Styling(size=Size(font_size="11px", padding_top="20px", margin_top="auto"), position=Position(alignment=Alignment.CENTER), color=Color(opacity="0.8", color_palette="default", border_top="1px solid rgba(255,255,255,0.2)")),
    component_id="iq3e8n",
    display_order=2,
    custom_attributes={"id": "iq3e8n"}
)
i1gti = ViewContainer(
    name="i1gti",
    description="nav container",
    view_elements={iiptx, iel0k, iq3e8n},
    styling=Styling(size=Size(width="250px", padding="20px", unit_size=UnitSize.PIXELS), position=Position(display="flex", overflow_y="auto"), color=Color(background_color="linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", text_color="white", color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column")),
    component_id="i1gti",
    tag_name="nav",
    display_order=0,
    custom_attributes={"id": "i1gti"}
)
i1gti_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column")
i1gti.layout = i1gti_layout
if3gj1 = Text(
    name="if3gj1",
    content="Interaction",
    description="Text element",
    styling=Styling(size=Size(font_size="32px", margin_top="0", margin_bottom="10px"), color=Color(text_color="#333", color_palette="default")),
    component_id="if3gj1",
    tag_name="h1",
    display_order=0,
    custom_attributes={"id": "if3gj1"}
)
io1pup = Text(
    name="io1pup",
    content="Manage Interaction data",
    description="Text element",
    styling=Styling(size=Size(margin_bottom="30px"), color=Color(text_color="#666", color_palette="default")),
    component_id="io1pup",
    tag_name="p",
    display_order=1,
    custom_attributes={"id": "io1pup"}
)
table_interaction_1_col_0 = FieldColumn(label="Created At", field=Interaction_created_at)
table_interaction_1_col_1 = FieldColumn(label="Subject", field=Interaction_subject)
table_interaction_1_col_2 = FieldColumn(label="Type", field=Interaction_type)
table_interaction_1_col_3 = FieldColumn(label="Content", field=Interaction_content)
table_interaction_1_col_4 = FieldColumn(label="Occurred At", field=Interaction_occurred_at)
table_interaction_1_col_5 = FieldColumn(label="Id", field=Interaction_id)
table_interaction_1_col_6 = FieldColumn(label="Direction", field=Interaction_direction)
table_interaction_1_col_7_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "performed_by")
table_interaction_1_col_7 = LookupColumn(label="Performed By", path=table_interaction_1_col_7_path, field=User_last_name)
table_interaction_1 = Table(
    name="table_interaction_1",
    title="Interaction List",
    primary_color="#2c3e50",
    show_header=True,
    striped_rows=False,
    show_pagination=True,
    rows_per_page=5,
    action_buttons=True,
    columns=[table_interaction_1_col_0, table_interaction_1_col_1, table_interaction_1_col_2, table_interaction_1_col_3, table_interaction_1_col_4, table_interaction_1_col_5, table_interaction_1_col_6, table_interaction_1_col_7],
    styling=Styling(size=Size(width="100%", min_height="400px", unit_size=UnitSize.PERCENTAGE), color=Color(color_palette="default", primary_color="#2c3e50")),
    component_id="table-interaction-1",
    display_order=2,
    custom_attributes={"chart-color": "#2c3e50", "chart-title": "Interaction List", "data-source": "53490d70-1ecb-4595-89f0-a2309285454e", "show-header": "true", "striped-rows": "false", "show-pagination": "true", "rows-per-page": "5", "action-buttons": "true", "columns": [{'field': 'created_at', 'label': 'Created At', 'columnType': 'field', '_expanded': False}, {'field': 'subject', 'label': 'Subject', 'columnType': 'field', '_expanded': False}, {'field': 'type', 'label': 'Type', 'columnType': 'field', '_expanded': False}, {'field': 'content', 'label': 'Content', 'columnType': 'field', '_expanded': False}, {'field': 'occurred_at', 'label': 'Occurred At', 'columnType': 'field', '_expanded': False}, {'field': 'id', 'label': 'Id', 'columnType': 'field', '_expanded': False}, {'field': 'direction', 'label': 'Direction', 'columnType': 'field', '_expanded': False}, {'field': 'performed_by', 'label': 'Performed By', 'columnType': 'lookup', 'lookupEntity': '278327b9-3f33-4fcc-aa98-c144c0933a65', 'lookupField': 'last_name', '_expanded': False}], "id": "table-interaction-1", "filter": ""}
)
domain_model_ref = globals().get('domain_model') or next((v for k, v in globals().items() if k.startswith('domain_model') and hasattr(v, 'get_class_by_name')), None)
table_interaction_1_binding_domain = None
if domain_model_ref is not None:
    table_interaction_1_binding_domain = domain_model_ref.get_class_by_name("Interaction")
if table_interaction_1_binding_domain:
    table_interaction_1_binding = DataBinding(domain_concept=table_interaction_1_binding_domain, name="InteractionDataBinding")
else:
    # Domain class 'Interaction' not resolved; data binding skipped.
    table_interaction_1_binding = None
if table_interaction_1_binding:
    table_interaction_1.data_binding = table_interaction_1_binding
iuzqk9 = ViewContainer(
    name="iuzqk9",
    description="main container",
    view_elements={if3gj1, io1pup, table_interaction_1},
    styling=Styling(size=Size(padding="40px"), position=Position(overflow_y="auto"), color=Color(background_color="#f5f5f5", color_palette="default"), layout=Layout(flex="1")),
    component_id="iuzqk9",
    tag_name="main",
    display_order=1,
    custom_attributes={"id": "iuzqk9"}
)
iuzqk9_layout = Layout(flex="1")
iuzqk9.layout = iuzqk9_layout
ixacu = ViewContainer(
    name="ixacu",
    description=" component",
    view_elements={i1gti, iuzqk9},
    styling=Styling(size=Size(height="100vh", font_family="Arial, sans-serif"), position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX)),
    component_id="ixacu",
    display_order=0,
    custom_attributes={"id": "ixacu"}
)
ixacu_layout = Layout(layout_type=LayoutType.FLEX)
ixacu.layout = ixacu_layout
wrapper_2.view_elements = {ixacu}


# Screen: wrapper_3
wrapper_3 = Screen(name="wrapper_3", description="Opportunity", view_elements=set(), route_path="/opportunity", screen_size="Medium")
wrapper_3.component_id = "page-opportunity-2"
i7mb6s = Text(
    name="i7mb6s",
    content="BESSER",
    description="Text element",
    styling=Styling(size=Size(font_size="24px", font_weight="bold", margin_top="0", margin_bottom="30px"), color=Color(color_palette="default")),
    component_id="i7mb6s",
    tag_name="h2",
    display_order=0,
    custom_attributes={"id": "i7mb6s"}
)
i1fdwi = Link(
    name="i1fdwi",
    description="Link element",
    label="Tag",
    url="/tag",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i1fdwi",
    tag_name="a",
    display_order=0,
    custom_attributes={"href": "/tag", "id": "i1fdwi"}
)
inugeg = Link(
    name="inugeg",
    description="Link element",
    label="Interaction",
    url="/interaction",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="inugeg",
    tag_name="a",
    display_order=1,
    custom_attributes={"href": "/interaction", "id": "inugeg"}
)
iut6fr = Link(
    name="iut6fr",
    description="Link element",
    label="Opportunity",
    url="/opportunity",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="rgba(255,255,255,0.2)", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iut6fr",
    tag_name="a",
    display_order=2,
    custom_attributes={"href": "/opportunity", "id": "iut6fr"}
)
id33ya = Link(
    name="id33ya",
    description="Link element",
    label="Contact",
    url="/contact",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="id33ya",
    tag_name="a",
    display_order=3,
    custom_attributes={"href": "/contact", "id": "id33ya"}
)
i69yw8 = Link(
    name="i69yw8",
    description="Link element",
    label="Company",
    url="/company",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i69yw8",
    tag_name="a",
    display_order=4,
    custom_attributes={"href": "/company", "id": "i69yw8"}
)
ivga7d = Link(
    name="ivga7d",
    description="Link element",
    label="User",
    url="/user",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ivga7d",
    tag_name="a",
    display_order=5,
    custom_attributes={"href": "/user", "id": "ivga7d"}
)
ieu13m = Link(
    name="ieu13m",
    description="Link element",
    label="ScoreHistory",
    url="/scorehistory",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ieu13m",
    tag_name="a",
    display_order=6,
    custom_attributes={"href": "/scorehistory", "id": "ieu13m"}
)
ix7m6i = Link(
    name="ix7m6i",
    description="Link element",
    label="EnrichmentLog",
    url="/enrichmentlog",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ix7m6i",
    tag_name="a",
    display_order=7,
    custom_attributes={"href": "/enrichmentlog", "id": "ix7m6i"}
)
i25cz4 = Link(
    name="i25cz4",
    description="Link element",
    label="GeneratedEmail",
    url="/generatedemail",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i25cz4",
    tag_name="a",
    display_order=8,
    custom_attributes={"href": "/generatedemail", "id": "i25cz4"}
)
i1eifw = Link(
    name="i1eifw",
    description="Link element",
    label="EmailTemplate",
    url="/emailtemplate",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i1eifw",
    tag_name="a",
    display_order=9,
    custom_attributes={"href": "/emailtemplate", "id": "i1eifw"}
)
i6o0ff = Link(
    name="i6o0ff",
    description="Link element",
    label="Task",
    url="/task",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i6o0ff",
    tag_name="a",
    display_order=10,
    custom_attributes={"href": "/task", "id": "i6o0ff"}
)
i3ufz6 = ViewContainer(
    name="i3ufz6",
    description=" component",
    view_elements={i1fdwi, inugeg, iut6fr, id33ya, i69yw8, ivga7d, ieu13m, ix7m6i, i25cz4, i1eifw, i6o0ff},
    styling=Styling(position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")),
    component_id="i3ufz6",
    display_order=1,
    custom_attributes={"id": "i3ufz6"}
)
i3ufz6_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")
i3ufz6.layout = i3ufz6_layout
iliksi = Text(
    name="iliksi",
    content="© 2026 BESSER. All rights reserved.",
    description="Text element",
    styling=Styling(size=Size(font_size="11px", padding_top="20px", margin_top="auto"), position=Position(alignment=Alignment.CENTER), color=Color(opacity="0.8", color_palette="default", border_top="1px solid rgba(255,255,255,0.2)")),
    component_id="iliksi",
    display_order=2,
    custom_attributes={"id": "iliksi"}
)
icjhn8 = ViewContainer(
    name="icjhn8",
    description="nav container",
    view_elements={i7mb6s, i3ufz6, iliksi},
    styling=Styling(size=Size(width="250px", padding="20px", unit_size=UnitSize.PIXELS), position=Position(display="flex", overflow_y="auto"), color=Color(background_color="linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", text_color="white", color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column")),
    component_id="icjhn8",
    tag_name="nav",
    display_order=0,
    custom_attributes={"id": "icjhn8"}
)
icjhn8_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column")
icjhn8.layout = icjhn8_layout
izmzig = Text(
    name="izmzig",
    content="Opportunity",
    description="Text element",
    styling=Styling(size=Size(font_size="32px", margin_top="0", margin_bottom="10px"), color=Color(text_color="#333", color_palette="default")),
    component_id="izmzig",
    tag_name="h1",
    display_order=0,
    custom_attributes={"id": "izmzig"}
)
i0kzeq = Text(
    name="i0kzeq",
    content="Manage Opportunity data",
    description="Text element",
    styling=Styling(size=Size(margin_bottom="30px"), color=Color(text_color="#666", color_palette="default")),
    component_id="i0kzeq",
    tag_name="p",
    display_order=1,
    custom_attributes={"id": "i0kzeq"}
)
table_opportunity_2_col_0 = FieldColumn(label="Expected Close Date", field=Opportunity_expected_close_date)
table_opportunity_2_col_1 = FieldColumn(label="Id", field=Opportunity_id)
table_opportunity_2_col_2 = FieldColumn(label="Probability", field=Opportunity_probability)
table_opportunity_2_col_3 = FieldColumn(label="Value", field=Opportunity_value)
table_opportunity_2_col_4 = FieldColumn(label="Closed At", field=Opportunity_closed_at)
table_opportunity_2_col_5 = FieldColumn(label="Stage", field=Opportunity_stage)
table_opportunity_2_col_6 = FieldColumn(label="Updated At", field=Opportunity_updated_at)
table_opportunity_2_col_7 = FieldColumn(label="Description", field=Opportunity_description)
table_opportunity_2_col_8 = FieldColumn(label="Created At", field=Opportunity_created_at)
table_opportunity_2_col_9 = FieldColumn(label="Title", field=Opportunity_title)
table_opportunity_2_col_10_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "tasks")
table_opportunity_2_col_10 = LookupColumn(label="Tasks", path=table_opportunity_2_col_10_path, field=Task_description)
table_opportunity_2_col_11_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "owner")
table_opportunity_2_col_11 = LookupColumn(label="Owner", path=table_opportunity_2_col_11_path, field=User_last_name)
table_opportunity_2_col_12_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "company")
table_opportunity_2_col_12 = LookupColumn(label="Company", path=table_opportunity_2_col_12_path, field=Company_website)
table_opportunity_2_col_13_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "contacts")
table_opportunity_2_col_13 = LookupColumn(label="Contacts", path=table_opportunity_2_col_13_path, field=Contact_updated_at)
table_opportunity_2 = Table(
    name="table_opportunity_2",
    title="Opportunity List",
    primary_color="#2c3e50",
    show_header=True,
    striped_rows=False,
    show_pagination=True,
    rows_per_page=5,
    action_buttons=True,
    columns=[table_opportunity_2_col_0, table_opportunity_2_col_1, table_opportunity_2_col_2, table_opportunity_2_col_3, table_opportunity_2_col_4, table_opportunity_2_col_5, table_opportunity_2_col_6, table_opportunity_2_col_7, table_opportunity_2_col_8, table_opportunity_2_col_9, table_opportunity_2_col_10, table_opportunity_2_col_11, table_opportunity_2_col_12, table_opportunity_2_col_13],
    styling=Styling(size=Size(width="100%", min_height="400px", unit_size=UnitSize.PERCENTAGE), color=Color(color_palette="default", primary_color="#2c3e50")),
    component_id="table-opportunity-2",
    display_order=2,
    custom_attributes={"chart-color": "#2c3e50", "chart-title": "Opportunity List", "data-source": "00c036a4-454e-4449-9a9b-6b3f03969d49", "show-header": "true", "striped-rows": "false", "show-pagination": "true", "rows-per-page": "5", "action-buttons": "true", "columns": [{'field': 'expected_close_date', 'label': 'Expected Close Date', 'columnType': 'field', '_expanded': False}, {'field': 'id', 'label': 'Id', 'columnType': 'field', '_expanded': False}, {'field': 'probability', 'label': 'Probability', 'columnType': 'field', '_expanded': False}, {'field': 'value', 'label': 'Value', 'columnType': 'field', '_expanded': False}, {'field': 'closed_at', 'label': 'Closed At', 'columnType': 'field', '_expanded': False}, {'field': 'stage', 'label': 'Stage', 'columnType': 'field', '_expanded': False}, {'field': 'updated_at', 'label': 'Updated At', 'columnType': 'field', '_expanded': False}, {'field': 'description', 'label': 'Description', 'columnType': 'field', '_expanded': False}, {'field': 'created_at', 'label': 'Created At', 'columnType': 'field', '_expanded': False}, {'field': 'title', 'label': 'Title', 'columnType': 'field', '_expanded': False}, {'field': 'tasks', 'label': 'Tasks', 'columnType': 'lookup', 'lookupEntity': 'fb02ddde-ec4d-4181-bc34-7cabad68eca1', 'lookupField': 'description', '_expanded': False}, {'field': 'owner', 'label': 'Owner', 'columnType': 'lookup', 'lookupEntity': '278327b9-3f33-4fcc-aa98-c144c0933a65', 'lookupField': 'last_name', '_expanded': False}, {'field': 'company', 'label': 'Company', 'columnType': 'lookup', 'lookupEntity': '9b3c4c9c-64b0-4937-98e5-bdddb729b882', 'lookupField': 'website', '_expanded': False}, {'field': 'contacts', 'label': 'Contacts', 'columnType': 'lookup', 'lookupEntity': 'aaddf56f-d3cb-4eab-8223-f08b85c10f1f', 'lookupField': 'updated_at', '_expanded': False}], "id": "table-opportunity-2", "filter": ""}
)
domain_model_ref = globals().get('domain_model') or next((v for k, v in globals().items() if k.startswith('domain_model') and hasattr(v, 'get_class_by_name')), None)
table_opportunity_2_binding_domain = None
if domain_model_ref is not None:
    table_opportunity_2_binding_domain = domain_model_ref.get_class_by_name("Opportunity")
if table_opportunity_2_binding_domain:
    table_opportunity_2_binding = DataBinding(domain_concept=table_opportunity_2_binding_domain, name="OpportunityDataBinding")
else:
    # Domain class 'Opportunity' not resolved; data binding skipped.
    table_opportunity_2_binding = None
if table_opportunity_2_binding:
    table_opportunity_2.data_binding = table_opportunity_2_binding
i20n9g = ViewContainer(
    name="i20n9g",
    description="main container",
    view_elements={izmzig, i0kzeq, table_opportunity_2},
    styling=Styling(size=Size(padding="40px"), position=Position(overflow_y="auto"), color=Color(background_color="#f5f5f5", color_palette="default"), layout=Layout(flex="1")),
    component_id="i20n9g",
    tag_name="main",
    display_order=1,
    custom_attributes={"id": "i20n9g"}
)
i20n9g_layout = Layout(flex="1")
i20n9g.layout = i20n9g_layout
ipxw1j = ViewContainer(
    name="ipxw1j",
    description=" component",
    view_elements={icjhn8, i20n9g},
    styling=Styling(size=Size(height="100vh", font_family="Arial, sans-serif"), position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX)),
    component_id="ipxw1j",
    display_order=0,
    custom_attributes={"id": "ipxw1j"}
)
ipxw1j_layout = Layout(layout_type=LayoutType.FLEX)
ipxw1j.layout = ipxw1j_layout
wrapper_3.view_elements = {ipxw1j}


# Screen: wrapper_4
wrapper_4 = Screen(name="wrapper_4", description="Contact", view_elements=set(), route_path="/contact", screen_size="Medium")
wrapper_4.component_id = "page-contact-3"
i3y5m9 = Text(
    name="i3y5m9",
    content="BESSER",
    description="Text element",
    styling=Styling(size=Size(font_size="24px", font_weight="bold", margin_top="0", margin_bottom="30px"), color=Color(color_palette="default")),
    component_id="i3y5m9",
    tag_name="h2",
    display_order=0,
    custom_attributes={"id": "i3y5m9"}
)
ixw7mp = Link(
    name="ixw7mp",
    description="Link element",
    label="Tag",
    url="/tag",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ixw7mp",
    tag_name="a",
    display_order=0,
    custom_attributes={"href": "/tag", "id": "ixw7mp"}
)
i8l4nn = Link(
    name="i8l4nn",
    description="Link element",
    label="Interaction",
    url="/interaction",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i8l4nn",
    tag_name="a",
    display_order=1,
    custom_attributes={"href": "/interaction", "id": "i8l4nn"}
)
i4uoo2 = Link(
    name="i4uoo2",
    description="Link element",
    label="Opportunity",
    url="/opportunity",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i4uoo2",
    tag_name="a",
    display_order=2,
    custom_attributes={"href": "/opportunity", "id": "i4uoo2"}
)
ivooxx = Link(
    name="ivooxx",
    description="Link element",
    label="Contact",
    url="/contact",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="rgba(255,255,255,0.2)", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ivooxx",
    tag_name="a",
    display_order=3,
    custom_attributes={"href": "/contact", "id": "ivooxx"}
)
iqri4y = Link(
    name="iqri4y",
    description="Link element",
    label="Company",
    url="/company",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iqri4y",
    tag_name="a",
    display_order=4,
    custom_attributes={"href": "/company", "id": "iqri4y"}
)
imlvx5 = Link(
    name="imlvx5",
    description="Link element",
    label="User",
    url="/user",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="imlvx5",
    tag_name="a",
    display_order=5,
    custom_attributes={"href": "/user", "id": "imlvx5"}
)
ixjj7l = Link(
    name="ixjj7l",
    description="Link element",
    label="ScoreHistory",
    url="/scorehistory",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ixjj7l",
    tag_name="a",
    display_order=6,
    custom_attributes={"href": "/scorehistory", "id": "ixjj7l"}
)
iiaom5 = Link(
    name="iiaom5",
    description="Link element",
    label="EnrichmentLog",
    url="/enrichmentlog",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iiaom5",
    tag_name="a",
    display_order=7,
    custom_attributes={"href": "/enrichmentlog", "id": "iiaom5"}
)
ivm1ak = Link(
    name="ivm1ak",
    description="Link element",
    label="GeneratedEmail",
    url="/generatedemail",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ivm1ak",
    tag_name="a",
    display_order=8,
    custom_attributes={"href": "/generatedemail", "id": "ivm1ak"}
)
i8qut5 = Link(
    name="i8qut5",
    description="Link element",
    label="EmailTemplate",
    url="/emailtemplate",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i8qut5",
    tag_name="a",
    display_order=9,
    custom_attributes={"href": "/emailtemplate", "id": "i8qut5"}
)
i6olia = Link(
    name="i6olia",
    description="Link element",
    label="Task",
    url="/task",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i6olia",
    tag_name="a",
    display_order=10,
    custom_attributes={"href": "/task", "id": "i6olia"}
)
i6agif = ViewContainer(
    name="i6agif",
    description=" component",
    view_elements={ixw7mp, i8l4nn, i4uoo2, ivooxx, iqri4y, imlvx5, ixjj7l, iiaom5, ivm1ak, i8qut5, i6olia},
    styling=Styling(position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")),
    component_id="i6agif",
    display_order=1,
    custom_attributes={"id": "i6agif"}
)
i6agif_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")
i6agif.layout = i6agif_layout
ichq6i = Text(
    name="ichq6i",
    content="© 2026 BESSER. All rights reserved.",
    description="Text element",
    styling=Styling(size=Size(font_size="11px", padding_top="20px", margin_top="auto"), position=Position(alignment=Alignment.CENTER), color=Color(opacity="0.8", color_palette="default", border_top="1px solid rgba(255,255,255,0.2)")),
    component_id="ichq6i",
    display_order=2,
    custom_attributes={"id": "ichq6i"}
)
i5cn6i = ViewContainer(
    name="i5cn6i",
    description="nav container",
    view_elements={i3y5m9, i6agif, ichq6i},
    styling=Styling(size=Size(width="250px", padding="20px", unit_size=UnitSize.PIXELS), position=Position(display="flex", overflow_y="auto"), color=Color(background_color="linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", text_color="white", color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column")),
    component_id="i5cn6i",
    tag_name="nav",
    display_order=0,
    custom_attributes={"id": "i5cn6i"}
)
i5cn6i_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column")
i5cn6i.layout = i5cn6i_layout
isujox = Text(
    name="isujox",
    content="Contact",
    description="Text element",
    styling=Styling(size=Size(font_size="32px", margin_top="0", margin_bottom="10px"), color=Color(text_color="#333", color_palette="default")),
    component_id="isujox",
    tag_name="h1",
    display_order=0,
    custom_attributes={"id": "isujox"}
)
it3852 = Text(
    name="it3852",
    content="Manage Contact data",
    description="Text element",
    styling=Styling(size=Size(margin_bottom="30px"), color=Color(text_color="#666", color_palette="default")),
    component_id="it3852",
    tag_name="p",
    display_order=1,
    custom_attributes={"id": "it3852"}
)
table_contact_3_col_0 = FieldColumn(label="Updated At", field=Contact_updated_at)
table_contact_3_col_1 = FieldColumn(label="Phone", field=Contact_phone)
table_contact_3_col_2 = FieldColumn(label="Lead Score Level", field=Contact_lead_score_level)
table_contact_3_col_3 = FieldColumn(label="Email", field=Contact_email)
table_contact_3_col_4 = FieldColumn(label="Lead Score", field=Contact_lead_score)
table_contact_3_col_5 = FieldColumn(label="Last Name", field=Contact_last_name)
table_contact_3_col_6 = FieldColumn(label="Profile Picture Url", field=Contact_profile_picture_url)
table_contact_3_col_7 = FieldColumn(label="First Name", field=Contact_first_name)
table_contact_3_col_8 = FieldColumn(label="Created At", field=Contact_created_at)
table_contact_3_col_9 = FieldColumn(label="Linkedin Url", field=Contact_linkedin_url)
table_contact_3_col_10 = FieldColumn(label="Id", field=Contact_id)
table_contact_3_col_11 = FieldColumn(label="Notes", field=Contact_notes)
table_contact_3_col_12 = FieldColumn(label="Job Title", field=Contact_job_title)
table_contact_3_col_13 = FieldColumn(label="Is Enriched", field=Contact_is_enriched)
table_contact_3_col_14_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "company")
table_contact_3_col_14 = LookupColumn(label="Company", path=table_contact_3_col_14_path, field=Company_website)
table_contact_3_col_15_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "created_by")
table_contact_3_col_15 = LookupColumn(label="Created By", path=table_contact_3_col_15_path, field=User_last_name)
table_contact_3_col_16_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "tags")
table_contact_3_col_16 = LookupColumn(label="Tags", path=table_contact_3_col_16_path, field=Tag_name)
table_contact_3_col_17_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "tasks")
table_contact_3_col_17 = LookupColumn(label="Tasks", path=table_contact_3_col_17_path, field=Task_description)
table_contact_3_col_18_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "enrichment_logs")
table_contact_3_col_18 = LookupColumn(label="Enrichment Logs", path=table_contact_3_col_18_path, field=EnrichmentLog_id)
table_contact_3_col_19_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "opportunities")
table_contact_3_col_19 = LookupColumn(label="Opportunities", path=table_contact_3_col_19_path, field=Opportunity_expected_close_date)
table_contact_3_col_20_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "generated_emails")
table_contact_3_col_20 = LookupColumn(label="Generated Emails", path=table_contact_3_col_20_path, field=GeneratedEmail_created_at)
table_contact_3 = Table(
    name="table_contact_3",
    title="Contact List",
    primary_color="#2c3e50",
    show_header=True,
    striped_rows=False,
    show_pagination=True,
    rows_per_page=5,
    action_buttons=True,
    columns=[table_contact_3_col_0, table_contact_3_col_1, table_contact_3_col_2, table_contact_3_col_3, table_contact_3_col_4, table_contact_3_col_5, table_contact_3_col_6, table_contact_3_col_7, table_contact_3_col_8, table_contact_3_col_9, table_contact_3_col_10, table_contact_3_col_11, table_contact_3_col_12, table_contact_3_col_13, table_contact_3_col_14, table_contact_3_col_15, table_contact_3_col_16, table_contact_3_col_17, table_contact_3_col_18, table_contact_3_col_19, table_contact_3_col_20],
    styling=Styling(size=Size(width="100%", min_height="400px", unit_size=UnitSize.PERCENTAGE), color=Color(color_palette="default", primary_color="#2c3e50")),
    component_id="table-contact-3",
    display_order=2,
    custom_attributes={"chart-color": "#2c3e50", "chart-title": "Contact List", "data-source": "aaddf56f-d3cb-4eab-8223-f08b85c10f1f", "show-header": "true", "striped-rows": "false", "show-pagination": "true", "rows-per-page": "5", "action-buttons": "true", "columns": [{'field': 'updated_at', 'label': 'Updated At', 'columnType': 'field', '_expanded': False}, {'field': 'phone', 'label': 'Phone', 'columnType': 'field', '_expanded': False}, {'field': 'lead_score_level', 'label': 'Lead Score Level', 'columnType': 'field', '_expanded': False}, {'field': 'email', 'label': 'Email', 'columnType': 'field', '_expanded': False}, {'field': 'lead_score', 'label': 'Lead Score', 'columnType': 'field', '_expanded': False}, {'field': 'last_name', 'label': 'Last Name', 'columnType': 'field', '_expanded': False}, {'field': 'profile_picture_url', 'label': 'Profile Picture Url', 'columnType': 'field', '_expanded': False}, {'field': 'first_name', 'label': 'First Name', 'columnType': 'field', '_expanded': False}, {'field': 'created_at', 'label': 'Created At', 'columnType': 'field', '_expanded': False}, {'field': 'linkedin_url', 'label': 'Linkedin Url', 'columnType': 'field', '_expanded': False}, {'field': 'id', 'label': 'Id', 'columnType': 'field', '_expanded': False}, {'field': 'notes', 'label': 'Notes', 'columnType': 'field', '_expanded': False}, {'field': 'job_title', 'label': 'Job Title', 'columnType': 'field', '_expanded': False}, {'field': 'is_enriched', 'label': 'Is Enriched', 'columnType': 'field', '_expanded': False}, {'field': 'company', 'label': 'Company', 'columnType': 'lookup', 'lookupEntity': '9b3c4c9c-64b0-4937-98e5-bdddb729b882', 'lookupField': 'website', '_expanded': False}, {'field': 'created_by', 'label': 'Created By', 'columnType': 'lookup', 'lookupEntity': '278327b9-3f33-4fcc-aa98-c144c0933a65', 'lookupField': 'last_name', '_expanded': False}, {'field': 'tags', 'label': 'Tags', 'columnType': 'lookup', 'lookupEntity': 'ff2a298c-cd09-447f-b868-72d4aa1d34dc', 'lookupField': 'name', '_expanded': False}, {'field': 'tasks', 'label': 'Tasks', 'columnType': 'lookup', 'lookupEntity': 'fb02ddde-ec4d-4181-bc34-7cabad68eca1', 'lookupField': 'description', '_expanded': False}, {'field': 'enrichment_logs', 'label': 'Enrichment Logs', 'columnType': 'lookup', 'lookupEntity': 'ef128c67-c532-4b67-b248-732b587445a7', 'lookupField': 'id', '_expanded': False}, {'field': 'opportunities', 'label': 'Opportunities', 'columnType': 'lookup', 'lookupEntity': '00c036a4-454e-4449-9a9b-6b3f03969d49', 'lookupField': 'expected_close_date', '_expanded': False}, {'field': 'generated_emails', 'label': 'Generated Emails', 'columnType': 'lookup', 'lookupEntity': '72d64804-891c-438b-bfed-6ecee0bfd085', 'lookupField': 'created_at', '_expanded': False}], "id": "table-contact-3", "filter": ""}
)
domain_model_ref = globals().get('domain_model') or next((v for k, v in globals().items() if k.startswith('domain_model') and hasattr(v, 'get_class_by_name')), None)
table_contact_3_binding_domain = None
if domain_model_ref is not None:
    table_contact_3_binding_domain = domain_model_ref.get_class_by_name("Contact")
if table_contact_3_binding_domain:
    table_contact_3_binding = DataBinding(domain_concept=table_contact_3_binding_domain, name="ContactDataBinding")
else:
    # Domain class 'Contact' not resolved; data binding skipped.
    table_contact_3_binding = None
if table_contact_3_binding:
    table_contact_3.data_binding = table_contact_3_binding
i4tyzw = ViewContainer(
    name="i4tyzw",
    description="main container",
    view_elements={isujox, it3852, table_contact_3},
    styling=Styling(size=Size(padding="40px"), position=Position(overflow_y="auto"), color=Color(background_color="#f5f5f5", color_palette="default"), layout=Layout(flex="1")),
    component_id="i4tyzw",
    tag_name="main",
    display_order=1,
    custom_attributes={"id": "i4tyzw"}
)
i4tyzw_layout = Layout(flex="1")
i4tyzw.layout = i4tyzw_layout
iflfki = ViewContainer(
    name="iflfki",
    description=" component",
    view_elements={i5cn6i, i4tyzw},
    styling=Styling(size=Size(height="100vh", font_family="Arial, sans-serif"), position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX)),
    component_id="iflfki",
    display_order=0,
    custom_attributes={"id": "iflfki"}
)
iflfki_layout = Layout(layout_type=LayoutType.FLEX)
iflfki.layout = iflfki_layout
wrapper_4.view_elements = {iflfki}


# Screen: wrapper_5
wrapper_5 = Screen(name="wrapper_5", description="Company", view_elements=set(), route_path="/company", screen_size="Medium")
wrapper_5.component_id = "page-company-4"
isjotn = Text(
    name="isjotn",
    content="BESSER",
    description="Text element",
    styling=Styling(size=Size(font_size="24px", font_weight="bold", margin_top="0", margin_bottom="30px"), color=Color(color_palette="default")),
    component_id="isjotn",
    tag_name="h2",
    display_order=0,
    custom_attributes={"id": "isjotn"}
)
i7s0mo = Link(
    name="i7s0mo",
    description="Link element",
    label="Tag",
    url="/tag",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i7s0mo",
    tag_name="a",
    display_order=0,
    custom_attributes={"href": "/tag", "id": "i7s0mo"}
)
i0ls1e = Link(
    name="i0ls1e",
    description="Link element",
    label="Interaction",
    url="/interaction",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i0ls1e",
    tag_name="a",
    display_order=1,
    custom_attributes={"href": "/interaction", "id": "i0ls1e"}
)
ifv8ru = Link(
    name="ifv8ru",
    description="Link element",
    label="Opportunity",
    url="/opportunity",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ifv8ru",
    tag_name="a",
    display_order=2,
    custom_attributes={"href": "/opportunity", "id": "ifv8ru"}
)
ipmkd2 = Link(
    name="ipmkd2",
    description="Link element",
    label="Contact",
    url="/contact",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ipmkd2",
    tag_name="a",
    display_order=3,
    custom_attributes={"href": "/contact", "id": "ipmkd2"}
)
ij26le = Link(
    name="ij26le",
    description="Link element",
    label="Company",
    url="/company",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="rgba(255,255,255,0.2)", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ij26le",
    tag_name="a",
    display_order=4,
    custom_attributes={"href": "/company", "id": "ij26le"}
)
i9ktd5 = Link(
    name="i9ktd5",
    description="Link element",
    label="User",
    url="/user",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i9ktd5",
    tag_name="a",
    display_order=5,
    custom_attributes={"href": "/user", "id": "i9ktd5"}
)
i14g5q = Link(
    name="i14g5q",
    description="Link element",
    label="ScoreHistory",
    url="/scorehistory",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i14g5q",
    tag_name="a",
    display_order=6,
    custom_attributes={"href": "/scorehistory", "id": "i14g5q"}
)
i143bx = Link(
    name="i143bx",
    description="Link element",
    label="EnrichmentLog",
    url="/enrichmentlog",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i143bx",
    tag_name="a",
    display_order=7,
    custom_attributes={"href": "/enrichmentlog", "id": "i143bx"}
)
i5qwuk = Link(
    name="i5qwuk",
    description="Link element",
    label="GeneratedEmail",
    url="/generatedemail",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i5qwuk",
    tag_name="a",
    display_order=8,
    custom_attributes={"href": "/generatedemail", "id": "i5qwuk"}
)
iplst9 = Link(
    name="iplst9",
    description="Link element",
    label="EmailTemplate",
    url="/emailtemplate",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iplst9",
    tag_name="a",
    display_order=9,
    custom_attributes={"href": "/emailtemplate", "id": "iplst9"}
)
i61t5q = Link(
    name="i61t5q",
    description="Link element",
    label="Task",
    url="/task",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i61t5q",
    tag_name="a",
    display_order=10,
    custom_attributes={"href": "/task", "id": "i61t5q"}
)
ibpcrk = ViewContainer(
    name="ibpcrk",
    description=" component",
    view_elements={i7s0mo, i0ls1e, ifv8ru, ipmkd2, ij26le, i9ktd5, i14g5q, i143bx, i5qwuk, iplst9, i61t5q},
    styling=Styling(position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")),
    component_id="ibpcrk",
    display_order=1,
    custom_attributes={"id": "ibpcrk"}
)
ibpcrk_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")
ibpcrk.layout = ibpcrk_layout
ix88zv = Text(
    name="ix88zv",
    content="© 2026 BESSER. All rights reserved.",
    description="Text element",
    styling=Styling(size=Size(font_size="11px", padding_top="20px", margin_top="auto"), position=Position(alignment=Alignment.CENTER), color=Color(opacity="0.8", color_palette="default", border_top="1px solid rgba(255,255,255,0.2)")),
    component_id="ix88zv",
    display_order=2,
    custom_attributes={"id": "ix88zv"}
)
ihy52x = ViewContainer(
    name="ihy52x",
    description="nav container",
    view_elements={isjotn, ibpcrk, ix88zv},
    styling=Styling(size=Size(width="250px", padding="20px", unit_size=UnitSize.PIXELS), position=Position(display="flex", overflow_y="auto"), color=Color(background_color="linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", text_color="white", color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column")),
    component_id="ihy52x",
    tag_name="nav",
    display_order=0,
    custom_attributes={"id": "ihy52x"}
)
ihy52x_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column")
ihy52x.layout = ihy52x_layout
i7l4xj = Text(
    name="i7l4xj",
    content="Company",
    description="Text element",
    styling=Styling(size=Size(font_size="32px", margin_top="0", margin_bottom="10px"), color=Color(text_color="#333", color_palette="default")),
    component_id="i7l4xj",
    tag_name="h1",
    display_order=0,
    custom_attributes={"id": "i7l4xj"}
)
ixk4gw = Text(
    name="ixk4gw",
    content="Manage Company data",
    description="Text element",
    styling=Styling(size=Size(margin_bottom="30px"), color=Color(text_color="#666", color_palette="default")),
    component_id="ixk4gw",
    tag_name="p",
    display_order=1,
    custom_attributes={"id": "ixk4gw"}
)
table_company_4_col_0 = FieldColumn(label="Website", field=Company_website)
table_company_4_col_1 = FieldColumn(label="Phone", field=Company_phone)
table_company_4_col_2 = FieldColumn(label="Name", field=Company_name)
table_company_4_col_3 = FieldColumn(label="Created At", field=Company_created_at)
table_company_4_col_4 = FieldColumn(label="Linkedin Url", field=Company_linkedin_url)
table_company_4_col_5 = FieldColumn(label="Description", field=Company_description)
table_company_4_col_6 = FieldColumn(label="Updated At", field=Company_updated_at)
table_company_4_col_7 = FieldColumn(label="Id", field=Company_id)
table_company_4_col_8 = FieldColumn(label="Country", field=Company_country)
table_company_4_col_9 = FieldColumn(label="Size", field=Company_size)
table_company_4_col_10 = FieldColumn(label="City", field=Company_city)
table_company_4_col_11 = FieldColumn(label="Industry", field=Company_industry)
table_company_4_col_12 = FieldColumn(label="Address", field=Company_address)
table_company_4_col_13_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "contacts")
table_company_4_col_13 = LookupColumn(label="Contacts", path=table_company_4_col_13_path, field=Contact_updated_at)
table_company_4_col_14_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "opportunities")
table_company_4_col_14 = LookupColumn(label="Opportunities", path=table_company_4_col_14_path, field=Opportunity_expected_close_date)
table_company_4_col_15_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "created_by")
table_company_4_col_15 = LookupColumn(label="Created By", path=table_company_4_col_15_path, field=User_last_name)
table_company_4_col_16_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "tags")
table_company_4_col_16 = LookupColumn(label="Tags", path=table_company_4_col_16_path, field=Tag_name)
table_company_4 = Table(
    name="table_company_4",
    title="Company List",
    primary_color="#2c3e50",
    show_header=True,
    striped_rows=False,
    show_pagination=True,
    rows_per_page=5,
    action_buttons=True,
    columns=[table_company_4_col_0, table_company_4_col_1, table_company_4_col_2, table_company_4_col_3, table_company_4_col_4, table_company_4_col_5, table_company_4_col_6, table_company_4_col_7, table_company_4_col_8, table_company_4_col_9, table_company_4_col_10, table_company_4_col_11, table_company_4_col_12, table_company_4_col_13, table_company_4_col_14, table_company_4_col_15, table_company_4_col_16],
    styling=Styling(size=Size(width="100%", min_height="400px", unit_size=UnitSize.PERCENTAGE), color=Color(color_palette="default", primary_color="#2c3e50")),
    component_id="table-company-4",
    display_order=2,
    custom_attributes={"chart-color": "#2c3e50", "chart-title": "Company List", "data-source": "9b3c4c9c-64b0-4937-98e5-bdddb729b882", "show-header": "true", "striped-rows": "false", "show-pagination": "true", "rows-per-page": "5", "action-buttons": "true", "columns": [{'field': 'website', 'label': 'Website', 'columnType': 'field', '_expanded': False}, {'field': 'phone', 'label': 'Phone', 'columnType': 'field', '_expanded': False}, {'field': 'name', 'label': 'Name', 'columnType': 'field', '_expanded': False}, {'field': 'created_at', 'label': 'Created At', 'columnType': 'field', '_expanded': False}, {'field': 'linkedin_url', 'label': 'Linkedin Url', 'columnType': 'field', '_expanded': False}, {'field': 'description', 'label': 'Description', 'columnType': 'field', '_expanded': False}, {'field': 'updated_at', 'label': 'Updated At', 'columnType': 'field', '_expanded': False}, {'field': 'id', 'label': 'Id', 'columnType': 'field', '_expanded': False}, {'field': 'country', 'label': 'Country', 'columnType': 'field', '_expanded': False}, {'field': 'size', 'label': 'Size', 'columnType': 'field', '_expanded': False}, {'field': 'city', 'label': 'City', 'columnType': 'field', '_expanded': False}, {'field': 'industry', 'label': 'Industry', 'columnType': 'field', '_expanded': False}, {'field': 'address', 'label': 'Address', 'columnType': 'field', '_expanded': False}, {'field': 'contacts', 'label': 'Contacts', 'columnType': 'lookup', 'lookupEntity': 'aaddf56f-d3cb-4eab-8223-f08b85c10f1f', 'lookupField': 'updated_at', '_expanded': False}, {'field': 'opportunities', 'label': 'Opportunities', 'columnType': 'lookup', 'lookupEntity': '00c036a4-454e-4449-9a9b-6b3f03969d49', 'lookupField': 'expected_close_date', '_expanded': False}, {'field': 'created_by', 'label': 'Created By', 'columnType': 'lookup', 'lookupEntity': '278327b9-3f33-4fcc-aa98-c144c0933a65', 'lookupField': 'last_name', '_expanded': False}, {'field': 'tags', 'label': 'Tags', 'columnType': 'lookup', 'lookupEntity': 'ff2a298c-cd09-447f-b868-72d4aa1d34dc', 'lookupField': 'name', '_expanded': False}], "id": "table-company-4", "filter": ""}
)
domain_model_ref = globals().get('domain_model') or next((v for k, v in globals().items() if k.startswith('domain_model') and hasattr(v, 'get_class_by_name')), None)
table_company_4_binding_domain = None
if domain_model_ref is not None:
    table_company_4_binding_domain = domain_model_ref.get_class_by_name("Company")
if table_company_4_binding_domain:
    table_company_4_binding = DataBinding(domain_concept=table_company_4_binding_domain, name="CompanyDataBinding")
else:
    # Domain class 'Company' not resolved; data binding skipped.
    table_company_4_binding = None
if table_company_4_binding:
    table_company_4.data_binding = table_company_4_binding
iabx6g = ViewContainer(
    name="iabx6g",
    description="main container",
    view_elements={i7l4xj, ixk4gw, table_company_4},
    styling=Styling(size=Size(padding="40px"), position=Position(overflow_y="auto"), color=Color(background_color="#f5f5f5", color_palette="default"), layout=Layout(flex="1")),
    component_id="iabx6g",
    tag_name="main",
    display_order=1,
    custom_attributes={"id": "iabx6g"}
)
iabx6g_layout = Layout(flex="1")
iabx6g.layout = iabx6g_layout
ituxtf = ViewContainer(
    name="ituxtf",
    description=" component",
    view_elements={ihy52x, iabx6g},
    styling=Styling(size=Size(height="100vh", font_family="Arial, sans-serif"), position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX)),
    component_id="ituxtf",
    display_order=0,
    custom_attributes={"id": "ituxtf"}
)
ituxtf_layout = Layout(layout_type=LayoutType.FLEX)
ituxtf.layout = ituxtf_layout
wrapper_5.view_elements = {ituxtf}


# Screen: wrapper_6
wrapper_6 = Screen(name="wrapper_6", description="User", view_elements=set(), route_path="/user", screen_size="Medium")
wrapper_6.component_id = "page-user-5"
itly6s = Text(
    name="itly6s",
    content="BESSER",
    description="Text element",
    styling=Styling(size=Size(font_size="24px", font_weight="bold", margin_top="0", margin_bottom="30px"), color=Color(color_palette="default")),
    component_id="itly6s",
    tag_name="h2",
    display_order=0,
    custom_attributes={"id": "itly6s"}
)
io6851 = Link(
    name="io6851",
    description="Link element",
    label="Tag",
    url="/tag",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="io6851",
    tag_name="a",
    display_order=0,
    custom_attributes={"href": "/tag", "id": "io6851"}
)
i7y3nv = Link(
    name="i7y3nv",
    description="Link element",
    label="Interaction",
    url="/interaction",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i7y3nv",
    tag_name="a",
    display_order=1,
    custom_attributes={"href": "/interaction", "id": "i7y3nv"}
)
i3hn9j = Link(
    name="i3hn9j",
    description="Link element",
    label="Opportunity",
    url="/opportunity",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i3hn9j",
    tag_name="a",
    display_order=2,
    custom_attributes={"href": "/opportunity", "id": "i3hn9j"}
)
i807an = Link(
    name="i807an",
    description="Link element",
    label="Contact",
    url="/contact",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i807an",
    tag_name="a",
    display_order=3,
    custom_attributes={"href": "/contact", "id": "i807an"}
)
ix57jg = Link(
    name="ix57jg",
    description="Link element",
    label="Company",
    url="/company",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ix57jg",
    tag_name="a",
    display_order=4,
    custom_attributes={"href": "/company", "id": "ix57jg"}
)
ikpfcf = Link(
    name="ikpfcf",
    description="Link element",
    label="User",
    url="/user",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="rgba(255,255,255,0.2)", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ikpfcf",
    tag_name="a",
    display_order=5,
    custom_attributes={"href": "/user", "id": "ikpfcf"}
)
idx6xg = Link(
    name="idx6xg",
    description="Link element",
    label="ScoreHistory",
    url="/scorehistory",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="idx6xg",
    tag_name="a",
    display_order=6,
    custom_attributes={"href": "/scorehistory", "id": "idx6xg"}
)
iqll9a = Link(
    name="iqll9a",
    description="Link element",
    label="EnrichmentLog",
    url="/enrichmentlog",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iqll9a",
    tag_name="a",
    display_order=7,
    custom_attributes={"href": "/enrichmentlog", "id": "iqll9a"}
)
iuyeby = Link(
    name="iuyeby",
    description="Link element",
    label="GeneratedEmail",
    url="/generatedemail",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iuyeby",
    tag_name="a",
    display_order=8,
    custom_attributes={"href": "/generatedemail", "id": "iuyeby"}
)
igryya = Link(
    name="igryya",
    description="Link element",
    label="EmailTemplate",
    url="/emailtemplate",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="igryya",
    tag_name="a",
    display_order=9,
    custom_attributes={"href": "/emailtemplate", "id": "igryya"}
)
i6vew9 = Link(
    name="i6vew9",
    description="Link element",
    label="Task",
    url="/task",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i6vew9",
    tag_name="a",
    display_order=10,
    custom_attributes={"href": "/task", "id": "i6vew9"}
)
ikfdq2 = ViewContainer(
    name="ikfdq2",
    description=" component",
    view_elements={io6851, i7y3nv, i3hn9j, i807an, ix57jg, ikpfcf, idx6xg, iqll9a, iuyeby, igryya, i6vew9},
    styling=Styling(position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")),
    component_id="ikfdq2",
    display_order=1,
    custom_attributes={"id": "ikfdq2"}
)
ikfdq2_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")
ikfdq2.layout = ikfdq2_layout
ix0nys = Text(
    name="ix0nys",
    content="© 2026 BESSER. All rights reserved.",
    description="Text element",
    styling=Styling(size=Size(font_size="11px", padding_top="20px", margin_top="auto"), position=Position(alignment=Alignment.CENTER), color=Color(opacity="0.8", color_palette="default", border_top="1px solid rgba(255,255,255,0.2)")),
    component_id="ix0nys",
    display_order=2,
    custom_attributes={"id": "ix0nys"}
)
irmob2 = ViewContainer(
    name="irmob2",
    description="nav container",
    view_elements={itly6s, ikfdq2, ix0nys},
    styling=Styling(size=Size(width="250px", padding="20px", unit_size=UnitSize.PIXELS), position=Position(display="flex", overflow_y="auto"), color=Color(background_color="linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", text_color="white", color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column")),
    component_id="irmob2",
    tag_name="nav",
    display_order=0,
    custom_attributes={"id": "irmob2"}
)
irmob2_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column")
irmob2.layout = irmob2_layout
ix4diz = Text(
    name="ix4diz",
    content="User",
    description="Text element",
    styling=Styling(size=Size(font_size="32px", margin_top="0", margin_bottom="10px"), color=Color(text_color="#333", color_palette="default")),
    component_id="ix4diz",
    tag_name="h1",
    display_order=0,
    custom_attributes={"id": "ix4diz"}
)
i51aeh = Text(
    name="i51aeh",
    content="Manage User data",
    description="Text element",
    styling=Styling(size=Size(margin_bottom="30px"), color=Color(text_color="#666", color_palette="default")),
    component_id="i51aeh",
    tag_name="p",
    display_order=1,
    custom_attributes={"id": "i51aeh"}
)
table_user_5_col_0 = FieldColumn(label="Last Name", field=User_last_name)
table_user_5_col_1 = FieldColumn(label="First Name", field=User_first_name)
table_user_5_col_2 = FieldColumn(label="Password Hash", field=User_password_hash)
table_user_5_col_3 = FieldColumn(label="Last Login", field=User_last_login)
table_user_5_col_4 = FieldColumn(label="Email", field=User_email)
table_user_5_col_5 = FieldColumn(label="Is Active", field=User_is_active)
table_user_5_col_6 = FieldColumn(label="Created At", field=User_created_at)
table_user_5_col_7 = FieldColumn(label="Id", field=User_id)
table_user_5_col_8 = FieldColumn(label="Role", field=User_role)
table_user_5_col_9_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "email_templates")
table_user_5_col_9 = LookupColumn(label="Email Templates", path=table_user_5_col_9_path, field=EmailTemplate_created_at)
table_user_5_col_10_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "created_contacts")
table_user_5_col_10 = LookupColumn(label="Created Contacts", path=table_user_5_col_10_path, field=Contact_updated_at)
table_user_5_col_11_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "owned_opportunities")
table_user_5_col_11 = LookupColumn(label="Owned Opportunities", path=table_user_5_col_11_path, field=Opportunity_expected_close_date)
table_user_5_col_12_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "generated_emails")
table_user_5_col_12 = LookupColumn(label="Generated Emails", path=table_user_5_col_12_path, field=GeneratedEmail_created_at)
table_user_5_col_13_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "interactions")
table_user_5_col_13 = LookupColumn(label="Interactions", path=table_user_5_col_13_path, field=Interaction_created_at)
table_user_5_col_14_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "tasks")
table_user_5_col_14 = LookupColumn(label="Tasks", path=table_user_5_col_14_path, field=Task_description)
table_user_5_col_15_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "created_companies")
table_user_5_col_15 = LookupColumn(label="Created Companies", path=table_user_5_col_15_path, field=Company_website)
table_user_5 = Table(
    name="table_user_5",
    title="User List",
    primary_color="#2c3e50",
    show_header=True,
    striped_rows=False,
    show_pagination=True,
    rows_per_page=5,
    action_buttons=True,
    columns=[table_user_5_col_0, table_user_5_col_1, table_user_5_col_2, table_user_5_col_3, table_user_5_col_4, table_user_5_col_5, table_user_5_col_6, table_user_5_col_7, table_user_5_col_8, table_user_5_col_9, table_user_5_col_10, table_user_5_col_11, table_user_5_col_12, table_user_5_col_13, table_user_5_col_14, table_user_5_col_15],
    styling=Styling(size=Size(width="100%", min_height="400px", unit_size=UnitSize.PERCENTAGE), color=Color(color_palette="default", primary_color="#2c3e50")),
    component_id="table-user-5",
    display_order=2,
    custom_attributes={"chart-color": "#2c3e50", "chart-title": "User List", "data-source": "278327b9-3f33-4fcc-aa98-c144c0933a65", "show-header": "true", "striped-rows": "false", "show-pagination": "true", "rows-per-page": "5", "action-buttons": "true", "columns": [{'field': 'last_name', 'label': 'Last Name', 'columnType': 'field', '_expanded': False}, {'field': 'first_name', 'label': 'First Name', 'columnType': 'field', '_expanded': False}, {'field': 'password_hash', 'label': 'Password Hash', 'columnType': 'field', '_expanded': False}, {'field': 'last_login', 'label': 'Last Login', 'columnType': 'field', '_expanded': False}, {'field': 'email', 'label': 'Email', 'columnType': 'field', '_expanded': False}, {'field': 'is_active', 'label': 'Is Active', 'columnType': 'field', '_expanded': False}, {'field': 'created_at', 'label': 'Created At', 'columnType': 'field', '_expanded': False}, {'field': 'id', 'label': 'Id', 'columnType': 'field', '_expanded': False}, {'field': 'role', 'label': 'Role', 'columnType': 'field', '_expanded': False}, {'field': 'email_templates', 'label': 'Email Templates', 'columnType': 'lookup', 'lookupEntity': '2d584aa6-7ca2-4d21-8510-57f70750bfb4', 'lookupField': 'created_at', '_expanded': False}, {'field': 'created_contacts', 'label': 'Created Contacts', 'columnType': 'lookup', 'lookupEntity': 'aaddf56f-d3cb-4eab-8223-f08b85c10f1f', 'lookupField': 'updated_at', '_expanded': False}, {'field': 'owned_opportunities', 'label': 'Owned Opportunities', 'columnType': 'lookup', 'lookupEntity': '00c036a4-454e-4449-9a9b-6b3f03969d49', 'lookupField': 'expected_close_date', '_expanded': False}, {'field': 'generated_emails', 'label': 'Generated Emails', 'columnType': 'lookup', 'lookupEntity': '72d64804-891c-438b-bfed-6ecee0bfd085', 'lookupField': 'created_at', '_expanded': False}, {'field': 'interactions', 'label': 'Interactions', 'columnType': 'lookup', 'lookupEntity': '53490d70-1ecb-4595-89f0-a2309285454e', 'lookupField': 'created_at', '_expanded': False}, {'field': 'tasks', 'label': 'Tasks', 'columnType': 'lookup', 'lookupEntity': 'fb02ddde-ec4d-4181-bc34-7cabad68eca1', 'lookupField': 'description', '_expanded': False}, {'field': 'created_companies', 'label': 'Created Companies', 'columnType': 'lookup', 'lookupEntity': '9b3c4c9c-64b0-4937-98e5-bdddb729b882', 'lookupField': 'website', '_expanded': False}], "id": "table-user-5", "filter": ""}
)
domain_model_ref = globals().get('domain_model') or next((v for k, v in globals().items() if k.startswith('domain_model') and hasattr(v, 'get_class_by_name')), None)
table_user_5_binding_domain = None
if domain_model_ref is not None:
    table_user_5_binding_domain = domain_model_ref.get_class_by_name("User")
if table_user_5_binding_domain:
    table_user_5_binding = DataBinding(domain_concept=table_user_5_binding_domain, name="UserDataBinding")
else:
    # Domain class 'User' not resolved; data binding skipped.
    table_user_5_binding = None
if table_user_5_binding:
    table_user_5.data_binding = table_user_5_binding
i0zqhb = ViewContainer(
    name="i0zqhb",
    description="main container",
    view_elements={ix4diz, i51aeh, table_user_5},
    styling=Styling(size=Size(padding="40px"), position=Position(overflow_y="auto"), color=Color(background_color="#f5f5f5", color_palette="default"), layout=Layout(flex="1")),
    component_id="i0zqhb",
    tag_name="main",
    display_order=1,
    custom_attributes={"id": "i0zqhb"}
)
i0zqhb_layout = Layout(flex="1")
i0zqhb.layout = i0zqhb_layout
i49bi8 = ViewContainer(
    name="i49bi8",
    description=" component",
    view_elements={irmob2, i0zqhb},
    styling=Styling(size=Size(height="100vh", font_family="Arial, sans-serif"), position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX)),
    component_id="i49bi8",
    display_order=0,
    custom_attributes={"id": "i49bi8"}
)
i49bi8_layout = Layout(layout_type=LayoutType.FLEX)
i49bi8.layout = i49bi8_layout
wrapper_6.view_elements = {i49bi8}


# Screen: wrapper_7
wrapper_7 = Screen(name="wrapper_7", description="ScoreHistory", view_elements=set(), route_path="/scorehistory", screen_size="Medium")
wrapper_7.component_id = "page-scorehistory-6"
if7xj3 = Text(
    name="if7xj3",
    content="BESSER",
    description="Text element",
    styling=Styling(size=Size(font_size="24px", font_weight="bold", margin_top="0", margin_bottom="30px"), color=Color(color_palette="default")),
    component_id="if7xj3",
    tag_name="h2",
    display_order=0,
    custom_attributes={"id": "if7xj3"}
)
iejhe7 = Link(
    name="iejhe7",
    description="Link element",
    label="Tag",
    url="/tag",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iejhe7",
    tag_name="a",
    display_order=0,
    custom_attributes={"href": "/tag", "id": "iejhe7"}
)
i0ke83 = Link(
    name="i0ke83",
    description="Link element",
    label="Interaction",
    url="/interaction",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i0ke83",
    tag_name="a",
    display_order=1,
    custom_attributes={"href": "/interaction", "id": "i0ke83"}
)
iww009 = Link(
    name="iww009",
    description="Link element",
    label="Opportunity",
    url="/opportunity",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iww009",
    tag_name="a",
    display_order=2,
    custom_attributes={"href": "/opportunity", "id": "iww009"}
)
iv6ydz = Link(
    name="iv6ydz",
    description="Link element",
    label="Contact",
    url="/contact",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iv6ydz",
    tag_name="a",
    display_order=3,
    custom_attributes={"href": "/contact", "id": "iv6ydz"}
)
iuutfu = Link(
    name="iuutfu",
    description="Link element",
    label="Company",
    url="/company",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iuutfu",
    tag_name="a",
    display_order=4,
    custom_attributes={"href": "/company", "id": "iuutfu"}
)
i9vvsr = Link(
    name="i9vvsr",
    description="Link element",
    label="User",
    url="/user",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i9vvsr",
    tag_name="a",
    display_order=5,
    custom_attributes={"href": "/user", "id": "i9vvsr"}
)
idleai = Link(
    name="idleai",
    description="Link element",
    label="ScoreHistory",
    url="/scorehistory",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="rgba(255,255,255,0.2)", text_color="white", color_palette="default", border_radius="4px")),
    component_id="idleai",
    tag_name="a",
    display_order=6,
    custom_attributes={"href": "/scorehistory", "id": "idleai"}
)
i8c3h9 = Link(
    name="i8c3h9",
    description="Link element",
    label="EnrichmentLog",
    url="/enrichmentlog",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i8c3h9",
    tag_name="a",
    display_order=7,
    custom_attributes={"href": "/enrichmentlog", "id": "i8c3h9"}
)
i7latf = Link(
    name="i7latf",
    description="Link element",
    label="GeneratedEmail",
    url="/generatedemail",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i7latf",
    tag_name="a",
    display_order=8,
    custom_attributes={"href": "/generatedemail", "id": "i7latf"}
)
i4bezp = Link(
    name="i4bezp",
    description="Link element",
    label="EmailTemplate",
    url="/emailtemplate",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i4bezp",
    tag_name="a",
    display_order=9,
    custom_attributes={"href": "/emailtemplate", "id": "i4bezp"}
)
iq3t2f = Link(
    name="iq3t2f",
    description="Link element",
    label="Task",
    url="/task",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iq3t2f",
    tag_name="a",
    display_order=10,
    custom_attributes={"href": "/task", "id": "iq3t2f"}
)
ipbm69 = ViewContainer(
    name="ipbm69",
    description=" component",
    view_elements={iejhe7, i0ke83, iww009, iv6ydz, iuutfu, i9vvsr, idleai, i8c3h9, i7latf, i4bezp, iq3t2f},
    styling=Styling(position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")),
    component_id="ipbm69",
    display_order=1,
    custom_attributes={"id": "ipbm69"}
)
ipbm69_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")
ipbm69.layout = ipbm69_layout
iz3fnk = Text(
    name="iz3fnk",
    content="© 2026 BESSER. All rights reserved.",
    description="Text element",
    styling=Styling(size=Size(font_size="11px", padding_top="20px", margin_top="auto"), position=Position(alignment=Alignment.CENTER), color=Color(opacity="0.8", color_palette="default", border_top="1px solid rgba(255,255,255,0.2)")),
    component_id="iz3fnk",
    display_order=2,
    custom_attributes={"id": "iz3fnk"}
)
i48kly = ViewContainer(
    name="i48kly",
    description="nav container",
    view_elements={if7xj3, ipbm69, iz3fnk},
    styling=Styling(size=Size(width="250px", padding="20px", unit_size=UnitSize.PIXELS), position=Position(display="flex", overflow_y="auto"), color=Color(background_color="linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", text_color="white", color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column")),
    component_id="i48kly",
    tag_name="nav",
    display_order=0,
    custom_attributes={"id": "i48kly"}
)
i48kly_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column")
i48kly.layout = i48kly_layout
i4c847 = Text(
    name="i4c847",
    content="ScoreHistory",
    description="Text element",
    styling=Styling(size=Size(font_size="32px", margin_top="0", margin_bottom="10px"), color=Color(text_color="#333", color_palette="default")),
    component_id="i4c847",
    tag_name="h1",
    display_order=0,
    custom_attributes={"id": "i4c847"}
)
izj6om = Text(
    name="izj6om",
    content="Manage ScoreHistory data",
    description="Text element",
    styling=Styling(size=Size(margin_bottom="30px"), color=Color(text_color="#666", color_palette="default")),
    component_id="izj6om",
    tag_name="p",
    display_order=1,
    custom_attributes={"id": "izj6om"}
)
table_scorehistory_6_col_0 = FieldColumn(label="Id", field=ScoreHistory_id)
table_scorehistory_6_col_1 = FieldColumn(label="Calculated At", field=ScoreHistory_calculated_at)
table_scorehistory_6_col_2 = FieldColumn(label="Reason", field=ScoreHistory_reason)
table_scorehistory_6_col_3 = FieldColumn(label="New Score", field=ScoreHistory_new_score)
table_scorehistory_6_col_4 = FieldColumn(label="Old Score", field=ScoreHistory_old_score)
table_scorehistory_6 = Table(
    name="table_scorehistory_6",
    title="ScoreHistory List",
    primary_color="#2c3e50",
    show_header=True,
    striped_rows=False,
    show_pagination=True,
    rows_per_page=5,
    action_buttons=True,
    columns=[table_scorehistory_6_col_0, table_scorehistory_6_col_1, table_scorehistory_6_col_2, table_scorehistory_6_col_3, table_scorehistory_6_col_4],
    styling=Styling(size=Size(width="100%", min_height="400px", unit_size=UnitSize.PERCENTAGE), color=Color(color_palette="default", primary_color="#2c3e50")),
    component_id="table-scorehistory-6",
    display_order=2,
    custom_attributes={"chart-color": "#2c3e50", "chart-title": "ScoreHistory List", "data-source": "e2ffed0e-86e2-4ef9-bb83-c7d2f87175a9", "show-header": "true", "striped-rows": "false", "show-pagination": "true", "rows-per-page": "5", "action-buttons": "true", "columns": [{'field': 'id', 'label': 'Id', 'columnType': 'field', '_expanded': False}, {'field': 'calculated_at', 'label': 'Calculated At', 'columnType': 'field', '_expanded': False}, {'field': 'reason', 'label': 'Reason', 'columnType': 'field', '_expanded': False}, {'field': 'new_score', 'label': 'New Score', 'columnType': 'field', '_expanded': False}, {'field': 'old_score', 'label': 'Old Score', 'columnType': 'field', '_expanded': False}], "id": "table-scorehistory-6", "filter": ""}
)
domain_model_ref = globals().get('domain_model') or next((v for k, v in globals().items() if k.startswith('domain_model') and hasattr(v, 'get_class_by_name')), None)
table_scorehistory_6_binding_domain = None
if domain_model_ref is not None:
    table_scorehistory_6_binding_domain = domain_model_ref.get_class_by_name("ScoreHistory")
if table_scorehistory_6_binding_domain:
    table_scorehistory_6_binding = DataBinding(domain_concept=table_scorehistory_6_binding_domain, name="ScoreHistoryDataBinding")
else:
    # Domain class 'ScoreHistory' not resolved; data binding skipped.
    table_scorehistory_6_binding = None
if table_scorehistory_6_binding:
    table_scorehistory_6.data_binding = table_scorehistory_6_binding
ijltwb = ViewContainer(
    name="ijltwb",
    description="main container",
    view_elements={i4c847, izj6om, table_scorehistory_6},
    styling=Styling(size=Size(padding="40px"), position=Position(overflow_y="auto"), color=Color(background_color="#f5f5f5", color_palette="default"), layout=Layout(flex="1")),
    component_id="ijltwb",
    tag_name="main",
    display_order=1,
    custom_attributes={"id": "ijltwb"}
)
ijltwb_layout = Layout(flex="1")
ijltwb.layout = ijltwb_layout
iddmah = ViewContainer(
    name="iddmah",
    description=" component",
    view_elements={i48kly, ijltwb},
    styling=Styling(size=Size(height="100vh", font_family="Arial, sans-serif"), position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX)),
    component_id="iddmah",
    display_order=0,
    custom_attributes={"id": "iddmah"}
)
iddmah_layout = Layout(layout_type=LayoutType.FLEX)
iddmah.layout = iddmah_layout
wrapper_7.view_elements = {iddmah}


# Screen: wrapper_8
wrapper_8 = Screen(name="wrapper_8", description="EnrichmentLog", view_elements=set(), route_path="/enrichmentlog", screen_size="Medium")
wrapper_8.component_id = "page-enrichmentlog-7"
inal0d = Text(
    name="inal0d",
    content="BESSER",
    description="Text element",
    styling=Styling(size=Size(font_size="24px", font_weight="bold", margin_top="0", margin_bottom="30px"), color=Color(color_palette="default")),
    component_id="inal0d",
    tag_name="h2",
    display_order=0,
    custom_attributes={"id": "inal0d"}
)
isoplk = Link(
    name="isoplk",
    description="Link element",
    label="Tag",
    url="/tag",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="isoplk",
    tag_name="a",
    display_order=0,
    custom_attributes={"href": "/tag", "id": "isoplk"}
)
ihyh5j = Link(
    name="ihyh5j",
    description="Link element",
    label="Interaction",
    url="/interaction",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ihyh5j",
    tag_name="a",
    display_order=1,
    custom_attributes={"href": "/interaction", "id": "ihyh5j"}
)
ime5ix = Link(
    name="ime5ix",
    description="Link element",
    label="Opportunity",
    url="/opportunity",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ime5ix",
    tag_name="a",
    display_order=2,
    custom_attributes={"href": "/opportunity", "id": "ime5ix"}
)
iuaxt3 = Link(
    name="iuaxt3",
    description="Link element",
    label="Contact",
    url="/contact",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iuaxt3",
    tag_name="a",
    display_order=3,
    custom_attributes={"href": "/contact", "id": "iuaxt3"}
)
imbbbv = Link(
    name="imbbbv",
    description="Link element",
    label="Company",
    url="/company",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="imbbbv",
    tag_name="a",
    display_order=4,
    custom_attributes={"href": "/company", "id": "imbbbv"}
)
ixii9i = Link(
    name="ixii9i",
    description="Link element",
    label="User",
    url="/user",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ixii9i",
    tag_name="a",
    display_order=5,
    custom_attributes={"href": "/user", "id": "ixii9i"}
)
in5i1p = Link(
    name="in5i1p",
    description="Link element",
    label="ScoreHistory",
    url="/scorehistory",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="in5i1p",
    tag_name="a",
    display_order=6,
    custom_attributes={"href": "/scorehistory", "id": "in5i1p"}
)
idsfku = Link(
    name="idsfku",
    description="Link element",
    label="EnrichmentLog",
    url="/enrichmentlog",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="rgba(255,255,255,0.2)", text_color="white", color_palette="default", border_radius="4px")),
    component_id="idsfku",
    tag_name="a",
    display_order=7,
    custom_attributes={"href": "/enrichmentlog", "id": "idsfku"}
)
ifd3xn = Link(
    name="ifd3xn",
    description="Link element",
    label="GeneratedEmail",
    url="/generatedemail",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ifd3xn",
    tag_name="a",
    display_order=8,
    custom_attributes={"href": "/generatedemail", "id": "ifd3xn"}
)
i4n1ei = Link(
    name="i4n1ei",
    description="Link element",
    label="EmailTemplate",
    url="/emailtemplate",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i4n1ei",
    tag_name="a",
    display_order=9,
    custom_attributes={"href": "/emailtemplate", "id": "i4n1ei"}
)
i7ln3e = Link(
    name="i7ln3e",
    description="Link element",
    label="Task",
    url="/task",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i7ln3e",
    tag_name="a",
    display_order=10,
    custom_attributes={"href": "/task", "id": "i7ln3e"}
)
i5ctcx = ViewContainer(
    name="i5ctcx",
    description=" component",
    view_elements={isoplk, ihyh5j, ime5ix, iuaxt3, imbbbv, ixii9i, in5i1p, idsfku, ifd3xn, i4n1ei, i7ln3e},
    styling=Styling(position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")),
    component_id="i5ctcx",
    display_order=1,
    custom_attributes={"id": "i5ctcx"}
)
i5ctcx_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")
i5ctcx.layout = i5ctcx_layout
i32ini = Text(
    name="i32ini",
    content="© 2026 BESSER. All rights reserved.",
    description="Text element",
    styling=Styling(size=Size(font_size="11px", padding_top="20px", margin_top="auto"), position=Position(alignment=Alignment.CENTER), color=Color(opacity="0.8", color_palette="default", border_top="1px solid rgba(255,255,255,0.2)")),
    component_id="i32ini",
    display_order=2,
    custom_attributes={"id": "i32ini"}
)
ii94og = ViewContainer(
    name="ii94og",
    description="nav container",
    view_elements={inal0d, i5ctcx, i32ini},
    styling=Styling(size=Size(width="250px", padding="20px", unit_size=UnitSize.PIXELS), position=Position(display="flex", overflow_y="auto"), color=Color(background_color="linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", text_color="white", color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column")),
    component_id="ii94og",
    tag_name="nav",
    display_order=0,
    custom_attributes={"id": "ii94og"}
)
ii94og_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column")
ii94og.layout = ii94og_layout
iqv6hz = Text(
    name="iqv6hz",
    content="EnrichmentLog",
    description="Text element",
    styling=Styling(size=Size(font_size="32px", margin_top="0", margin_bottom="10px"), color=Color(text_color="#333", color_palette="default")),
    component_id="iqv6hz",
    tag_name="h1",
    display_order=0,
    custom_attributes={"id": "iqv6hz"}
)
iboxhu = Text(
    name="iboxhu",
    content="Manage EnrichmentLog data",
    description="Text element",
    styling=Styling(size=Size(margin_bottom="30px"), color=Color(text_color="#666", color_palette="default")),
    component_id="iboxhu",
    tag_name="p",
    display_order=1,
    custom_attributes={"id": "iboxhu"}
)
table_enrichmentlog_7_col_0 = FieldColumn(label="Id", field=EnrichmentLog_id)
table_enrichmentlog_7_col_1 = FieldColumn(label="Error Message", field=EnrichmentLog_error_message)
table_enrichmentlog_7_col_2 = FieldColumn(label="Is Successful", field=EnrichmentLog_is_successful)
table_enrichmentlog_7_col_3 = FieldColumn(label="Linkedin Url", field=EnrichmentLog_linkedin_url)
table_enrichmentlog_7_col_4 = FieldColumn(label="Enriched At", field=EnrichmentLog_enriched_at)
table_enrichmentlog_7_col_5_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "contact")
table_enrichmentlog_7_col_5 = LookupColumn(label="Contact", path=table_enrichmentlog_7_col_5_path, field=Contact_updated_at)
table_enrichmentlog_7 = Table(
    name="table_enrichmentlog_7",
    title="EnrichmentLog List",
    primary_color="#2c3e50",
    show_header=True,
    striped_rows=False,
    show_pagination=True,
    rows_per_page=5,
    action_buttons=True,
    columns=[table_enrichmentlog_7_col_0, table_enrichmentlog_7_col_1, table_enrichmentlog_7_col_2, table_enrichmentlog_7_col_3, table_enrichmentlog_7_col_4, table_enrichmentlog_7_col_5],
    styling=Styling(size=Size(width="100%", min_height="400px", unit_size=UnitSize.PERCENTAGE), color=Color(color_palette="default", primary_color="#2c3e50")),
    component_id="table-enrichmentlog-7",
    display_order=2,
    custom_attributes={"chart-color": "#2c3e50", "chart-title": "EnrichmentLog List", "data-source": "ef128c67-c532-4b67-b248-732b587445a7", "show-header": "true", "striped-rows": "false", "show-pagination": "true", "rows-per-page": "5", "action-buttons": "true", "columns": [{'field': 'id', 'label': 'Id', 'columnType': 'field', '_expanded': False}, {'field': 'error_message', 'label': 'Error Message', 'columnType': 'field', '_expanded': False}, {'field': 'is_successful', 'label': 'Is Successful', 'columnType': 'field', '_expanded': False}, {'field': 'linkedin_url', 'label': 'Linkedin Url', 'columnType': 'field', '_expanded': False}, {'field': 'enriched_at', 'label': 'Enriched At', 'columnType': 'field', '_expanded': False}, {'field': 'contact', 'label': 'Contact', 'columnType': 'lookup', 'lookupEntity': 'aaddf56f-d3cb-4eab-8223-f08b85c10f1f', 'lookupField': 'updated_at', '_expanded': False}], "id": "table-enrichmentlog-7", "filter": ""}
)
domain_model_ref = globals().get('domain_model') or next((v for k, v in globals().items() if k.startswith('domain_model') and hasattr(v, 'get_class_by_name')), None)
table_enrichmentlog_7_binding_domain = None
if domain_model_ref is not None:
    table_enrichmentlog_7_binding_domain = domain_model_ref.get_class_by_name("EnrichmentLog")
if table_enrichmentlog_7_binding_domain:
    table_enrichmentlog_7_binding = DataBinding(domain_concept=table_enrichmentlog_7_binding_domain, name="EnrichmentLogDataBinding")
else:
    # Domain class 'EnrichmentLog' not resolved; data binding skipped.
    table_enrichmentlog_7_binding = None
if table_enrichmentlog_7_binding:
    table_enrichmentlog_7.data_binding = table_enrichmentlog_7_binding
il8nz4 = ViewContainer(
    name="il8nz4",
    description="main container",
    view_elements={iqv6hz, iboxhu, table_enrichmentlog_7},
    styling=Styling(size=Size(padding="40px"), position=Position(overflow_y="auto"), color=Color(background_color="#f5f5f5", color_palette="default"), layout=Layout(flex="1")),
    component_id="il8nz4",
    tag_name="main",
    display_order=1,
    custom_attributes={"id": "il8nz4"}
)
il8nz4_layout = Layout(flex="1")
il8nz4.layout = il8nz4_layout
ibzwfm = ViewContainer(
    name="ibzwfm",
    description=" component",
    view_elements={ii94og, il8nz4},
    styling=Styling(size=Size(height="100vh", font_family="Arial, sans-serif"), position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX)),
    component_id="ibzwfm",
    display_order=0,
    custom_attributes={"id": "ibzwfm"}
)
ibzwfm_layout = Layout(layout_type=LayoutType.FLEX)
ibzwfm.layout = ibzwfm_layout
wrapper_8.view_elements = {ibzwfm}


# Screen: wrapper_9
wrapper_9 = Screen(name="wrapper_9", description="GeneratedEmail", view_elements=set(), route_path="/generatedemail", screen_size="Medium")
wrapper_9.component_id = "page-generatedemail-8"
iqjrso = Text(
    name="iqjrso",
    content="BESSER",
    description="Text element",
    styling=Styling(size=Size(font_size="24px", font_weight="bold", margin_top="0", margin_bottom="30px"), color=Color(color_palette="default")),
    component_id="iqjrso",
    tag_name="h2",
    display_order=0,
    custom_attributes={"id": "iqjrso"}
)
iz10ia = Link(
    name="iz10ia",
    description="Link element",
    label="Tag",
    url="/tag",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iz10ia",
    tag_name="a",
    display_order=0,
    custom_attributes={"href": "/tag", "id": "iz10ia"}
)
ibwhqk = Link(
    name="ibwhqk",
    description="Link element",
    label="Interaction",
    url="/interaction",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ibwhqk",
    tag_name="a",
    display_order=1,
    custom_attributes={"href": "/interaction", "id": "ibwhqk"}
)
icybrw = Link(
    name="icybrw",
    description="Link element",
    label="Opportunity",
    url="/opportunity",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="icybrw",
    tag_name="a",
    display_order=2,
    custom_attributes={"href": "/opportunity", "id": "icybrw"}
)
i7r9il = Link(
    name="i7r9il",
    description="Link element",
    label="Contact",
    url="/contact",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i7r9il",
    tag_name="a",
    display_order=3,
    custom_attributes={"href": "/contact", "id": "i7r9il"}
)
i978bb = Link(
    name="i978bb",
    description="Link element",
    label="Company",
    url="/company",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="i978bb",
    tag_name="a",
    display_order=4,
    custom_attributes={"href": "/company", "id": "i978bb"}
)
il6em3 = Link(
    name="il6em3",
    description="Link element",
    label="User",
    url="/user",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="il6em3",
    tag_name="a",
    display_order=5,
    custom_attributes={"href": "/user", "id": "il6em3"}
)
ido3a1 = Link(
    name="ido3a1",
    description="Link element",
    label="ScoreHistory",
    url="/scorehistory",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ido3a1",
    tag_name="a",
    display_order=6,
    custom_attributes={"href": "/scorehistory", "id": "ido3a1"}
)
iqrn4u = Link(
    name="iqrn4u",
    description="Link element",
    label="EnrichmentLog",
    url="/enrichmentlog",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iqrn4u",
    tag_name="a",
    display_order=7,
    custom_attributes={"href": "/enrichmentlog", "id": "iqrn4u"}
)
iskyed = Link(
    name="iskyed",
    description="Link element",
    label="GeneratedEmail",
    url="/generatedemail",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="rgba(255,255,255,0.2)", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iskyed",
    tag_name="a",
    display_order=8,
    custom_attributes={"href": "/generatedemail", "id": "iskyed"}
)
ibjn1i = Link(
    name="ibjn1i",
    description="Link element",
    label="EmailTemplate",
    url="/emailtemplate",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="ibjn1i",
    tag_name="a",
    display_order=9,
    custom_attributes={"href": "/emailtemplate", "id": "ibjn1i"}
)
iviogf = Link(
    name="iviogf",
    description="Link element",
    label="Task",
    url="/task",
    styling=Styling(size=Size(padding="10px 15px", text_decoration="none", margin_bottom="5px"), position=Position(display="block"), color=Color(background_color="transparent", text_color="white", color_palette="default", border_radius="4px")),
    component_id="iviogf",
    tag_name="a",
    display_order=10,
    custom_attributes={"href": "/task", "id": "iviogf"}
)
ityuf4 = ViewContainer(
    name="ityuf4",
    description=" component",
    view_elements={iz10ia, ibwhqk, icybrw, i7r9il, i978bb, il6em3, ido3a1, iqrn4u, iskyed, ibjn1i, iviogf},
    styling=Styling(position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")),
    component_id="ityuf4",
    display_order=1,
    custom_attributes={"id": "ityuf4"}
)
ityuf4_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column", flex="1")
ityuf4.layout = ityuf4_layout
i3mn6m = Text(
    name="i3mn6m",
    content="© 2026 BESSER. All rights reserved.",
    description="Text element",
    styling=Styling(size=Size(font_size="11px", padding_top="20px", margin_top="auto"), position=Position(alignment=Alignment.CENTER), color=Color(opacity="0.8", color_palette="default", border_top="1px solid rgba(255,255,255,0.2)")),
    component_id="i3mn6m",
    display_order=2,
    custom_attributes={"id": "i3mn6m"}
)
itkref = ViewContainer(
    name="itkref",
    description="nav container",
    view_elements={iqjrso, ityuf4, i3mn6m},
    styling=Styling(size=Size(width="250px", padding="20px", unit_size=UnitSize.PIXELS), position=Position(display="flex", overflow_y="auto"), color=Color(background_color="linear-gradient(135deg, #4b3c82 0%, #5a3d91 100%)", text_color="white", color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX, flex_direction="column")),
    component_id="itkref",
    tag_name="nav",
    display_order=0,
    custom_attributes={"id": "itkref"}
)
itkref_layout = Layout(layout_type=LayoutType.FLEX, flex_direction="column")
itkref.layout = itkref_layout
ilpfe2 = Text(
    name="ilpfe2",
    content="GeneratedEmail",
    description="Text element",
    styling=Styling(size=Size(font_size="32px", margin_top="0", margin_bottom="10px"), color=Color(text_color="#333", color_palette="default")),
    component_id="ilpfe2",
    tag_name="h1",
    display_order=0,
    custom_attributes={"id": "ilpfe2"}
)
iscwpw = Text(
    name="iscwpw",
    content="Manage GeneratedEmail data",
    description="Text element",
    styling=Styling(size=Size(margin_bottom="30px"), color=Color(text_color="#666", color_palette="default")),
    component_id="iscwpw",
    tag_name="p",
    display_order=1,
    custom_attributes={"id": "iscwpw"}
)
table_generatedemail_8_col_0 = FieldColumn(label="Created At", field=GeneratedEmail_created_at)
table_generatedemail_8_col_1 = FieldColumn(label="Sent At", field=GeneratedEmail_sent_at)
table_generatedemail_8_col_2 = FieldColumn(label="Is Sent", field=GeneratedEmail_is_sent)
table_generatedemail_8_col_3 = FieldColumn(label="Body", field=GeneratedEmail_body)
table_generatedemail_8_col_4 = FieldColumn(label="Subject", field=GeneratedEmail_subject)
table_generatedemail_8_col_5 = FieldColumn(label="Id", field=GeneratedEmail_id)
table_generatedemail_8_col_6_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "template")
table_generatedemail_8_col_6 = LookupColumn(label="Template", path=table_generatedemail_8_col_6_path, field=EmailTemplate_created_at)
table_generatedemail_8_col_7_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "created_by")
table_generatedemail_8_col_7 = LookupColumn(label="Created By", path=table_generatedemail_8_col_7_path, field=User_last_name)
table_generatedemail_8_col_8_path = next(end for assoc in domain_model.associations for end in assoc.ends if end.name == "contact")
table_generatedemail_8_col_8 = LookupColumn(label="Contact", path=table_generatedemail_8_col_8_path, field=Contact_updated_at)
table_generatedemail_8 = Table(
    name="table_generatedemail_8",
    title="GeneratedEmail List",
    primary_color="#2c3e50",
    show_header=True,
    striped_rows=False,
    show_pagination=True,
    rows_per_page=5,
    action_buttons=True,
    columns=[table_generatedemail_8_col_0, table_generatedemail_8_col_1, table_generatedemail_8_col_2, table_generatedemail_8_col_3, table_generatedemail_8_col_4, table_generatedemail_8_col_5, table_generatedemail_8_col_6, table_generatedemail_8_col_7, table_generatedemail_8_col_8],
    styling=Styling(size=Size(width="100%", min_height="400px", unit_size=UnitSize.PERCENTAGE), color=Color(color_palette="default", primary_color="#2c3e50")),
    component_id="table-generatedemail-8",
    display_order=2,
    custom_attributes={"chart-color": "#2c3e50", "chart-title": "GeneratedEmail List", "data-source": "72d64804-891c-438b-bfed-6ecee0bfd085", "show-header": "true", "striped-rows": "false", "show-pagination": "true", "rows-per-page": "5", "action-buttons": "true", "columns": [{'field': 'created_at', 'label': 'Created At', 'columnType': 'field', '_expanded': False}, {'field': 'sent_at', 'label': 'Sent At', 'columnType': 'field', '_expanded': False}, {'field': 'is_sent', 'label': 'Is Sent', 'columnType': 'field', '_expanded': False}, {'field': 'body', 'label': 'Body', 'columnType': 'field', '_expanded': False}, {'field': 'subject', 'label': 'Subject', 'columnType': 'field', '_expanded': False}, {'field': 'id', 'label': 'Id', 'columnType': 'field', '_expanded': False}, {'field': 'template', 'label': 'Template', 'columnType': 'lookup', 'lookupEntity': '2d584aa6-7ca2-4d21-8510-57f70750bfb4', 'lookupField': 'created_at', '_expanded': False}, {'field': 'created_by', 'label': 'Created By', 'columnType': 'lookup', 'lookupEntity': '278327b9-3f33-4fcc-aa98-c144c0933a65', 'lookupField': 'last_name', '_expanded': False}, {'field': 'contact', 'label': 'Contact', 'columnType': 'lookup', 'lookupEntity': 'aaddf56f-d3cb-4eab-8223-f08b85c10f1f', 'lookupField': 'updated_at', '_expanded': False}], "id": "table-generatedemail-8", "filter": ""}
)
domain_model_ref = globals().get('domain_model') or next((v for k, v in globals().items() if k.startswith('domain_model') and hasattr(v, 'get_class_by_name')), None)
table_generatedemail_8_binding_domain = None
if domain_model_ref is not None:
    table_generatedemail_8_binding_domain = domain_model_ref.get_class_by_name("GeneratedEmail")
if table_generatedemail_8_binding_domain:
    table_generatedemail_8_binding = DataBinding(domain_concept=table_generatedemail_8_binding_domain, name="GeneratedEmailDataBinding")
else:
    # Domain class 'GeneratedEmail' not resolved; data binding skipped.
    table_generatedemail_8_binding = None
if table_generatedemail_8_binding:
    table_generatedemail_8.data_binding = table_generatedemail_8_binding
i83kti = ViewContainer(
    name="i83kti",
    description="main container",
    view_elements={ilpfe2, iscwpw, table_generatedemail_8},
    styling=Styling(size=Size(padding="40px"), position=Position(overflow_y="auto"), color=Color(background_color="#f5f5f5", color_palette="default"), layout=Layout(flex="1")),
    component_id="i83kti",
    tag_name="main",
    display_order=1,
    custom_attributes={"id": "i83kti"}
)
i83kti_layout = Layout(flex="1")
i83kti.layout = i83kti_layout
ibpb5f = ViewContainer(
    name="ibpb5f",
    description=" component",
    view_elements={itkref, i83kti},
    styling=Styling(size=Size(height="100vh", font_family="Arial, sans-serif"), position=Position(display="flex"), color=Color(color_palette="default"), layout=Layout(layout_type=LayoutType.FLEX)),
    component_id="ibpb5f",
    display_order=0,
    custom_attributes={"id": "ibpb5f"}
)
ibpb5f_layout = Layout(layout_type=LayoutType.FLEX)
ibpb5f.layout = ibpb5f_layout
wrapper_9.view_elements = {ibpb5f}

gui_module = Module(
    name="GUI_Module",
    screens={wrapper, wrapper_10, wrapper_11, wrapper_2, wrapper_3, wrapper_4, wrapper_5, wrapper_6, wrapper_7, wrapper_8, wrapper_9}
)

# GUI Model

gui_model = GUIModel(
    name="GUI",
    package="",
    versionCode="1.0",
    versionName="1.0",
    modules={gui_module},
    description="GUI"
)


######################
# PROJECT DEFINITION #
######################

from besser.BUML.metamodel.project import Project
from besser.BUML.metamodel.structural.structural import Metadata

metadata = Metadata(description="")
project = Project(
    name="Class_Diagram",
    models=[user_model, gui_model],
    owner="",
    metadata=metadata
)
