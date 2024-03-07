
from besser.BUML.metamodel.gui import *
from besser.BUML.metamodel.structural import *
from besser.generators.python_classes import Python_Generator
from besser.generators.django import DjangoGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.sql import SQLGenerator
from tests.library_test.output.classes import *


"""The following model illustrates the output of the BUML model, demonstrating how users can display information about library 
concepts within the BUML model using concepts in the GUI of mobile apps."""

####################### BUML model  ############################################################################################################

# Primitive DataTypes
t_str: PrimitiveDataType = PrimitiveDataType("str")

# Library definition
library_name: Property = Property(name="name", property_type=t_str)
address: Property = Property(name="address", property_type=t_str)
library: Class = Class (name="Library", attributes={library_name, address})


######################## GUI model #############################################################################################

#DataSource definition
datasource_library: ModelElement = ModelElement(name="AwesomeLibrary", dataSourceClass=library, fields={library_name, address})

# List definition
List_name: List.name="LibraryList"
MyList: List=List(name=List_name, description="", list_sources=[datasource_library])

#Button definition:
#Button1:
Button1_name: Button.name="viewListButton"
Button1_label: Button.Label="View List"
Button1: Button=Button(name=Button1_name, description= "", Label=Button1_label)

#Button2:
Button2_name: Button.name="cancelButton"
Button2_label: Button.Label="Cancel"
Button2: Button=Button(name=Button2_name, description= "", Label=Button2_label)

#ViewComponent definition
view_component_name: ViewComponent.name= "my_component"
view_component_descriotion: ViewComponent.description = "description"
ViewComponent1: ViewComponent=ViewComponent(view_component_name,view_component_descriotion)

#Screen definition
screen_name: Screen.name= "Student List Page"
screen_x_dpi: Screen.x_dpi="x_dpi"
screen_y_dpi: Screen.y_dpi="y_dpi"
screen_size: Screen.size="SmallScreen"
screen_components:Screen.components=[MyList,Button1,Button2]
MyScreen: Screen=Screen(name=screen_name, description="", x_dpi=screen_x_dpi, y_dpi= screen_y_dpi,size=screen_size,components=screen_components)

#Module definition:
module_name: Module.name="module_name"
module_screens: Module.screens=[MyScreen]
MyModule: Module=Module(name=module_name, screens=module_screens)


#Application definition:
application_name:Application.name="Library List App"
application_package: Application.package="com.example.librarylistapp"
application_version_Code: Application.versionCode="1"
application_version_name: Application.versionName="1.0"
application_description: Application.description="This is a simple Android app"
application_screenCompatibility:Application.screenCompatibility=False
MyApp: Application=Application(name=application_name, package=application_package,versionCode=application_version_Code,versionName=application_version_name,description=application_description,screenCompatibility=application_screenCompatibility,modules=[MyModule])


print("This App has" + " "+ str(len(MyScreen.components)) +" components")

for source in MyList.list_sources:
    
    print("The concept of " + source.name + " is characterized by two key features, as outlined below:")
    for field in source.fields:
      print(field.name)
    print("   ")