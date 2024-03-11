
from besser.BUML.metamodel.gui import *
from besser.BUML.metamodel.structural import *
from besser.generators.python_classes import Python_Generator
from besser.generators.django import DjangoGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator
from besser.generators.sql import SQLGenerator


###################################################
#   Library class - structural model definition   #
###################################################

# Primitive DataTypes
t_str: PrimitiveDataType = PrimitiveDataType("str")

# Library definition
library_name: Property = Property(name="name", property_type=t_str)
address: Property = Property(name="address", property_type=t_str)
library: Class = Class (name="Library", attributes={library_name, address})

##################################
#      GUI model definition      #
##################################

#DataSource definition
datasource_library: ModelElement = ModelElement(name="AwesomeLibrary", dataSourceClass=library, fields={library_name, address})

# List definition
MyList: List=List(name="LibraryList", description="A comprehensive collection of libraries", list_sources={datasource_library})

#Buttons definition:
#Button1:
Button1: Button=Button(name="viewListButton", description= "Explore the Complete List", Label="View List")

#Button2:
Button2: Button=Button(name="cancelButton", description="Abort the Current Operation" , Label="Cancel")

#ViewComponent definition
ViewComponent1: ViewComponent=ViewComponent(name="my_component", description= "Detailed Information at a Glance")

#Screen definition
MyScreen: Screen=Screen(name="Library List Page", description="Discover nearby libraries with names and addresses", x_dpi="x_dpi", y_dpi= "y_dpi", size="SmallScreen", components={MyList,Button1,Button2})

#Module definition:
MyModule: Module=Module(name="module_name", screens={MyScreen})

#Application definition:
MyApp: Application=Application(name="Library List App", package="com.example.librarylistapp",versionCode="1",versionName="1.0",description="This is a simple Android app",screenCompatibility=False,modules={MyModule})


print("This App has" + " "+ str(len(MyScreen.components)) +" components")

for source in MyList.list_sources:
    
    print("The concept of " + source.name + " is characterized by two key features, as outlined below:")
    for field in source.fields:
      print(field.name)
    print("   ")