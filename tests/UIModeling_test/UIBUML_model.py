
from besser.BUML.metamodel.structural import *


# Primitive DataTypes
t_int: PrimitiveDataType = PrimitiveDataType("int")
t_str: PrimitiveDataType = PrimitiveDataType("str")


# Student attributes definition
student_value: Property = Property(name="value", property_type=t_str)
student_label: Property = Property(name="label", property_type=t_str)
student_age: Property = Property(name="age", property_type=t_str)
student_grade: Property = Property(name="grade", property_type=t_int)


# Student class definition
student: Class = Class (name="Student", attributes={student_value, student_label, student_age, student_grade})


#ModelElementDataSource definition:
modelElement_dataSource1_name: ModelElementDataSource.name="ModelElementDataSource1"
modelElement_dataSource1_dataSource_class: ModelElementDataSource.dataSourceClass = student
modelElement_dataSource1: ModelElementDataSource= ModelElementDataSource(name=modelElement_dataSource1_name, dataSourceClass=modelElement_dataSource1_dataSource_class)



#CollectionSourceType definition:
collection_type1_name: CollectionSourceType.name="Type1"
collection_type1_type:CollectionSourceType.type="Array"
collection_type1: CollectionSourceType=CollectionSourceType(name=collection_type1_name, type=collection_type1_type)

#DataSource definition
data_source_name: DataSource.name = "StudentSource"
data_source1: DataSource = DataSource(name=data_source_name)


#CollectionDataSource definition:
collection_datasource1_name: CollectionDataSource = "Collection1"
collection_datasource1_type: CollectionDataSource.type=collection_type1
collection_datasource1: CollectionDataSource=CollectionDataSource(name=collection_datasource1_name, type=collection_datasource1_type)

# ListItems definition:
#ListItem1:
List_item1_name: ListItem.name = "Student Name Item1"
List_item1_source: ListItem.item_source =modelElement_dataSource1.dataSourceClass
List_item1_fields: ListItem.fields = student.attributes
List_item1_value: student_value= "123"
List_item1_label: student_label="John Doe"
List_item1_age: student_age="20"
List_item1_grade : student_grade= "19"
ListItem1: ListItem = ListItem(name=List_item1_name, item_source=List_item1_source, fields=List_item1_fields)


#ListItem2:
List_item2_name: ListItem.name = "Student Name Item2"
List_item2_source: ListItem.item_source =modelElement_dataSource1.dataSourceClass
List_item2_fields: ListItem.fields = [student_age, student_grade]
List_item2_value: student_value= "546"
List_item2_label: student_label="Jane Smith"
List_item2_age: student_age="18"
List_item2_grade : student_grade= "17"
ListItem2: ListItem = ListItem(name=List_item2_name, item_source=List_item2_source, fields=List_item2_fields)


#ListItem3:
List_item3_name: ListItem.name = "Student Name Item3"
List_item3_source: ListItem.item_source =collection_datasource1
List_item3_fields: ListItem.fields = [student_value, student_label, student_age]
List_item3_value: student_value = "789"
List_item3_label: student_label = "Alex Johnson"
List_item3_age: student_age="25"
List_item3_grade : student_grade= "15"
ListItem3: ListItem = ListItem(name=List_item3_name, item_source=List_item3_source, fields=List_item3_fields)


# List definition
List_name: List.name="StudentList"
List_items: List.list_items=[ListItem1, ListItem2, ListItem3]
MyList: List=List(name=List_name, list_sources=[data_source1], list_items=List_items)


#Button definition:
#Button1:
Button1_name: Button.name="viewListButton"
Button1_label: Button.Label="View List"
Button1: Button=Button(name=Button1_name, Label=Button1_label)

#Button2:
Button2_name: Button.name="cancelButton"
Button2_label: Button.Label="Cancel"
Button2: Button=Button(name=Button2_name, Label=Button2_label)



#ViewComponent definition
view_component_name: ViewComponent.name= "my_component"
view_component_descriotion: ViewComponent.description = "description"
ViewComponent1: ViewComponent=ViewComponent(view_component_name,view_component_descriotion)


#Screen definition
screen_name: Screen.name= "Student List Page"
screen_x_dpi: Screen.x_dpi="x_dpi"
screen_y_dpi: Screen.y_dpi="y_dpi"
screen_type: Screen.screenType="smallScreen"
screen_components:Screen.components=[MyList,Button1,Button2]
MyScreen: Screen=Screen(name=screen_name, x_dpi=screen_x_dpi, y_dpi= screen_y_dpi,screenType=screen_type,components=screen_components)


#Module definition:
module_name: Module.name="module_name"
module_screens: Module.screens=[MyScreen]
MyModule: Module=Module(name=module_name, screens=module_screens)


#Application definition:
application_name:Application.name="Student List App"
application_package: Application.package="com.example.studentlistapp"
application_version_Code: Application.versionCode="1"
application_version_name: Application.versionName="1.0"
application_description: Application.description="This is a simple Android app"
application_screenCompatibility:Application.screenCompatibility=False
MyApp: Application=Application(name=application_name, package=application_package,versionCode=application_version_Code,versionName=application_version_name,description=application_description,screenCompatibility=application_screenCompatibility,modules=[MyModule])





print("This App has " + " "+ str(len(MyScreen.components)) +" components")

print("The fields in ListItem1: ")
for field in ListItem1.fields:
    print (field.name)

print("The fields in ListItem2: ")
for field in ListItem2.fields:
    print (field.name)

print("The fields in ListItem3: ")
for field in ListItem3.fields:
    print (field.name)


    