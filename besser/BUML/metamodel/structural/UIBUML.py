from besser.BUML.metamodel.structural import NamedElement, Class, Property


# FileSourceType
class FileSourceType:
    def __init__(self, name: str, type: str):
        self.type: str = type
        self.name: str = name

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def type(self) -> str:
        return self.__type

    @type.setter
    def type(self, type: str):
        if type not in ['FileSystem', 'LocalStorage', 'DatabaseFileSystem']:
            raise ValueError("Invalid value of type")
            self.__type = type

    def __repr__(self):
        return f'FileSourceType({self.name}, type={self.type})'



# CollectionSourceType
class CollectionSourceType:
    def __init__(self, name: str, type: str):
        self.type: str = type
        self.name: str = name

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def type(self) -> str:
        return self.__type

    @type.setter
    def type(self, type: str):
        if type not in ['List', 'Table', 'Tree', 'Grid', 'Array', 'Stack']:
            raise ValueError("Invalid value of type")
            self.__type = type

    def __repr__(self):
        return f'CollectionSourceType({self.name}, type={self.type})'



#DataSource
class DataSource:
    def __init__(self, name: str):
        self.name: str = name

    @property
    def name(self) -> str:
      return self.__name

    @name.setter
    def name(self, name: str):
      self.__name = name

    def __repr__(self):
      return f'DataSource({self.name})'



#ModelElementDataSource
class ModelElementDataSource(DataSource):
    def __init__(self, name: str, dataSourceClass: Class):
        self.name: str = name

    @property
    def name(self) -> str:
      return self.__name

    @name.setter
    def name(self, name: str):
      self.__name = name

    @property
    def dataSourceClass(self) -> Class:
        return self.__dataSourceClass

    @dataSourceClass.setter
    def name(self, dataSourceClass: Class):
        self.__dataSourceClass = dataSourceClass

    def __repr__(self):
      return f'ModelElementDataSource({self.name}, {self.dataSourceClass})'


#FileDataSource
class FileDataSource(DataSource):
    def __init__(self, name: str, type:FileSourceType):
        self.name: str = name
        self.type: FileSourceType = type

    @property
    def name(self) -> str:
      return self.__name

    @name.setter
    def name(self, name: str):
      self.__name = name

    @property
    def type(self) -> FileSourceType:
      return self.__type

    @type.setter
    def type(self, type: FileSourceType):
      self.__type = type

    def __repr__(self):
      return f'FileDataSource({self.name}, {self.type})'


#CollectionDataSource
class CollectionDataSource(DataSource):
    def __init__(self, name: str, type:CollectionSourceType):
        self.name: str = name
        self.type: CollectionSourceType = type

    @property
    def name(self) -> str:
      return self.__name

    @name.setter
    def name(self, name: str):
      self.__name = name

    @property
    def type(self) -> CollectionSourceType:
      return self.__type

    @type.setter
    def type(self, type: CollectionSourceType):
      self.__type = type

    def __repr__(self):
      return f'CollectionDataSource({self.name}, {self.type})'


# ListItem are owned by a List
class ListItem():
    def __init__(self, name: str, item_source: DataSource, fields: set[Property]):
        self.name: str = name
        self.item_source: DataSource = item_source
        self.fields: set[Property]= fields

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name


    @property
    def item_source(self) -> DataSource:
        return self.__item_source

    @item_source.setter
    def item_source(self, item_source: DataSource):
        self.__item_source = item_source

    @property
    def fields(self) -> set[Property]:
        return self.__fields

    @fields.setter
    def fields(self, fields: set[Property]):
        self.__fields = fields


    def __repr__(self):
        return f'ListItem({self.name},item_source={self.item_source}, fields={self.fields})'

#ViewElement
class ViewElement(NamedElement):
    def __init__(self, name: str):
        super().__init__(name)
        self.name: str = name

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    def __repr__(self):
        return f'ViewElement({self.name})'


#ViewComponent
class ViewComponent(ViewElement):
    def __init__(self, name: str, description: str):
        self.description: str = description
        self.name: str = name

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def description(self) -> str:
        return self.__description

    @description.setter
    def description(self, description: str):
        self.__description = description

    def __repr__(self):
        return f'ViewComponent({self.name}, description={self.description})'


#Screen
class Screen:
    def __init__(self, name: str, components: set[ViewComponent], x_dpi: str, y_dpi: str, screenType: str):
        self.name: str = name
        self.x_dpi: str = x_dpi
        self.y_dpi: str = y_dpi
        self.screenTy: str = screenType
        self.components: set[ViewComponent] = components

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name


    @property
    def components(self) -> set[ViewComponent]:
        return self.__components

    @components.setter
    def components(self, components: set[ViewComponent]):
        self.__components = components

    @property
    def x_dpi(self) -> str:
      return self.__x_pdi

    @x_dpi.setter
    def x_dpi(self, x_dpi: str):
      self.__x_pdi = x_dpi

    @property
    def y_dpi(self) -> str:
      return self.__y_pdi

    @y_dpi.setter
    def y_dpi(self, y_dpi: str):
     self.__y_pdi = y_dpi

    @property
    def screenType(self) -> str:
      return self.__screenType


    @screenType.setter
    def screenType(self, screenType: str):
      if screenType not in ['SmallScreen', 'MediumScreen', 'LargScreen', 'xLargeScreen']:
        raise ValueError("Invalid value of ScreenType")
        self.__screenType = screenType

    def __repr__(self):
      return f'Screen({self.name}, {self.x_dpi}, {self.y_dpi}, {self.screenType}, {self.components})'



#Module
class Module(NamedElement):
    def __init__(self, name: str, screens: set[Screen]):
        self.screens: set[Screen] = screens
        self.name: str = name

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def screens(self) -> set[Screen]:
        return self.__screens

    @screens.setter
    def screens(self, screens: set[Screen]):
       self.__screens = screens

    def __repr__(self):
        return f'Module({self.name}, {self.screens})'


# List is a type of ViewComponent
class List(ViewComponent):
    def __init__(self, name: str, list_items: set[ListItem], list_sources: set[DataSource]):
        self.name=name
        self.list_items: ListItem = list_items
        self.list_sources: set[DataSource] = list_sources

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def list_items(self) -> set[ListItem]:
        return self.__list_items

    @list_items.setter
    def list_items(self, list_items: set[ListItem]):
        self.__list_items = list_items

    @property
    def list_sources(self) -> set[DataSource]:
        return self.__list_sources

    @list_sources.setter
    def list_sources(self, list_sources: set[DataSource]):
        self.__list_sources = list_sources

    def __repr__(self):
     return f'List({self.name},{self.list_items}, {self.list_sources})'


# Button is a type of ViewComponent
class Button(ViewComponent):
    def __init__(self, name: str, Label: str):
        self.name=name
        self.Label=Label

    @property
    def Label(self) -> str:
        return self.__Label

    @Label.setter
    def Label(self, Label: str):
        self.__Label = Label

    @property
    def name(self) -> str:
        return self.__name

    @Label.setter
    def name(self, name: str):
        self.__name = name

    def __repr__(self):
     return f'Button({self.Label},{self.name})'


#Application
class Application(NamedElement):
    def __init__(self, name: str, package: str, versionCode: str, versionName: str, modules: set[Module], description: str, screenCompatibility: bool = False):
        self.name: str = name
        self.description: str = description
        self.package: str = package
        self.versionCode: str = versionCode
        self.versionName: str = versionName
        self.screenCompatibility: str = screenCompatibility
        self.modules: set[Module] = modules

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    @property
    def package(self) -> str:
        return self.__package

    @package.setter
    def package(self, package: str):
        self.__package = package

    @property
    def versionCode(self) -> str:
        return self.__versionCode

    @versionCode.setter
    def versionCode(self, versionCode: str):
        self.__versionCode = versionCode

    @property
    def versionName(self) -> str:
        return self.__versionName

    @versionName.setter
    def versionName(self, versionName: str):
        self.__versionName = versionName

    @property
    def description(self) -> str:
        return self.__description

    @description.setter
    def description(self, description: str):
        self.__description = description

    @property
    def screenCompatibility(self) -> bool:
        return self.__screenCompatibility

    @screenCompatibility.setter
    def screenCompatibility(self, screenCompatibility: bool):
        self.__screenCompatibility = screenCompatibility

    @property
    def modules(self) -> set[Module]:
        return self.modules

    @modules.setter
    def modules(self, modules: set[Module]):
       self.__modules = modules

    def __repr__(self):
        return f'Application({self.name}, {self.package}, {self.versionCode}, {self.versionName},{self.description},{self.screenCompatibility}, {self.modules})'





