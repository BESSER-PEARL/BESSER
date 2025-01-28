class InitRoot:
    def __init__(self, context: str, variable_name:str, root,type:str):
        self.root = root
        self.context: str = context
        self.type: str= type
        self.variable_name: str = variable_name

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, type: str):
        """str: Set the root."""
        self.__type = type
    @property
    def root(self):
        return self.__root

    @root.setter
    def root(self, root: str):
        """str: Set the root."""
        self.__root = root

    @property
    def context(self):
        return self.__context

    @context.setter
    def context(self, context: str):
        """str: Set the context"""
        self.__context = context

    @property
    def variable_name(self):
        return self.__variable_name

    @variable_name.setter
    def variable_name(self, variable_name: str):
        """str: Set the context"""
        self.__variable_name = variable_name