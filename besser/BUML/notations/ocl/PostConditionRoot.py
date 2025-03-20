class PostConditionRoot:
    def __init__(self, context: str, function_name: str, root):
        self.root = root
        self.function_name: str = function_name
        self.context: str = context

    @property
    def function_name(self):
        return self.__function_name

    @function_name.setter
    def function_name(self, function_name: str):
        """str: Set the root."""
        self.__function_name = function_name

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