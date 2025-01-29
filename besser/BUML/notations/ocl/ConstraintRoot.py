class ConstraintRoot:
    def __init__(self, context: str,  root):
        self.root = root
        self.context: str = context

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
