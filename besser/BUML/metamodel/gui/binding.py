from enum import Enum
from besser.BUML.metamodel.structural import Class, Property, Constraint
from besser.BUML.metamodel.structural.structural import Element


class DataAggregation(Enum):
    """Aggregation types for dashboard metrics and chart data."""
    SUM = "sum"
    AVG = "average"
    COUNT = "count"
    MIN = "minimum"
    MAX = "maximum"
    MEDIAN = "median"
    FIRST = "first"
    LAST = "last"


class DataBinding(Element):
    """
    DataBinding references a domain concept (Class) to bind data to a GUI element.
    
    Args:
        name (str, optional): Name of the data binding.
        domain_concept (Class): The domain concept to bind data from.
        visualization_attrs (Property, optional): The attribute of the GUI element to bind data to. Defaults to None.
        label_field (Property, optional): The field in the domain concept to use as a label. Defaults to None.
        data_field (Property, optional): The specific field in the domain concept to bind data from. Defaults to None.
        data_filter (Constraint, optional): Filter constraint for the data. Defaults to None.

    Attributes:
        name (str): Name of the data binding.
        domain_concept (Class): The domain concept to bind data from.
        visualization_attrs (Property): The attribute of the GUI element to bind data to.
        label_field (Property): The field in the domain concept to use as a label.
        data_field (Property): The specific field in the domain concept to bind data from.
        data_filter (Constraint): Filter constraint for the data.
    """
    def __init__(self, domain_concept: Class, name: str = None, visualization_attrs: set[Property] = None,
                 label_field: Property = None, data_field: Property = None,
                 data_filter: Constraint = None):
        super().__init__()
        self.name = name or f"{domain_concept.name if domain_concept else 'Unknown'}DataBinding"
        self.domain_concept = domain_concept
        self.visualization_attrs = visualization_attrs if visualization_attrs is not None else set()
        self.label_field = label_field
        self.data_field = data_field
        self.data_filter = data_filter

    @property
    def name(self) -> str:
        """str: Get the name."""
        return self._name

    @name.setter
    def name(self, name: str):
        """str: Set the name."""
        self._name = name

    @property
    def domain_concept(self) -> Class:
        """Class: Get the domain concept."""
        return self._domain_concept

    @domain_concept.setter
    def domain_concept(self, domain_concept: Class):
        """Class: Set the domain concept."""
        if not isinstance(domain_concept, Class):
            raise TypeError("domain_concept must be an instance of Class")
        self._domain_concept = domain_concept

    @property
    def visualization_attrs(self) -> set[Property]:
        """Property: Get the visualization attributes."""
        return self._visualization_attrs

    @visualization_attrs.setter
    def visualization_attrs(self, visualization_attrs: set[Property]):
        """Property: Set the visualization attributes."""
        if visualization_attrs is not None and not isinstance(visualization_attrs, set):
            raise TypeError("visualization_attrs must be a set of Property or None")
        self._visualization_attrs = visualization_attrs

    @property
    def label_field(self) -> Property:
        """Property: Get the label field."""
        return self._label_field

    @label_field.setter
    def label_field(self, label_field: Property):
        """Property: Set the label field."""
        if label_field is not None and not isinstance(label_field, Property):
            raise TypeError("label_field must be an instance of Property or None")
        self._label_field = label_field

    @property
    def data_field(self) -> Property:
        """Property: Get the data field."""
        return self._data_field

    @data_field.setter
    def data_field(self, data_field: Property):
        """Property: Set the data field."""
        if data_field is not None and not isinstance(data_field, Property):
            raise TypeError("data_field must be an instance of Property or None")
        self._data_field = data_field

    @property
    def data_filter(self) -> Constraint:
        """Constraint: Get the data filter constraint."""
        return self._data_filter

    @data_filter.setter
    def data_filter(self, data_filter: Constraint):
        """Constraint: Set the data filter constraint."""
        if data_filter is not None and not isinstance(data_filter, Constraint):
            raise TypeError("data_filter must be an instance of Constraint or None")
        self._data_filter = data_filter

    def __str__(self):
        return (
            f"DataBinding(domain_concept={self.domain_concept}, visualization_attrs={self.visualization_attrs},"
            f"label_field={self.label_field}, data_field={self.data_field}, data_filter={self.data_filter})"
        )
