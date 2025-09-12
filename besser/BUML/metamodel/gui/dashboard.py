from besser.BUML.metamodel.gui.graphical_ui import ViewComponent
from besser.BUML.metamodel.structural import Property

class Chart(ViewComponent):
    """Represents a chart component in the dashboard.

    Args:
        name (str): The name of the chart.

    Attributes:
        name (str): The name of the chart.
    """

    def __init__(self, name: str):
        super().__init__(name)

class LineChart(Chart):
    """Represents a line chart component in the dashboard.

    Args:
        name (str): The name of the line chart.
        x_axis (Property): The x-axis property of the line chart.
        y_axis (Property): The y-axis property of the line chart.

    Attributes:
        name (str): The name of the line chart.
        x_axis (Property): The x-axis property of the line chart.
        y_axis (Property): The y-axis property of the line chart.
    """

    def __init__(self, name: str, x_axis: Property, y_axis: Property):
        super().__init__(name)
        self.x_axis = x_axis
        self.y_axis = y_axis

    @property
    def x_axis(self) -> Property:
        """Property: Get the x-axis of the line chart."""
        return self._x_axis

    @x_axis.setter
    def x_axis(self, value: Property):
        """Property: Set the x-axis of the line chart."""
        self._x_axis = value

    @property
    def y_axis(self) -> Property:
        """Property: Get the y-axis of the line chart."""
        return self._y_axis

    @y_axis.setter
    def y_axis(self, value: Property):
        """Property: Set the y-axis of the line chart."""
        self._y_axis = value

    def __repr__(self):
        return f"LineChart(name={self.name}, x_axis={self.x_axis.name}, y_axis={self.y_axis.name})"

class BarChart(Chart):
    """Represents a bar chart component in the dashboard.

    Args:
        name (str): The name of the bar chart.

    Attributes:
        name (str): The name of the bar chart.
    """

    def __init__(self, name: str):
        super().__init__(name)

class PieChart(Chart):
    """Represents a pie chart component in the dashboard.

    Args:
        name (str): The name of the pie chart.

    Attributes:
        name (str): The name of the pie chart.
    """

    def __init__(self, name: str):
        super().__init__(name)

class RadarChart(Chart):
    """Represents a radar chart component in the dashboard.

    Args:
        name (str): The name of the radar chart.

    Attributes:
        name (str): The name of the radar chart.
    """

    def __init__(self, name: str):
        super().__init__(name)


class RadialBarChart(Chart):
    """Represents a radial bar chart component in the dashboard.

    Args:
        name (str): The name of the radial bar chart.

    Attributes:
        name (str): The name of the radial bar chart.
    """

    def __init__(self, name: str):
        super().__init__(name)

class Map(ViewComponent):
    """Represents a map component in the dashboard.

    Args:
        name (str): The name of the map.

    Attributes:
        name (str): The name of the map.
    """

    def __init__(self, name: str, data: list):
        super().__init__(name)
        self.data = data

class WorldMap(Map):
    """Represents a world map component in the dashboard.

    Args:
        name (str): The name of the world map.
        data (list): The data to be displayed on the world map.

    Attributes:
        name (str): The name of the world map.
        data (list): The data to be displayed on the world map.
    """

    def __init__(self, name: str, data: list):
        super().__init__(name, data)


class LocationMap(Map):
    """Represents a location map component in the dashboard.

    Args:
        name (str): The name of the location map.
        data (list): The data to be displayed on the location map.

    Attributes:
        name (str): The name of the location map.
        data (list): The data to be displayed on the location map.
    """

    def __init__(self, name: str, data: list):
        super().__init__(name, data)
