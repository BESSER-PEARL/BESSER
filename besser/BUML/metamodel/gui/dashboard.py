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
        line_width (int): The width of the line in the chart.

    Attributes:
        name (str): The name of the line chart.
        x_axis (Property): The x-axis property of the line chart.
        y_axis (Property): The y-axis property of the line chart.
        line_width (int): The width of the line in the chart.
    """

    def __init__(self, name: str, x_axis: Property, y_axis: Property, line_width: int = 2):
        super().__init__(name)
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.line_width = line_width

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

    @property
    def line_width(self) -> int:
        """Property: Get the line width of the line chart."""
        return self._line_width

    @line_width.setter
    def line_width(self, value: int):
        """Property: Set the line width of the line chart."""
        self._line_width = value

    def __repr__(self):
        return (
            f"LineChart(name={self.name}, x_axis={self.x_axis.name}, y_axis={self.y_axis.name}, "
            f"line_width={self.line_width})"
        )

class BarChart(Chart):
    """Represents a bar chart component in the dashboard.

    Args:
        name (str): The name of the bar chart.
        x_axis (Property): The x-axis property of the bar chart.
        y_axis (Property): The y-axis property of the bar chart.
        bar_width (int): The width of the bars in the chart.

    Attributes:
        name (str): The name of the bar chart.
        x_axis (Property): The x-axis property of the bar chart.
        y_axis (Property): The y-axis property of the bar chart.
        bar_width (int): The width of the bars in the chart.
    """

    def __init__(self, name: str, x_axis: Property, y_axis: Property, bar_width: int = 30):
        super().__init__(name)
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.bar_width = bar_width

    @property
    def x_axis(self) -> Property:
        """Property: Get the x-axis of the bar chart."""
        return self._x_axis

    @x_axis.setter
    def x_axis(self, value: Property):
        """Property: Set the x-axis of the bar chart."""
        self._x_axis = value

    @property
    def y_axis(self) -> Property:
        """Property: Get the y-axis of the bar chart."""
        return self._y_axis

    @y_axis.setter
    def y_axis(self, value: Property):
        """Property: Set the y-axis of the bar chart."""
        self._y_axis = value

    @property
    def bar_width(self) -> int:
        """Property: Get the bar width of the bar chart."""
        return self._bar_width

    @bar_width.setter
    def bar_width(self, value: int):
        """Property: Set the bar width of the bar chart."""
        self._bar_width = value

    def __repr__(self):
        return (
            f"BarChart(name={self.name}, x_axis={self.x_axis.name}, y_axis={self.y_axis.name}, "
            f"bar_width={self.bar_width})"
        )

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
