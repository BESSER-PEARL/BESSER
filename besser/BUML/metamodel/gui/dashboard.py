from besser.BUML.metamodel.gui.graphical_ui import ViewComponent
from besser.BUML.metamodel.gui.style import Alignment

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
        line_width (int): The width of the line in the chart.

    Attributes:
        name (str): The name of the line chart.
        line_width (int): The width of the line in the chart.
    """

    def __init__(self, name: str, line_width: int = 2):
        super().__init__(name)
        self.line_width = line_width

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
            f"LineChart(name={self.name}, line_width={self.line_width})"
        )

class BarChart(Chart):
    """Represents a bar chart component in the dashboard.

    Args:
        name (str): The name of the bar chart.
        bar_width (int): The width of the bars in the chart.

    Attributes:
        name (str): The name of the bar chart.
        bar_width (int): The width of the bars in the chart.
    """

    def __init__(self, name: str, bar_width: int = 30):
        super().__init__(name)
        self.bar_width = bar_width

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
            f"BarChart(name={self.name}, bar_width={self.bar_width})"
        )

class PieChart(Chart):
    """Represents a pie chart component in the dashboard.

    Args:
        name (str): The name of the pie chart.
        show_legend (bool): Whether to show the legend in the pie chart.
        legend_position (Alignment): The position of the legend in the pie chart.
        show_labels (bool): Whether to show labels in the pie chart.
        label_position (Alignment): The position of the labels in the pie chart.
        padding_angle (int): The padding angle between slices in the pie chart.

    Attributes:
        name (str): The name of the pie chart.
        show_legend (bool): Whether to show the legend in the pie chart.
        legend_position (Alignment): The position of the legend in the pie chart.
        show_labels (bool): Whether to show labels in the pie chart.
        label_position (Alignment): The position of the labels in the pie chart.
        padding_angle (int): The padding angle between slices in the pie chart.
    """

    def __init__(self, name: str, show_legend: bool = True,
                 legend_position: Alignment = Alignment.LEFT, show_labels: bool = True,
                 label_position: Alignment = Alignment.INSIDE, padding_angle: int = 0):
        super().__init__(name)
        self.show_legend: bool = show_legend
        self.legend_position: Alignment = legend_position
        self.show_labels: bool = show_labels
        self.label_position: Alignment = label_position
        self.padding_angle: int = padding_angle

    @property
    def show_legend(self) -> bool:
        """Property: Get whether to show the legend of the pie chart."""
        return self._show_legend

    @show_legend.setter
    def show_legend(self, value: bool):
        """Property: Set whether to show the legend of the pie chart."""
        self._show_legend = value

    @property
    def legend_position(self) -> Alignment:
        """Property: Get the legend position of the pie chart."""
        return self._legend_position

    @legend_position.setter
    def legend_position(self, value: Alignment):
        """Property: Set the legend position of the pie chart."""
        self._legend_position = value

    @property
    def show_labels(self) -> bool:
        """Property: Get whether to show the labels of the pie chart."""
        return self._show_labels

    @show_labels.setter
    def show_labels(self, value: bool): 
        """Property: Set whether to show the labels of the pie chart."""
        self._show_labels = value

    @property
    def label_position(self) -> Alignment:
        """Property: Get the label position of the pie chart."""
        return self._label_position

    @label_position.setter
    def label_position(self, value: Alignment):
        """Property: Set the label position of the pie chart."""
        self._label_position = value

    @property
    def padding_angle(self) -> int:
        """Property: Get the padding angle of the pie chart."""
        return self._padding_angle

    @padding_angle.setter
    def padding_angle(self, value: int):
        """Property: Set the padding angle of the pie chart."""
        self._padding_angle = value

    def __repr__(self):
        return (
            f"PieChart(name={self.name}, show_legend={self.show_legend}, "
            f"legend_position={self.legend_position}, show_labels={self.show_labels}, "
            f"label_position={self.label_position}, padding_angle={self.padding_angle})"
        )

class RadarChart(Chart):
    """Represents a radar chart component in the dashboard.

    Args:
        name (str): The name of the radar chart.
        show_grid (bool): Whether to show the grid in the radar chart.
        show_tooltip (bool): Whether to show the tooltip in the radar chart.
        show_radius_axis (bool): Whether to show the radius axis in the radar chart.

    Attributes:
        name (str): The name of the radar chart.
        show_grid (bool): Whether to show the grid in the radar chart.
        show_tooltip (bool): Whether to show the tooltip in the radar chart.
        show_radius_axis (bool): Whether to show the radius axis in the radar chart.
    """

    def __init__(self, name: str, show_grid: bool = True, show_tooltip: bool = True,
                 show_radius_axis: bool = True):
        super().__init__(name)
        self.show_grid = show_grid
        self.show_tooltip = show_tooltip
        self.show_radius_axis = show_radius_axis

    @property
    def show_grid(self) -> bool:
        """Property: Get the show_grid option of the radar chart."""
        return self._show_grid

    @show_grid.setter
    def show_grid(self, value: bool):
        """Property: Set the show_grid option of the radar chart."""
        self._show_grid = value

    @property
    def show_tooltip(self) -> bool:
        """Property: Get the show_tooltip option of the radar chart."""
        return self._show_tooltip

    @show_tooltip.setter
    def show_tooltip(self, value: bool):
        """Property: Set the show_tooltip option of the radar chart."""
        self._show_tooltip = value

    @property
    def show_radius_axis(self) -> bool:
        """Property: Get the show_radius_axis option of the radar chart."""
        return self._show_radius_axis

    @show_radius_axis.setter
    def show_radius_axis(self, value: bool):
        """Property: Set the show_radius_axis option of the radar chart."""
        self._show_radius_axis = value

    def __repr__(self):
        return (
            f"RadarChart(name={self.name}, show_grid={self.show_grid}, show_tooltip={self.show_tooltip}, "
            f"show_radius_axis={self.show_radius_axis})"
        )

class RadialBarChart(Chart):
    """Represents a radial bar chart component in the dashboard.

    Args:
        name (str): The name of the radial bar chart.
        start_angle (int): The start angle of the radial bar chart.
        end_angle (int): The end angle of the radial bar chart.
        
    Attributes:
        name (str): The name of the radial bar chart.
        start_angle (int): The start angle of the radial bar chart.
        end_angle (int): The end angle of the radial bar chart.
    """

    def __init__(self, name: str, start_angle: int, end_angle: int):
        super().__init__(name)
        self.start_angle = start_angle
        self.end_angle = end_angle

    @property
    def start_angle(self) -> int:
        """Property: Get the start angle of the radial bar chart."""
        return self._start_angle

    @start_angle.setter
    def start_angle(self, value: int):
        """Property: Set the start angle of the radial bar chart."""
        self._start_angle = value

    @property
    def end_angle(self) -> int:
        """Property: Get the end angle of the radial bar chart."""
        return self._end_angle

    @end_angle.setter
    def end_angle(self, value: int):
        """Property: Set the end angle of the radial bar chart."""
        self._end_angle = value

    def __repr__(self):
        return (
            f"RadialBarChart(name={self.name}, start_angle={self.start_angle}, end_angle={self.end_angle})"
        )

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
