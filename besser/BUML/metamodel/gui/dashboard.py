from typing import Optional, Sequence

from besser.BUML.metamodel.gui.graphical_ui import ViewComponent
from besser.BUML.metamodel.gui.style import Alignment

class Chart(ViewComponent):
    """Represents a chart component in the dashboard.

    Args:
        name (str): The name of the chart.
        title (str | None): Optional display title of the chart.
        primary_color (str | None): Optional primary color used by the chart.
    """

    def __init__(
        self,
        name: str,
        title: Optional[str] = None,
        primary_color: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.title = title
        self.primary_color = primary_color

    @property
    def title(self) -> Optional[str]:
        """Optional[str]: Display title for the chart."""
        return self._title

    @title.setter
    def title(self, value: Optional[str]):
        self._title = value

    @property
    def primary_color(self) -> Optional[str]:
        """Optional[str]: Main color used by the chart."""
        return self._primary_color

    @primary_color.setter
    def primary_color(self, value: Optional[str]):
        self._primary_color = value

class LineChart(Chart):
    """Represents a line chart component in the dashboard.

    Args:
        name (str): The name of the line chart.
        line_width (int): The width of the line in the chart.
        show_grid (bool): Whether to show the grid in the chart.
        show_legend (bool): Whether to show the legend in the chart.
        show_tooltip (bool): Whether to show tooltips.
        curve_type (str): The curve type ('linear', 'monotone', 'step').
        animate (bool): Whether to animate the chart.
        legend_position (str): Position of the legend ('top', 'right', 'bottom', 'left').
        grid_color (str): Color of the grid lines.
        dot_size (int): Size of the data point dots.

    Attributes:
        name (str): The name of the line chart.
        line_width (int): The width of the line in the chart.
        show_grid (bool): Whether to show the grid.
        show_legend (bool): Whether to show the legend.
        show_tooltip (bool): Whether to show tooltips.
        curve_type (str): The curve type.
        animate (bool): Whether to animate the chart.
        legend_position (str): Position of the legend.
        grid_color (str): Color of the grid lines.
        dot_size (int): Size of the data point dots.
    """

    def __init__(
        self,
        name: str,
        line_width: int = 2,
        show_grid: bool = True,
        show_legend: bool = True,
        show_tooltip: bool = True,
        curve_type: str = "monotone",
        animate: bool = True,
        legend_position: str = "top",
        grid_color: str = "#e0e0e0",
        dot_size: int = 5,
        title: Optional[str] = None,
        primary_color: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, title=title, primary_color=primary_color, **kwargs)
        self.line_width = line_width
        self.show_grid = show_grid
        self.show_legend = show_legend
        self.show_tooltip = show_tooltip
        self.curve_type = curve_type
        self.animate = animate
        self.legend_position = legend_position
        self.grid_color = grid_color
        self.dot_size = dot_size

    @property
    def line_width(self) -> int:
        """Property: Get the line width of the line chart."""
        return self._line_width

    @line_width.setter
    def line_width(self, value: int):
        """Property: Set the line width of the line chart."""
        self._line_width = value

    @property
    def show_grid(self) -> bool:
        """Property: Get whether to show grid."""
        return self._show_grid

    @show_grid.setter
    def show_grid(self, value: bool):
        """Property: Set whether to show grid."""
        self._show_grid = value

    @property
    def show_legend(self) -> bool:
        """Property: Get whether to show legend."""
        return self._show_legend

    @show_legend.setter
    def show_legend(self, value: bool):
        """Property: Set whether to show legend."""
        self._show_legend = value

    @property
    def show_tooltip(self) -> bool:
        """Property: Get whether to show tooltip."""
        return self._show_tooltip

    @show_tooltip.setter
    def show_tooltip(self, value: bool):
        """Property: Set whether to show tooltip."""
        self._show_tooltip = value

    @property
    def curve_type(self) -> str:
        """Property: Get the curve type."""
        return self._curve_type

    @curve_type.setter
    def curve_type(self, value: str):
        """Property: Set the curve type."""
        self._curve_type = value

    @property
    def animate(self) -> bool:
        """Property: Get whether to animate."""
        return self._animate

    @animate.setter
    def animate(self, value: bool):
        """Property: Set whether to animate."""
        self._animate = value

    @property
    def legend_position(self) -> str:
        """Property: Get legend position."""
        return self._legend_position

    @legend_position.setter
    def legend_position(self, value: str):
        """Property: Set legend position."""
        self._legend_position = value

    @property
    def grid_color(self) -> str:
        """Property: Get grid color."""
        return self._grid_color

    @grid_color.setter
    def grid_color(self, value: str):
        """Property: Set grid color."""
        self._grid_color = value

    @property
    def dot_size(self) -> int:
        """Property: Get dot size."""
        return self._dot_size

    @dot_size.setter
    def dot_size(self, value: int):
        """Property: Set dot size."""
        self._dot_size = value

    def __repr__(self):
        return (
            f"LineChart(name={self.name}, line_width={self.line_width}, "
            f"show_grid={self.show_grid}, show_legend={self.show_legend}, "
            f"curve_type={self.curve_type})"
        )

class BarChart(Chart):
    """Represents a bar chart component in the dashboard.

    Args:
        name (str): The name of the bar chart.
        bar_width (int): The width of the bars in the chart.
        orientation (str): Orientation of bars ('vertical' or 'horizontal').
        show_grid (bool): Whether to show the grid.
        show_legend (bool): Whether to show the legend.
        show_tooltip (bool): Whether to show tooltips.
        stacked (bool): Whether to stack bars.
        animate (bool): Whether to animate the chart.
        legend_position (str): Position of the legend.
        grid_color (str): Color of the grid lines.
        bar_gap (int): Gap between bars.

    Attributes:
        name (str): The name of the bar chart.
        bar_width (int): The width of the bars in the chart.
        orientation (str): Orientation of bars.
        show_grid (bool): Whether to show the grid.
        show_legend (bool): Whether to show the legend.
        show_tooltip (bool): Whether to show tooltips.
        stacked (bool): Whether to stack bars.
        animate (bool): Whether to animate.
        legend_position (str): Position of the legend.
        grid_color (str): Color of the grid.
        bar_gap (int): Gap between bars.
    """

    def __init__(
        self,
        name: str,
        bar_width: int = 30,
        orientation: str = "vertical",
        show_grid: bool = True,
        show_legend: bool = True,
        show_tooltip: bool = True,
        stacked: bool = False,
        animate: bool = True,
        legend_position: str = "top",
        grid_color: str = "#e0e0e0",
        bar_gap: int = 4,
        title: Optional[str] = None,
        primary_color: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, title=title, primary_color=primary_color, **kwargs)
        self.bar_width = bar_width
        self.orientation = orientation
        self.show_grid = show_grid
        self.show_legend = show_legend
        self.show_tooltip = show_tooltip
        self.stacked = stacked
        self.animate = animate
        self.legend_position = legend_position
        self.grid_color = grid_color
        self.bar_gap = bar_gap

    @property
    def bar_width(self) -> int:
        """Property: Get the bar width of the bar chart."""
        return self._bar_width

    @bar_width.setter
    def bar_width(self, value: int):
        """Property: Set the bar width of the bar chart."""
        self._bar_width = value

    @property
    def orientation(self) -> str:
        """Property: Get orientation."""
        return self._orientation

    @orientation.setter
    def orientation(self, value: str):
        """Property: Set orientation."""
        self._orientation = value

    @property
    def show_grid(self) -> bool:
        """Property: Get whether to show grid."""
        return self._show_grid

    @show_grid.setter
    def show_grid(self, value: bool):
        """Property: Set whether to show grid."""
        self._show_grid = value

    @property
    def show_legend(self) -> bool:
        """Property: Get whether to show legend."""
        return self._show_legend

    @show_legend.setter
    def show_legend(self, value: bool):
        """Property: Set whether to show legend."""
        self._show_legend = value

    @property
    def show_tooltip(self) -> bool:
        """Property: Get whether to show tooltip."""
        return self._show_tooltip

    @show_tooltip.setter
    def show_tooltip(self, value: bool):
        """Property: Set whether to show tooltip."""
        self._show_tooltip = value

    @property
    def stacked(self) -> bool:
        """Property: Get whether bars are stacked."""
        return self._stacked

    @stacked.setter
    def stacked(self, value: bool):
        """Property: Set whether bars are stacked."""
        self._stacked = value

    @property
    def animate(self) -> bool:
        """Property: Get whether to animate."""
        return self._animate

    @animate.setter
    def animate(self, value: bool):
        """Property: Set whether to animate."""
        self._animate = value

    @property
    def legend_position(self) -> str:
        """Property: Get legend position."""
        return self._legend_position

    @legend_position.setter
    def legend_position(self, value: str):
        """Property: Set legend position."""
        self._legend_position = value

    @property
    def grid_color(self) -> str:
        """Property: Get grid color."""
        return self._grid_color

    @grid_color.setter
    def grid_color(self, value: str):
        """Property: Set grid color."""
        self._grid_color = value

    @property
    def bar_gap(self) -> int:
        """Property: Get bar gap."""
        return self._bar_gap

    @bar_gap.setter
    def bar_gap(self, value: int):
        """Property: Set bar gap."""
        self._bar_gap = value

    def __repr__(self):
        return (
            f"BarChart(name={self.name}, bar_width={self.bar_width}, "
            f"orientation={self.orientation}, stacked={self.stacked})"
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
        inner_radius (int): Inner radius for donut charts (0-100).
        outer_radius (int): Outer radius percentage (0-100).
        start_angle (int): Start angle in degrees.
        end_angle (int): End angle in degrees.

    Attributes:
        name (str): The name of the pie chart.
        show_legend (bool): Whether to show the legend in the pie chart.
        legend_position (Alignment): The position of the legend in the pie chart.
        show_labels (bool): Whether to show labels in the pie chart.
        label_position (Alignment): The position of the labels in the pie chart.
        padding_angle (int): The padding angle between slices in the pie chart.
        inner_radius (int): Inner radius for donut charts.
        outer_radius (int): Outer radius percentage.
        start_angle (int): Start angle in degrees.
        end_angle (int): End angle in degrees.
    """

    def __init__(
        self,
        name: str,
        show_legend: bool = True,
        legend_position: Alignment = Alignment.LEFT,
        show_labels: bool = True,
        label_position: Alignment = Alignment.INSIDE,
        padding_angle: int = 0,
        inner_radius: int = 0,
        outer_radius: int = 80,
        start_angle: int = 0,
        end_angle: int = 360,
        title: Optional[str] = None,
        primary_color: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, title=title, primary_color=primary_color, **kwargs)
        self.show_legend: bool = show_legend
        self.legend_position: Alignment = legend_position
        self.show_labels: bool = show_labels
        self.label_position: Alignment = label_position
        self.padding_angle: int = padding_angle
        self.inner_radius: int = inner_radius
        self.outer_radius: int = outer_radius
        self.start_angle: int = start_angle
        self.end_angle: int = end_angle

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

    @property
    def inner_radius(self) -> int:
        """Property: Get the inner radius."""
        return self._inner_radius

    @inner_radius.setter
    def inner_radius(self, value: int):
        """Property: Set the inner radius."""
        self._inner_radius = value

    @property
    def outer_radius(self) -> int:
        """Property: Get the outer radius."""
        return self._outer_radius

    @outer_radius.setter
    def outer_radius(self, value: int):
        """Property: Set the outer radius."""
        self._outer_radius = value

    @property
    def start_angle(self) -> int:
        """Property: Get the start angle."""
        return self._start_angle

    @start_angle.setter
    def start_angle(self, value: int):
        """Property: Set the start angle."""
        self._start_angle = value

    @property
    def end_angle(self) -> int:
        """Property: Get the end angle."""
        return self._end_angle

    @end_angle.setter
    def end_angle(self, value: int):
        """Property: Set the end angle."""
        self._end_angle = value

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
        show_legend (bool): Whether to show the legend.
        legend_position (str): Position of the legend.
        dot_size (int): Size of data point dots.
        grid_type (str): Type of grid ('polygon' or 'circle').
        stroke_width (int): Width of the stroke lines.

    Attributes:
        name (str): The name of the radar chart.
        show_grid (bool): Whether to show the grid in the radar chart.
        show_tooltip (bool): Whether to show the tooltip in the radar chart.
        show_radius_axis (bool): Whether to show the radius axis in the radar chart.
        show_legend (bool): Whether to show the legend.
        legend_position (str): Position of the legend.
        dot_size (int): Size of data point dots.
        grid_type (str): Type of grid.
        stroke_width (int): Width of the stroke lines.
    """

    def __init__(
        self,
        name: str,
        show_grid: bool = True,
        show_tooltip: bool = True,
        show_radius_axis: bool = True,
        show_legend: bool = True,
        legend_position: str = "top",
        dot_size: int = 3,
        grid_type: str = "polygon",
        stroke_width: int = 2,
        title: Optional[str] = None,
        primary_color: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, title=title, primary_color=primary_color, **kwargs)
        self.show_grid = show_grid
        self.show_tooltip = show_tooltip
        self.show_radius_axis = show_radius_axis
        self.show_legend = show_legend
        self.legend_position = legend_position
        self.dot_size = dot_size
        self.grid_type = grid_type
        self.stroke_width = stroke_width

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

    @property
    def show_legend(self) -> bool:
        """Property: Get whether to show legend."""
        return self._show_legend

    @show_legend.setter
    def show_legend(self, value: bool):
        """Property: Set whether to show legend."""
        self._show_legend = value

    @property
    def legend_position(self) -> str:
        """Property: Get legend position."""
        return self._legend_position

    @legend_position.setter
    def legend_position(self, value: str):
        """Property: Set legend position."""
        self._legend_position = value

    @property
    def dot_size(self) -> int:
        """Property: Get dot size."""
        return self._dot_size

    @dot_size.setter
    def dot_size(self, value: int):
        """Property: Set dot size."""
        self._dot_size = value

    @property
    def grid_type(self) -> str:
        """Property: Get grid type."""
        return self._grid_type

    @grid_type.setter
    def grid_type(self, value: str):
        """Property: Set grid type."""
        self._grid_type = value

    @property
    def stroke_width(self) -> int:
        """Property: Get stroke width."""
        return self._stroke_width

    @stroke_width.setter
    def stroke_width(self, value: int):
        """Property: Set stroke width."""
        self._stroke_width = value

    def __repr__(self):
        return (
            f"RadarChart(name={self.name}, show_grid={self.show_grid}, show_tooltip={self.show_tooltip}, "
            f"show_radius_axis={self.show_radius_axis}, grid_type={self.grid_type})"
        )

class RadialBarChart(Chart):
    """Represents a radial bar chart component in the dashboard.

    Args:
        name (str): The name of the radial bar chart.
        start_angle (int): The start angle of the radial bar chart.
        end_angle (int): The end angle of the radial bar chart.
        inner_radius (int): Inner radius percentage (0-100).
        outer_radius (int): Outer radius percentage (0-100).
        show_legend (bool): Whether to show the legend.
        legend_position (str): Position of the legend.
        show_tooltip (bool): Whether to show tooltips.
        
    Attributes:
        name (str): The name of the radial bar chart.
        start_angle (int): The start angle of the radial bar chart.
        end_angle (int): The end angle of the radial bar chart.
        inner_radius (int): Inner radius percentage.
        outer_radius (int): Outer radius percentage.
        show_legend (bool): Whether to show the legend.
        legend_position (str): Position of the legend.
        show_tooltip (bool): Whether to show tooltips.
    """

    def __init__(
        self,
        name: str,
        start_angle: int = 0,
        end_angle: int = 360,
        inner_radius: int = 30,
        outer_radius: int = 80,
        show_legend: bool = True,
        legend_position: str = "top",
        show_tooltip: bool = True,
        title: Optional[str] = None,
        primary_color: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, title=title, primary_color=primary_color, **kwargs)
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.show_legend = show_legend
        self.legend_position = legend_position
        self.show_tooltip = show_tooltip

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

    @property
    def inner_radius(self) -> int:
        """Property: Get the inner radius."""
        return self._inner_radius

    @inner_radius.setter
    def inner_radius(self, value: int):
        """Property: Set the inner radius."""
        self._inner_radius = value

    @property
    def outer_radius(self) -> int:
        """Property: Get the outer radius."""
        return self._outer_radius

    @outer_radius.setter
    def outer_radius(self, value: int):
        """Property: Set the outer radius."""
        self._outer_radius = value

    @property
    def show_legend(self) -> bool:
        """Property: Get whether to show legend."""
        return self._show_legend

    @show_legend.setter
    def show_legend(self, value: bool):
        """Property: Set whether to show legend."""
        self._show_legend = value

    @property
    def legend_position(self) -> str:
        """Property: Get legend position."""
        return self._legend_position

    @legend_position.setter
    def legend_position(self, value: str):
        """Property: Set legend position."""
        self._legend_position = value

    @property
    def show_tooltip(self) -> bool:
        """Property: Get whether to show tooltip."""
        return self._show_tooltip

    @show_tooltip.setter
    def show_tooltip(self, value: bool):
        """Property: Set whether to show tooltip."""
        self._show_tooltip = value

    def __repr__(self):
        return (
            f"RadialBarChart(name={self.name}, start_angle={self.start_angle}, "
            f"end_angle={self.end_angle}, inner_radius={self.inner_radius})"
        )

class TableChart(Chart):
    """Represents a tabular chart component in the dashboard.

    Args:
        name (str): The name of the table chart.
        show_header (bool): Whether to render the table header.
        striped_rows (bool): Whether to alternate row background colors.
        show_pagination (bool): Whether to display pagination information.
        rows_per_page (int): Number of rows shown per page.
        title (str | None): Optional title of the table chart.
        primary_color (str | None): Optional primary color used for the header/background.

    Attributes:
        show_header (bool): Whether to render the table header.
        striped_rows (bool): Whether to alternate row backgrounds.
        show_pagination (bool): Whether to display pagination controls.
        rows_per_page (int): Number of rows per page when pagination is enabled.
    """

    def __init__(
        self,
        name: str,
        show_header: bool = True,
        striped_rows: bool = False,
        show_pagination: bool = False,
        rows_per_page: int = 5,
        title: Optional[str] = None,
        primary_color: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        super().__init__(name, title=title, primary_color=primary_color, **kwargs)
        self.show_header = show_header
        self.striped_rows = striped_rows
        self.show_pagination = show_pagination
        self.rows_per_page = rows_per_page
        self.columns = list(columns or [])

    @property
    def show_header(self) -> bool:
        """Property: Get whether the table header is displayed."""
        return self._show_header

    @show_header.setter
    def show_header(self, value: bool):
        """Property: Set whether the table header is displayed."""
        self._show_header = bool(value)

    @property
    def striped_rows(self) -> bool:
        """Property: Get whether striped rows are enabled."""
        return self._striped_rows

    @striped_rows.setter
    def striped_rows(self, value: bool):
        """Property: Set whether striped rows are enabled."""
        self._striped_rows = bool(value)

    @property
    def show_pagination(self) -> bool:
        """Property: Get whether pagination info is displayed."""
        return self._show_pagination

    @show_pagination.setter
    def show_pagination(self, value: bool):
        """Property: Set whether pagination info is displayed."""
        self._show_pagination = bool(value)

    @property
    def rows_per_page(self) -> int:
        """Property: Get the number of rows per page."""
        return self._rows_per_page

    @rows_per_page.setter
    def rows_per_page(self, value: int):
        """Property: Set the number of rows per page (minimum 1)."""
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            int_value = 5
        self._rows_per_page = max(1, int_value)

    @property
    def columns(self) -> list[str]:
        """Property: Get the configured column names."""
        return self._columns

    @columns.setter
    def columns(self, value: Sequence[str]):
        """Property: Set the configured column names."""
        if value is None:
            self._columns = []
            return
        self._columns = [str(item) for item in value if item]

    def __repr__(self):
        return (
            f"TableChart(name={self.name}, show_header={self.show_header}, "
            f"striped_rows={self.striped_rows}, rows_per_page={self.rows_per_page}, "
            f"columns={self.columns})"
        )

class MetricCard(ViewComponent):
    """Represents a metric card / KPI card component in the dashboard.
    
    Args:
        name (str): The name of the metric card.
        metric_title (str): The title displayed on the metric card.
        format (str): The format for displaying the value ('number', 'currency', 'percentage', 'time').
        value_color (str): The color of the value text.
        value_size (int): The font size of the value text.
        show_trend (bool): Whether to show the trend indicator.
        positive_color (str): Color for positive trends.
        negative_color (str): Color for negative trends.
        title (str | None): Optional title for the metric card container.
        primary_color (str | None): Optional primary color.
    
    Attributes:
        metric_title (str): The title displayed on the metric card.
        format (str): The format for displaying the value.
        value_color (str): The color of the value text.
        value_size (int): The font size of the value text.
        show_trend (bool): Whether to show the trend indicator.
        positive_color (str): Color for positive trends.
        negative_color (str): Color for negative trends.
    """
    
    def __init__(
        self,
        name: str,
        metric_title: str = "Metric Title",
        format: str = "number",
        value_color: str = "#2c3e50",
        value_size: int = 32,
        show_trend: bool = True,
        positive_color: str = "#27ae60",
        negative_color: str = "#e74c3c",
        title: Optional[str] = None,
        primary_color: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.metric_title = metric_title
        self.format = format
        self.value_color = value_color
        self.value_size = value_size
        self.show_trend = show_trend
        self.positive_color = positive_color
        self.negative_color = negative_color
        self.title = title
        self.primary_color = primary_color
    
    @property
    def metric_title(self) -> str:
        """Property: Get the metric title."""
        return self._metric_title
    
    @metric_title.setter
    def metric_title(self, value: str):
        """Property: Set the metric title."""
        self._metric_title = value
    
    @property
    def format(self) -> str:
        """Property: Get the display format."""
        return self._format
    
    @format.setter
    def format(self, value: str):
        """Property: Set the display format."""
        self._format = value
    
    @property
    def value_color(self) -> str:
        """Property: Get the value color."""
        return self._value_color
    
    @value_color.setter
    def value_color(self, value: str):
        """Property: Set the value color."""
        self._value_color = value
    
    @property
    def value_size(self) -> int:
        """Property: Get the value font size."""
        return self._value_size
    
    @value_size.setter
    def value_size(self, value: int):
        """Property: Set the value font size."""
        self._value_size = value
    
    @property
    def show_trend(self) -> bool:
        """Property: Get whether to show trend."""
        return self._show_trend
    
    @show_trend.setter
    def show_trend(self, value: bool):
        """Property: Set whether to show trend."""
        self._show_trend = value
    
    @property
    def positive_color(self) -> str:
        """Property: Get the positive trend color."""
        return self._positive_color
    
    @positive_color.setter
    def positive_color(self, value: str):
        """Property: Set the positive trend color."""
        self._positive_color = value
    
    @property
    def negative_color(self) -> str:
        """Property: Get the negative trend color."""
        return self._negative_color
    
    @negative_color.setter
    def negative_color(self, value: str):
        """Property: Set the negative trend color."""
        self._negative_color = value
    
    @property
    def title(self) -> Optional[str]:
        """Optional[str]: Optional title for the metric card container."""
        return self._title
    
    @title.setter
    def title(self, value: Optional[str]):
        """Property: Set the title."""
        self._title = value
    
    @property
    def primary_color(self) -> Optional[str]:
        """Optional[str]: Optional primary color."""
        return self._primary_color
    
    @primary_color.setter
    def primary_color(self, value: Optional[str]):
        """Property: Set the primary color."""
        self._primary_color = value
    
    def __repr__(self):
        return (
            f"MetricCard(name={self.name}, metric_title={self.metric_title}, "
            f"format={self.format}, show_trend={self.show_trend})"
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
