import pytest

from besser.BUML.metamodel.gui.binding import DataAggregation, DataBinding
from besser.BUML.metamodel.structural import Class


def test_aggregation_is_optional():
    assert DataBinding(domain_concept=Class(name="Task")).aggregation is None


@pytest.mark.parametrize("value, expected", [
    (DataAggregation.SUM, DataAggregation.SUM),
    ("sum", DataAggregation.SUM),
    ("avg", DataAggregation.AVG),
    ("average", DataAggregation.AVG),
    ("COUNT", DataAggregation.COUNT),
    ("min", DataAggregation.MIN),
    ("maximum", DataAggregation.MAX),
])
def test_aggregation_accepts_the_member_or_its_editor_name(value, expected):
    assert DataBinding(domain_concept=Class(name="Task"), aggregation=value).aggregation is expected


def test_an_unknown_aggregation_is_rejected():
    with pytest.raises(ValueError):
        DataBinding(domain_concept=Class(name="Task"), aggregation="total")
