"""``DataBinding.endpoint``: where a chart or metric card reads its records.

Without it a chart could only read the CRUD list of its class (``/<class>/``),
so a dashboard over a large table downloaded every row and aggregated in the
browser. An endpoint is a backend path or an absolute http(s) URL.
"""
import pytest

from besser.BUML.metamodel.gui.binding import DataBinding, is_valid_endpoint
from besser.BUML.metamodel.structural import Class, FloatType, Property


def _sale():
    amount = Property(name="amount", type=FloatType)
    return Class(name="Sale", attributes={amount})


def test_no_endpoint_by_default():
    assert DataBinding(domain_concept=_sale()).endpoint is None


@pytest.mark.parametrize("endpoint", [
    "/stats/sales",
    "/stats/sales?year=2025&by=region",
    "/",
    "https://api.example.org/v1/sales",
    "HTTP://localhost:8000/sales/",
])
def test_a_path_or_an_absolute_url_is_kept(endpoint):
    assert DataBinding(domain_concept=_sale(), endpoint=endpoint).endpoint == endpoint


def test_surrounding_spaces_are_stripped():
    assert DataBinding(domain_concept=_sale(), endpoint="  /stats/sales  ").endpoint == "/stats/sales"


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_endpoint_means_none(blank):
    assert DataBinding(domain_concept=_sale(), endpoint=blank).endpoint is None


@pytest.mark.parametrize("endpoint", [
    "stats/sales",          # no leading slash: relative to what?
    "ftp://host/sales",
    "/stats/sales by region",
    "javascript:alert(1)",
    "//evil.example.org/sales",  # scheme-relative: another host
    "http://",                   # no host
    "https:///sales",
    "/stats/sales\x00",          # control characters
    "/stats/\x07sales",
    "/stats/sa\u200bles",        # invisible format characters
    "/stats/sa\u2028les",
])
def test_anything_else_is_refused(endpoint):
    with pytest.raises(ValueError, match="Invalid endpoint"):
        DataBinding(domain_concept=_sale(), endpoint=endpoint)


def test_a_non_string_is_refused():
    with pytest.raises(TypeError, match="endpoint must be a string"):
        DataBinding(domain_concept=_sale(), endpoint=42)


def test_the_setter_validates_too():
    binding = DataBinding(domain_concept=_sale())
    binding.endpoint = "/stats/sales"
    assert binding.endpoint == "/stats/sales"
    binding.endpoint = None
    assert binding.endpoint is None
    with pytest.raises(ValueError):
        binding.endpoint = "stats"


def test_is_valid_endpoint_matches_the_setter():
    assert is_valid_endpoint("/stats/sales")
    assert is_valid_endpoint("https://api.example.org/v1")
    assert not is_valid_endpoint("//evil.example.org")
    assert not is_valid_endpoint("")
    assert not is_valid_endpoint(None)


def test_str_names_the_endpoint_and_the_aggregation():
    text = str(DataBinding(domain_concept=_sale(), aggregation="sum", endpoint="/stats/sales"))
    assert "endpoint=/stats/sales" in text
    assert "aggregation=DataAggregation.SUM" in text
