"""
Comprehensive Tests for OCL Constraint Utilities

This module tests the OCL constraint parsing and validation code generation
functionality in besser/generators/pydantic_classes/ocl_utils.py
"""

import pytest
from besser.BUML.metamodel.structural import (
    DomainModel, Constraint,
)
from besser.generators.pydantic_classes.ocl_utils import (
    parse_ocl_constraint, build_constraints_map,
    _fallback_parse, _parse_value, get_constraints_for_class
)


# ============================================================================
# Test Fixtures
# ============================================================================

# player_class and team_class are provided by tests/conftest.py


@pytest.fixture
def domain_model(player_team_domain_model):
    """Alias the shared Player/Team domain model fixture."""
    return player_team_domain_model


# ============================================================================
# Tests for _parse_value
# ============================================================================

class TestParseValue:
    """Tests for the _parse_value helper function."""

    @pytest.mark.parametrize("raw, expected_value, expected_repr, expected_type", [
        ("42",           42,             "42",             "int"),
        ("-10",          -10,            "-10",            "int"),
        ("3.14",         3.14,           "3.14",           "float"),
        ("'hello'",      "hello",        "'hello'",        "str"),
        ("True",         True,           "True",           "bool"),
        ("false",        False,          "False",          "bool"),
        ("someVariable", "someVariable", "someVariable",   "unknown"),
    ], ids=["integer", "negative-int", "float", "string", "bool-true", "bool-false", "unknown"])
    def test_parse_value(self, raw, expected_value, expected_repr, expected_type):
        result = _parse_value(raw)
        assert result['value'] == expected_value
        assert result['repr'] == expected_repr
        assert result['type'] == expected_type


# ============================================================================
# Tests for _fallback_parse
# ============================================================================

class TestFallbackParse:
    """Tests for the regex-based fallback parser."""

    # --- Operator parsing (parametrized) --------------------------------------

    @pytest.mark.parametrize(
        "expression, expected_property, expected_ocl_op, expected_py_op, expected_val",
        [
            ("context Player inv inv1: self.age > 10",                 "age",          ">",  ">",  10),
            ("context Player inv: self.age < 100",                     "age",          "<",  "<",  100),
            ("context Player inv min_age: self.age >= 18",             "age",          ">=", ">=", 18),
            ("context Player inv max_jersey: self.jerseyNumber <= 99", "jerseyNumber", "<=", "<=", 99),
            ("context Player inv: self.jerseyNumber = 10",             "jerseyNumber", "=",  "==", 10),
            ("context Player inv: self.jerseyNumber <> 0",             "jerseyNumber", "<>", "!=", 0),
        ],
        ids=[">", "<", ">=", "<=", "=", "<>"],
    )
    def test_parse_comparison_operators(
        self, expression, expected_property, expected_ocl_op, expected_py_op, expected_val
    ):
        result = _fallback_parse(expression)
        assert result is not None
        assert result['property'] == expected_property
        assert result['operator'] == expected_ocl_op
        assert result['python_operator'] == expected_py_op
        assert result['value'] == expected_val

    # --- Value type detection (parametrized) ----------------------------------

    @pytest.mark.parametrize("expression, expected_prop, expected_val, expected_type", [
        ("context Player inv: self.salary > 50000.50", "salary", 50000.50, "float"),
        ("context Player inv: self.name <> ''",        "name",   "",       "str"),
        ("context Player inv: self.active = True",     "active", True,     "bool"),
    ], ids=["float-value", "string-value", "bool-value"])
    def test_parse_value_types(self, expression, expected_prop, expected_val, expected_type):
        result = _fallback_parse(expression)
        assert result is not None
        assert result['property'] == expected_prop
        assert result['value'] == expected_val
        assert result['value_type'] == expected_type

    # --- Invalid expressions return None (parametrized) -----------------------

    @pytest.mark.parametrize("expression", [
        "this is not a valid OCL expression",
        "context Player inv: age > 10",
    ], ids=["garbage", "no-self-prefix"])
    def test_invalid_expression_returns_none(self, expression):
        result = _fallback_parse(expression)
        assert result is None

    def test_parse_compound_and_same_property(self):
        """Test parsing compound constraint with the same property."""
        result = _fallback_parse(
            "context Publication inv: self.year > 1500 and self.year < 2100"
        )
        assert result is not None
        assert result['property'] == 'year'
        assert result['python_expression'] == 'v > 1500 and v < 2100'
        assert result['message'] == 'year must be > 1500 and < 2100'

    def test_compound_multiple_properties_returns_none(self):
        """Test that compound constraints with multiple properties are skipped."""
        result = _fallback_parse(
            "context Player inv: self.age > 10 and self.jerseyNumber < 99"
        )
        assert result is None


# ============================================================================
# Tests for parse_ocl_constraint
# ============================================================================

class TestParseOclConstraint:
    """Tests for the main OCL constraint parser."""
    
    def test_parse_simple_constraint(self, player_class, domain_model):
        """Test parsing a simple greater-than constraint."""
        constraint = Constraint(
            name="age_constraint",
            context=player_class,
            expression="context Player inv: self.age > 18",
            language="OCL"
        )
        domain_model.constraints = {constraint}
        
        result = parse_ocl_constraint(constraint, domain_model)
        assert result is not None
        assert result['property'] == 'age'
        assert result['python_operator'] == '>'
        assert result['value'] == 18
    
    def test_parse_named_invariant(self, player_class, domain_model):
        """Test parsing constraint with named invariant."""
        constraint = Constraint(
            name="min_jersey",
            context=player_class,
            expression="context Player inv minJersey: self.jerseyNumber >= 1",
            language="OCL"
        )
        domain_model.constraints = {constraint}
        
        result = parse_ocl_constraint(constraint, domain_model)
        assert result is not None
        assert result['property'] == 'jerseyNumber'
        assert result['python_operator'] == '>='
        assert result['value'] == 1


# ============================================================================
# Tests for get_constraints_for_class
# ============================================================================

class TestGetConstraintsForClass:
    """Tests for retrieving constraints for a specific class."""
    
    def test_get_single_constraint(self, player_class, team_class, domain_model):
        """Test getting a single constraint for a class."""
        constraint = Constraint(
            name="age_constraint",
            context=player_class,
            expression="context Player inv: self.age > 10",
            language="OCL"
        )
        domain_model.constraints = {constraint}
        
        result = get_constraints_for_class(domain_model.constraints, "Player", domain_model)
        assert len(result) == 1
        assert result[0]['property'] == 'age'
        assert result[0]['constraint_name'] == 'age_constraint'
    
    def test_get_multiple_constraints(self, player_class, domain_model):
        """Test getting multiple constraints for a class."""
        constraints = {
            Constraint(
                name="min_age",
                context=player_class,
                expression="context Player inv: self.age > 10",
                language="OCL"
            ),
            Constraint(
                name="max_age",
                context=player_class,
                expression="context Player inv: self.age < 65",
                language="OCL"
            ),
            Constraint(
                name="jersey_range",
                context=player_class,
                expression="context Player inv: self.jerseyNumber >= 1",
                language="OCL"
            )
        }
        domain_model.constraints = constraints
        
        result = get_constraints_for_class(domain_model.constraints, "Player", domain_model)
        assert len(result) == 3
        properties = {r['property'] for r in result}
        assert 'age' in properties
        assert 'jerseyNumber' in properties
    
    def test_filter_by_class(self, player_class, team_class, domain_model):
        """Test that constraints are filtered by class."""
        constraints = {
            Constraint(
                name="player_constraint",
                context=player_class,
                expression="context Player inv: self.age > 10",
                language="OCL"
            ),
            Constraint(
                name="team_constraint",
                context=team_class,
                expression="context Team inv: self.name <> ''",
                language="OCL"
            )
        }
        domain_model.constraints = constraints
        
        player_results = get_constraints_for_class(domain_model.constraints, "Player", domain_model)
        team_results = get_constraints_for_class(domain_model.constraints, "Team", domain_model)
        
        assert len(player_results) == 1
        assert player_results[0]['property'] == 'age'
        
        assert len(team_results) == 1
        assert team_results[0]['property'] == 'name'
    
    def test_filter_by_language(self, player_class, domain_model):
        """Test that only OCL constraints are returned."""
        constraints = {
            Constraint(
                name="ocl_constraint",
                context=player_class,
                expression="context Player inv: self.age > 10",
                language="OCL"
            ),
            Constraint(
                name="sql_constraint",
                context=player_class,
                expression="CHECK (age > 10)",
                language="SQL"
            )
        }
        domain_model.constraints = constraints
        
        result = get_constraints_for_class(domain_model.constraints, "Player", domain_model)
        assert len(result) == 1
        assert result[0]['constraint_name'] == 'ocl_constraint'


# ============================================================================
# Tests for build_constraints_map
# ============================================================================

class TestBuildConstraintsMap:
    """Tests for building the constraints map."""
    
    def test_empty_constraints(self, domain_model):
        """Test building map with no constraints."""
        domain_model.constraints = set()
        result = build_constraints_map(domain_model)
        assert result == {}
    
    def test_single_class_constraints(self, player_class, domain_model):
        """Test building map with constraints for one class."""
        domain_model.constraints = {
            Constraint(
                name="age_check",
                context=player_class,
                expression="context Player inv: self.age > 10",
                language="OCL"
            )
        }
        
        result = build_constraints_map(domain_model)
        assert "Player" in result
        assert len(result["Player"]) == 1
        assert "Team" not in result  # Team has no constraints
    
    def test_multiple_class_constraints(self, player_class, team_class, domain_model):
        """Test building map with constraints for multiple classes."""
        domain_model.constraints = {
            Constraint(
                name="player_age",
                context=player_class,
                expression="context Player inv: self.age > 10",
                language="OCL"
            ),
            Constraint(
                name="team_name",
                context=team_class,
                expression="context Team inv: self.name <> ''",
                language="OCL"
            )
        }
        
        result = build_constraints_map(domain_model)
        assert "Player" in result
        assert "Team" in result
        assert len(result["Player"]) == 1
        assert len(result["Team"]) == 1


# ============================================================================
# Integration Tests - Full Pipeline
# ============================================================================

class TestFullPipeline:
    """Integration tests for the complete constraint parsing pipeline."""
    
    def test_constraint_to_validator_info(self, player_class, domain_model):
        """Test the full pipeline from constraint to validator info."""
        constraint = Constraint(
            name="min_age",
            context=player_class,
            expression="context Player inv minAge: self.age > 10",
            language="OCL"
        )
        domain_model.constraints = {constraint}
        
        constraints_map = build_constraints_map(domain_model)
        
        assert "Player" in constraints_map
        player_constraints = constraints_map["Player"]
        assert len(player_constraints) == 1
        
        c = player_constraints[0]
        assert c['property'] == 'age'
        assert c['python_operator'] == '>'
        assert c['value'] == 10
        assert c['value_repr'] == '10'
        assert c['constraint_name'] == 'min_age'
    
    def test_multiple_constraints_same_property(self, player_class, domain_model):
        """Test multiple constraints on the same property (e.g., range check)."""
        domain_model.constraints = {
            Constraint(
                name="min_age",
                context=player_class,
                expression="context Player inv: self.age > 10",
                language="OCL"
            ),
            Constraint(
                name="max_age",
                context=player_class,
                expression="context Player inv: self.age < 65",
                language="OCL"
            )
        }
        
        constraints_map = build_constraints_map(domain_model)
        player_constraints = constraints_map["Player"]
        
        assert len(player_constraints) == 2
        
        # Both should be for 'age' property
        properties = [c['property'] for c in player_constraints]
        assert properties == ['age', 'age']
        
        # Check operators
        operators = {c['python_operator'] for c in player_constraints}
        assert '>' in operators
        assert '<' in operators


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.parametrize("expression, expected_property, expected_value", [
        ("context Player inv: self.age > 0",              "age",           0),
        ("context Player inv: self.salary < 1000000",     "salary",        1000000),
        ("context Player inv:   self.age   >   10",       "age",           10),
        ("context Player inv: self.jersey_number >= 1",   "jersey_number", 1),
    ], ids=["zero-value", "large-integer", "extra-whitespace", "underscore-property"])
    def test_edge_case_values(self, expression, expected_property, expected_value):
        result = _fallback_parse(expression)
        assert result is not None
        assert result['property'] == expected_property
        assert result['value'] == expected_value

    def test_empty_string_constraint(self):
        """Test constraint checking for empty string."""
        result = _fallback_parse("context Team inv: self.name <> ''")
        assert result is not None
        assert result['property'] == 'name'
        assert result['value'] == ''
        assert result['python_operator'] == '!='
