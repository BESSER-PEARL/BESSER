"""
Comprehensive Tests for OCL Constraint Utilities

This module tests the OCL constraint parsing and validation code generation
functionality in besser/generators/pydantic_classes/ocl_utils.py
"""

import pytest
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Constraint,
    IntegerType, StringType, FloatType, BooleanType,
    BinaryAssociation, Multiplicity
)
from besser.generators.pydantic_classes.ocl_utils import (
    parse_ocl_constraint, build_constraints_map, 
    _fallback_parse, _parse_value, get_constraints_for_class
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def player_class():
    """Create a Player class for testing."""
    return Class(name="Player", attributes={
        Property(name="age", type=IntegerType),
        Property(name="name", type=StringType),
        Property(name="salary", type=FloatType),
        Property(name="active", type=BooleanType),
        Property(name="jerseyNumber", type=IntegerType),
    })


@pytest.fixture
def team_class():
    """Create a Team class for testing."""
    return Class(name="Team", attributes={
        Property(name="name", type=StringType),
        Property(name="city", type=StringType),
    })


@pytest.fixture
def domain_model(player_class, team_class):
    """Create a domain model with Player and Team classes."""
    return DomainModel(
        name="TestModel",
        types={player_class, team_class}
    )


# ============================================================================
# Tests for _parse_value
# ============================================================================

class TestParseValue:
    """Tests for the _parse_value helper function."""
    
    def test_parse_integer(self):
        """Test parsing integer values."""
        result = _parse_value("42")
        assert result['value'] == 42
        assert result['repr'] == "42"
        assert result['type'] == 'int'
    
    def test_parse_negative_integer(self):
        """Test parsing negative integer values."""
        result = _parse_value("-10")
        assert result['value'] == -10
        assert result['repr'] == "-10"
        assert result['type'] == 'int'
    
    def test_parse_float(self):
        """Test parsing float values."""
        result = _parse_value("3.14")
        assert result['value'] == 3.14
        assert result['repr'] == "3.14"
        assert result['type'] == 'float'
    
    def test_parse_string_with_quotes(self):
        """Test parsing string values with single quotes."""
        result = _parse_value("'hello'")
        assert result['value'] == "hello"
        assert result['repr'] == "'hello'"
        assert result['type'] == 'str'
    
    def test_parse_boolean_true(self):
        """Test parsing boolean True."""
        result = _parse_value("True")
        assert result['value'] == True
        assert result['repr'] == "True"
        assert result['type'] == 'bool'
    
    def test_parse_boolean_false(self):
        """Test parsing boolean False."""
        result = _parse_value("false")  # lowercase
        assert result['value'] == False
        assert result['repr'] == "False"
        assert result['type'] == 'bool'
    
    def test_parse_unknown(self):
        """Test parsing unknown value type."""
        result = _parse_value("someVariable")
        assert result['value'] == "someVariable"
        assert result['repr'] == "someVariable"
        assert result['type'] == 'unknown'


# ============================================================================
# Tests for _fallback_parse
# ============================================================================

class TestFallbackParse:
    """Tests for the regex-based fallback parser."""
    
    def test_parse_greater_than(self):
        """Test parsing greater than constraint."""
        result = _fallback_parse("context Player inv inv1: self.age > 10")
        assert result is not None
        assert result['property'] == 'age'
        assert result['operator'] == '>'
        assert result['python_operator'] == '>'
        assert result['value'] == 10
        assert result['value_type'] == 'int'
    
    def test_parse_less_than(self):
        """Test parsing less than constraint."""
        result = _fallback_parse("context Player inv: self.age < 100")
        assert result is not None
        assert result['property'] == 'age'
        assert result['operator'] == '<'
        assert result['python_operator'] == '<'
        assert result['value'] == 100
    
    def test_parse_greater_or_equal(self):
        """Test parsing greater than or equal constraint."""
        result = _fallback_parse("context Player inv min_age: self.age >= 18")
        assert result is not None
        assert result['property'] == 'age'
        assert result['operator'] == '>='
        assert result['python_operator'] == '>='
        assert result['value'] == 18
    
    def test_parse_less_or_equal(self):
        """Test parsing less than or equal constraint."""
        result = _fallback_parse("context Player inv max_jersey: self.jerseyNumber <= 99")
        assert result is not None
        assert result['property'] == 'jerseyNumber'
        assert result['operator'] == '<='
        assert result['python_operator'] == '<='
        assert result['value'] == 99
    
    def test_parse_equality(self):
        """Test parsing equality constraint (OCL = maps to Python ==)."""
        result = _fallback_parse("context Player inv: self.jerseyNumber = 10")
        assert result is not None
        assert result['property'] == 'jerseyNumber'
        assert result['operator'] == '='
        assert result['python_operator'] == '=='
        assert result['value'] == 10
    
    def test_parse_not_equal(self):
        """Test parsing not equal constraint (OCL <> maps to Python !=)."""
        result = _fallback_parse("context Player inv: self.jerseyNumber <> 0")
        assert result is not None
        assert result['property'] == 'jerseyNumber'
        assert result['operator'] == '<>'
        assert result['python_operator'] == '!='
        assert result['value'] == 0
    
    def test_parse_float_value(self):
        """Test parsing constraint with float value."""
        result = _fallback_parse("context Player inv: self.salary > 50000.50")
        assert result is not None
        assert result['property'] == 'salary'
        assert result['value'] == 50000.50
        assert result['value_type'] == 'float'
    
    def test_parse_string_value(self):
        """Test parsing constraint with string value."""
        result = _fallback_parse("context Player inv: self.name <> ''")
        assert result is not None
        assert result['property'] == 'name'
        assert result['value'] == ''
        assert result['value_type'] == 'str'
    
    def test_parse_boolean_value(self):
        """Test parsing constraint with boolean value."""
        result = _fallback_parse("context Player inv: self.active = True")
        assert result is not None
        assert result['property'] == 'active'
        assert result['value'] == True
        assert result['value_type'] == 'bool'
    
    def test_invalid_expression_returns_none(self):
        """Test that invalid expressions return None."""
        result = _fallback_parse("this is not a valid OCL expression")
        assert result is None
    
    def test_no_self_prefix_returns_none(self):
        """Test that expressions without 'self.' prefix return None."""
        result = _fallback_parse("context Player inv: age > 10")
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
    
    def test_zero_value(self):
        """Test constraint with zero value."""
        result = _fallback_parse("context Player inv: self.age > 0")
        assert result is not None
        assert result['value'] == 0
    
    def test_large_integer(self):
        """Test constraint with large integer value."""
        result = _fallback_parse("context Player inv: self.salary < 1000000")
        assert result is not None
        assert result['value'] == 1000000
    
    def test_whitespace_handling(self):
        """Test constraint with extra whitespace."""
        result = _fallback_parse("context Player inv:   self.age   >   10")
        assert result is not None
        assert result['property'] == 'age'
        assert result['value'] == 10
    
    def test_underscore_in_property(self):
        """Test constraint with underscore in property name."""
        result = _fallback_parse("context Player inv: self.jersey_number >= 1")
        assert result is not None
        assert result['property'] == 'jersey_number'
    
    def test_empty_string_constraint(self):
        """Test constraint checking for empty string."""
        result = _fallback_parse("context Team inv: self.name <> ''")
        assert result is not None
        assert result['property'] == 'name'
        assert result['value'] == ''
        assert result['python_operator'] == '!='
