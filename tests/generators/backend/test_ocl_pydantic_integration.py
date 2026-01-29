"""
Integration Tests for OCL Constraint Generated Pydantic Validators

These tests verify that the Pydantic validators generated from OCL constraints
work correctly at runtime, rejecting invalid values and accepting valid ones.
"""

import pytest
import os
import tempfile
import sys
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Constraint,
    IntegerType, StringType, FloatType, BooleanType,
    BinaryAssociation, Multiplicity
)
from besser.generators.pydantic_classes import PydanticGenerator


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def player_class():
    """Create a Player class with various attributes."""
    return Class(name="Player", attributes={
        Property(name="age", type=IntegerType),
        Property(name="name", type=StringType),
        Property(name="salary", type=FloatType),
        Property(name="jerseyNumber", type=IntegerType),
    })


@pytest.fixture
def team_class():
    """Create a Team class."""
    return Class(name="Team", attributes={
        Property(name="name", type=StringType),
        Property(name="city", type=StringType),
    })


@pytest.fixture
def simple_domain_model(player_class, team_class):
    """Create a simple domain model with Player and Team."""
    association = BinaryAssociation(
        name="team_player",
        ends={
            Property(name="team", type=team_class, multiplicity=Multiplicity(1, 1)),
            Property(name="players", type=player_class, multiplicity=Multiplicity(0, 9999)),
        }
    )
    return DomainModel(
        name="TestModel",
        types={player_class, team_class},
        associations={association}
    )


def generate_and_load_pydantic_classes(domain_model, output_dir):
    """
    Generate Pydantic classes and load them as a module.
    
    Returns a dict containing the generated classes.
    """
    generator = PydanticGenerator(
        model=domain_model,
        backend=True,
        output_dir=output_dir
    )
    generator.generate()
    
    # Read the generated file
    file_path = os.path.join(output_dir, "pydantic_classes.py")
    with open(file_path, 'r') as f:
        code = f.read()
    
    # Execute the generated code and return the namespace
    namespace = {}
    exec(code, namespace)
    return namespace


# ============================================================================
# Tests for Greater Than Constraint (>)
# ============================================================================

class TestGreaterThanConstraint:
    """Tests for greater than (>) constraints."""
    
    def test_greater_than_accepts_valid_value(self, player_class, simple_domain_model):
        """Test that values greater than the constraint are accepted."""
        simple_domain_model.constraints = {
            Constraint(
                name="min_age",
                context=player_class,
                expression="context Player inv: self.age > 10",
                language="OCL"
            )
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            
            # age = 11 should be valid (11 > 10)
            player = PlayerCreate(age=11, name="Test", salary=50000.0, jerseyNumber=7, team=1)
            assert player.age == 11
    
    def test_greater_than_rejects_boundary_value(self, player_class, simple_domain_model):
        """Test that the boundary value is rejected."""
        simple_domain_model.constraints = {
            Constraint(
                name="min_age",
                context=player_class,
                expression="context Player inv: self.age > 10",
                language="OCL"
            )
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            
            # age = 10 should be rejected (10 is not > 10)
            with pytest.raises(Exception) as exc_info:
                PlayerCreate(age=10, name="Test", salary=50000.0, jerseyNumber=7, team=1)
            assert "age" in str(exc_info.value).lower()
    
    def test_greater_than_rejects_lower_value(self, player_class, simple_domain_model):
        """Test that values below the constraint are rejected."""
        simple_domain_model.constraints = {
            Constraint(
                name="min_age",
                context=player_class,
                expression="context Player inv: self.age > 10",
                language="OCL"
            )
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            
            # age = 5 should be rejected
            with pytest.raises(Exception):
                PlayerCreate(age=5, name="Test", salary=50000.0, jerseyNumber=7, team=1)


# ============================================================================
# Tests for Greater Than or Equal Constraint (>=)
# ============================================================================

class TestGreaterOrEqualConstraint:
    """Tests for greater than or equal (>=) constraints."""
    
    def test_gte_accepts_boundary_value(self, player_class, simple_domain_model):
        """Test that the boundary value is accepted for >=."""
        simple_domain_model.constraints = {
            Constraint(
                name="min_jersey",
                context=player_class,
                expression="context Player inv: self.jerseyNumber >= 1",
                language="OCL"
            )
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            
            # jerseyNumber = 1 should be valid (1 >= 1)
            player = PlayerCreate(age=25, name="Test", salary=50000.0, jerseyNumber=1, team=1)
            assert player.jerseyNumber == 1
    
    def test_gte_rejects_lower_value(self, player_class, simple_domain_model):
        """Test that values below the constraint are rejected."""
        simple_domain_model.constraints = {
            Constraint(
                name="min_jersey",
                context=player_class,
                expression="context Player inv: self.jerseyNumber >= 1",
                language="OCL"
            )
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            
            # jerseyNumber = 0 should be rejected
            with pytest.raises(Exception):
                PlayerCreate(age=25, name="Test", salary=50000.0, jerseyNumber=0, team=1)


# ============================================================================
# Tests for Less Than Constraint (<)
# ============================================================================

class TestLessThanConstraint:
    """Tests for less than (<) constraints."""
    
    def test_less_than_accepts_valid_value(self, player_class, simple_domain_model):
        """Test that values less than the constraint are accepted."""
        simple_domain_model.constraints = {
            Constraint(
                name="max_age",
                context=player_class,
                expression="context Player inv: self.age < 65",
                language="OCL"
            )
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            
            # age = 40 should be valid
            player = PlayerCreate(age=40, name="Test", salary=50000.0, jerseyNumber=7, team=1)
            assert player.age == 40
    
    def test_less_than_rejects_boundary_value(self, player_class, simple_domain_model):
        """Test that the boundary value is rejected."""
        simple_domain_model.constraints = {
            Constraint(
                name="max_age",
                context=player_class,
                expression="context Player inv: self.age < 65",
                language="OCL"
            )
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            
            # age = 65 should be rejected (65 is not < 65)
            with pytest.raises(Exception):
                PlayerCreate(age=65, name="Test", salary=50000.0, jerseyNumber=7, team=1)


# ============================================================================
# Tests for Less Than or Equal Constraint (<=)
# ============================================================================

class TestLessOrEqualConstraint:
    """Tests for less than or equal (<=) constraints."""
    
    def test_lte_accepts_boundary_value(self, player_class, simple_domain_model):
        """Test that the boundary value is accepted for <=."""
        simple_domain_model.constraints = {
            Constraint(
                name="max_jersey",
                context=player_class,
                expression="context Player inv: self.jerseyNumber <= 99",
                language="OCL"
            )
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            
            # jerseyNumber = 99 should be valid
            player = PlayerCreate(age=25, name="Test", salary=50000.0, jerseyNumber=99, team=1)
            assert player.jerseyNumber == 99
    
    def test_lte_rejects_higher_value(self, player_class, simple_domain_model):
        """Test that values above the constraint are rejected."""
        simple_domain_model.constraints = {
            Constraint(
                name="max_jersey",
                context=player_class,
                expression="context Player inv: self.jerseyNumber <= 99",
                language="OCL"
            )
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            
            # jerseyNumber = 100 should be rejected
            with pytest.raises(Exception):
                PlayerCreate(age=25, name="Test", salary=50000.0, jerseyNumber=100, team=1)


# ============================================================================
# Tests for Multiple Constraints
# ============================================================================

class TestMultipleConstraints:
    """Tests for multiple constraints on the same class."""
    
    def test_range_constraint_accepts_valid(self, player_class, simple_domain_model):
        """Test that values within a range are accepted."""
        simple_domain_model.constraints = {
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
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            
            # age = 30 should be valid (10 < 30 < 65)
            player = PlayerCreate(age=30, name="Test", salary=50000.0, jerseyNumber=7, team=1)
            assert player.age == 30
    
    def test_range_constraint_rejects_too_low(self, player_class, simple_domain_model):
        """Test that values below the range are rejected."""
        simple_domain_model.constraints = {
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
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            
            # age = 5 should be rejected
            with pytest.raises(Exception):
                PlayerCreate(age=5, name="Test", salary=50000.0, jerseyNumber=7, team=1)
    
    def test_range_constraint_rejects_too_high(self, player_class, simple_domain_model):
        """Test that values above the range are rejected."""
        simple_domain_model.constraints = {
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
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            
            # age = 70 should be rejected
            with pytest.raises(Exception):
                PlayerCreate(age=70, name="Test", salary=50000.0, jerseyNumber=7, team=1)
    
    def test_multiple_properties_constrained(self, player_class, simple_domain_model):
        """Test constraints on multiple properties."""
        simple_domain_model.constraints = {
            Constraint(
                name="min_age",
                context=player_class,
                expression="context Player inv: self.age > 10",
                language="OCL"
            ),
            Constraint(
                name="valid_jersey",
                context=player_class,
                expression="context Player inv: self.jerseyNumber >= 1",
                language="OCL"
            )
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            
            # Valid: age=25 (>10), jerseyNumber=10 (>=1)
            player = PlayerCreate(age=25, name="Test", salary=50000.0, jerseyNumber=10, team=1)
            assert player.age == 25
            assert player.jerseyNumber == 10
            
            # Invalid age
            with pytest.raises(Exception):
                PlayerCreate(age=5, name="Test", salary=50000.0, jerseyNumber=10, team=1)
            
            # Invalid jersey
            with pytest.raises(Exception):
                PlayerCreate(age=25, name="Test", salary=50000.0, jerseyNumber=0, team=1)


# ============================================================================
# Tests for No Constraints
# ============================================================================

class TestNoConstraints:
    """Tests for classes without constraints."""
    
    def test_no_constraints_accepts_any_value(self, player_class, team_class, simple_domain_model):
        """Test that classes without constraints accept any value."""
        simple_domain_model.constraints = set()  # No constraints
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            
            # Any values should be valid
            player = PlayerCreate(age=-100, name="", salary=-1.0, jerseyNumber=0, team=1)
            assert player.age == -100


# ============================================================================
# Tests for Constraint on Different Classes
# ============================================================================

class TestConstraintsDifferentClasses:
    """Tests for constraints on different classes."""
    
    def test_constraint_only_affects_target_class(self, player_class, team_class, simple_domain_model):
        """Test that constraints only affect their target class."""
        simple_domain_model.constraints = {
            Constraint(
                name="min_age",
                context=player_class,
                expression="context Player inv: self.age > 10",
                language="OCL"
            )
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            classes = generate_and_load_pydantic_classes(simple_domain_model, tmpdir)
            PlayerCreate = classes['PlayerCreate']
            TeamCreate = classes['TeamCreate']
            
            # Player constraint should be enforced
            with pytest.raises(Exception):
                PlayerCreate(age=5, name="Test", salary=50000.0, jerseyNumber=1, team=1)
            
            # Team should not be affected (no validators on Team)
            team = TeamCreate(name="TeamA", city="CityA")
            assert team.name == "TeamA"
