"""
Tests for type checking in DomainModel and Association classes.
This test ensures that type errors are caught at construction time,
not at runtime when accessing attributes.
"""
import pytest
from besser.BUML.metamodel.structural import *


def test_domain_model_associations_type_check():
    """Test that DomainModel raises TypeError when associations contain non-Association instances."""
    # Create classes
    movie_class = Class(name="Movie", attributes=set())
    star_class = Class(name="Star", attributes=set())
    
    # Create Property instances (ends)
    movie_starring = Property(
        name="starring", owner=None, type=movie_class, multiplicity=Multiplicity(0, "*"))
    star_starred_in = Property(
        name="starredIn", owner=None, type=star_class, multiplicity=Multiplicity(0, "*"))
    
    # This should raise TypeError because we're passing Property instances instead of Association instances
    with pytest.raises(TypeError) as excinfo:
        model_cinema = DomainModel(
            name="Cinema",
            types={movie_class, star_class},
            associations={movie_starring, star_starred_in}  # Wrong: these are Property, not Association
        )
    
    assert "Expected Association instance, but got Property instance" in str(excinfo.value)


def test_domain_model_types_type_check():
    """Test that DomainModel raises TypeError when types contain non-Type instances."""
    # Create a valid class
    movie_class = Class(name="Movie", attributes=set())
    
    # Create Property instances (which are not Type instances)
    invalid_type = Property(name="invalid", owner=None, type=movie_class)
    
    # This should raise TypeError
    with pytest.raises(TypeError) as excinfo:
        model = DomainModel(
            name="TestModel",
            types={movie_class, invalid_type}  # Wrong: Property is not a Type
        )
    
    assert "Expected Type instance, but got Property instance" in str(excinfo.value)


def test_domain_model_generalizations_type_check():
    """Test that DomainModel raises TypeError when generalizations contain non-Generalization instances."""
    class1 = Class(name="Class1", attributes=set())
    class2 = Class(name="Class2", attributes=set())
    
    # Create a valid generalization
    valid_gen = Generalization(general=class1, specific=class2)
    
    # Create Property instance (not a Generalization)
    invalid_gen = Property(name="invalid", owner=None, type=class1)
    
    # This should raise TypeError
    with pytest.raises(TypeError) as excinfo:
        model = DomainModel(
            name="TestModel",
            types={class1, class2},
            generalizations={valid_gen, invalid_gen}  # Wrong: Property is not a Generalization
        )
    
    assert "Expected Generalization instance, but got Property instance" in str(excinfo.value)


def test_domain_model_packages_type_check():
    """Test that DomainModel raises TypeError when packages contain non-Package instances."""
    class1 = Class(name="Class1", attributes=set())
    
    # Create a valid package
    valid_package = Package(name="Package1", elements={class1})
    
    # Create Property instance (not a Package)
    invalid_package = Property(name="invalid", owner=None, type=class1)
    
    # This should raise TypeError
    with pytest.raises(TypeError) as excinfo:
        model = DomainModel(
            name="TestModel",
            types={class1},
            packages={valid_package, invalid_package}  # Wrong: Property is not a Package
        )
    
    assert "Expected Package instance, but got Property instance" in str(excinfo.value)


def test_domain_model_constraints_type_check():
    """Test that DomainModel raises TypeError when constraints contain non-Constraint instances."""
    class1 = Class(name="Class1", attributes=set())
    
    # Create a valid constraint
    valid_constraint = Constraint(name="Constraint1", context=class1, expression="self.name <> ''", language="OCL")
    
    # Create Property instance (not a Constraint)
    invalid_constraint = Property(name="invalid", owner=None, type=class1)
    
    # This should raise TypeError
    with pytest.raises(TypeError) as excinfo:
        model = DomainModel(
            name="TestModel",
            types={class1},
            constraints={valid_constraint, invalid_constraint}  # Wrong: Property is not a Constraint
        )
    
    assert "Expected Constraint instance, but got Property instance" in str(excinfo.value)


def test_association_ends_type_check():
    """Test that Association raises TypeError when ends contain non-Property instances."""
    class1 = Class(name="Class1", attributes=set())
    class2 = Class(name="Class2", attributes=set())
    
    # Create valid ends
    valid_end1 = Property(name="end1", owner=None, type=class1, multiplicity=Multiplicity(0, 1))
    
    # Create an invalid end (using a Class instead of Property)
    invalid_end = class2  # Wrong: this is a Class, not a Property
    
    # This should raise TypeError
    with pytest.raises(TypeError) as excinfo:
        association = BinaryAssociation(name="TestAssoc", ends={valid_end1, invalid_end})
    
    assert "Expected Property instance, but got Class instance" in str(excinfo.value)


def test_domain_model_correct_types_accepted():
    """Test that DomainModel accepts correct types without errors."""
    # Create classes
    movie_class = Class(name="Movie", attributes=set())
    star_class = Class(name="Star", attributes=set())
    studio_class = Class(name="Studio", attributes=set())
    
    # Create association ends
    movie_starring = Property(
        name="starring", owner=None, type=movie_class, multiplicity=Multiplicity(0, "*"))
    star_starred_in = Property(
        name="starredIn", owner=None, type=star_class, multiplicity=Multiplicity(0, "*"))
    
    movie_owned_by = Property(
        name="ownedBy", owner=None, type=studio_class, multiplicity=Multiplicity(1, 1))
    studio_owns = Property(
        name="owns", owner=None, type=movie_class, multiplicity=Multiplicity(0, "*"))
    
    # Create associations (correct way)
    relation_stars = BinaryAssociation(
        name="stars", ends={movie_starring, star_starred_in})
    relation_owns = BinaryAssociation(
        name="owns", ends={movie_owned_by, studio_owns})
    
    # Create generalization
    generalization = Generalization(general=movie_class, specific=studio_class)
    
    # Create package
    package = Package(name="Package1", elements={movie_class})
    
    # Create constraint
    constraint = Constraint(name="Constraint1", context=movie_class, expression="self.name <> ''", language="OCL")
    
    # This should work without errors
    model_cinema = DomainModel(
        name="Cinema",
        types={movie_class, studio_class, star_class},
        associations={relation_stars, relation_owns},  # Correct: these are Association instances
        generalizations={generalization},
        packages={package},
        constraints={constraint}
    )
    
    # Verify the model was created correctly
    assert len(model_cinema.associations) == 2
    assert relation_stars in model_cinema.associations
    assert relation_owns in model_cinema.associations
    
    # Verify we can iterate over associations and access ends without errors
    for association in model_cinema.associations:
        assert len(association.ends) == 2
        assert hasattr(association, 'ends')


def test_association_setter_type_check():
    """Test that the associations setter also performs type checking."""
    class1 = Class(name="Class1", attributes=set())
    class2 = Class(name="Class2", attributes=set())
    
    # Create valid model
    model = DomainModel(name="TestModel", types={class1, class2})
    
    # Try to set associations with wrong type
    invalid_association = Property(name="invalid", owner=None, type=class1)
    
    with pytest.raises(TypeError) as excinfo:
        model.associations = {invalid_association}
    
    assert "Expected Association instance, but got Property instance" in str(excinfo.value)


def test_types_setter_type_check():
    """Test that the types setter also performs type checking."""
    # Create valid model
    model = DomainModel(name="TestModel")
    
    class1 = Class(name="Class1", attributes=set())
    invalid_type = Property(name="invalid", owner=None, type=class1)
    
    with pytest.raises(TypeError) as excinfo:
        model.types = {class1, invalid_type}
    
    assert "Expected Type instance, but got Property instance" in str(excinfo.value)


def test_generalizations_setter_type_check():
    """Test that the generalizations setter also performs type checking."""
    class1 = Class(name="Class1", attributes=set())
    class2 = Class(name="Class2", attributes=set())
    
    # Create valid model
    model = DomainModel(name="TestModel", types={class1, class2})
    
    invalid_gen = Property(name="invalid", owner=None, type=class1)
    
    with pytest.raises(TypeError) as excinfo:
        model.generalizations = {invalid_gen}
    
    assert "Expected Generalization instance, but got Property instance" in str(excinfo.value)


def test_packages_setter_type_check():
    """Test that the packages setter also performs type checking."""
    class1 = Class(name="Class1", attributes=set())
    
    # Create valid model
    model = DomainModel(name="TestModel", types={class1})
    
    invalid_package = Property(name="invalid", owner=None, type=class1)
    
    with pytest.raises(TypeError) as excinfo:
        model.packages = {invalid_package}
    
    assert "Expected Package instance, but got Property instance" in str(excinfo.value)


def test_constraints_setter_type_check():
    """Test that the constraints setter also performs type checking."""
    class1 = Class(name="Class1", attributes=set())
    
    # Create valid model
    model = DomainModel(name="TestModel", types={class1})
    
    invalid_constraint = Property(name="invalid", owner=None, type=class1)
    
    with pytest.raises(TypeError) as excinfo:
        model.constraints = {invalid_constraint}
    
    assert "Expected Constraint instance, but got Property instance" in str(excinfo.value)


def test_ends_setter_type_check():
    """Test that the ends setter also performs type checking."""
    class1 = Class(name="Class1", attributes=set())
    class2 = Class(name="Class2", attributes=set())
    
    # Create valid ends
    valid_end1 = Property(name="end1", owner=None, type=class1, multiplicity=Multiplicity(0, 1))
    valid_end2 = Property(name="end2", owner=None, type=class2, multiplicity=Multiplicity(0, 1))
    
    # Create association
    association = BinaryAssociation(name="TestAssoc", ends={valid_end1, valid_end2})
    
    # Try to set ends with wrong type
    invalid_end = class1  # Class instance, not Property
    
    with pytest.raises(TypeError) as excinfo:
        association.ends = {valid_end1, invalid_end}
    
    assert "Expected Property instance, but got Class instance" in str(excinfo.value)
