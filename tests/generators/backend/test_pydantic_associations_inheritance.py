import pytest
import os
from besser.generators.pydantic_classes import PydanticGenerator
from besser.BUML.metamodel.structural import DomainModel, Class, Property, \
    IntegerType, StringType, Multiplicity, BinaryAssociation, Generalization


def test_inheritance():
    """Test that inheritance relationships are properly generated"""
    # Create parent class
    parent = Class(name="Parent", attributes={
        Property(name="parent_attr", type=StringType),
        Property(name="id", type=IntegerType)
    })
    
    # Create child class
    child = Class(name="Child", attributes={
        Property(name="child_attr", type=StringType),
        Property(name="id", type=IntegerType)  # This should be filtered out as it's inherited
    })
    
    # Create generalization
    generalization = Generalization(general=parent, specific=child)
    
    domain_model = DomainModel(name="TestModel", types={parent, child})
    output_file = 'output/pydantic_classes.py'
    pydantic_model = PydanticGenerator(model=domain_model, backend=False)
    pydantic_model.generate()
    
    assert os.path.exists(output_file), "The file was not created."
    
    with open(output_file, 'r') as file:
        content = file.read()
    
    # Check that Parent class is generated correctly
    assert 'class Parent(BaseModel):' in content, "Parent class definition is incorrect."
    assert 'parent_attr: str' in content, "Parent attribute is missing."
    
    # Check that Child class inherits from Parent
    assert 'class Child(Parent):' in content, "Child class does not inherit from Parent."
    assert 'child_attr: str' in content, "Child attribute is missing."
    
    # Check that inherited 'id' is not duplicated in Child
    lines = content.split('\n')
    child_section = []
    in_child = False
    for line in lines:
        if 'class Child(Parent):' in line:
            in_child = True
        elif in_child and line.startswith('class '):
            break
        elif in_child:
            child_section.append(line)
    
    # The child section should only have child_attr, not id (id is inherited)
    child_content = '\n'.join(child_section)
    assert child_content.count('id:') == 0, "Child class should not re-declare inherited 'id' attribute."
    
    os.remove(output_file)


def test_one_to_many_association():
    """Test 1:N association generation"""
    employee = Class(name="Employee", attributes={Property(name="name", type=StringType)})
    department = Class(name="Department", attributes={Property(name="dept_name", type=StringType)})
    
    # 1 Department has many Employees
    association = BinaryAssociation(name="works_in", ends={
        Property(name="department", type=department, multiplicity=Multiplicity(1, 1)),
        Property(name="employees", type=employee, multiplicity=Multiplicity(0, "*"))
    })
    
    domain_model = DomainModel(name="TestModel", types={employee, department}, associations={association})
    output_file = 'output/pydantic_classes.py'
    pydantic_model = PydanticGenerator(model=domain_model, backend=False)
    pydantic_model.generate()
    
    assert os.path.exists(output_file), "The file was not created."
    
    with open(output_file, 'r') as file:
        content = file.read()
    
    # Employee should have a reference to Department (N:1 from Employee's perspective)
    assert 'department: "Department"  # N:1 Relationship' in content, "Employee-Department association is missing."
    
    # Department should have a list of Employees (1:N from Department's perspective)
    assert 'employees: List["Employee"]  # 1:N Relationship' in content, "Department-Employee association is missing."
    
    os.remove(output_file)


def test_many_to_many_association():
    """Test N:M association generation"""
    student = Class(name="Student", attributes={Property(name="name", type=StringType)})
    course = Class(name="Course", attributes={Property(name="title", type=StringType)})
    
    # N:M relationship
    association = BinaryAssociation(name="enrollment", ends={
        Property(name="students", type=student, multiplicity=Multiplicity(0, "*")),
        Property(name="courses", type=course, multiplicity=Multiplicity(0, "*"))
    })
    
    domain_model = DomainModel(name="TestModel", types={student, course}, associations={association})
    output_file = 'output/pydantic_classes.py'
    pydantic_model = PydanticGenerator(model=domain_model, backend=False)
    pydantic_model.generate()
    
    assert os.path.exists(output_file), "The file was not created."
    
    with open(output_file, 'r') as file:
        content = file.read()
    
    # Both should have List[int] for N:M relationship (when nested_creations=False)
    assert 'courses: List[int]  # N:M Relationship' in content, "Student-Course association is missing."
    assert 'students: List[int]  # N:M Relationship' in content, "Course-Student association is missing."
    
    os.remove(output_file)


def test_one_to_one_association():
    """Test 1:1 association generation"""
    person = Class(name="Person", attributes={Property(name="name", type=StringType)})
    passport = Class(name="Passport", attributes={Property(name="number", type=StringType)})
    
    # 1:1 relationship
    association = BinaryAssociation(name="has_passport", ends={
        Property(name="person", type=person, multiplicity=Multiplicity(1, 1)),
        Property(name="passport", type=passport, multiplicity=Multiplicity(1, 1))
    })
    
    domain_model = DomainModel(name="TestModel", types={person, passport}, associations={association})
    output_file = 'output/pydantic_classes.py'
    pydantic_model = PydanticGenerator(model=domain_model, backend=False)
    pydantic_model.generate()
    
    assert os.path.exists(output_file), "The file was not created."
    
    with open(output_file, 'r') as file:
        content = file.read()
    
    # Both should have references to each other
    assert 'passport: "Passport"  # 1:1 Relationship' in content, "Person-Passport association is missing."
    assert 'person: "Person"  # 1:1 Relationship' in content, "Passport-Person association is missing."
    
    os.remove(output_file)


def test_combined_inheritance_and_associations():
    """Test that both inheritance and associations work together"""
    # Create the Unit class (parent)
    unit = Class(name="Unit", attributes={
        Property(name="size", type=IntegerType),
        Property(name="id", type=IntegerType)
    })
    
    # Create the Department class (child of Unit)
    department = Class(name="Department", attributes={
        Property(name="name", type=StringType),
        Property(name="id", type=IntegerType)
    })
    
    # Create the Employee class
    employee = Class(name="Employee", attributes={
        Property(name="salary", type=IntegerType),
        Property(name="id", type=IntegerType)
    })
    
    # Create generalization
    generalization = Generalization(general=unit, specific=department)
    
    # Create association between Employee and Department
    association = BinaryAssociation(name="manages", ends={
        Property(name="boss", type=employee, multiplicity=Multiplicity(1, 1)),
        Property(name="manages", type=department, multiplicity=Multiplicity(0, "*"))
    })
    
    domain_model = DomainModel(
        name="CompanyModel", 
        types={employee, department, unit}, 
        associations={association}
    )
    
    output_file = 'output/pydantic_classes.py'
    pydantic_model = PydanticGenerator(model=domain_model, backend=False)
    pydantic_model.generate()
    
    assert os.path.exists(output_file), "The file was not created."
    
    with open(output_file, 'r') as file:
        content = file.read()
    
    # Check inheritance
    assert 'class Unit(BaseModel):' in content, "Unit class definition is incorrect."
    assert 'class Department(Unit):' in content, "Department does not inherit from Unit."
    
    # Check that Department doesn't duplicate inherited attributes
    lines = content.split('\n')
    dept_section = []
    in_dept = False
    for line in lines:
        if 'class Department(Unit):' in line:
            in_dept = True
        elif in_dept and line.startswith('class '):
            break
        elif in_dept:
            dept_section.append(line)
    
    dept_content = '\n'.join(dept_section)
    assert dept_content.count('size:') == 0, "Department should not re-declare inherited 'size' attribute."
    
    # Check associations
    assert 'manages: List["Department"]  # 1:N Relationship' in content, "Employee-Department association is missing."
    assert 'boss: "Employee"  # N:1 Relationship' in content, "Department-Employee association is missing."
    
    os.remove(output_file)


def test_backend_mode_associations():
    """Test that associations work correctly in backend mode"""
    employee = Class(name="Employee", attributes={Property(name="name", type=StringType)})
    department = Class(name="Department", attributes={Property(name="dept_name", type=StringType)})
    
    # 1 Department has many Employees
    association = BinaryAssociation(name="works_in", ends={
        Property(name="department", type=department, multiplicity=Multiplicity(1, 1)),
        Property(name="employees", type=employee, multiplicity=Multiplicity(0, "*"))
    })
    
    domain_model = DomainModel(name="TestModel", types={employee, department}, associations={association})
    output_file = 'output/pydantic_classes.py'
    pydantic_model = PydanticGenerator(model=domain_model, backend=True)
    pydantic_model.generate()
    
    assert os.path.exists(output_file), "The file was not created."
    
    with open(output_file, 'r') as file:
        content = file.read()
    
    # In backend mode, classes should have "Create" suffix
    assert 'class EmployeeCreate(BaseModel):' in content, "EmployeeCreate class is missing."
    assert 'class DepartmentCreate(BaseModel):' in content, "DepartmentCreate class is missing."
    
    # Associations should use int type in backend mode
    assert 'department: int  # N:1 Relationship' in content, "Employee-Department association is incorrect in backend mode."
    assert 'employees: Optional[List[int]] = None  # 1:N Relationship' in content, "Department-Employee association is incorrect in backend mode."
    
    os.remove(output_file)
