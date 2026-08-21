import os

from besser.BUML.metamodel.action_language.action_language import Nothing
from besser.generators.backend import BackendGenerator
from besser.BUML.metamodel.structural import (
    DomainModel,
    Class,
    Property,
    Enumeration,
    EnumerationLiteral,
    StringType,
    IntegerType,
    FloatType,
    DateTimeType,
    BinaryAssociation,
    Multiplicity,
    Generalization, Method, MethodImplementationType,
)


def test_backend_generator_full_uml(tmpdir):
    """
    End-to-end backend generation for a mixed UML model:
    - Enumerations
    - Deep inheritance (Person <- Employee <- Manager)
    - N:1, 1:N, and N:M associations
    """
    role_enum = Enumeration(
        name="Role",
        literals={
            EnumerationLiteral(name="STAFF"),
            EnumerationLiteral(name="LEAD"),
        },
    )

    person = Class(
        name="Person",
        attributes={
            Property(name="name", type=StringType),
            Property(name="email", type=StringType),
            Property(name="created_at", type=DateTimeType),
        },
    )
    employee = Class(
        name="Employee",
        attributes={
            Property(name="salary", type=FloatType),
            Property(name="role", type=role_enum),
        },
    )
    manager = Class(
        name="Manager",
        attributes={
            Property(name="level", type=IntegerType),
        },
    )
    department = Class(
        name="Department",
        attributes={
            Property(name="title", type=StringType),
        },
    )
    project = Class(
        name="Project",
        attributes={
            Property(name="code", type=StringType),
        },
    )

    gen_person_employee = Generalization(general=person, specific=employee)
    gen_employee_manager = Generalization(general=employee, specific=manager)

    employee_department = BinaryAssociation(
        name="EmployeeDepartment",
        ends={
            Property(name="department", type=department, multiplicity=Multiplicity(1, 1)),
            Property(name="employees", type=employee, multiplicity=Multiplicity(0, "*")),
        },
    )
    manager_project = BinaryAssociation(
        name="ManagerProject",
        ends={
            Property(name="projects", type=project, multiplicity=Multiplicity(0, "*")),
            Property(name="managers", type=manager, multiplicity=Multiplicity(0, "*")),
        },
    )

    model = DomainModel(
        name="FullBackendModel",
        types={person, employee, manager, department, project, role_enum},
        associations={employee_department, manager_project},
        generalizations={gen_person_employee, gen_employee_manager},
    )

    output_dir = tmpdir.mkdir("output_full")
    generator = BackendGenerator(model=model, output_dir=str(output_dir))
    generator.generate()

    api_file = os.path.join(str(output_dir), "main_api.py")
    pydantic_file = os.path.join(str(output_dir), "pydantic_classes.py")
    sqlalchemy_file = os.path.join(str(output_dir), "sql_alchemy.py")

    assert os.path.isfile(api_file)
    assert os.path.isfile(pydantic_file)
    assert os.path.isfile(sqlalchemy_file)

    with open(api_file, "r", encoding="utf-8") as f:
        api_code = f.read()
    with open(pydantic_file, "r", encoding="utf-8") as f:
        pydantic_code = f.read()
    with open(sqlalchemy_file, "r", encoding="utf-8") as f:
        sqlalchemy_code = f.read()

    # Pydantic: enums and inheritance
    assert "class Role(Enum):" in pydantic_code
    assert "STAFF = \"STAFF\"" in pydantic_code
    assert "class PersonCreate(BaseModel):" in pydantic_code
    assert "class EmployeeCreate(PersonCreate):" in pydantic_code
    assert "class ManagerCreate(EmployeeCreate):" in pydantic_code

    # Pydantic: relationship fields
    assert "department: int  # N:1 Relationship (mandatory)" in pydantic_code
    assert "employees: Optional[List[int]] = None  # 1:N Relationship" in pydantic_code
    assert "projects: Optional[List[int]] = None  # N:M Relationship (optional)" in pydantic_code
    assert "managers: Optional[List[int]] = None  # N:M Relationship (optional)" in pydantic_code

    # SQLAlchemy: inheritance + FK + join table
    assert "class Employee(Person):" in sqlalchemy_code
    assert "class Manager(Employee):" in sqlalchemy_code
    assert "department_id: Mapped_[int] = mapped_column(ForeignKey_(\"department.id\")" in sqlalchemy_code
    assert "managerproject = Table_(" in sqlalchemy_code

    # REST API: constructor uses inherited attributes
    assert "name=manager_data.name" in api_code
    assert "salary=manager_data.salary" in api_code
    assert "level=manager_data.level" in api_code


def test_backend_generator_full_uml_and_implem(tmpdir):
    """
    End-to-end backend generation for a mixed UML model:
    - Enumerations
    - Deep inheritance (Person <- Employee <- Manager)
    - N:1, 1:N, and N:M associations
    - Method implementations
    """
    role_enum = Enumeration(
        name="Role",
        literals={
            EnumerationLiteral(name="STAFF"),
            EnumerationLiteral(name="LEAD"),
        },
    )

    person = Class(
        name="Person",
        attributes={
            Property(name="name", type=StringType),
            Property(name="email", type=StringType),
            Property(name="created_at", type=DateTimeType),
        },
    )
    employee = Class(
        name="Employee",
        attributes={
            Property(name="salary", type=FloatType),
            Property(name="role", type=role_enum),
        },
    )
    manager = Class(
        name="Manager",
        attributes={
            Property(name="level", type=IntegerType),
        },
    )
    manager_promote: Method = Method(name="promote",
                                                parameters=[], type=None,
                                                implementation_type=MethodImplementationType.BAL)
    manager_promote.code = """def promote() -> nothing {
        this.salary = 1.1 * this.salary;
        this.level = this.level +1
    }"""
    manager.add_method(manager_promote)

    department = Class(
        name="Department",
        attributes={
            Property(name="title", type=StringType),
        },
    )
    project = Class(
        name="Project",
        attributes={
            Property(name="code", type=StringType),
        },
    )

    gen_person_employee = Generalization(general=person, specific=employee)
    gen_employee_manager = Generalization(general=employee, specific=manager)

    employee_department = BinaryAssociation(
        name="EmployeeDepartment",
        ends={
            Property(name="department", type=department, multiplicity=Multiplicity(1, 1)),
            Property(name="employees", type=employee, multiplicity=Multiplicity(0, "*")),
        },
    )
    manager_project = BinaryAssociation(
        name="ManagerProject",
        ends={
            Property(name="projects", type=project, multiplicity=Multiplicity(0, "*")),
            Property(name="managers", type=manager, multiplicity=Multiplicity(0, "*")),
        },
    )

    model = DomainModel(
        name="FullBackendModel",
        types={person, employee, manager, department, project, role_enum},
        associations={employee_department, manager_project},
        generalizations={gen_person_employee, gen_employee_manager},
    )

    output_dir = tmpdir.mkdir("output_full")
    generator = BackendGenerator(model=model, output_dir=str(output_dir))
    generator.generate()

    api_file = os.path.join(str(output_dir), "main_api.py")
    pydantic_file = os.path.join(str(output_dir), "pydantic_classes.py")
    sqlalchemy_file = os.path.join(str(output_dir), "sql_alchemy.py")

    assert os.path.isfile(api_file)
    assert os.path.isfile(pydantic_file)
    assert os.path.isfile(sqlalchemy_file)

    with open(api_file, "r", encoding="utf-8") as f:
        api_code = f.read()
    with open(pydantic_file, "r", encoding="utf-8") as f:
        pydantic_code = f.read()
    with open(sqlalchemy_file, "r", encoding="utf-8") as f:
        sqlalchemy_code = f.read()

    # Pydantic: enums and inheritance
    assert "class Role(Enum):" in pydantic_code
    assert "STAFF = \"STAFF\"" in pydantic_code
    assert "class PersonCreate(BaseModel):" in pydantic_code
    assert "class EmployeeCreate(PersonCreate):" in pydantic_code
    assert "class ManagerCreate(EmployeeCreate):" in pydantic_code

    # Pydantic: relationship fields
    assert "department: int  # N:1 Relationship (mandatory)" in pydantic_code
    assert "employees: Optional[List[int]] = None  # 1:N Relationship" in pydantic_code
    assert "projects: Optional[List[int]] = None  # N:M Relationship (optional)" in pydantic_code
    assert "managers: Optional[List[int]] = None  # N:M Relationship (optional)" in pydantic_code

    # SQLAlchemy: inheritance + FK + join table
    assert "class Employee(Person):" in sqlalchemy_code
    assert "class Manager(Employee):" in sqlalchemy_code
    assert "department_id: Mapped_[int] = mapped_column(ForeignKey_(\"department.id\")" in sqlalchemy_code
    assert "managerproject = Table_(" in sqlalchemy_code

    # REST API: constructor uses inherited attributes
    assert "name=manager_data.name" in api_code
    assert "salary=manager_data.salary" in api_code
    assert "level=manager_data.level" in api_code

    # REST API: Methods implementation
    assert "inst_to_update = _manager_object" in api_code
    assert "await update_manager(inst_to_update.id, ManagerCreate(created_at = inst_to_update.created_at, email = inst_to_update.email, level = inst_to_update.level, name = inst_to_update.name, role = inst_to_update.role, salary = (1.1 * _manager_object.salary), department = inst_to_update.department, projects = inst_to_update.projects), database)" in api_code
    assert "await update_manager(inst_to_update.id, ManagerCreate(created_at = inst_to_update.created_at, email = inst_to_update.email, level = (_manager_object.level + 1), name = inst_to_update.name, role = inst_to_update.role, salary = inst_to_update.salary, department = inst_to_update.department, projects = inst_to_update.projects), database)" in api_code


def test_backend_generator_one_to_one_optional_field(tmpdir):
    """
    Ensure the non-FK side of a 1:1 relationship still gets a field,
    and optionality follows the end multiplicity.
    """
    device = Class(
        name="ADevice",
        attributes={
            Property(name="name", type=StringType),
        },
    )
    serial = Class(
        name="BSerial",
        attributes={
            Property(name="code", type=StringType),
        },
    )

    device_serial = BinaryAssociation(
        name="DeviceSerial",
        ends={
            Property(name="device", type=device, multiplicity=Multiplicity(0, 1)),
            Property(name="serial", type=serial, multiplicity=Multiplicity(1, 1)),
        },
    )

    model = DomainModel(
        name="OneToOneOptionalModel",
        types={device, serial},
        associations={device_serial},
    )

    output_dir = tmpdir.mkdir("output_one_to_one")
    generator = BackendGenerator(model=model, output_dir=str(output_dir))
    generator.generate()

    pydantic_file = os.path.join(str(output_dir), "pydantic_classes.py")
    with open(pydantic_file, "r", encoding="utf-8") as f:
        pydantic_code = f.read()

    # FK side (ADevice) should be mandatory
    assert "class ADeviceCreate(BaseModel):" in pydantic_code
    assert "serial: int  # 1:1 Relationship (mandatory)" in pydantic_code

    # Non-FK side (BSerial) should still have the relationship field (optional)
    assert "class BSerialCreate(BaseModel):" in pydantic_code
    assert "device: Optional[int] = None  # 1:1 Relationship (optional)" in pydantic_code


def test_backend_generator_nested_creations_nm(tmpdir):
    """
    Validate nested_creations=True adds Union[Create, int] types for N:M relationships.
    """
    student = Class(name="Student", attributes={Property(name="name", type=StringType)})
    course = Class(name="Course", attributes={Property(name="title", type=StringType)})

    enrollment = BinaryAssociation(
        name="Enrollment",
        ends={
            Property(name="students", type=student, multiplicity=Multiplicity(0, "*")),
            Property(name="courses", type=course, multiplicity=Multiplicity(0, "*")),
        },
    )

    model = DomainModel(
        name="NestedCreationsModel",
        types={student, course},
        associations={enrollment},
    )

    output_dir = tmpdir.mkdir("output_nested")
    generator = BackendGenerator(model=model, output_dir=str(output_dir), nested_creations=True)
    generator.generate()

    pydantic_file = os.path.join(str(output_dir), "pydantic_classes.py")
    with open(pydantic_file, "r", encoding="utf-8") as f:
        pydantic_code = f.read()

    assert "class StudentCreate(BaseModel):" in pydantic_code
    assert "courses: Optional[List[Union[\"CourseCreate\", int]]] = None  # N:M Relationship" in pydantic_code
    assert "class CourseCreate(BaseModel):" in pydantic_code
    assert "students: Optional[List[Union[\"StudentCreate\", int]]] = None  # N:M Relationship" in pydantic_code
