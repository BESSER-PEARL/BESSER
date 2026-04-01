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


def _read(output_dir, *parts):
    """Helper to read a generated file from the layered output structure."""
    path = os.path.join(str(output_dir), *parts)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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

    # Check layered structure exists
    app_dir = os.path.join(str(output_dir), "app")
    assert os.path.isdir(os.path.join(app_dir, "models"))
    assert os.path.isdir(os.path.join(app_dir, "schemas"))
    assert os.path.isdir(os.path.join(app_dir, "routers"))
    assert os.path.isfile(os.path.join(app_dir, "main.py"))
    assert os.path.isfile(os.path.join(app_dir, "config.py"))
    assert os.path.isfile(os.path.join(app_dir, "database.py"))

    # Read per-entity files
    enums_model = _read(output_dir, "app", "models", "_enums.py")
    employee_schema = _read(output_dir, "app", "schemas", "employee.py")
    person_schema = _read(output_dir, "app", "schemas", "person.py")
    manager_schema = _read(output_dir, "app", "schemas", "manager.py")
    department_schema = _read(output_dir, "app", "schemas", "department.py")
    project_schema = _read(output_dir, "app", "schemas", "project.py")
    employee_model = _read(output_dir, "app", "models", "employee.py")
    manager_model = _read(output_dir, "app", "models", "manager.py")
    models_init = _read(output_dir, "app", "models", "__init__.py")
    manager_router = _read(output_dir, "app", "routers", "manager.py")

    # Enums defined in models/_enums.py
    assert "class Role(enum.Enum):" in enums_model
    assert 'STAFF = "STAFF"' in enums_model

    # Pydantic: inheritance
    assert "class PersonCreate(" in person_schema
    assert "class EmployeeCreate(PersonCreate):" in employee_schema
    assert "class ManagerCreate(EmployeeCreate):" in manager_schema

    # Pydantic: relationship fields
    assert "department: int  # N:1 Relationship (mandatory)" in employee_schema
    assert "employees: Optional[List[int]] = None  # 1:N Relationship" in department_schema
    assert "projects: Optional[List[int]] = None  # N:M Relationship (optional)" in manager_schema
    assert "managers: Optional[List[int]] = None  # N:M Relationship (optional)" in project_schema

    # SQLAlchemy: inheritance + FK
    assert "class Employee(Person):" in employee_model
    assert "class Manager(Employee):" in manager_model
    assert "department_id: Mapped[int] = mapped_column(ForeignKey(\"department.id\")" in employee_model

    # SQLAlchemy: join table in models/__init__.py
    assert "managerproject = Table(" in models_init

    # REST API: constructor uses inherited attributes
    assert "name=manager_data.name" in manager_router
    assert "salary=manager_data.salary" in manager_router
    assert "level=manager_data.level" in manager_router


def test_backend_generator_full_uml_and_implem(tmpdir):
    """
    End-to-end backend generation with method implementations (BAL).
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

    enums_model = _read(output_dir, "app", "models", "_enums.py")
    person_schema = _read(output_dir, "app", "schemas", "person.py")
    employee_schema = _read(output_dir, "app", "schemas", "employee.py")
    manager_schema = _read(output_dir, "app", "schemas", "manager.py")
    department_schema = _read(output_dir, "app", "schemas", "department.py")
    project_schema = _read(output_dir, "app", "schemas", "project.py")
    employee_model = _read(output_dir, "app", "models", "employee.py")
    manager_model = _read(output_dir, "app", "models", "manager.py")
    models_init = _read(output_dir, "app", "models", "__init__.py")
    manager_router = _read(output_dir, "app", "routers", "manager.py")

    # Enums defined in models/_enums.py
    assert "class Role(enum.Enum):" in enums_model
    assert 'STAFF = "STAFF"' in enums_model

    # Pydantic: inheritance
    assert "class PersonCreate(" in person_schema
    assert "class EmployeeCreate(PersonCreate):" in employee_schema
    assert "class ManagerCreate(EmployeeCreate):" in manager_schema

    # Pydantic: relationship fields
    assert "department: int  # N:1 Relationship (mandatory)" in employee_schema
    assert "employees: Optional[List[int]] = None  # 1:N Relationship" in department_schema
    assert "projects: Optional[List[int]] = None  # N:M Relationship (optional)" in manager_schema
    assert "managers: Optional[List[int]] = None  # N:M Relationship (optional)" in project_schema

    # SQLAlchemy: inheritance + FK + join table
    assert "class Employee(Person):" in employee_model
    assert "class Manager(Employee):" in manager_model
    assert "department_id: Mapped[int] = mapped_column(ForeignKey(\"department.id\")" in employee_model
    assert "managerproject = Table(" in models_init

    # REST API: constructor uses inherited attributes
    assert "name=manager_data.name" in manager_router
    assert "salary=manager_data.salary" in manager_router
    assert "level=manager_data.level" in manager_router

    # REST API: Methods implementation
    assert "inst_to_update = _manager_object" in manager_router
    assert "await update_manager(inst_to_update.id, ManagerCreate(created_at = inst_to_update.created_at, email = inst_to_update.email, level = inst_to_update.level, name = inst_to_update.name, role = inst_to_update.role, salary = (1.1 * _manager_object.salary), department = inst_to_update.department, projects = inst_to_update.projects), database)" in manager_router
    assert "await update_manager(inst_to_update.id, ManagerCreate(created_at = inst_to_update.created_at, email = inst_to_update.email, level = (_manager_object.level + 1), name = inst_to_update.name, role = inst_to_update.role, salary = inst_to_update.salary, department = inst_to_update.department, projects = inst_to_update.projects), database)" in manager_router


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

    adevice_schema = _read(output_dir, "app", "schemas", "adevice.py")
    bserial_schema = _read(output_dir, "app", "schemas", "bserial.py")

    # FK side (ADevice) should be mandatory
    assert "class ADeviceCreate(" in adevice_schema
    assert "serial: int  # 1:1 Relationship (mandatory)" in adevice_schema

    # Non-FK side (BSerial) should still have the relationship field (optional)
    assert "class BSerialCreate(" in bserial_schema
    assert "device: Optional[int] = None  # 1:1 Relationship (optional)" in bserial_schema


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

    student_schema = _read(output_dir, "app", "schemas", "student.py")
    course_schema = _read(output_dir, "app", "schemas", "course.py")

    assert "class StudentCreate(" in student_schema
    assert 'courses: Optional[List[Union["CourseCreate", int]]] = None  # N:M Relationship' in student_schema
    assert "class CourseCreate(" in course_schema
    assert 'students: Optional[List[Union["StudentCreate", int]]] = None  # N:M Relationship' in course_schema
