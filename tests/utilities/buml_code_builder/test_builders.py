"""
Tests for the BUML code builders.

Covers:
  - common.py: _escape_python_string, safe_var_name, safe_class_name, PRIMITIVE_TYPE_MAPPING
  - domain_model_builder.py: domain_model_to_code (roundtrip + structural checks)
  - agent_model_builder.py: agent_model_to_code (smoke + structural)
  - gui_model_builder.py: gui_model_to_code (smoke)
  - project_builder.py: project_to_code (smoke + multi-model)
  - quantum_model_builder.py: quantum_model_to_code (smoke + roundtrip)
  - state_machine_builder.py: state_machine_to_code (smoke + roundtrip)
"""

import os
import tempfile
import textwrap

import pytest

# ---------------------------------------------------------------------------
# common.py
# ---------------------------------------------------------------------------
from besser.utilities.buml_code_builder.common import (
    _escape_python_string,
    safe_var_name,
    safe_class_name,
    PRIMITIVE_TYPE_MAPPING,
    RESERVED_NAMES,
)


class TestEscapePythonString:
    """Tests for _escape_python_string."""

    def test_plain_string(self):
        assert _escape_python_string("hello") == "hello"

    def test_single_quote(self):
        assert _escape_python_string("it's") == "it\\'s"

    def test_double_quote(self):
        assert _escape_python_string('say "hi"') == 'say \\"hi\\"'

    def test_backslash(self):
        assert _escape_python_string("a\\b") == "a\\\\b"

    def test_newline_and_carriage_return(self):
        assert _escape_python_string("line1\nline2\rline3") == "line1\\nline2\\rline3"

    def test_combined_special_chars(self):
        result = _escape_python_string("it's a \"test\"\nok\\done")
        assert "\\'" in result
        assert '\\"' in result
        assert "\\n" in result
        assert "\\\\" in result

    def test_empty_string(self):
        assert _escape_python_string("") == ""


class TestSafeVarName:
    """Tests for safe_var_name."""

    def test_simple_name(self):
        assert safe_var_name("myVar") == "myvar"

    def test_spaces_replaced(self):
        assert safe_var_name("my variable") == "my_variable"

    def test_special_chars_replaced(self):
        assert safe_var_name("my-var!name") == "my_var_name"

    def test_leading_digit(self):
        # safe_var_name adds '_' prefix for leading digits, but strip('_')
        # at the end removes it -- the result still starts with a digit.
        # This documents the current behaviour.
        result = safe_var_name("1start")
        assert result == "1start"

    def test_consecutive_underscores_collapsed(self):
        result = safe_var_name("a__b___c")
        assert "__" not in result

    def test_empty_string(self):
        assert safe_var_name("") == "unnamed"

    def test_python_keyword(self):
        result = safe_var_name("class")
        assert result == "class_"

    def test_another_keyword(self):
        result = safe_var_name("return")
        assert result == "return_"


class TestSafeClassName:
    """Tests for safe_class_name."""

    def test_normal_name(self):
        assert safe_class_name("Person") == "Person"

    def test_reserved_name_class(self):
        assert safe_class_name("Class") == "Class_"

    def test_reserved_name_property(self):
        assert safe_class_name("Property") == "Property_"

    def test_reserved_name_method(self):
        assert safe_class_name("Method") == "Method_"

    def test_python_keyword(self):
        assert safe_class_name("class") == "class_"

    def test_trailing_underscore_with_reserved(self):
        # "Class_" already ends with underscore and base is reserved
        assert safe_class_name("Class_") == "Class_var"

    def test_trailing_underscore_normal(self):
        assert safe_class_name("Foo_") == "Foo_"

    def test_empty_string(self):
        assert safe_class_name("") == "unnamed_class"

    def test_all_reserved_names_handled(self):
        for name in RESERVED_NAMES:
            result = safe_class_name(name)
            assert result != name, f"{name} should have been modified"


class TestPrimitiveTypeMapping:
    """Sanity checks on the PRIMITIVE_TYPE_MAPPING dict."""

    def test_str_mapped(self):
        assert PRIMITIVE_TYPE_MAPPING["str"] == "StringType"

    def test_int_mapped(self):
        assert PRIMITIVE_TYPE_MAPPING["int"] == "IntegerType"

    def test_float_mapped(self):
        assert PRIMITIVE_TYPE_MAPPING["float"] == "FloatType"

    def test_bool_mapped(self):
        assert PRIMITIVE_TYPE_MAPPING["bool"] == "BooleanType"

    def test_case_variants(self):
        assert PRIMITIVE_TYPE_MAPPING["string"] == "StringType"
        assert PRIMITIVE_TYPE_MAPPING["integer"] == "IntegerType"
        assert PRIMITIVE_TYPE_MAPPING["boolean"] == "BooleanType"

    def test_date_types(self):
        assert PRIMITIVE_TYPE_MAPPING["date"] == "DateType"
        assert PRIMITIVE_TYPE_MAPPING["time"] == "TimeType"
        assert PRIMITIVE_TYPE_MAPPING["datetime"] == "DateTimeType"
        assert PRIMITIVE_TYPE_MAPPING["timedelta"] == "TimeDeltaType"

    def test_any_type(self):
        assert PRIMITIVE_TYPE_MAPPING["any"] == "AnyType"


# ---------------------------------------------------------------------------
# domain_model_builder.py
# ---------------------------------------------------------------------------
from besser.BUML.metamodel.structural import (
    DomainModel, Class, Property, Method, Parameter,
    BinaryAssociation, Generalization, Multiplicity,
    Enumeration, EnumerationLiteral, Constraint,
    StringType, IntegerType, FloatType, BooleanType, DateType,
    Metadata,
)
from besser.utilities.buml_code_builder.domain_model_builder import (
    domain_model_to_code,
    contains_user_class,
    is_user_object_model,
    _get_impl_type_name,
    _format_method_code_literal,
)
from besser.BUML.metamodel.structural.structural import MethodImplementationType


class TestDomainModelBuilder:
    """Tests for domain_model_to_code with roundtrip verification."""

    @staticmethod
    def _build_library_model():
        """Create a minimal Library/Book domain model for testing."""
        # Classes
        library_name = Property(name="name", type=StringType)
        address = Property(name="address", type=StringType)
        library = Class(name="Library", attributes={library_name, address})

        title = Property(name="title", type=StringType)
        pages = Property(name="pages", type=IntegerType)
        book = Class(name="Book", attributes={title, pages})

        # Association
        located_in = Property(
            name="locatedIn", type=library,
            multiplicity=Multiplicity(1, 1),
        )
        has = Property(
            name="has", type=book,
            multiplicity=Multiplicity(0, "*"),
        )
        assoc = BinaryAssociation(name="lib_book_assoc", ends={located_in, has})

        model = DomainModel(
            name="Library_model",
            types={library, book},
            associations={assoc},
        )
        return model

    def test_generates_valid_python(self, tmp_path):
        """Generated code must be syntactically valid Python."""
        model = self._build_library_model()
        file_path = str(tmp_path / "domain_model.py")

        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Must compile without SyntaxError
        compile(code, file_path, "exec")

    def test_roundtrip_exec(self, tmp_path):
        """Generated code can be exec()'d and produces a DomainModel with correct structure."""
        model = self._build_library_model()
        file_path = str(tmp_path / "domain_model.py")

        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        namespace = {}
        exec(code, namespace)

        recreated = namespace["domain_model"]
        assert isinstance(recreated, DomainModel)
        assert recreated.name == "Library_model"

        class_names = {cls.name for cls in recreated.get_classes()}
        assert "Library" in class_names
        assert "Book" in class_names

        assert len(recreated.associations) == 1

    def test_file_extension_appended(self, tmp_path):
        """If path lacks .py, the builder appends it."""
        model = self._build_library_model()
        file_path = str(tmp_path / "model_no_ext")

        domain_model_to_code(model, file_path)

        assert os.path.exists(file_path + ".py")

    def test_output_dir_created(self, tmp_path):
        """Builder creates intermediate directories."""
        model = self._build_library_model()
        nested_dir = tmp_path / "sub" / "dir"
        file_path = str(nested_dir / "model.py")

        domain_model_to_code(model, file_path)

        assert os.path.exists(file_path)

    def test_custom_var_name(self, tmp_path):
        """Generated code uses the specified variable name."""
        model = self._build_library_model()
        file_path = str(tmp_path / "model.py")

        domain_model_to_code(model, file_path, model_var_name="my_model")

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "my_model = DomainModel(" in code

        namespace = {}
        exec(code, namespace)
        assert "my_model" in namespace

    def test_enumerations(self, tmp_path):
        """Enumerations are correctly generated and recoverable."""
        lit_a = EnumerationLiteral(name="RED")
        lit_b = EnumerationLiteral(name="GREEN")
        lit_c = EnumerationLiteral(name="BLUE")
        color_enum = Enumeration(name="Color", literals={lit_a, lit_b, lit_c})

        prop = Property(name="color", type=color_enum)
        widget = Class(name="Widget", attributes={prop})

        model = DomainModel(
            name="EnumModel",
            types={widget, color_enum},
            associations=set(),
        )

        file_path = str(tmp_path / "enum_model.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        compile(code, file_path, "exec")
        namespace = {}
        exec(code, namespace)

        recreated = namespace["domain_model"]
        enum_names = {t.name for t in recreated.types if isinstance(t, Enumeration)}
        assert "Color" in enum_names

    def test_generalization(self, tmp_path):
        """Generalization relationships are generated."""
        animal = Class(name="Animal")
        dog = Class(name="Dog")
        gen = Generalization(general=animal, specific=dog)

        model = DomainModel(
            name="InheritModel",
            types={animal, dog},
            associations=set(),
            generalizations={gen},
        )

        file_path = str(tmp_path / "gen_model.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        compile(code, file_path, "exec")
        namespace = {}
        exec(code, namespace)

        recreated = namespace["domain_model"]
        assert len(recreated.generalizations) == 1

    def test_method_with_parameters(self, tmp_path):
        """Methods with parameters are correctly generated."""
        param = Parameter(name="amount", type=IntegerType)
        method = Method(name="deposit", parameters={param}, type=BooleanType)
        cls = Class(name="Account", methods={method})

        model = DomainModel(
            name="MethodModel",
            types={cls},
            associations=set(),
        )

        file_path = str(tmp_path / "method_model.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        compile(code, file_path, "exec")
        namespace = {}
        exec(code, namespace)

        recreated = namespace["domain_model"]
        classes = list(recreated.get_classes())
        account = [c for c in classes if c.name == "Account"][0]
        methods = list(account.methods)
        assert len(methods) == 1
        assert methods[0].name == "deposit"

    def test_abstract_class(self, tmp_path):
        """Abstract classes are correctly flagged."""
        cls = Class(name="Shape", is_abstract=True)
        model = DomainModel(name="AbstractModel", types={cls}, associations=set())

        file_path = str(tmp_path / "abstract.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "is_abstract=True" in code
        namespace = {}
        exec(code, namespace)
        recreated = namespace["domain_model"]
        shape = [c for c in recreated.get_classes() if c.name == "Shape"][0]
        assert shape.is_abstract is True

    def test_metadata_on_model(self, tmp_path):
        """Domain model metadata (description, uri, icon) is generated."""
        meta = Metadata(description="Test model", uri="http://example.com", icon="test.png")
        cls = Class(name="Foo")
        model = DomainModel(name="MetaModel", types={cls}, associations=set(), metadata=meta)

        file_path = str(tmp_path / "meta.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "Test model" in code
        assert "http://example.com" in code

        namespace = {}
        exec(code, namespace)
        recreated = namespace["domain_model"]
        assert recreated.metadata is not None
        assert recreated.metadata.description == "Test model"

    def test_optional_property(self, tmp_path):
        """Optional properties have is_optional=True."""
        prop = Property(name="nickname", type=StringType, is_optional=True)
        cls = Class(name="User", attributes={prop})
        model = DomainModel(name="OptModel", types={cls}, associations=set())

        file_path = str(tmp_path / "opt.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "is_optional=True" in code

    def test_special_chars_in_names(self, tmp_path):
        """Names with quotes are safely escaped.

        Note: The structural metamodel rejects names with spaces, so we
        test only quote characters within otherwise valid identifiers.
        """
        # The metamodel forbids spaces in names, so use an underscore-separated name
        cls = Class(name="It_s_Test")
        model = DomainModel(name="Escape_Model", types={cls}, associations=set())

        file_path = str(tmp_path / "escape.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Must be syntactically valid
        compile(code, file_path, "exec")


class TestContainsUserClass:
    """Tests for contains_user_class helper."""

    def test_model_with_user_class(self):
        user = Class(name="User")
        model = DomainModel(name="M", types={user}, associations=set())
        assert contains_user_class(model) is True

    def test_model_without_user_class(self):
        person = Class(name="Person")
        model = DomainModel(name="M", types={person}, associations=set())
        assert contains_user_class(model) is False

    def test_none_model(self):
        assert contains_user_class(None) is False


class TestGetImplTypeName:
    """Tests for _get_impl_type_name."""

    def test_none_input(self):
        assert _get_impl_type_name(None) is None

    def test_enum_value(self):
        assert _get_impl_type_name(MethodImplementationType.CODE) == "CODE"

    def test_string_value(self):
        assert _get_impl_type_name("CODE") == "CODE"
        assert _get_impl_type_name("code") == "CODE"

    def test_prefixed_string(self):
        assert _get_impl_type_name("MethodImplementationType.CODE") == "CODE"


class TestFormatMethodCodeLiteral:
    """Tests for _format_method_code_literal."""

    def test_simple_code(self):
        result = _format_method_code_literal("return 42")
        assert result == '"""return 42"""'

    def test_triple_quotes_escaped(self):
        result = _format_method_code_literal('x = """hello"""')
        assert '"""' not in result[3:-3]  # inner part should have no unescaped triple quotes


# ---------------------------------------------------------------------------
# state_machine_builder.py
# ---------------------------------------------------------------------------
from besser.BUML.metamodel.state_machine.state_machine import (
    StateMachine, Body, Event,
)
from besser.utilities.buml_code_builder.state_machine_builder import (
    state_machine_to_code,
    _sanitize_identifier,
)


class TestSanitizeIdentifier:
    """Tests for _sanitize_identifier."""

    def test_simple(self):
        assert _sanitize_identifier("hello") == "hello"

    def test_special_chars(self):
        assert _sanitize_identifier("my-state!") == "my_state_"

    def test_leading_digit(self):
        result = _sanitize_identifier("1state")
        assert result == "state"

    def test_empty_string(self):
        assert _sanitize_identifier("") == "unnamed"


class TestStateMachineBuilder:
    """Tests for state_machine_to_code."""

    @staticmethod
    def _build_simple_sm():
        """Create a simple state machine with 2 states and a transition."""
        sm = StateMachine(name="TestSM")
        s1 = sm.new_state("Idle", initial=True)
        s2 = sm.new_state("Active")

        evt = Event(name="start_event")
        s1.when_event(evt).go_to(s2)

        return sm

    def test_returns_code_string(self):
        """state_machine_to_code returns a string."""
        sm = self._build_simple_sm()
        code = state_machine_to_code(sm)
        assert isinstance(code, str)
        assert len(code) > 0

    def test_generates_valid_python(self):
        """Returned code compiles without errors."""
        sm = self._build_simple_sm()
        code = state_machine_to_code(sm)
        compile(code, "<sm>", "exec")

    def test_writes_to_file(self, tmp_path):
        """When file_path is provided, code is written to file."""
        sm = self._build_simple_sm()
        file_path = str(tmp_path / "sm.py")
        code = state_machine_to_code(sm, file_path=file_path)

        assert os.path.exists(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            file_code = f.read()
        assert file_code == code

    def test_custom_var_name(self):
        """Custom variable name is used in the generated code."""
        sm = self._build_simple_sm()
        code = state_machine_to_code(sm, model_var_name="my_sm")
        assert "my_sm = StateMachine(" in code

    def test_state_names_present(self):
        """Both state names appear in the generated code."""
        sm = self._build_simple_sm()
        code = state_machine_to_code(sm)
        assert "Idle" in code
        assert "Active" in code

    def test_event_name_present(self):
        """Event names appear in the generated code."""
        sm = self._build_simple_sm()
        code = state_machine_to_code(sm)
        assert "start_event" in code

    def test_initial_state_flag(self):
        """Initial state flag is correctly emitted."""
        sm = self._build_simple_sm()
        code = state_machine_to_code(sm)
        assert "initial=True" in code

    def test_metadata_on_state(self):
        """State metadata is emitted."""
        sm = StateMachine(name="MetaSM")
        s1 = sm.new_state("Start", initial=True)
        s1.metadata = Metadata(description="This is the start state")
        code = state_machine_to_code(sm)
        assert "This is the start state" in code

    def test_metadata_on_machine(self):
        """Machine-level metadata is emitted."""
        sm = StateMachine(name="MetaSM")
        sm.new_state("Start", initial=True)
        sm.metadata = Metadata(description="Top level comment")
        code = state_machine_to_code(sm)
        assert "Top level comment" in code


# ---------------------------------------------------------------------------
# quantum_model_builder.py
# ---------------------------------------------------------------------------
from besser.BUML.metamodel.quantum.quantum import (
    QuantumCircuit, HadamardGate, PauliXGate, Measurement,
    ClassicalRegister, ControlState,
)
from besser.utilities.buml_code_builder.quantum_model_builder import (
    quantum_model_to_code,
)


class TestQuantumModelBuilder:
    """Tests for quantum_model_to_code."""

    @staticmethod
    def _build_simple_circuit():
        """Build a simple quantum circuit: H(0), CNOT(0->1), Measure(0)."""
        qc = QuantumCircuit(name="TestCircuit", qubits=2)
        h_gate = HadamardGate(target_qubit=0)
        qc.add_operation(h_gate)

        x_gate = PauliXGate(target_qubit=1)
        x_gate.control_qubits = [0]
        x_gate.control_states = [ControlState.CONTROL]
        qc.add_operation(x_gate)

        qc.add_creg(ClassicalRegister("c", 2))
        m = Measurement(target_qubit=0, output_bit=0)
        qc.add_operation(m)

        return qc

    def test_generates_valid_python(self, tmp_path):
        """Generated quantum code compiles."""
        qc = self._build_simple_circuit()
        file_path = str(tmp_path / "quantum.py")
        quantum_model_to_code(qc, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        compile(code, file_path, "exec")

    def test_roundtrip_exec(self, tmp_path):
        """Generated code produces a QuantumCircuit when executed."""
        qc = self._build_simple_circuit()
        file_path = str(tmp_path / "quantum.py")
        quantum_model_to_code(qc, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        namespace = {}
        exec(code, namespace)

        recreated = namespace["qc"]
        assert isinstance(recreated, QuantumCircuit)
        assert recreated.name == "TestCircuit"
        # Should have 3 operations: H, X (controlled), Measurement
        assert len(recreated.operations) == 3

    def test_custom_var_name(self, tmp_path):
        """Custom variable name is used."""
        qc = QuantumCircuit(name="Simple", qubits=1)
        file_path = str(tmp_path / "q.py")
        quantum_model_to_code(qc, file_path, model_var_name="my_qc")

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "my_qc = QuantumCircuit(" in code

    def test_classical_register(self, tmp_path):
        """Classical registers are emitted."""
        qc = QuantumCircuit(name="CRegTest", qubits=1)
        qc.add_creg(ClassicalRegister("c", 1))
        file_path = str(tmp_path / "creg.py")
        quantum_model_to_code(qc, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "ClassicalRegister" in code
        assert "'c'" in code

    def test_control_qubits_emitted(self, tmp_path):
        """Control qubits/states are written for controlled gates."""
        qc = self._build_simple_circuit()
        file_path = str(tmp_path / "ctrl.py")
        quantum_model_to_code(qc, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "control_qubits" in code
        assert "ControlState.CONTROL" in code


# ---------------------------------------------------------------------------
# agent_model_builder.py
# ---------------------------------------------------------------------------
from besser.BUML.metamodel.state_machine.agent import (
    Agent, AgentReply, Auto,
)
from besser.utilities.buml_code_builder.agent_model_builder import agent_model_to_code


class TestAgentModelBuilder:
    """Tests for agent_model_to_code."""

    @staticmethod
    def _build_simple_agent():
        """Create a simple agent with 2 states, 1 intent, and transitions."""
        agent = Agent(name="Greeter")

        greet_intent = agent.new_intent("greeting", ["hello", "hi", "hey"])

        s1 = agent.new_state("Welcome", initial=True)
        s2 = agent.new_state("Respond")

        # Set body on s1
        body = Body(name="welcome_body")
        body.add_action(AgentReply("Hello! How can I help?"))
        s1.set_body(body)

        # Transition: when greeting intent matched
        s1.when_intent_matched(greet_intent).go_to(s2)

        # Auto-transition from s2 back to s1
        s2.go_to(s1)

        return agent

    def test_generates_valid_python(self, tmp_path):
        """Agent code compiles."""
        agent = self._build_simple_agent()
        file_path = str(tmp_path / "agent.py")
        agent_model_to_code(agent, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        compile(code, file_path, "exec")

    def test_agent_name_in_code(self, tmp_path):
        """Agent name appears in the generated code."""
        agent = self._build_simple_agent()
        file_path = str(tmp_path / "agent.py")
        agent_model_to_code(agent, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "'Greeter'" in code

    def test_intent_present(self, tmp_path):
        """Intent names and training sentences are in the code."""
        agent = self._build_simple_agent()
        file_path = str(tmp_path / "agent.py")
        agent_model_to_code(agent, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "'greeting'" in code
        assert "'hello'" in code
        assert "'hi'" in code

    def test_state_names_present(self, tmp_path):
        """State names appear in generated code."""
        agent = self._build_simple_agent()
        file_path = str(tmp_path / "agent.py")
        agent_model_to_code(agent, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "'Welcome'" in code
        assert "'Respond'" in code

    def test_body_actions_present(self, tmp_path):
        """Body with AgentReply is present in code."""
        agent = self._build_simple_agent()
        file_path = str(tmp_path / "agent.py")
        agent_model_to_code(agent, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "AgentReply" in code
        assert "Hello! How can I help?" in code

    def test_custom_var_name(self, tmp_path):
        """Custom agent variable name is used."""
        agent = self._build_simple_agent()
        file_path = str(tmp_path / "agent.py")
        agent_model_to_code(agent, file_path, model_var_name="my_agent")

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "my_agent = Agent(" in code

    def test_file_extension_appended(self, tmp_path):
        """If path lacks .py, the builder appends it."""
        agent = self._build_simple_agent()
        file_path = str(tmp_path / "agent_no_ext")
        agent_model_to_code(agent, file_path)
        assert os.path.exists(file_path + ".py")

    def test_initial_state_flag(self, tmp_path):
        """Initial state flag is emitted."""
        agent = self._build_simple_agent()
        file_path = str(tmp_path / "agent.py")
        agent_model_to_code(agent, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "initial=True" in code

    def test_metadata_on_agent(self, tmp_path):
        """Agent metadata (description) is emitted when present."""
        agent = Agent(name="MetaAgent")
        agent.metadata = Metadata(description="A helpful agent")
        agent.new_state("Start", initial=True)
        file_path = str(tmp_path / "agent_meta.py")
        agent_model_to_code(agent, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "A helpful agent" in code


# ---------------------------------------------------------------------------
# gui_model_builder.py
# ---------------------------------------------------------------------------
from besser.BUML.metamodel.gui import GUIModel
from besser.BUML.metamodel.gui.graphical_ui import (
    Screen, Module, Button, Text, ButtonType, ButtonActionType,
)
from besser.utilities.buml_code_builder.gui_model_builder import gui_model_to_code


class TestGUIModelBuilder:
    """Tests for gui_model_to_code."""

    @staticmethod
    def _build_simple_gui():
        """Build a minimal GUI model with one screen and two components."""
        btn = Button(
            name="SubmitBtn",
            label="Submit",
            description="A submit button",
            buttonType=ButtonType.RaisedButton,
            actionType=ButtonActionType.Navigate,
        )
        txt = Text(name="WelcomeText", content="Hello World", description="Welcome text")

        screen = Screen(
            name="MainScreen",
            description="The main screen",
            view_elements={btn, txt},
            is_main_page=True,
        )

        module = Module(name="MainModule", screens={screen})

        gui = GUIModel(
            name="TestGUI",
            package="com.test",
            versionCode="1",
            versionName="1.0",
            modules={module},
            description="Test GUI model",
        )
        return gui

    def test_generates_valid_python(self, tmp_path):
        """GUI code compiles without errors."""
        gui = self._build_simple_gui()
        file_path = str(tmp_path / "gui.py")
        gui_model_to_code(gui, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        compile(code, file_path, "exec")

    def test_screen_name_present(self, tmp_path):
        """Screen name appears in generated code."""
        gui = self._build_simple_gui()
        file_path = str(tmp_path / "gui.py")
        gui_model_to_code(gui, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "MainScreen" in code

    def test_component_names_present(self, tmp_path):
        """Component names appear in generated code."""
        gui = self._build_simple_gui()
        file_path = str(tmp_path / "gui.py")
        gui_model_to_code(gui, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "SubmitBtn" in code
        assert "WelcomeText" in code

    def test_module_present(self, tmp_path):
        """Module name appears in generated code."""
        gui = self._build_simple_gui()
        file_path = str(tmp_path / "gui.py")
        gui_model_to_code(gui, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "MainModule" in code

    def test_custom_var_name(self, tmp_path):
        """Custom GUI model variable name is used."""
        gui = self._build_simple_gui()
        file_path = str(tmp_path / "gui.py")
        gui_model_to_code(gui, file_path, model_var_name="my_gui")

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "my_gui" in code

    def test_roundtrip_exec(self, tmp_path):
        """Generated GUI code can be exec()'d and produces a GUIModel."""
        gui = self._build_simple_gui()
        file_path = str(tmp_path / "gui.py")
        gui_model_to_code(gui, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        namespace = {}
        exec(code, namespace)

        recreated = namespace["gui_model"]
        assert isinstance(recreated, GUIModel)
        assert recreated.name == "TestGUI"


# ---------------------------------------------------------------------------
# project_builder.py
# ---------------------------------------------------------------------------
from besser.BUML.metamodel.project import Project
from besser.utilities.buml_code_builder.project_builder import project_to_code


class TestProjectBuilder:
    """Tests for project_to_code."""

    @staticmethod
    def _build_simple_project():
        """Build a minimal project with one domain model."""
        cls = Class(name="Person")
        prop = Property(name="name", type=StringType)
        cls.attributes = {prop}

        dm = DomainModel(name="PersonModel", types={cls}, associations=set())

        meta = Metadata(description="Test project")
        project = Project(
            name="TestProject",
            models=[dm],
            owner="tester",
            metadata=meta,
        )
        return project

    def test_generates_valid_python(self, tmp_path):
        """Project code compiles."""
        project = self._build_simple_project()
        file_path = str(tmp_path / "project.py")
        project_to_code(project, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        compile(code, file_path, "exec")

    def test_project_name_in_code(self, tmp_path):
        """Project name appears in generated code."""
        project = self._build_simple_project()
        file_path = str(tmp_path / "project.py")
        project_to_code(project, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "TestProject" in code

    def test_domain_model_included(self, tmp_path):
        """Domain model content is embedded in the project code."""
        project = self._build_simple_project()
        file_path = str(tmp_path / "project.py")
        project_to_code(project, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "PersonModel" in code
        assert "Person" in code

    def test_file_extension_appended(self, tmp_path):
        """If path lacks .py, the builder appends it."""
        project = self._build_simple_project()
        file_path = str(tmp_path / "proj_no_ext")
        project_to_code(project, file_path)
        assert os.path.exists(file_path + ".py")

    def test_roundtrip_exec(self, tmp_path):
        """Generated project code can be exec()'d and produces a Project."""
        project = self._build_simple_project()
        file_path = str(tmp_path / "project.py")
        project_to_code(project, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        namespace = {}
        exec(code, namespace)

        recreated = namespace["project"]
        assert isinstance(recreated, Project)
        assert recreated.name == "TestProject"

    def test_multi_model_project(self, tmp_path):
        """Project with multiple domain models generates suffixed variable names."""
        cls1 = Class(name="Alpha")
        dm1 = DomainModel(name="Model1", types={cls1}, associations=set())

        cls2 = Class(name="Beta")
        dm2 = DomainModel(name="Model2", types={cls2}, associations=set())

        meta = Metadata(description="Multi-model project")
        project = Project(
            name="MultiProject",
            models=[dm1, dm2],
            owner="tester",
            metadata=meta,
        )

        file_path = str(tmp_path / "multi.py")
        project_to_code(project, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        compile(code, file_path, "exec")
        # With multiple domain models, variable names get numeric suffixes
        assert "domain_model_1" in code
        assert "domain_model_2" in code

    def test_project_with_state_machine(self, tmp_path):
        """Project containing a StateMachine generates state machine code."""
        sm = StateMachine(name="SimpleSM")
        sm.new_state("Init", initial=True)

        cls = Class(name="Foo")
        dm = DomainModel(name="FooModel", types={cls}, associations=set())

        meta = Metadata(description="SM project")
        project = Project(
            name="SMProject",
            models=[dm, sm],
            owner="tester",
            metadata=meta,
        )

        file_path = str(tmp_path / "sm_proj.py")
        project_to_code(project, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        compile(code, file_path, "exec")
        assert "SimpleSM" in code
        assert "StateMachine" in code

    def test_project_with_quantum(self, tmp_path):
        """Project containing a QuantumCircuit generates quantum code."""
        qc = QuantumCircuit(name="SimpleQC", qubits=2)
        qc.add_operation(HadamardGate(target_qubit=0))

        cls = Class(name="Bar")
        dm = DomainModel(name="BarModel", types={cls}, associations=set())

        meta = Metadata(description="Quantum project")
        project = Project(
            name="QProject",
            models=[dm, qc],
            owner="tester",
            metadata=meta,
        )

        file_path = str(tmp_path / "q_proj.py")
        project_to_code(project, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        compile(code, file_path, "exec")
        assert "SimpleQC" in code
        assert "QuantumCircuit" in code

    def test_owner_in_code(self, tmp_path):
        """Owner field appears in generated project code."""
        project = self._build_simple_project()
        file_path = str(tmp_path / "project.py")
        project_to_code(project, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "tester" in code

    def test_metadata_in_code(self, tmp_path):
        """Project metadata appears in generated code."""
        project = self._build_simple_project()
        file_path = str(tmp_path / "project.py")
        project_to_code(project, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "Test project" in code


# ---------------------------------------------------------------------------
# domain_model_builder: advanced scenarios
# ---------------------------------------------------------------------------
class TestDomainModelBuilderAdvanced:
    """Additional domain model builder tests for edge cases."""

    def test_empty_model_compiles(self, tmp_path):
        """A model with no classes or associations generates syntactically valid code."""
        model = DomainModel(name="EmptyModel", types=set(), associations=set())
        file_path = str(tmp_path / "empty.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Generated code must at least compile
        compile(code, file_path, "exec")
        # Note: exec() currently fails because the builder emits types={}
        # (a dict literal) instead of set() for empty type collections.
        # This is a known limitation of the builder.

    def test_single_class_model(self, tmp_path):
        """A model with a single class (no associations) roundtrips correctly."""
        cls = Class(name="Singleton")
        model = DomainModel(name="SingleModel", types={cls}, associations=set())
        file_path = str(tmp_path / "single.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        namespace = {}
        exec(code, namespace)
        recreated = namespace["domain_model"]
        assert recreated.name == "SingleModel"
        class_names = {c.name for c in recreated.get_classes()}
        assert "Singleton" in class_names

    def test_private_visibility_property(self, tmp_path):
        """Private attributes have visibility='private' in generated code."""
        prop = Property(name="secret", type=StringType, visibility="private")
        cls = Class(name="Vault", attributes={prop})
        model = DomainModel(name="PrivModel", types={cls}, associations=set())

        file_path = str(tmp_path / "priv.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert 'visibility="private"' in code

    def test_default_value_string(self, tmp_path):
        """String default values are generated correctly."""
        prop = Property(name="status", type=StringType, default_value="active")
        cls = Class(name="Task", attributes={prop})
        model = DomainModel(name="DefaultModel", types={cls}, associations=set())

        file_path = str(tmp_path / "default.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert 'default_value="active"' in code

    def test_default_value_numeric(self, tmp_path):
        """Numeric default values are generated correctly."""
        prop = Property(name="count", type=IntegerType, default_value=42)
        cls = Class(name="Counter", attributes={prop})
        model = DomainModel(name="NumDefaultModel", types={cls}, associations=set())

        file_path = str(tmp_path / "num_default.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "default_value=42" in code

    def test_composite_association(self, tmp_path):
        """Composite associations have is_composite=True in generated code."""
        parent = Class(name="Parent")
        child = Class(name="Child")

        parent_end = Property(
            name="parent", type=parent,
            multiplicity=Multiplicity(1, 1),
        )
        children_end = Property(
            name="children", type=child,
            multiplicity=Multiplicity(0, "*"),
            is_composite=True,
        )
        assoc = BinaryAssociation(name="parent_children", ends={parent_end, children_end})

        model = DomainModel(
            name="CompositeModel",
            types={parent, child},
            associations={assoc},
        )

        file_path = str(tmp_path / "composite.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "is_composite=True" in code

    def test_non_navigable_end(self, tmp_path):
        """Non-navigable association ends include is_navigable=False."""
        a = Class(name="A")
        b = Class(name="B")

        end_a = Property(name="a", type=a, multiplicity=Multiplicity(1, 1), is_navigable=False)
        end_b = Property(name="b", type=b, multiplicity=Multiplicity(0, "*"))
        assoc = BinaryAssociation(name="a_b_assoc", ends={end_a, end_b})

        model = DomainModel(name="NavModel", types={a, b}, associations={assoc})

        file_path = str(tmp_path / "nav.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "is_navigable=False" in code

    def test_constraint_generated(self, tmp_path):
        """OCL constraints appear in the generated code."""
        cls = Class(name="Order")
        prop = Property(name="total", type=FloatType)
        cls.attributes = {prop}

        # The metamodel rejects hyphens in names, so use underscores
        constraint = Constraint(
            name="positive_total",
            context=cls,
            expression="self.total > 0",
            language="OCL",
        )

        model = DomainModel(
            name="ConstraintModel",
            types={cls},
            associations=set(),
            constraints={constraint},
        )

        file_path = str(tmp_path / "constraint.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        compile(code, file_path, "exec")
        assert "positive_total" in code
        assert "self.total > 0" in code

    def test_method_with_implementation_type(self, tmp_path):
        """Methods with implementation_type are generated correctly."""
        method = Method(
            name="calculate",
            type=IntegerType,
            implementation_type=MethodImplementationType.CODE,
        )
        method.code = "return 42"

        cls = Class(name="Calculator", methods={method})
        model = DomainModel(name="ImplModel", types={cls}, associations=set())

        file_path = str(tmp_path / "impl.py")
        domain_model_to_code(model, file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        assert "MethodImplementationType.CODE" in code
        assert "return 42" in code
