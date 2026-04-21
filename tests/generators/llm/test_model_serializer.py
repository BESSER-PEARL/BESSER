"""Tests for the LLM model serializer."""

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    Constraint,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    Generalization,
    Metadata,
    Method,
    Multiplicity,
    Parameter,
    PrimitiveDataType,
    Property,
    UNLIMITED_MAX_MULTIPLICITY,
    AssociationClass,
)
from besser.generators.llm.model_serializer import (
    serialize_domain_model,
    serialize_object_model,
    serialize_quantum_circuit,
    serialize_state_machines,
)


class TestSerializeDomainModel:

    def _build_blog_model(self) -> DomainModel:
        """Build a realistic blog model for testing."""
        StringType = PrimitiveDataType("str")
        IntegerType = PrimitiveDataType("int")
        BooleanType = PrimitiveDataType("bool")

        status = Enumeration(name="PostStatus", literals={
            EnumerationLiteral(name="DRAFT"),
            EnumerationLiteral(name="PUBLISHED"),
            EnumerationLiteral(name="ARCHIVED"),
        })

        user = Class(name="User", metadata=Metadata(description="A registered user"))
        user.attributes = {
            Property(name="id", type=IntegerType, is_id=True),
            Property(name="email", type=StringType),
            Property(name="username", type=StringType),
            Property(name="bio", type=StringType, is_optional=True),
        }

        post = Class(name="Post")
        post.attributes = {
            Property(name="id", type=IntegerType, is_id=True),
            Property(name="title", type=StringType),
            Property(name="content", type=StringType),
            Property(name="published", type=BooleanType),
        }
        post.methods = {
            Method(name="word_count", type=IntegerType),
        }

        comment = Class(name="Comment")
        comment.attributes = {
            Property(name="id", type=IntegerType, is_id=True),
            Property(name="text", type=StringType),
        }

        user_post = BinaryAssociation(name="User_Post", ends={
            Property(name="author", type=user, multiplicity=Multiplicity(1, 1)),
            Property(name="posts", type=post, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
        })
        post_comment = BinaryAssociation(name="Post_Comment", ends={
            Property(name="post", type=post, multiplicity=Multiplicity(1, 1)),
            Property(name="comments", type=comment, multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)),
        })

        gen = Generalization(general=user, specific=post)

        constraint = Constraint(
            name="valid_email",
            context=user,
            expression="self.email->matches('[a-zA-Z0-9.]+@[a-zA-Z0-9.]+')",
            language="OCL",
        )

        return DomainModel(
            name="BlogApp",
            types={user, post, comment, status},
            associations={user_post, post_comment},
            generalizations={gen},
            constraints={constraint},
        )

    def test_basic_structure(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        assert result["name"] == "BlogApp"
        assert "classes" in result
        assert "enumerations" in result
        assert "associations" in result
        assert "generalizations" in result
        assert "constraints" in result

    def test_classes_sorted_by_name(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        names = [c["name"] for c in result["classes"]]
        assert names == sorted(names)

    def test_class_attributes(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        user = next(c for c in result["classes"] if c["name"] == "User")
        attr_names = {a["name"] for a in user["attributes"]}
        assert "email" in attr_names
        assert "username" in attr_names
        assert "bio" in attr_names
        assert "id" in attr_names

        # Check id is marked
        id_attr = next(a for a in user["attributes"] if a["name"] == "id")
        assert id_attr.get("is_id") is True

        # Check optional
        bio_attr = next(a for a in user["attributes"] if a["name"] == "bio")
        assert bio_attr.get("is_optional") is True

    def test_methods_serialized(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        post = next(c for c in result["classes"] if c["name"] == "Post")
        assert "methods" in post
        wc = next(m for m in post["methods"] if m["name"] == "word_count")
        assert wc["return_type"] == "int"

    def test_enumerations(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        assert len(result["enumerations"]) == 1
        enum = result["enumerations"][0]
        assert enum["name"] == "PostStatus"
        assert set(enum["literals"]) == {"DRAFT", "PUBLISHED", "ARCHIVED"}

    def test_associations(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        assoc_names = {a["name"] for a in result["associations"]}
        assert "User_Post" in assoc_names
        assert "Post_Comment" in assoc_names

        user_post = next(a for a in result["associations"] if a["name"] == "User_Post")
        assert len(user_post["ends"]) == 2
        author_end = next(e for e in user_post["ends"] if e["role"] == "author")
        assert author_end["multiplicity"] == "1..1"
        posts_end = next(e for e in user_post["ends"] if e["role"] == "posts")
        assert posts_end["multiplicity"] == "0..*"

    def test_generalizations(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        assert len(result["generalizations"]) == 1
        assert result["generalizations"][0]["parent"] == "User"
        assert result["generalizations"][0]["child"] == "Post"

    def test_constraints(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        assert len(result["constraints"]) == 1
        c = result["constraints"][0]
        assert c["context"] == "User"
        assert "email" in c["expression"]

    def test_metadata_included(self):
        model = self._build_blog_model()
        result = serialize_domain_model(model)

        user = next(c for c in result["classes"] if c["name"] == "User")
        assert user["metadata"]["description"] == "A registered user"

    def test_empty_model(self):
        model = DomainModel(name="Empty")
        result = serialize_domain_model(model)
        assert result["name"] == "Empty"
        assert "classes" not in result
        assert "associations" not in result

    def test_compact_output(self):
        """Verify only non-empty sections are included (saves tokens)."""
        StringType = PrimitiveDataType("str")
        cls = Class(name="Simple")
        cls.attributes = {Property(name="name", type=StringType)}
        model = DomainModel(name="Minimal", types={cls})
        result = serialize_domain_model(model)

        assert "associations" not in result
        assert "generalizations" not in result
        assert "constraints" not in result
        assert "enumerations" not in result

    def test_inherited_attributes_flattened(self):
        """A child class should expose its parent's attributes under
        ``inherited_attributes`` with a ``from`` field pointing at the
        ancestor that declared them."""
        StringType = PrimitiveDataType("str")
        IntegerType = PrimitiveDataType("int")

        animal = Class(name="Animal")
        animal.attributes = {
            Property(name="species", type=StringType),
            Property(name="age", type=IntegerType),
        }
        dog = Class(name="Dog")
        dog.attributes = {Property(name="breed", type=StringType)}
        Generalization(general=animal, specific=dog)

        model = DomainModel(name="Zoo", types={animal, dog})
        result = serialize_domain_model(model)

        dog_data = next(c for c in result["classes"] if c["name"] == "Dog")
        assert "inherited_attributes" in dog_data
        inherited_names = {a["name"] for a in dog_data["inherited_attributes"]}
        assert inherited_names == {"species", "age"}
        for entry in dog_data["inherited_attributes"]:
            assert entry["from"] == "Animal"

    def test_inherited_methods_skip_overrides(self):
        """An inherited method should NOT appear in ``inherited_methods``
        when the child declares a same-named override."""
        StringType = PrimitiveDataType("str")

        base = Class(name="Shape")
        base.methods = {
            Method(name="area", type=StringType),
            Method(name="describe", type=StringType),
        }
        square = Class(name="Square")
        square.methods = {Method(name="area", type=StringType)}  # override
        Generalization(general=base, specific=square)

        model = DomainModel(name="Shapes", types={base, square})
        result = serialize_domain_model(model)

        square_data = next(c for c in result["classes"] if c["name"] == "Square")
        inherited_names = {m["name"] for m in square_data.get("inherited_methods", [])}
        assert "describe" in inherited_names
        assert "area" not in inherited_names  # overridden, skipped


class _FakeClassifier:
    def __init__(self, name):
        self.name = name


class _FakeDataValue:
    def __init__(self, value):
        self.value = value


class _FakeSlot:
    def __init__(self, attr_name, value):
        self.attribute = _FakeClassifier(attr_name)
        self.value = _FakeDataValue(value)


class _FakeObject:
    def __init__(self, name, classifier_name, slots=None):
        self.name_ = name
        self.name = name
        self.classifier = _FakeClassifier(classifier_name)
        self.slots = slots or []


class _FakeObjectModel:
    def __init__(self, name, objects=None, links=None):
        self.name = name
        self.objects = objects or set()
        self.links = links or set()


class TestSerializeObjectModel:

    def test_none_returns_none(self):
        assert serialize_object_model(None) is None

    def test_empty_model_returns_none(self):
        """An object model with no objects is collapsed to None — keeps
        the prompt clean instead of emitting an empty section."""
        model = _FakeObjectModel("Empty", objects=set())
        assert serialize_object_model(model) is None

    def test_basic_object_serialization(self):
        obj = _FakeObject("user_alice", "User", slots=[
            _FakeSlot("email", "alice@example.com"),
            _FakeSlot("age", 30),
        ])
        model = _FakeObjectModel("Seed", objects={obj})
        result = serialize_object_model(model)

        assert result["name"] == "Seed"
        assert len(result["objects"]) == 1
        entry = result["objects"][0]
        assert entry["name"] == "user_alice"
        assert entry["class"] == "User"
        slot_map = {s["attribute"]: s["value"] for s in entry["slots"]}
        assert slot_map == {"email": "alice@example.com", "age": 30}

    def test_non_json_values_stringified(self):
        """Unusual slot values (e.g. datetime, custom types) must still
        serialize as strings rather than raising."""
        import datetime as _dt

        obj = _FakeObject("o", "Event", slots=[
            _FakeSlot("when", _dt.datetime(2024, 1, 1, 12, 0, 0)),
        ])
        model = _FakeObjectModel("Calendar", objects={obj})
        result = serialize_object_model(model)
        slot = result["objects"][0]["slots"][0]
        assert isinstance(slot["value"], str)
        assert "2024" in slot["value"]


class _FakeTransition:
    def __init__(self, dest_name, event=None, conditions=None):
        self.dest = _FakeClassifier(dest_name)
        self.event = _FakeClassifier(event) if event else None
        self.conditions = [_FakeClassifier(c) for c in (conditions or [])]


class _FakeState:
    def __init__(self, name, initial=False, final=False, transitions=None):
        self.name = name
        self.initial = initial
        self.final = final
        self.transitions = transitions or []


class _FakeStateMachine:
    def __init__(self, name, states=None, properties=None):
        self.name = name
        self.states = states or []
        self.properties = properties or []


class TestSerializeStateMachines:

    def test_none_returns_none(self):
        assert serialize_state_machines(None) is None
        assert serialize_state_machines([]) is None

    def test_single_instance_wrapped_in_list(self):
        """Passing a lone state machine should behave identically to
        passing a 1-element list."""
        sm = _FakeStateMachine("OrderFSM", states=[
            _FakeState("Draft", initial=True),
            _FakeState("Closed", final=True),
        ])
        result = serialize_state_machines(sm)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "OrderFSM"

    def test_transitions_captured(self):
        draft = _FakeState("Draft", initial=True, transitions=[
            _FakeTransition("Submitted", event="submit", conditions=["is_valid"]),
        ])
        submitted = _FakeState("Submitted", transitions=[
            _FakeTransition("Closed"),
        ])
        closed = _FakeState("Closed", final=True)
        sm = _FakeStateMachine("OrderFSM", states=[draft, submitted, closed])

        result = serialize_state_machines([sm])
        states = result[0]["states"]
        draft_entry = next(s for s in states if s["name"] == "Draft")
        assert draft_entry["initial"] is True
        assert draft_entry["transitions"] == [
            {"to": "Submitted", "event": "submit", "conditions": ["is_valid"]},
        ]
        closed_entry = next(s for s in states if s["name"] == "Closed")
        assert closed_entry["final"] is True
        assert "transitions" not in closed_entry  # no outgoing


class _FakeRegister:
    def __init__(self, name, size):
        self.name = name
        self.size = size


class _FakeOp:
    def __init__(self, op_type, name, targets, controls=None):
        self._type = op_type
        self.name = name
        self.target_qubits = targets
        self.control_qubits = controls or []

    @property
    def __class__(self):  # pragma: no cover — tricked type() lookup
        return type(self._type, (object,), {})


class _FakeCircuit:
    def __init__(self, name, qregs=None, cregs=None, operations=None):
        self.name = name
        self.qregs = qregs or []
        self.cregs = cregs or []
        self.operations = operations or []
        self.num_qubits = sum(q.size for q in self.qregs)
        self.num_clbits = sum(c.size for c in self.cregs)


class TestSerializeQuantumCircuit:

    def test_none_returns_none(self):
        assert serialize_quantum_circuit(None) is None

    def test_register_sizes_captured(self):
        circuit = _FakeCircuit(
            "Bell",
            qregs=[_FakeRegister("q", 2)],
            cregs=[_FakeRegister("c", 2)],
        )
        result = serialize_quantum_circuit(circuit)
        assert result["name"] == "Bell"
        assert result["num_qubits"] == 2
        assert result["num_clbits"] == 2
        assert result["qregs"] == [{"name": "q", "size": 2}]
        assert result["cregs"] == [{"name": "c", "size": 2}]

    def test_operations_captured(self):
        # Use real class hierarchy — the serializer reads ``type(op).__name__``
        class Hadamard:
            def __init__(self):
                self.name = "H"
                self.target_qubits = [0]
                self.control_qubits = []
        h = Hadamard()

        circuit = _FakeCircuit("H", operations=[h])
        result = serialize_quantum_circuit(circuit)
        assert len(result["operations"]) == 1
        op = result["operations"][0]
        assert op["type"] == "Hadamard"
        assert op["targets"] == [0]
        assert "controls" not in op  # empty list suppressed
