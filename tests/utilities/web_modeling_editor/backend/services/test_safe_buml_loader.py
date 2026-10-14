"""
Tests for the AST-based safe loader that replaces ``exec()`` in the
BUML -> JSON converters.

The loader must:
  - Execute well-formed BUML files (happy path) producing the same local
    namespace the old ``exec()`` would have.
  - Reject any construct outside the small allowlist: imports, function
    definitions, dunder access, unknown name calls, attribute access whose
    root is not a previously assigned or pre-approved name, etc.
"""

import datetime as _real_datetime

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Enumeration, PrimitiveDataType, Property,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._safe_buml_loader import (
    SAFE_DATETIME_MODULE,
    SafeBumlLoaderError,
    safe_load_buml,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def allowed_names():
    """A representative allowlist similar to the class-diagram converter's."""
    return {
        "Class": Class,
        "DomainModel": DomainModel,
        "Enumeration": Enumeration,
        "Property": Property,
        "PrimitiveDataType": PrimitiveDataType,
        "StringType": PrimitiveDataType("str"),
        "set": set,
        "list": list,
        "dict": dict,
        "tuple": tuple,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    """Well-formed BUML content must execute and produce the expected model."""

    def test_simple_class_declaration(self, allowed_names):
        content = (
            "Person = Class(name='Person')\n"
            "Animal = Class(name='Animal')\n"
            "domain_model = DomainModel(name='Test', types={Person, Animal})\n"
        )
        result = safe_load_buml(content, allowed_names)
        assert isinstance(result["domain_model"], DomainModel)
        assert result["domain_model"].name == "Test"
        assert {t.name for t in result["domain_model"].types} >= {"Person", "Animal"}

    def test_method_call_on_declared_var(self, allowed_names):
        """``x.types.add(cls)`` should be allowed because ``x`` is declared."""
        content = (
            "Person = Class(name='Person')\n"
            "domain_model = DomainModel(name='T')\n"
            "domain_model.types.add(Person)\n"
        )
        result = safe_load_buml(content, allowed_names)
        type_names = {t.name for t in result["domain_model"].types}
        assert "Person" in type_names

    def test_negative_number_and_string_concat(self, allowed_names):
        """UnaryOp and BinOp(Add) should be allowed for the common patterns."""
        content = (
            "prefix = 'My'\n"
            "name = prefix + 'Class'\n"
            "Cls = Class(name=name)\n"
        )
        result = safe_load_buml(content, allowed_names)
        assert result["Cls"].name == "MyClass"

    def test_dict_and_list_containers(self, allowed_names):
        content = (
            "attrs = {'x': 1, 'y': 2}\n"
            "items = [1, 2, 3]\n"
            "tup = (1, 2)\n"
            "s = {1, 2}\n"
        )
        result = safe_load_buml(content, allowed_names)
        assert result["attrs"] == {"x": 1, "y": 2}
        assert result["items"] == [1, 2, 3]
        assert result["tup"] == (1, 2)
        assert result["s"] == {1, 2}

    def test_annotated_assignment(self, allowed_names):
        """Our BUML builders emit ``x: T = value`` heavily; must be accepted."""
        content = (
            "Person_name: Property = Property(name='name', type=StringType)\n"
            "Person = Class(name='Person')\n"
            "Person.attributes = {Person_name}\n"
        )
        result = safe_load_buml(content, allowed_names)
        assert isinstance(result["Person"], Class)
        assert isinstance(result["Person_name"], Property)

    def test_subscript_access(self, allowed_names):
        """Subscript (``list[0]``, ``dict['k']``) should be allowed."""
        content = (
            "items = [1, 2, 3]\n"
            "first = items[0]\n"
        )
        result = safe_load_buml(content, allowed_names)
        assert result["first"] == 1


# ---------------------------------------------------------------------------
# Rejected constructs
# ---------------------------------------------------------------------------

class TestRejectedConstructs:
    """Disallowed AST shapes must raise ``SafeBumlLoaderError``."""

    def test_import_statement_raises(self, allowed_names):
        content = "import os\n"
        with pytest.raises(SafeBumlLoaderError):
            safe_load_buml(content, allowed_names)

    def test_from_import_statement_raises(self, allowed_names):
        content = "from os import system\n"
        with pytest.raises(SafeBumlLoaderError):
            safe_load_buml(content, allowed_names)

    def test_function_definition_raises(self, allowed_names):
        content = "def evil():\n    return 42\n"
        with pytest.raises(SafeBumlLoaderError):
            safe_load_buml(content, allowed_names)

    def test_class_definition_raises(self, allowed_names):
        content = "class Evil:\n    pass\n"
        with pytest.raises(SafeBumlLoaderError):
            safe_load_buml(content, allowed_names)

    def test_dunder_attribute_access_raises(self, allowed_names):
        """Classic ``__class__.__mro__`` introspection must be blocked."""
        content = (
            "Person = Class(name='Person')\n"
            "leaked = Person.__class__\n"
        )
        with pytest.raises(SafeBumlLoaderError, match=r"dunder|forbidden|__class__"):
            safe_load_buml(content, allowed_names)

    def test_dunder_via_chain_raises(self, allowed_names):
        content = (
            "Person = Class(name='Person')\n"
            "leaked = Person.__class__.__mro__\n"
        )
        with pytest.raises(SafeBumlLoaderError):
            safe_load_buml(content, allowed_names)

    def test_call_to_unknown_name_raises(self, allowed_names):
        """Calls to names not in the allowlist must be blocked."""
        content = "os.system('rm -rf /')\n"
        with pytest.raises(SafeBumlLoaderError, match=r"unknown|os"):
            safe_load_buml(content, allowed_names)

    def test_bare_eval_call_raises(self, allowed_names):
        content = "eval('1+1')\n"
        with pytest.raises(SafeBumlLoaderError):
            safe_load_buml(content, allowed_names)

    def test_attribute_on_unknown_name_raises(self, allowed_names):
        """Reading ``foo.bar`` when ``foo`` isn't declared/allowed must fail."""
        content = "x = unknown_obj.whatever\n"
        with pytest.raises(SafeBumlLoaderError, match=r"unknown"):
            safe_load_buml(content, allowed_names)

    def test_lambda_raises(self, allowed_names):
        content = "f = lambda x: x\n"
        with pytest.raises(SafeBumlLoaderError):
            safe_load_buml(content, allowed_names)

    def test_for_loop_raises(self, allowed_names):
        content = "for i in [1, 2]:\n    x = i\n"
        with pytest.raises(SafeBumlLoaderError):
            safe_load_buml(content, allowed_names)

    def test_if_statement_raises(self, allowed_names):
        content = "if True:\n    x = 1\n"
        with pytest.raises(SafeBumlLoaderError):
            safe_load_buml(content, allowed_names)

    def test_list_comprehension_raises(self, allowed_names):
        content = "x = [i for i in [1, 2]]\n"
        with pytest.raises(SafeBumlLoaderError):
            safe_load_buml(content, allowed_names)

    def test_bitshift_binop_raises(self, allowed_names):
        content = "x = 1 << 2\n"
        with pytest.raises(SafeBumlLoaderError, match=r"binary operator"):
            safe_load_buml(content, allowed_names)

    def test_multiply_binop_raises(self, allowed_names):
        """Only Add is allowed; Mult should fail."""
        content = "x = 2 * 3\n"
        with pytest.raises(SafeBumlLoaderError, match=r"binary operator"):
            safe_load_buml(content, allowed_names)

    def test_while_loop_raises(self, allowed_names):
        content = "while True:\n    x = 1\n"
        with pytest.raises(SafeBumlLoaderError):
            safe_load_buml(content, allowed_names)

    def test_try_except_raises(self, allowed_names):
        content = "try:\n    x = 1\nexcept Exception:\n    pass\n"
        with pytest.raises(SafeBumlLoaderError):
            safe_load_buml(content, allowed_names)

    def test_with_statement_raises(self, allowed_names):
        content = "with open('/etc/passwd') as f:\n    data = f.read()\n"
        with pytest.raises(SafeBumlLoaderError):
            safe_load_buml(content, allowed_names)

    def test_assignment_to_dunder_raises(self, allowed_names):
        content = "__evil__ = 1\n"
        with pytest.raises(SafeBumlLoaderError, match=r"dunder|underscore"):
            safe_load_buml(content, allowed_names)

    def test_builtins_cannot_be_reached(self, allowed_names):
        """Even if user tries to reach __builtins__, it should fail."""
        content = "x = __builtins__\n"
        with pytest.raises(SafeBumlLoaderError, match=r"dunder|underscore|__builtins__"):
            safe_load_buml(content, allowed_names)

    def test_syntax_error_propagates(self, allowed_names):
        """Genuinely invalid Python should raise SyntaxError, not swallow it."""
        content = "this is not valid python ===\n"
        with pytest.raises(SyntaxError):
            safe_load_buml(content, allowed_names)


# ---------------------------------------------------------------------------
# Literal receivers and string formatting
# ---------------------------------------------------------------------------
# A format string resolves attribute lookups at runtime, out of the AST walk's
# sight, so a method call on a literal must never reach the interpreter.

@pytest.mark.parametrize("source", [
    'x = "{0}".format(Class)\n',
    'x = "{a}".format_map(dict())\n',
    "x = 'abc'.upper()\n",
    "x = (1).bit_length()\n",
    "x = Class.mro()\n",
])
def test_attribute_access_on_literals_and_formatting_is_refused(allowed_names, source):
    with pytest.raises(SafeBumlLoaderError):
        safe_load_buml(source, allowed_names)


def test_attribute_calls_on_declared_objects_still_load(allowed_names):
    source = (
        "book = Class(name='Book')\n"
        "model = DomainModel(name='Library', types={book})\n"
        "names = model.types\n"
    )
    namespace = safe_load_buml(source, allowed_names)
    assert namespace["model"].name == "Library"


# ---------------------------------------------------------------------------
# Writes through injected (process-wide) objects
# ---------------------------------------------------------------------------
# Everything in ``allowed_names`` is shared by every upload the process
# handles. An assignment to one of its attributes would outlive the request
# and poison every later import, for every user.

_TS = 'x = datetime.datetime.fromisoformat("2024-01-02T03:04:05")\n'


@pytest.fixture
def with_datetime(allowed_names):
    return {**allowed_names, "datetime": SAFE_DATETIME_MODULE}


@pytest.mark.parametrize("source", [
    "datetime.datetime = 1\n",
    "datetime.datetime: int = 1\n",
    "Class.add_attribute = 1\n",
    "StringType.name = 'poisoned'\n",
    "StringType.metadata.description = 'poisoned'\n",
    "datetime.datetime.fromisoformat = 1\n",
    # Aliasing a shared object through a script variable must not help.
    "d = datetime\nd.datetime = 1\n",
    "k = Class\nk.add_attribute = 1\n",
    "s = [StringType][0]\ns.name = 'poisoned'\n",
    "a = b = datetime\nb.datetime = 1\n",
])
def test_writes_to_injected_objects_are_refused(with_datetime, source):
    with pytest.raises(SafeBumlLoaderError):
        safe_load_buml(source, with_datetime)
    # Nothing leaked into the shared objects.
    assert SAFE_DATETIME_MODULE.datetime is _real_datetime.datetime
    assert with_datetime["StringType"].name == "str"
    assert callable(Class.add_attribute)


@pytest.mark.parametrize("source", [
    "del datetime.datetime\n",
    "Class.x += 1\n",
    "x, datetime.datetime = 1, 2\n",
])
def test_delete_augmented_and_unpacking_writes_are_refused(with_datetime, source):
    with pytest.raises(SafeBumlLoaderError):
        safe_load_buml(source, with_datetime)
    assert SAFE_DATETIME_MODULE.datetime is _real_datetime.datetime


def test_fresh_load_after_rejected_poisoning_still_works(with_datetime):
    with pytest.raises(SafeBumlLoaderError):
        safe_load_buml("datetime.datetime = 1\n", with_datetime)
    result = safe_load_buml(_TS, with_datetime)
    assert result["x"] == _real_datetime.datetime(2024, 1, 2, 3, 4, 5)


def test_each_load_gets_its_own_datetime_namespace(with_datetime):
    # Belt and braces: even a mutation the validator missed stays per-call.
    first = safe_load_buml("d = datetime\n", with_datetime)["d"]
    second = safe_load_buml("d = datetime\n", with_datetime)["d"]
    assert first is not SAFE_DATETIME_MODULE
    assert first is not second
    assert first.datetime is _real_datetime.datetime


def test_attribute_writes_on_script_objects_still_load(allowed_names):
    # The shape every builder emits: ``<declared var>.<attr> = <value>``,
    # including inside the ``except NameError: pass`` guard.
    source = (
        "book = Class(name='Book')\n"
        "book.is_abstract = True\n"
        "model = DomainModel(name='Library', types={book})\n"
        "model.name = 'Renamed'\n"
        "try:\n"
        "    book.synonyms = undefined_name\n"
        "except NameError:\n"
        "    pass\n"
    )
    namespace = safe_load_buml(source, allowed_names)
    assert namespace["book"].is_abstract is True
    assert namespace["model"].name == "Renamed"
    assert set(namespace) == {"book", "model"}


def test_builder_export_with_attribute_writes_round_trips(tmp_path):
    """A real ``domain_model_to_code`` export assigns ``cls.attributes = {...}``,
    ``method.code = ...`` and, under the NameError guard,
    ``method.state_machine = sm``; it must still load."""
    from besser.BUML.metamodel.structural import (
        Method, MethodImplementationType, StringType as RealStringType,
    )
    from besser.utilities.buml_code_builder import domain_model_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
        parse_buml_content,
    )

    book = Class(name="Book")
    book.attributes = {Property(name="title", type=RealStringType)}
    book.methods = {
        Method(name="describe", code="def describe(self):\n    return self.title\n"),
        Method(name="flow", implementation_type=MethodImplementationType.STATE_MACHINE),
    }
    out = tmp_path / "model.py"
    domain_model_to_code(DomainModel(name="Library", types={book}), str(out))
    source = out.read_text(encoding="utf-8")
    assert ".attributes=" in source and ".code = " in source and ".state_machine = sm" in source

    loaded = parse_buml_content(source)
    loaded_book = loaded.get_class_by_name("Book")
    assert {a.name for a in loaded_book.attributes} == {"title"}
    describe = next(m for m in loaded_book.methods if m.name == "describe")
    assert "return self.title" in describe.code


# ---------------------------------------------------------------------------
# Process-wide primitive types
# ---------------------------------------------------------------------------
# ``PrimitiveDataType.name.fset(StringType, "int")`` in a project upload renamed
# the real StringType; every later ``DomainModel(...)`` then failed with
# duplicate type names, for every user, until the process restarted.

_PROJECT_TEMPLATE = (
    "# STRUCTURAL MODEL #\n"
    'domain_model = DomainModel(name="D")\n'
    "{payload}\n"
    "\n"
    "# GUI MODEL #\n"
    "x = 1\n"
    "\n"
    'project = Project(name="p", models=[domain_model, gui_model], owner="o")\n'
)

_POISONING_PAYLOADS = [
    'PrimitiveDataType.name.fset(StringType, "int")',
    'f = PrimitiveDataType.name.fset\nf(IntegerType, "str")',
    "Class.timestamp.fset(Class, 5)",
    'p = Property(name="a", type="str")\nt = p.type\nt.name = "int"',
    'Property.type_mapping.pop("int")',
]


@pytest.fixture
def real_primitives():
    """Restore the process-wide primitives so a failure here cannot cascade."""
    from besser.BUML.metamodel.structural import structural

    names = {
        t: getattr(structural, t).name for t in ("StringType", "IntegerType")
    }
    mapping = dict(structural.TypedElement.type_mapping)
    yield structural
    for type_name, name in names.items():
        getattr(structural, type_name).name = name
    structural.TypedElement.type_mapping.clear()
    structural.TypedElement.type_mapping.update(mapping)
    for owner in (structural.Class, structural.Element):
        if "_Element__timestamp" in vars(owner):
            delattr(owner, "_Element__timestamp")


def _assert_primitives_untouched(structural):
    assert structural.StringType.name == "str"
    assert structural.IntegerType.name == "int"
    assert structural.TypedElement.type_mapping["int"] is structural.IntegerType
    assert "_Element__timestamp" not in vars(structural.Class)
    structural.DomainModel(name="Later", types={structural.Class(name="A")})


@pytest.mark.parametrize("payload", _POISONING_PAYLOADS)
def test_project_upload_cannot_rename_shared_primitive_types(real_primitives, payload):
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.project_converter import (
        project_to_json,
    )

    try:
        project_to_json(_PROJECT_TEMPLATE.format(payload=payload))
    except Exception:  # the import itself may fail; the shared state must not change
        pass
    _assert_primitives_untouched(real_primitives)


@pytest.mark.parametrize("payload", _POISONING_PAYLOADS)
def test_class_upload_cannot_rename_shared_primitive_types(real_primitives, payload):
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
        parse_buml_content,
    )

    try:
        parse_buml_content(payload + '\ndomain_model = DomainModel(name="D")\n')
    except Exception:
        pass
    _assert_primitives_untouched(real_primitives)


@pytest.mark.parametrize("source", [
    "p = PrimitiveDataType.name.setter\n",
    "p = PrimitiveDataType.name.fget\n",
    "p = PrimitiveDataType.name.fdel\n",
    "p = Property.type_mapping\n",
])
def test_property_accessors_and_type_mapping_are_refused(allowed_names, source):
    with pytest.raises(SafeBumlLoaderError):
        safe_load_buml(source, allowed_names)


def test_gui_domain_context_gets_its_own_primitive_types(monkeypatch):
    from besser.BUML.metamodel.structural import structural
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json import (
        gui_diagram_converter,
    )

    seen = []
    real_loader = gui_diagram_converter.safe_load_buml

    def recording_loader(content, names):
        seen.append(names)
        return real_loader(content, names)

    monkeypatch.setattr(gui_diagram_converter, "safe_load_buml", recording_loader)
    gui_diagram_converter._parse_gui_model("x = 1\n", context_code='domain_model = DomainModel(name="D")\n')
    context_names = seen[0]
    for type_name in ("StringType", "IntegerType", "DateTimeType"):
        assert context_names[type_name] is not getattr(structural, type_name)
        assert context_names[type_name].name == getattr(structural, type_name).name
