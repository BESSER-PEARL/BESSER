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

import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, Enumeration, PrimitiveDataType, Property,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json._safe_buml_loader import (
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
