"""B-OCL ``->closure`` iterator (BESSER-PEARL/BESSER#635).

OCL 2.4 §7.6.5 defines ``closure`` with the select/reject syntax, §8.3.7
(IteratorExp closure) fixes its well-formedness: one iterator, a body typed
as the source element type (or a collection of it), and a result that is an
OrderedSet for ordered sources and a Set otherwise.
"""

import pytest

from besser.BUML.metamodel.ocl.ocl import (
    IteratorExp, LoopExp, OCLConstraint, OperationCallExpression,
    OrderedSetType, PropertyCallExpression, SetType, VariableExp,
)
from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, IntegerType, Multiplicity, Property,
)
from besser.BUML.notations.ocl import parse_ocl, pretty_print
from besser.BUML.notations.ocl.error_handling import BOCLSyntaxError


@pytest.fixture(scope="module")
def genealogy_model() -> DomainModel:
    """``Person`` with the ``is_child_of`` self-association (``parents`` 0..*),
    plus a ``Family`` whose ``members`` are persons (element type != context)."""
    person = Class("Person", attributes={Property("age", IntegerType)})
    family = Class("Family")
    is_child_of = BinaryAssociation("is_child_of", ends={
        Property("person_1", person, multiplicity=Multiplicity(1, 1), is_navigable=False),
        Property("parents", person, multiplicity=Multiplicity(0, "*")),
    })
    has_members = BinaryAssociation("has_members", ends={
        Property("family", family, multiplicity=Multiplicity(0, 1), is_navigable=False),
        Property("members", person, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(
        "Genealogy", types={person, family}, associations={is_child_of, has_members},
    )


@pytest.mark.parametrize(
    "text",
    [
        "context Person inv Person_inv_3_1 :  self.parents->closure(parents)->excludes(self)",
    ],
)
def test_closure_constraint_parses(genealogy_model, text):
    constraint = parse_ocl(text, genealogy_model)
    assert isinstance(constraint, OCLConstraint)


def _closure_of(constraint) -> LoopExp:
    loop = constraint.ast.source
    assert isinstance(loop, LoopExp) and loop.name == "closure"
    return loop


def test_short_form_ast(genealogy_model):
    constraint = parse_ocl(
        "context Person inv: self.parents->closure(parents)->excludes(self)",
        genealogy_model,
    )
    assert isinstance(constraint.ast, OperationCallExpression)
    assert constraint.ast.operation == "EXCLUDES"
    loop = _closure_of(constraint)
    assert isinstance(loop.source, PropertyCallExpression)
    assert loop.source.property.name == "parents"
    assert loop.iterator == []
    (body,) = loop.body
    assert isinstance(body, Property) and body.name == "parents"


@pytest.mark.parametrize("text", [
    "context Person inv: self.parents->closure(p | p.parents)->excludes(self)",
    "context Person inv: self.parents->closure(p : Person | p.parents)->excludes(self)",
])
def test_explicit_iterator_ast(genealogy_model, text):
    loop = _closure_of(parse_ocl(text, genealogy_model))
    (iterator,) = loop.iterator
    assert isinstance(iterator, IteratorExp)
    assert iterator.name == "p" and iterator.type.name == "Person"
    (body,) = loop.body
    assert isinstance(body, PropertyCallExpression)
    assert body.property.name == "parents"
    assert isinstance(body.source, VariableExp) and body.source.name == "p"


@pytest.mark.parametrize("text", [
    "context Person inv: self.parents->closure(parents)->size() >= 0",
    "context Person inv: self.parents->closure(p | p.parents)->size() >= 0",
    "context Person inv: self->closure(parents)->size() >= 1",
    "context Person inv: Person::allInstances()->closure(parents)->size() >= 0",
    "context Person inv: self.parents->closure(parents)->closure(parents)->size() >= 0",
])
def test_unordered_source_gives_set(genealogy_model, text):
    loop = parse_ocl(text, genealogy_model).ast.arguments[0].source
    assert loop.name == "closure"
    assert isinstance(loop.type, SetType)
    assert not isinstance(loop.type, OrderedSetType)
    assert loop.type.name == "Set(Person)"


@pytest.mark.parametrize("text", [
    "context Person inv: self.parents->subOrderedSet(1, 2)->closure(parents)->size() >= 0",
    "context Person inv: self.parents->subSequence(1, 2)->closure(p | p.parents)->size() >= 0",
    "context Person inv: self.parents->append(self)->closure(parents)->size() >= 0",
])
def test_ordered_source_gives_ordered_set(genealogy_model, text):
    loop = parse_ocl(text, genealogy_model).ast.arguments[0].source
    assert loop.name == "closure"
    assert isinstance(loop.type, OrderedSetType)
    assert loop.type.name == "OrderedSet(Person)"


@pytest.mark.parametrize("source", [
    "context Person inv: self.parents->closure(parents)->size() >= 0",
    "context Person inv: self.parents->closure(p | p.parents)->size() >= 0",
    "context Person inv: self.parents->forAll(p | p.parents->closure(parents)->size() >= 0)",
])
def test_round_trip_identity(genealogy_model, source):
    assert pretty_print(parse_ocl(source, genealogy_model)) == source


def test_nested_in_forall(genealogy_model):
    constraint = parse_ocl(
        "context Person inv: self.parents->forAll(p | p.parents->closure(q | q.parents)->excludes(p))",
        genealogy_model,
    )
    forall = constraint.ast
    assert isinstance(forall, LoopExp) and forall.name == "forAll"
    excludes = forall.body[0]
    loop = excludes.source
    assert loop.name == "closure"
    assert isinstance(loop.source.source, VariableExp) and loop.source.source.name == "p"
    assert loop.iterator[0].type.name == "Person"
    assert isinstance(excludes.arguments[0], VariableExp)
    assert excludes.arguments[0].name == "p"


def test_closure_result_feeds_typed_iterator(genealogy_model):
    """Closure keeps the source element type, so a following iterator is typed."""
    forall = parse_ocl(
        "context Person inv: self.parents->closure(parents)->forAll(a | a.age >= 0)",
        genealogy_model,
    ).ast
    assert forall.iterator[0].type.name == "Person"
    assert forall.body[0].arguments[0].property.owner.name == "Person"


def test_short_form_body_resolves_against_element_type(genealogy_model):
    """OCL 2.4 §7.8: the implicit source of ``parents`` is the implicit
    iterator (a Person), not ``self`` (a Family)."""
    loop = parse_ocl(
        "context Family inv: self.members->closure(parents)->size() >= 0",
        genealogy_model,
    ).ast.arguments[0].source
    body = loop.body[0]
    assert isinstance(body, Property)
    assert body.name == "parents" and body.type.name == "Person"
    assert loop.type.name == "Set(Person)"


def test_short_form_body_chain_resolves_against_element_type(genealogy_model):
    loop = parse_ocl(
        "context Family inv: self.members->closure(parents.parents)->size() >= 0",
        genealogy_model,
    ).ast.arguments[0].source
    body = loop.body[0]
    assert isinstance(body, PropertyCallExpression)
    assert body.property.name == "parents"
    assert isinstance(body.source, Property) and body.source.name == "parents"


@pytest.mark.parametrize("text, fragment", [
    # Unknown role, explicit iterator.
    ("context Person inv: self.parents->closure(p | p.ancestors)->size() >= 0", "ancestors"),
    # Unknown role, short form (used to become a dangling VariableExp).
    ("context Person inv: self.parents->closure(ancestors)->size() >= 0", "ancestors"),
    # Body of the wrong type: an Integer, not a Person.
    ("context Person inv: self.parents->closure(p | p.age)->size() >= 0", "must yield"),
    ("context Person inv: self.parents->closure(age)->size() >= 0", "must yield"),
    # Body of an unrelated class.
    ("context Family inv: self.members->closure(p | self)->size() >= 0", "must yield"),
    # closure takes exactly one iterator variable.
    ("context Person inv: self.parents->closure(a, b | a.parents)->size() >= 0", "one iterator"),
])
def test_invalid_body_is_rejected(genealogy_model, text, fragment):
    with pytest.raises(BOCLSyntaxError) as excinfo:
        parse_ocl(text, genealogy_model)
    assert fragment in str(excinfo.value)


def test_body_of_a_subclass_is_accepted():
    """A body typed as a subclass of the element type conforms to it."""
    from besser.BUML.metamodel.structural import Generalization
    person, teacher = Class("Person"), Class("Teacher")
    mentoring = BinaryAssociation("mentoring", ends={
        Property("mentee", person, multiplicity=Multiplicity(0, "*"), is_navigable=False),
        Property("mentor", teacher, multiplicity=Multiplicity(0, 1)),
    })
    model = DomainModel(
        "School", types={person, teacher}, associations={mentoring},
        generalizations={Generalization(general=person, specific=teacher)},
    )
    loop = parse_ocl(
        "context Person inv: self->closure(p | p.mentor)->size() >= 1", model,
    ).ast.arguments[0].source
    assert loop.type.name == "Set(Person)"


@pytest.fixture(scope="module")
def shop_model() -> DomainModel:
    """A ``Shop`` with an attribute literally named ``closure``."""
    shop = Class("Shop", attributes={Property("closure", IntegerType)})
    branches = BinaryAssociation("has_branches", ends={
        Property("owner", shop, multiplicity=Multiplicity(0, 1), is_navigable=False),
        Property("branches", shop, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel("Shops", types={shop}, associations={branches})


@pytest.mark.parametrize("text", [
    "context Shop inv: self.closure > 0",
    "context Shop inv: self.branches->forAll(b | b.closure > 0)",
    "context Shop inv: self.branches->closure(branches)->forAll(b | b.closure >= self.closure)",
])
def test_attribute_named_closure_still_navigates(shop_model, text):
    """``closure`` is a keyword token, so ``.closure`` needs the same
    keyword-property fallback as ``size`` and ``date``."""
    from besser.BUML.metamodel.ocl import walk
    constraint = parse_ocl(text, shop_model)
    navigations = [n for n in walk(constraint.ast)
                   if isinstance(n, PropertyCallExpression) and n.property.name == "closure"]
    assert navigations and all(n.property.type is IntegerType for n in navigations)
    assert pretty_print(constraint) == text


def test_attribute_named_closure_in_base_visitor(shop_model):
    """The evaluator's visitor (``BOCLVisitorImpl``) resolves it too."""
    from antlr4 import CommonTokenStream, InputStream
    from besser.BUML.notations.ocl.BOCLLexer import BOCLLexer
    from besser.BUML.notations.ocl.BOCLParser import BOCLParser
    from besser.BUML.notations.ocl.visitor import BOCLVisitorImpl
    shop = next(t for t in shop_model.types if t.name == "Shop")
    tree = BOCLParser(CommonTokenStream(BOCLLexer(InputStream(
        "context Shop inv: self.closure > 0")))).oclFile()
    ast = BOCLVisitorImpl(shop_model, None, shop).visit(tree)
    assert isinstance(ast.arguments[0], Property) and ast.arguments[0].name == "closure"
