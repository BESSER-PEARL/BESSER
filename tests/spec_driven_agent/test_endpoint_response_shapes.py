"""The endpoint manifest says what the generated create/update routes RETURN.

Seven live runs (Claude Sonnet 5.5, r9) each lost 1-3 turns reading routers to learn
that ``router.py.j2`` wraps a create response as ``{"<entity>": {...}, "<end>_ids": [...]}``
when the class has a navigable end to many, and returns the bare record otherwise; three
still guessed ``{{0.id}}`` wrong in ``test_api``. The derivation is checked here against
the generator's own output, and was replayed against all seven saved workspaces.
"""
import ast
from unittest import mock

from besser.BUML.metamodel.structural import (
    AssociationClass, BinaryAssociation, Class, DomainModel, Generalization, Multiplicity, Property,
    StringType,
)
from besser.generators.backend import BackendGenerator
from besser.spec_driven_agent.agent.prompt_builder import build_endpoint_manifest, create_response_keys
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from tests.spec_driven_agent.shelf_app import GENERATOR, generate_backend, shelf_model


def _model() -> DomainModel:
    """One class per template branch: 1:N, N:M, to-one only, non-navigable, inherited, association class,
    a class linked to itself (N:M and 1:N), two N:M links to one class, an abstract class."""
    def cls(name, is_abstract=False):
        return Class(name=name, attributes={Property(name="label", type=StringType)}, is_abstract=is_abstract)

    owner, item, tag, note, animal, dog, vet, course, student = map(cls, (
        "Owner", "Item", "Tag", "Note", "Animal", "Dog", "Vet", "Course", "Student"))
    person, team, node, leaf, circle = map(cls, ("Person", "Team", "Node", "Leaf", "Circle"))
    shape = cls("Shape", is_abstract=True)
    associations = {
        BinaryAssociation(name="owns", ends={
            Property(name="items", type=item, multiplicity=Multiplicity(0, "*")),
            Property(name="owner", type=owner, multiplicity=Multiplicity(1, 1))}),
        BinaryAssociation(name="tagged", ends={
            Property(name="tags", type=tag, multiplicity=Multiplicity(0, "*")),
            Property(name="tagged_items", type=item, multiplicity=Multiplicity(0, "*"))}),
        BinaryAssociation(name="annotates", ends={
            Property(name="notes", type=note, multiplicity=Multiplicity(0, "*"), is_navigable=False),
            Property(name="vet", type=vet, multiplicity=Multiplicity(0, 1))}),
        BinaryAssociation(name="treats", ends={
            Property(name="patients", type=animal, multiplicity=Multiplicity(0, "*")),
            Property(name="doctors", type=vet, multiplicity=Multiplicity(0, "*"))}),
        BinaryAssociation(name="friendship", ends={
            Property(name="friends", type=person, multiplicity=Multiplicity(0, "*")),
            Property(name="friend_of", type=person, multiplicity=Multiplicity(0, "*"))}),
        BinaryAssociation(name="membership", ends={
            Property(name="members", type=person, multiplicity=Multiplicity(0, "*")),
            Property(name="member_teams", type=team, multiplicity=Multiplicity(0, "*"))}),
        BinaryAssociation(name="leadership", ends={
            Property(name="leaders", type=person, multiplicity=Multiplicity(0, "*")),
            Property(name="led_teams", type=team, multiplicity=Multiplicity(0, "*"))}),
        # The single end sorts before the collection end.
        BinaryAssociation(name="tree", ends={
            Property(name="branch", type=node, multiplicity=Multiplicity(0, 1)),
            Property(name="twigs", type=node, multiplicity=Multiplicity(0, "*"))}),
    }
    enrolment = BinaryAssociation(name="enrolment", ends={
        Property(name="students", type=student, multiplicity=Multiplicity(0, "*")),
        Property(name="courses", type=course, multiplicity=Multiplicity(0, "*"))})
    enrolled = AssociationClass(name="Enrolled", attributes={Property(name="grade", type=StringType)},
                                association=enrolment)
    return DomainModel(name="Shapes",
                       types={owner, item, tag, note, animal, dog, vet, course, student, enrolled,
                              person, team, node, leaf, shape, circle},
                       associations=associations | {enrolment},
                       generalizations={Generalization(general=animal, specific=dog),
                                        Generalization(general=node, specific=leaf),
                                        Generalization(general=shape, specific=circle)})


def _router_keys(router_source: str, function: str) -> list[str]:
    """The top-level keys a generated route returns, in order; [] for the bare record."""
    for node in ast.parse(router_source).body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == function:
            for statement in ast.walk(node):
                if (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Dict)
                        and any(getattr(target, "id", "") == "response_data" for target in statement.targets)):
                    return [key.value for key in statement.value.keys]
            return []
    raise AssertionError(f"{function} not generated")


def test_every_predicted_key_matches_the_generated_routes(tmp_path):
    model = _model()
    BackendGenerator(model, output_dir=str(tmp_path)).generate()
    keys = create_response_keys(model)
    for name, create in keys.items():
        source = (tmp_path / "routers" / f"{name.lower()}.py").read_text(encoding="utf-8")
        # An abstract class has no create route.
        written = ("update",) if name == "Shape" else ("create", "update")
        for function in (f"{verb}_{name.lower()}" for verb in written):
            assert _router_keys(source, function) == create, (name, function)
        get = [] if name == "Enrolled" else [name.lower(), *create[1:]]
        assert _router_keys(source, f"get_{name.lower()}") == get, name
    # N:M lists are keyed by the other CLASS, 1:N and association-class links by the other END.
    assert keys["Tag"] == ["tag", "item_ids"]
    assert keys["Item"] == ["item", "tag_ids"]
    assert keys["Owner"] == ["owner", "items_ids"]
    assert keys["Course"] == ["course", "students_ids"]
    assert keys["Dog"] == ["dog", "vet_ids"]
    assert keys["Note"] == keys["Enrolled"] == []
    # A class linked to itself lists both ends; two N:M links to one class are keyed by their ends.
    assert keys["Person"] == ["person", "friend_of_ids", "friends_ids", "led_teams_ids", "member_teams_ids"]
    assert keys["Team"] == ["team", "leaders_ids", "members_ids"]
    assert keys["Node"] == ["node", "twigs_ids"]
    assert keys["Leaf"] == ["leaf", "twigs_ids"]
    assert keys["Shape"] == keys["Circle"] == []
    assert "async def create_shape" not in (tmp_path / "routers" / "shape.py").read_text(encoding="utf-8")


def test_manifest_names_the_actual_keys_and_the_reference_to_use(tmp_path):
    model = shelf_model(author=True)
    generate_backend(tmp_path, model)
    plain = build_endpoint_manifest(str(tmp_path))
    shaped = build_endpoint_manifest(str(tmp_path), response_model=model)
    assert "Response shapes" not in plain
    assert "for Author {author, works_ids}; Library {library, books_ids}" in shaped
    assert "{{N.author.id}}" in shaped and "the bare record ({{N.id}})" in shaped
    # GET by id wraps the entity whose POST is bare too.
    assert 'also one whose POST is bare (GET /book/1/ -> {"book": {...}})' in shaped


def _system_prompt(tmp_path, generator):
    model = shelf_model(author=True)
    generate_backend(tmp_path, model)
    orch = LLMOrchestrator(llm_client=mock.MagicMock(), domain_model=model, output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False)
    orch._generator_used = generator
    return orch._build_system_prompt("Build the app")


def test_the_prompt_carries_the_shapes_only_for_the_router_template_scaffold(tmp_path):
    assert "Response shapes (generated routers)" in _system_prompt(tmp_path / "a", GENERATOR)
    # Another scaffold's routers are not that template's: no claim about them.
    assert "Response shapes" not in _system_prompt(tmp_path / "b", "generate_django")
