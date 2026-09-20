"""Tests for `besser.generators.alloy.instance_generator.alloy_solver`.

These tests invoke every module-level function and every class of
`alloy_solver.py` directly in Python, without going through the FastAPI
web-editor endpoint. Object-diagram generation tests drive `AlloySolver` with
the **real** Alloy Analyzer (``java -jar alloy.jar``) as an external process;
they are skipped when the jar or a JRE is not available. Pure helper functions
are unit-tested directly.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    Constraint,
    DomainModel,
    Generalization,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
)
from besser.generators.alloy.alloy_utils_generator import (
    build_inheritance_and_attribute_maps,
    process_associations,
    translate_constraints,
)
from besser.generators.alloy.instance_generator.alloy_analyzer_executor import (
    AlloyAnalyzerExecutor,
    AlloyResult,
)
from besser.generators.alloy.instance_generator.alloy_solver import (
    AlloySolver,
)
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.object_diagram_converter import (
    object_buml_to_json,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def team_player_model():
    """Team 1 -- 3..4 Player."""
    team = Class(name="Team")
    player = Class(name="Player")
    team.attributes = {Property(name="name", type=StringType)}
    player.attributes = {
        Property(name="name", type=StringType),
        Property(name="age", type=IntegerType),
    }
    plays_for = BinaryAssociation(
        name="PlaysFor",
        ends={
            Property(name="players", type=player, multiplicity=Multiplicity(3, 4)),
            Property(name="team", type=team, multiplicity=Multiplicity(1, 1)),
        },
    )
    return DomainModel(name="TeamModel", types={team, player}, associations={plays_for})


@pytest.fixture
def person_model():
    """A minimal one-class model: Person(name: str)."""
    person = Class(name="Person")
    person.attributes = {Property(name="name", type=StringType)}
    return DomainModel(name="PersonModel", types={person})


def _alloy_real():
    """True when the real Alloy Analyzer (alloy.jar + java) is available.

    The object-generation tests that go through ``AlloySolver`` need to run the
    actual ``java -jar alloy.jar`` call; if the jar or a JRE is missing they are
    skipped rather than failing.
    """
    try:
        executor = AlloyAnalyzerExecutor()
    except RuntimeError:
        return False
    return bool(
        getattr(executor, "alloy_jar_path", None)
        and getattr(executor, "java_path", None)
    )


# ---------------------------------------------------------------------------
# build_inheritance_and_attribute_maps
# ---------------------------------------------------------------------------


class TestBuildInheritanceAndAttributeMaps:

    def test_class_without_parents_maps_to_underscore(self, team_player_model):
        inherits_from, data, basic_signatures, sigs_nv = build_inheritance_and_attribute_maps(team_player_model)

        assert inherits_from["Team"] == ["_"]
        assert inherits_from["Player"] == ["_"]
        assert "name:str" in data["Team"]
        assert "age:int" in data["Player"]
        assert {"str", "int"} <= basic_signatures
        assert "Team" in sigs_nv and "Player" in sigs_nv

    def test_generalization_records_parent_name(self):
        animal = Class(name="Animal")
        dog = Class(name="Dog")
        animal.attributes = {Property(name="name", type=StringType)}
        genealogy = Generalization(general=animal, specific=dog)
        model = DomainModel(
            name="Zoo", types={animal, dog}, generalizations={genealogy}
        )

        inherits_from, _, _, _ = build_inheritance_and_attribute_maps(model)

        assert inherits_from["Dog"] == ["Animal"]
        assert inherits_from["Animal"] == ["_"]


# ---------------------------------------------------------------------------
# process_associations
# ---------------------------------------------------------------------------


class TestProcessAssociations:

    def test_association_ends_added_to_both_sides(self, team_player_model):
        _, data, _, _ = build_inheritance_and_attribute_maps(team_player_model)

        facts = process_associations(team_player_model, data)

        assert "players:Player" in data["Team"]
        assert "team:Team" in data["Player"]
        assert len(facts) >= 1


# ---------------------------------------------------------------------------
# translate_constraints
# ---------------------------------------------------------------------------


class TestTranslateConstraints:

    def test_ocl_expression_is_replaced_by_alloy_fact(self, team_player_model):
        inherits_from, data, _, _ = build_inheritance_and_attribute_maps(team_player_model)
        process_associations(team_player_model, data)

        player_class = team_player_model.get_class_by_name("Player")
        constraint = Constraint(
            name="AgePositive",
            context=player_class,
            expression="context Player inv AgePositive: self.age > 0",
            language="OCL",
        )
        team_player_model.constraints = {constraint}

        translate_constraints(team_player_model, inherits_from, data, enums={})

        assert "Player_age > 0" in constraint.expression
        assert "fact{" in constraint.expression


# ---------------------------------------------------------------------------
# AlloySolver
# ---------------------------------------------------------------------------


class TestAlloySolverConstruction:

    def test_generates_als_file_and_sanitizes_names(self, tmpdir):
        # Valid BUML name (no spaces/hyphens) that is still invalid for Alloy.
        team = Class(name="Clase_Número")
        team.attributes = {Property(name="1attr", type=StringType)}
        model = DomainModel(name="M", types={team})

        solver = AlloySolver(model=model, output_dir=str(tmpdir.mkdir("out")))

        assert os.path.isfile(solver.specification)
        with open(solver.specification, encoding="utf-8") as f:
            als_content = f.read()
        # The generated .als must use sanitized, Alloy-valid identifiers.
        assert re.search(r"sig\s+Clase_Nmero", als_content)
        assert re.search(r"Clase_Nmero__1attr", als_content)
        # The original model must NOT be mutated.
        original = model.classes_sorted_by_inheritance()[0]
        assert original.name == "Clase_Número"
        assert {a.name for a in original.attributes} == {"1attr"}


class TestAlloyAnalyzerExecutorResolution:

    def test_uses_besser_alloy_jar_env_var(self, tmp_path, monkeypatch):
        fake_jar = tmp_path / "alloy.jar"
        fake_jar.write_text("", encoding="utf-8")
        monkeypatch.setenv("BESSER_ALLOY_JAR", str(fake_jar))

        assert AlloyAnalyzerExecutor().alloy_jar_path == str(fake_jar)

    def test_raises_when_env_var_invalid(self, monkeypatch):
        # An invalid BESSER_ALLOY_JAR must not silently fall back: the executor
        # cannot run without the analyzer jar.
        monkeypatch.setenv("BESSER_ALLOY_JAR", "/nonexistent/alloy.jar")

        with pytest.raises(RuntimeError):
            AlloyAnalyzerExecutor()


class TestAlloySolverReceiptParsing:

    def test_receipt_missing_reports_error(self, tmp_path):
        exec_output_dir = tmp_path / "exec_out"
        exec_output_dir.mkdir()
        fake_result = subprocess.CompletedProcess(
            args=["java"], returncode=0, stdout="boom", stderr=""
        )

        executor = AlloyAnalyzerExecutor.__new__(AlloyAnalyzerExecutor)
        executor.output_dir = str(exec_output_dir)

        with pytest.raises(RuntimeError) as exc_info:
            executor._check_satisfiability_in_alloy_receipt_json(fake_result)
        # The real Alloy output must be surfaced so upstream failures are
        # diagnosable instead of showing a generic "no receipt.json" message.
        assert "no receipt.json" in str(exc_info.value)
        assert "boom" in str(exc_info.value)


class TestAlloySolverPipelineWithoutEndpoint:
    """Exercises the full "generate an object diagram" pipeline directly on
    AlloySolver, running the real Alloy Analyzer (alloy.jar) as an external
    process. Every test builds a fresh model and writes the generated .als and
    instance XML into a temporary directory."""
    scope = 3

    @pytest.fixture(autouse=True)
    def _skip_unless_alloy(self):
        if not _alloy_real():
            pytest.skip("Real Alloy Analyzer (alloy.jar + java) not available")

    def _reference_model(self):
        return {
            "elements": {
                "elem_1": {
                    "name": "Person",
                    "type": "Class",
                    "attributes": {"attr_1": {"name": "name", "type": "str"}},
                }
            },
            "relationships": {},
        }

    def test_check_consistency(self, person_model, tmpdir):
        solver = AlloySolver(model=person_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        result = solver.check_consistency()

        assert result == AlloyResult.SAT

    def test_object_diagrams_string_only_model_scope_8(self, tmpdir):
        """Regression: a str-only model (no Int attributes) at scope >= 8 used to
        fail because Alloy's default integer bitwidth (4) caps sequence length
        at 7, while the generated run command requested an 8-element seq."""
        person = Class(name="Person")
        person.attributes = {Property(name="name", type=StringType)}
        model = DomainModel(name="StrOnlyModel", types={person})

        solver = AlloySolver(model=model, output_dir=str(tmpdir.mkdir("out")), scope=8)

        result, codes = solver.generate_object_diagrams()

        assert result == AlloyResult.SAT
        assert codes
        assert 'Person("Person_' in codes[0]

    def test_check_consistency_scope_8_string_model(self, person_model, tmpdir):
        solver = AlloySolver(model=person_model, output_dir=str(tmpdir.mkdir("out")), scope=8)

        result = solver.check_consistency()

        assert result == AlloyResult.SAT

    def test_generate_object_diagrams(self, person_model, tmpdir):
        solver = AlloySolver(model=person_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        result, codes = solver.generate_object_diagrams()

        assert result == AlloyResult.SAT
        assert isinstance(codes, list)
        assert len(codes) >= 1
        assert 'Person("Person_' in codes[0]
        assert "ObjectModel(" in codes[0]

    def test_generate_class_and_object_model_writes_file(self, person_model, tmpdir):
        solver = AlloySolver(model=person_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        result = solver.generate_class_and_object_model()

        assert result == AlloyResult.SAT
        outfile = Path(solver.output_dir) / "buml_class_object_model.py"
        assert outfile.is_file()
        content = outfile.read_text(encoding="utf-8")
        assert "# OBJECT MODEL #" in content
        assert 'Person("Person_' in content

    def test_generate_object_diagrams_multiple_instances(self, person_model, tmpdir):
        solver = AlloySolver(model=person_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        result, codes = solver.generate_object_diagrams(num_instances=3)

        assert result == AlloyResult.SAT
        assert isinstance(codes, list)
        assert 1 <= len(codes) <= 3
        assert "ObjectModel(" in codes[0]

    def test_generate_object_diagram_json(self, person_model, tmpdir):
        """The default dialect is the "editor" dialect, which must yield a
        parsable ObjectDiagram JSON when fed to the web editor converter."""
        solver = AlloySolver(model=person_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        reference_model = self._reference_model()
        result, codes = solver.generate_object_diagrams()

        assert result == AlloyResult.SAT
        assert codes
        obj_json = object_buml_to_json(codes[0], reference_model)

        assert obj_json is not None
        assert "elements" in obj_json
        object_names = {
            elem["name"]
            for elem in obj_json["elements"].values()
            if elem.get("type") == "ObjectName"
        }
        assert any(name.startswith("Person_") for name in object_names)

    def test_generate_object_diagrams_returns_empty_when_unsat(self, person_model, tmpdir):
        # A model is guaranteed to be unsatisfiable by adding two contradictory
        # OCL invariants, which are translated to mutually exclusive Alloy facts.
        from besser.BUML.metamodel.structural import Constraint

        person = Class(name="Person")
        person.attributes = {
            Property(name="name", type=StringType),
            Property(name="age", type=IntegerType),
        }
        positive = Constraint(
            name="Positive",
            context=person,
            expression="context Person inv Positive: self.age > 0",
            language="OCL",
        )
        negative = Constraint(
            name="Negative",
            context=person,
            expression="context Person inv Negative: self.age < 0",
            language="OCL",
        )
        unsat_model = DomainModel(
            name="UnsatModel", types={person}, constraints={positive, negative}
        )

        solver = AlloySolver(model=unsat_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        result, codes = solver.generate_object_diagrams()

        assert result == AlloyResult.UNSAT
        assert codes == []


# ---------------------------------------------------------------------------
# AlloySolver – pipeline with Team/Player rich model
# ---------------------------------------------------------------------------


class TestAlloySolverInstanceGenerationRichModel:
    """Full pipeline tests using the Team/Player model, running the real Alloy
    Analyzer. Teams have 3..4 players, so a satisfying instance always contains
    at least one Team and three Player objects; exact object counts and labels
    are left unchecked because Alloy picks them non-deterministically."""
    scope = 8

    @pytest.fixture(autouse=True)
    def _skip_unless_alloy(self):
        if not _alloy_real():
            pytest.skip("Real Alloy Analyzer (alloy.jar + java) not available")

    def test_generate_object_diagrams_team_player(self, team_player_model, tmpdir):
        solver = AlloySolver(model=team_player_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        result, codes = solver.generate_object_diagrams()

        assert result == AlloyResult.SAT
        assert isinstance(codes, list)
        assert len(codes) >= 1
        assert re.search(r'^\w+_obj = Team\("Team_', codes[0], re.MULTILINE)
        assert re.search(r'^\w+_obj = Player\("Player_', codes[0], re.MULTILINE)

    def test_generate_object_diagrams_includes_attributes(self, team_player_model, tmpdir):
        solver = AlloySolver(model=team_player_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        result, codes = solver.generate_object_diagrams(for_editor=False)

        assert result == AlloyResult.SAT
        assert "'name':" in codes[0]
        assert "'age':" in codes[0]

    def test_generate_object_diagrams_includes_association(self, team_player_model, tmpdir):
        solver = AlloySolver(model=team_player_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        result, codes = solver.generate_object_diagrams(for_editor=False)

        assert result == AlloyResult.SAT
        # The executable dialect preserves many-valued links through setattr.
        assert "setattr(" in codes[0]
        # The association should connect team to players or vice versa
        assert "team" in codes[0]
        assert "player" in codes[0]

    def test_generate_object_diagrams_object_model_contains_all(self, team_player_model, tmpdir):
        solver = AlloySolver(model=team_player_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        result, codes = solver.generate_object_diagrams()

        assert result == AlloyResult.SAT
        assert "ObjectModel(" in codes[0]
        # Team and Player objects must appear in the ObjectModel constructor
        assert "Team(" in codes[0]
        assert "Player(" in codes[0]

    def test_generate_class_and_object_model_team_player(self, team_player_model, tmpdir):
        solver = AlloySolver(model=team_player_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        result = solver.generate_class_and_object_model()

        assert result == AlloyResult.SAT
        outfile = Path(solver.output_dir) / "buml_class_object_model.py"
        integrated = outfile.read_text(encoding="utf-8")
        assert "# OBJECT MODEL #" in integrated
        assert 'Team("Team_' in integrated
        assert 'Player("Player_' in integrated

    def test_generate_class_and_object_model_executes(self, team_player_model, tmpdir):
        # The integrated model is a valid .py script: it can be exec'd to
        # reconstruct the class diagram + object model from the real Alloy
        # instance.
        solver = AlloySolver(model=team_player_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        result = solver.generate_class_and_object_model()

        assert result == AlloyResult.SAT
        outfile = Path(solver.output_dir) / "buml_class_object_model.py"
        integrated = outfile.read_text(encoding="utf-8")

        namespace = {}
        exec(compile(integrated, "<integrated>", "exec"), namespace)  # noqa: S102

        object_model = namespace["object_model"]
        assert object_model.name == "Object_Diagram"
        objects = object_model.objects
        assert len(objects) >= 1
        class_names = {obj.classifier.name for obj in objects}
        assert "Team" in class_names
        assert "Player" in class_names

    def test_generate_class_and_object_model_object_links(self, team_player_model, tmpdir):
        """The integrated file's OBJECT MODEL section must re-import into the
        editor keeping the object relationships (ObjectLinks).

        Regression test: ``generate_class_and_object_model`` used to emit the
        object model in the executable dialect (``setattr`` calls), which the
        web editor's ``object_buml_to_json`` only parses into ObjectLink
        relationships when the assignment is a plain ``obj.role = target``."""
        solver = AlloySolver(model=team_player_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        result = solver.generate_class_and_object_model()

        assert result == AlloyResult.SAT
        outfile = Path(solver.output_dir) / "buml_class_object_model.py"
        integrated = outfile.read_text(encoding="utf-8")
        obj_section = integrated.split("# OBJECT MODEL #", 1)[1].split("######################", 1)[0]

        reference_model = {
            "elements": {
                "e_team": {"name": "Team", "type": "Class", "attributes": {}},
                "e_player": {"name": "Player", "type": "Class", "attributes": {}},
            },
            "relationships": {
                "r_plays_for": {
                    "type": "ClassBidirectional",
                    "source": {"role": "players"},
                    "target": {"role": "team"},
                },
            },
        }

        obj_json = object_buml_to_json(obj_section, reference_model)

        assert obj_json is not None
        links = [
            rel for rel in obj_json["relationships"].values()
            if rel.get("type") == "ObjectLink"
        ]
        assert links, "the imported object diagram must contain ObjectLink relationships"

    def test_generate_object_diagram_json_team_player(self, team_player_model, tmpdir):
        """Editor-dialect code must convert to an ObjectDiagram JSON that keeps
        the objects and their attributes."""
        solver = AlloySolver(model=team_player_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        reference_model = {
            "elements": {
                "e_team": {
                    "name": "Team",
                    "type": "Class",
                    "attributes": {"a1": {"name": "name", "type": "str"}},
                },
                "e_player": {
                    "name": "Player",
                    "type": "Class",
                    "attributes": {
                        "a1": {"name": "name", "type": "str"},
                        "a2": {"name": "age", "type": "int"},
                    },
                },
            },
            "relationships": {},
        }
        result, codes = solver.generate_object_diagrams()

        assert result == AlloyResult.SAT
        assert codes
        obj_json = object_buml_to_json(codes[0], reference_model)

        assert obj_json is not None
        assert "elements" in obj_json
        object_names = {
            elem["name"]
            for elem in obj_json["elements"].values()
            if elem.get("type") == "ObjectName"
        }
        assert any(name.startswith("Team_") for name in object_names)
        assert any(name.startswith("Player_") for name in object_names)
        assert any(
            elem.get("type") == "ObjectAttribute"
            for elem in obj_json["elements"].values()
        )

    def test_check_consistency(self, team_player_model, tmpdir):
        solver = AlloySolver(model=team_player_model, output_dir=str(tmpdir.mkdir("out")), scope=self.scope)

        result = solver.check_consistency()

        assert result == AlloyResult.SAT