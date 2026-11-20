"""The lab agent's prompts, the ``submit_plan`` tool and the outcome text."""

import os

import pytest

from besser.spec_driven_agent.lab.contracts import (
    FRONTEND_SUBMODULE_PATH,
    REPO_BESSER,
    REPO_FRONTEND,
    REPO_MODELING_AGENT,
    ChangeKind,
    CheckResult,
    ChecklistItem,
    FileChange,
    Plan,
    RepoChange,
    RepoSpec,
)
from besser.spec_driven_agent.lab.prompts import (
    MODEL_SPEC_SCHEMA,
    build_system_prompt,
    describe_outcome,
    first_user_message,
    format_model_spec,
    format_plan,
    submit_plan_tool,
)

# The brief caps the system prompt at roughly 1,200 words.
_MAX_PROMPT_WORDS = 1200


@pytest.fixture
def repos(tmp_path):
    root = str(tmp_path / "lab")
    besser = os.path.join(root, "besser")
    return {
        REPO_MODELING_AGENT: RepoSpec(REPO_MODELING_AGENT, os.path.join(root, "modeling-agent"), "develop",
                                      "BESSER-PEARL", "modeling-agent"),
        REPO_BESSER: RepoSpec(REPO_BESSER, besser, "development", "BESSER-PEARL", "BESSER"),
        REPO_FRONTEND: RepoSpec(REPO_FRONTEND, os.path.join(besser, *FRONTEND_SUBMODULE_PATH.split("/")), "develop",
                                "BESSER-PEARL", "BESSER-Web-Modeling-Editor"),
    }


def _note_spec():
    return {
        "classes": [
            {"name": "Element", "is_abstract": True},
            {"name": "Note", "attributes": [{"name": "text", "type": "str"}, {"name": "color", "type": "Color"}],
             "visibility": "public"},
            {"name": "Board"},
        ],
        "enumerations": [{"name": "Color", "literals": ["YELLOW", "BLUE"]}],
        "associations": [{"name": "board_notes", "source": "Note", "target": "Board",
                          "source_multiplicity": "0..*", "target_multiplicity": "1", "composition": True}],
        "generalizations": [{"child": "Note", "parent": "Element"}],
    }


def _plan_arguments(**overrides):
    arguments = {
        "kind": "metamodel",
        "title": "  Add a note element  ",
        "summary": "Adds a Note metaclass.",
        "steps": ["Add Note to besser/BUML/metamodel/structural/structural.py", "Test it"],
        "repos": [REPO_MODELING_AGENT, REPO_BESSER, REPO_BESSER],
        "questions": ["Should notes have a color?", "  "],
        "model_spec": _note_spec(),
    }
    arguments.update(overrides)
    return arguments


def _submit(arguments):
    plans = []
    tool = submit_plan_tool(plans.append)
    return tool.handler(arguments), plans


# ---- system prompt ----------------------------------------------------------------------------------------


def test_plan_prompt_names_the_repositories_relative_to_their_folder(repos):
    prompt = build_system_prompt(mode="plan", repos=repos)

    assert "- besser: `besser` (BESSER-PEARL/BESSER, base branch development)" in prompt
    assert f"- frontend: `besser/{FRONTEND_SUBMODULE_PATH}`" in prompt
    assert "- modeling_agent: `modeling-agent` (BESSER-PEARL/modeling-agent, base branch develop)" in prompt
    # The editor goes first: the order pull requests are opened in.
    assert prompt.index("- frontend:") < prompt.index("- besser:") < prompt.index("- modeling_agent:")
    assert str(repos[REPO_BESSER].path) not in prompt


def test_relative_repository_paths_are_kept(repos):
    relative = {REPO_BESSER: RepoSpec(REPO_BESSER, "besser", "development", "BESSER-PEARL", "BESSER")}

    assert "- besser: `besser` (" in build_system_prompt(mode="plan", repos=relative)


def test_prompt_says_guidance_paths_are_relative_to_each_repository(repos):
    # The kind guidance writes `besser/generators/registry.py`; as a tool path that is <root>/besser/generators/.
    prompt = build_system_prompt(mode="build", repos=repos, kind=ChangeKind.GENERATOR,
                                 guidance="- Register it in `besser/generators/registry.py`.")

    assert "Paths in the steps below and in each repository's own docs are relative to that repository" in prompt
    assert "`besser/besser/generators/registry.py`" in prompt
    assert prompt.index("relative to that repository") < prompt.index("## This change")


def test_plan_prompt_asks_for_a_plan_and_states_the_rules(repos):
    prompt = build_system_prompt(mode="plan", repos=repos)

    assert "CLAUDE.md" in prompt
    assert "call `submit_plan` once" in prompt
    assert "model_spec: required for a metamodel or diagram_type change" in prompt
    assert "Writing tools and the shell are refused" in prompt
    assert "Never commit, push" in prompt
    assert ".github/workflows/" in prompt
    assert "secrets" in prompt
    assert "run_checks" not in prompt


def test_build_prompt_carries_the_kind_guidance_notes_and_model(repos):
    prompt = build_system_prompt(mode="build", repos=repos, kind=ChangeKind.GENERATOR,
                                 notes="## Request 1\nAdd a note generator.\n",
                                 guidance="1. Package in besser/generators/<name>/",
                                 model_spec=_note_spec())

    assert "## This step: build" in prompt
    assert "`run_checks`" in prompt
    assert "submit_plan" not in prompt
    assert "## This change: generator (a new code generator)\n1. Package in besser/generators/<name>/" in prompt
    assert "## Notes of this change\n## Request 1\nAdd a note generator." in prompt
    assert "## Model of the extension\nClasses:\n- Element (abstract)" in prompt


def test_kind_without_guidance_still_names_the_kind(repos):
    prompt = build_system_prompt(mode="plan", repos=repos, kind=ChangeKind.BUGFIX)

    assert prompt.endswith("## This change: bugfix (a fix of a defect)")


@pytest.mark.parametrize("mode", ["plan", "build"])
def test_system_prompt_stays_short(repos, mode):
    assert len(build_system_prompt(mode=mode, repos=repos, kind=ChangeKind.METAMODEL).split()) < _MAX_PROMPT_WORDS


def test_unknown_step_is_refused(repos):
    with pytest.raises(ValueError, match="Unknown lab step"):
        build_system_prompt(mode="review", repos=repos)
    with pytest.raises(ValueError, match="Unknown lab step"):
        first_user_message(message="hi", mode="review", plan=None, model_spec_text=None)


# ---- submit_plan --------------------------------------------------------------------------------------------


def test_submit_plan_tool_is_read_only_and_named():
    tool = submit_plan_tool(lambda plan: None)

    assert tool.schema["name"] == "submit_plan"
    assert tool.writes is False
    assert tool.schema["input_schema"]["properties"]["model_spec"] is MODEL_SPEC_SCHEMA


def test_valid_arguments_become_a_plan():
    result, plans = _submit(_plan_arguments())

    assert result.startswith("Plan submitted.")
    [plan] = plans
    assert plan.kind is ChangeKind.METAMODEL
    assert plan.title == "Add a note element"
    assert plan.repos == [REPO_BESSER, REPO_MODELING_AGENT]
    assert plan.questions == ["Should notes have a color?"]
    assert plan.steps[0].startswith("Add Note")


def test_model_spec_is_normalized():
    _, [plan] = _submit(_plan_arguments())

    spec = plan.model_spec
    assert set(spec) == {"classes", "enumerations", "associations", "generalizations"}
    assert spec["classes"][1] == {"name": "Note", "is_abstract": False,
                                  "attributes": [{"name": "text", "type": "str"}, {"name": "color", "type": "Color"}]}
    assert spec["classes"][2] == {"name": "Board", "is_abstract": False, "attributes": []}
    assert spec["associations"][0]["composition"] is True


def test_plan_without_model_spec_for_a_generator():
    _, [plan] = _submit(_plan_arguments(kind="generator", model_spec=None))

    assert plan.model_spec is None


def test_every_problem_is_reported_and_no_plan_is_passed_on():
    plans = []
    tool = submit_plan_tool(plans.append)

    with pytest.raises(ValueError) as raised:
        tool.handler({"kind": "feature", "title": "x" * 73, "summary": " ", "steps": [], "repos": ["docs"]})

    message = str(raised.value)
    assert message.startswith("The plan was not submitted. Fix these and call submit_plan again:")
    for expected in ("kind: 'feature' is not one of generator,", "title: at most 72 characters",
                     "summary: a non-empty string is required.", "steps: at least one step is required.",
                     "repos: 'docs' is not one of frontend, besser, modeling_agent."):
        assert expected in message
    assert plans == []


def test_a_request_already_met_is_a_plan_that_changes_nothing():
    """E2E C-D4: "the request is already met" had to name a repository, so a change and three branches were made."""
    _, [plan] = _submit(_plan_arguments(kind="metamodel", repos=[], steps=[], model_spec=None,
                                        summary="get_build_sha already says why unknown means not stamped."))

    assert plan.changes_nothing and plan.steps == [] and plan.model_spec is None


def test_steps_without_a_repository_are_refused():
    plans = []
    with pytest.raises(ValueError) as raised:
        submit_plan_tool(plans.append).handler(_plan_arguments(repos=[]))

    assert "repos: the repositories the steps change are required" in str(raised.value) and plans == []


def test_plan_step_executor_offers_submit_plan_and_returns_refusals_as_errors(tmp_path):
    from besser.spec_driven_agent.lab.tooling import LabToolExecutor

    plans = []
    executor = LabToolExecutor(str(tmp_path), read_only=True, extra_tools=[submit_plan_tool(plans.append)])
    try:
        assert "submit_plan" in {schema["name"] for schema in executor.schemas}

        text, is_error = executor.execute("submit_plan", _plan_arguments(steps=[]))
        assert is_error and "steps: at least one step is required." in text and plans == []

        text, is_error = executor.execute("submit_plan", _plan_arguments())
        assert not is_error and text.startswith("Plan submitted.") and len(plans) == 1
    finally:
        executor.close()


@pytest.mark.parametrize("kind", ["metamodel", "diagram_type"])
def test_metamodel_and_diagram_type_plans_need_a_model(kind):
    with pytest.raises(ValueError, match=f"model_spec: a {kind} change needs the class diagram"):
        _submit(_plan_arguments(kind=kind, model_spec=None))


@pytest.mark.parametrize("change, expected", [
    (lambda s: s["classes"][1]["attributes"].append({"name": "size", "type": "Integer"}),
     "model_spec.classes[1].attributes[2].type: 'Integer' is neither a primitive type"),
    (lambda s: s["classes"][1]["attributes"].append({"name": "text", "type": "str"}),
     "model_spec.classes[1].attributes[2].name: 'text' is already an attribute of this class."),
    (lambda s: s["classes"].append({"name": "Color"}),
     "model_spec.enumerations[0].name: 'Color' names two classes or enumerations."),
    (lambda s: s["classes"].append({"name": "Sticky note"}),
     "model_spec.classes[3].name: 'Sticky note' is not a Python identifier"),
    (lambda s: s["associations"][0].update(target="Wall"),
     "model_spec.associations[0].target: 'Wall' is not a class of this diagram."),
    (lambda s: s["associations"][0].update(target="Color"),
     "model_spec.associations[0].target: 'Color' is not a class of this diagram."),
    (lambda s: s["associations"][0].update(source_multiplicity="many"),
     "model_spec.associations[0].source_multiplicity: 'many' is not a multiplicity"),
    (lambda s: s["associations"][0].update(source_multiplicity="*..1"),
     "model_spec.associations[0].source_multiplicity: '*..1' is not a multiplicity"),
    (lambda s: s["associations"][0].update(target_multiplicity="3..1"),
     "model_spec.associations[0].target_multiplicity: in '3..1' the lower bound exceeds the upper bound."),
    (lambda s: s["associations"][0].pop("name"),
     "model_spec.associations[0].name: a non-empty string is required."),
    (lambda s: s["generalizations"].append({"child": "Board", "parent": "Board"}),
     "model_spec.generalizations[1]: a class cannot generalize itself."),
    (lambda s: s["enumerations"][0].update(literals=["YELLOW", "YELLOW"]),
     "model_spec.enumerations[0].literals: a literal is listed twice."),
    (lambda s: s["enumerations"][0].update(literals=[]),
     "model_spec.enumerations[0].literals: at least one literal is required."),
    (lambda s: s.update(classes=[]), "model_spec.classes: at least one class is required."),
    (lambda s: s.update(associations="none"), "model_spec.associations: a list is required."),
    # Specs the editor's class diagram converter cannot build (a generalization cycle crashed the proposal).
    (lambda s: s["generalizations"].extend([{"child": "Element", "parent": "Board"},
                                             {"child": "Board", "parent": "Note"}]),
     "model_spec.generalizations: Board, Element, Note generalize each other in a cycle;"),
    (lambda s: s["associations"].append({"name": "board_notes", "source": "Board", "target": "Note",
                                         "source_multiplicity": "1", "target_multiplicity": "0..*"}),
     "model_spec.associations[1].name: 'board_notes' names two associations."),
    (lambda s: s["associations"][0].update(target_multiplicity="0"),
     "model_spec.associations[0].target_multiplicity: in '0' the upper bound is 0;"),
    (lambda s: s["associations"][0].update(source_multiplicity="0..0"),
     "model_spec.associations[0].source_multiplicity: in '0..0' the upper bound is 0;"),
])
def test_model_spec_problems_are_reported(change, expected):
    spec = _note_spec()
    change(spec)

    with pytest.raises(ValueError) as raised:
        _submit(_plan_arguments(model_spec=spec))
    assert expected in str(raised.value)


@pytest.mark.parametrize("multiplicity", ["1", "0..1", "0..*", "1..*", "*", "2..5"])
def test_valid_multiplicities_are_accepted(multiplicity):
    spec = _note_spec()
    spec["associations"][0]["source_multiplicity"] = multiplicity

    _, [plan] = _submit(_plan_arguments(model_spec=spec))
    assert plan.model_spec["associations"][0]["source_multiplicity"] == multiplicity


def test_schemas_are_valid_json_schema_and_agree_with_the_checks():
    jsonschema = pytest.importorskip("jsonschema")
    tool_schema = submit_plan_tool(lambda plan: None).schema["input_schema"]
    jsonschema.Draft202012Validator.check_schema(tool_schema)

    jsonschema.validate(_plan_arguments(), tool_schema)
    bad = _note_spec()
    bad["associations"][0]["source_multiplicity"] = "many"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, MODEL_SPEC_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_plan_arguments(kind="feature"), tool_schema)


# ---- text forms -----------------------------------------------------------------------------------------------


def test_format_model_spec():
    _, [plan] = _submit(_plan_arguments())

    assert format_model_spec(plan.model_spec) == "\n".join([
        "Classes:",
        "- Element (abstract)",
        "- Note: text: str, color: Color",
        "- Board",
        "Enumerations:",
        "- Color: YELLOW, BLUE",
        "Associations:",
        "- board_notes: Note [0..*] -- Board [1], composition (Board is the whole)",
        "Generalizations:",
        "- Note is a Element",
    ])


def _plan(**overrides):
    fields = {"kind": ChangeKind.GENERATOR, "title": "Add a note generator", "summary": "Writes notes.",
              "steps": ["Add the package", "Register it"], "repos": [REPO_BESSER], "questions": ["Which suffix?"]}
    fields.update(overrides)
    return Plan(**fields)


def test_format_plan():
    assert format_plan(_plan()) == "\n".join([
        "Title: Add a note generator", "Kind: generator", "Repositories: besser", "Summary: Writes notes.",
        "Steps:", "1. Add the package", "2. Register it", "Open questions:", "- Which suffix?",
    ])


def test_first_message_of_a_new_plan_is_the_request():
    assert first_user_message(message=" Add a note generator ", mode="plan", plan=None,
                              model_spec_text=None) == "## The user's message\nAdd a note generator"


def test_first_message_of_a_revised_plan_carries_the_plan_and_the_diagram():
    text = first_user_message(message="Use .md files", mode="plan", plan=_plan(), model_spec_text="Classes:\n- Note")

    assert "## The current plan\nRevise it to answer the message, then call submit_plan.\nTitle: Add a note" in text
    # E2E A-1: a rename the user made on the canvas was lost because the revision started from the old plan's model.
    assert ("## The class diagram of this change in the editor\nIt is the model of the extension as the user "
            "corrected it after review: the revised plan's model_spec starts from it, not from the earlier plan's "
            "model in the notes, and keeps the user's corrections unless the message asks to change them."
            "\nClasses:\n- Note") in text


def test_first_message_of_a_build_follows_the_reviewed_model():
    text = first_user_message(message="Go ahead", mode="build", plan=_plan(), model_spec_text="Classes:\n- Note")

    assert text.startswith("## The user's message\nGo ahead\n\n## The approved plan\nTitle: Add a note generator")
    assert "as the user left it after review: where it differs from the plan, follow it.\nClasses:\n- Note" in text


# ---- outcome ----------------------------------------------------------------------------------------------------


def test_outcome_lists_files_per_repository_checks_and_open_items():
    changes = [
        RepoChange(REPO_BESSER, "lab/jordi/note", "development", (
            FileChange("tests/generators/note/test_note.py", "added"),
            FileChange("besser/generators/registry.py", "modified"),
            FileChange("besser/generators/note/__init__.py", "added"),
        )),
        RepoChange(REPO_MODELING_AGENT, "lab/jordi/note", "develop", ()),
        RepoChange(REPO_FRONTEND, "lab/jordi/note", "develop", (FileChange("packages/i18n/en/webapp.json",
                                                                            "modified"),)),
    ]
    checks = [
        CheckResult("besser:tests", REPO_BESSER, "Tests of the touched areas", True, summary="12 passed"),
        CheckResult("besser:ruff", REPO_BESSER, "Ruff", False, summary="2 errors"),
        CheckResult("frontend:types", REPO_FRONTEND, "Type check", False, skipped=True, summary="npx not found"),
    ]
    checklist = [
        ChecklistItem("generator.registry", "Registered in registry.py", True),
        ChecklistItem("generator.docs", "Docs page in a toctree", False),
        ChecklistItem("i18n", "New strings in webapp.json", None),
    ]

    text = describe_outcome(changes, checks, checklist, "  Added the note generator.\n")

    assert text == "\n".join([
        "frontend (branch lab/jordi/note, base develop): 1 modified",
        "  modified packages/i18n/en/webapp.json",
        "besser (branch lab/jordi/note, base development): 2 added, 1 modified",
        "  added besser/generators/note/__init__.py",
        "  added tests/generators/note/test_note.py",
        "  modified besser/generators/registry.py",
        "Checks: 1 passed, 1 failed, 1 skipped.",
        "  failed Ruff (besser): 2 errors",
        "  skipped Type check (frontend): npx not found",
        "Checklist: 1 of 3 done, 1 not judged.",
        "  open: Docs page in a toctree",
        "  not judged: New strings in webapp.json",
        "",
        "Added the note generator.",
    ])
    assert describe_outcome(list(reversed(changes)), checks, checklist, "Added the note generator.") == text


def test_outcome_without_changes_or_checks():
    assert describe_outcome([], [], [], "") == "No files changed.\nChecks: none ran."


def test_outcome_counts_files_beyond_the_listed_ones():
    files = tuple(FileChange(f"tests/generators/snapshots/s{i:02}.json", "modified") for i in range(30))

    lines = describe_outcome([RepoChange(REPO_BESSER, "lab/a/b", "development", files)], [], [], "").splitlines()

    assert lines[0] == "besser (branch lab/a/b, base development): 30 modified"
    assert lines[25] == "  modified tests/generators/snapshots/s24.json"
    assert lines[26] == "  and 5 more files"
