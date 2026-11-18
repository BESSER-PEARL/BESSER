"""A downloaded run says what was built and what was checked.

Live evidence (run 0e794481): the ZIP held the code, the attribution trace and
a provenance file naming only the version and the model. The recipe, which
holds the request, the target and every validation finding, is never shipped,
so the download could not be reviewed. The request itself stays out: this file
also goes into GitHub pushes, public ones included.
"""

from besser.utilities.provenance import generation_provenance_markdown
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import (
    run_summary_markdown,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.sse_events import (
    VerificationCounts, VerificationItem, VerificationReport,
)

# No known provider prefix: only the run's own key list can redact it.
KEY = "abcd1234efgh5678ijkl9012mnop"


def _report():
    return VerificationReport(
        verified=[VerificationItem(kind="requirement", what="Create a task")],
        shippedUnenforced=[VerificationItem(
            kind="requirement", what="Users can create, view, edit and delete Tag records",
            why="tags can only be added")],
        notVerified=[VerificationItem(kind="check", what="the Python compiler", why="no toolchain")],
        counts=VerificationCounts(verified=1, notVerified=1, shippedUnenforced=1),
    )


def _summary(**recipe):
    base = {
        "instructions": f"quiero una app de windows 11 local para mis tareas, key {KEY}",
        "target": {"kind": "other", "description": "Windows 11 desktop todo app",
                   "has_user_interface": True},
        "generator_used": None,
    }
    return "\n".join(run_summary_markdown({**base, **recipe}, _report(), secrets=(KEY,)))


def test_summary_names_target_and_verification_but_not_the_request():
    text = _summary()
    assert "quiero una app" not in text and "Request" not in text
    assert "Built as: `Windows 11 desktop todo app`" in text
    assert "none, written from scratch" in text
    assert "1 confirmed by running the app's API, 0 found in the code but not run, 1 not checked, 1 checked and missing" in text
    assert "edit and delete Tag records`: `tags can only be added`" in text


def test_summary_never_ships_the_run_key():
    report = _report()
    report.notVerified[0].why = f"the probe was refused with key {KEY}"
    summary = run_summary_markdown({"generator_used": None}, report, secrets=(KEY,))
    assert "the probe was refused with key" in "\n".join(summary)
    assert KEY not in "\n".join(summary)


def test_provenance_file_carries_the_summary():
    text = generation_provenance_markdown("spec_driven_agent", {}, ["## This run", "", "- Built as: desktop app"])
    assert "## This run" in text
    assert text.index("## This run") < text.index("Regenerating from the model")
    assert "## This run" not in generation_provenance_markdown("django", {})


def test_model_text_is_shown_as_code_never_as_markup():
    # Review: the file goes to GitHub, where a model-written link or HTML would render.
    report = _report()
    report.shippedUnenforced[0].why = "see [docs](https://evil.example) <img src=x> `tick`"
    text = "\n".join(run_summary_markdown({"generator_used": "generate_django\n# Pwn"}, report))
    assert "`see [docs](https://evil.example) <img src=x> 'tick'`" in text
    assert "\n# Pwn" not in text and "`generate_django # Pwn`" in text


def test_without_the_run_key_list_a_prefixless_key_would_leak():
    # Shows the secrets argument is what protects a key with no known prefix.
    report = _report()
    report.notVerified[0].why = f"refused with key {KEY}"
    assert KEY in "\n".join(run_summary_markdown({"generator_used": None}, report, secrets=()))


def test_a_packaged_backend_says_where_its_run_instructions_are(tmp_path):
    """A backend on its own gets a README, and the run summary points to it."""
    import zipfile

    from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import SmartGenerationRunner
    from tests.spec_driven_agent.test_frontend_scaffold_repair import _MAIN_API_PY
    from tests.utilities.web_modeling_editor.backend._runtime import make_runtime
    from tests.utilities.web_modeling_editor.backend.spec_driven.test_runner import _build_request

    result = tmp_path / "besser_llm_example" / "result"
    (result / "backend").mkdir(parents=True)
    (result / "backend" / "main_api.py").write_text(_MAIN_API_PY, encoding="utf-8")
    (result / "backend" / "requirements.txt").write_text("fastapi\nuvicorn\n", encoding="utf-8")
    runner = SmartGenerationRunner(_build_request(), runtime=make_runtime())
    runner.temp_dir = str(result.parent)

    _done, entry = runner._package_result(str(result))

    with zipfile.ZipFile(entry.file_path) as archive:
        readme = archive.read("README.md").decode("utf-8")
        summary = archive.read("BESSER_GENERATION.md").decode("utf-8")
    assert "cd backend" in readme and "python main_api.py" in readme
    assert "- How to run it: see `README.md`" in summary


def test_the_summary_points_to_no_readme_that_is_not_there():
    assert not [line for line in run_summary_markdown({}, VerificationReport()) if "README" in line]
