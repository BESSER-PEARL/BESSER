"""The gap planner reads the modeling agent's notes as context, not as the user's words.

Live evidence, p6-wastewater-dashboard (r9-p61, r11-p61, r12-p61): the request is
a dashboard; the "Assistant's notes" above it mention user authentication. The
planner was sent the whole instructions as ``USER REQUEST``, so its rule "only if
the user's own words asked for authentication" matched the notes, and 3-4 auth
tasks ("per user request for 'user authentication'") took 25-30% of each run. The
requirements ledger reads only the verbatim request and found no auth requirement.
"""

import besser.spec_driven_agent.pipeline.orchestrator as orchestrator_module
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator
from besser.spec_driven_agent.planning import gap_analyzer
from besser.spec_driven_agent.planning import requirements_ledger as ledger
from tests.spec_driven_agent.shelf_app import GENERATOR, generate_backend, shelf_model
from tests.spec_driven_agent.test_gap_planner_knows_the_target import _EndsTurn, _Planner

NOTES = ("## Assistant's notes (may be inaccurate)\n\nUse these only for choices the user confirmed in the "
         "conversation that are missing from their request below.\n\nGenerate a full web application with "
         "user authentication, using SQLite as the backend. Include features for user login and session "
         "management, ensuring secure access to the application.")
REQUEST = ("I am planning to make a dashboard for the wastewater virus tracking to show some data on a map "
           "and key metrics. ONce I click on a geojson part it should show the relevant plot")
P6 = (f"{NOTES}\n\n## The user's original request, verbatim\n\n"
      f"This is the authority. Where the summary above is shorter or differs, this text wins.\n\n{REQUEST}")


def test_the_split_follows_the_verbatim_marker():
    assert ledger.original_request(P6) == REQUEST
    assert ledger.assistant_notes(P6) == NOTES
    assert ledger.assistant_notes(REQUEST) == ""


def test_the_planner_prompt_labels_the_notes_apart_from_the_request():
    planner = _Planner()
    gap_analyzer.analyze_gaps_via_llm(
        instructions=REQUEST, assistant_notes=NOTES, generator_used="generate_fastapi_backend",
        domain_model=None, inventory="", llm_client=planner, workspace_files=[], action_endpoints=[])
    prompt = planner.prompts[0]
    request = prompt.split("USER REQUEST:\n", 1)[1].split("\n\n", 1)[0]
    assert request == REQUEST and "authentication" not in request
    assert "ASSISTANT'S NOTES" in prompt and "NOT the user's words" in prompt and "user authentication" in prompt


def test_the_orchestrator_sends_the_request_and_the_notes_apart(tmp_path, monkeypatch):
    generate_backend(tmp_path, shelf_model(author=True))
    seen = {}

    def planner(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(orchestrator_module, "analyze_gaps_via_llm", planner)
    orch = LLMOrchestrator(llm_client=_EndsTurn(), domain_model=shelf_model(author=True), output_dir=str(tmp_path),
                           enable_tracing=False, enable_checkpointing=False, enable_import_smoke_check=False,
                           enable_requirements_ledger=False, use_streaming=False, max_turns=1)
    orch._generator_used = GENERATOR
    orch._instructions = P6
    orch._run_phase2(P6)
    assert seen["instructions"] == REQUEST and seen["assistant_notes"] == NOTES

