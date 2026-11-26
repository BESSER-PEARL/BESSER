"""build_generation_trace records the files an LLM produced with AI evidence, not rule evidence."""

from datetime import datetime

from besser.BUML.metamodel.traceability import AIEvidence, CodeGenerator, LLM, RuleEvidence
from besser.utilities.generation_trace import (
    ElementRef, FileTrace, GeneratedFile, InputModel, TraceRegion, TraceSource, build_generation_trace,
)

NOW = datetime(2026, 10, 1)
REGION = TraceRegion("s1", "State", "idle", "t#state", 1, 1)
SOURCE = TraceSource(ElementRef("a1", "Agent", "bot"), "t#agent")


def _files():
    return [
        GeneratedFile("bot.py", "idle = 1\n", FileTrace((REGION,), (SOURCE,))),
        GeneratedFile("config.yaml", "x: 1\n", FileTrace((), (SOURCE,))),
    ]


def _build(**options):
    return build_generation_trace(
        _files(), models=[InputModel(ElementRef("diagram-1", "Agent", "bot"))], generator="agent",
        generator_version="1", actor_name="Ada", max_stored_content_bytes=1000, started_at=NOW, ended_at=NOW,
        **options)


def _evidence_by_target(trace):
    result = {}
    for link in trace.trace_links:
        (evidence,) = link.evidence
        result.setdefault(link.target.name.split(" ")[0], set()).add(type(evidence))
    return result


def test_without_ai_files_every_link_is_rule_evidence_of_a_single_generator_call():
    trace = _build()
    assert trace.validate()["success"]
    assert {type(e) for link in trace.trace_links for e in link.evidence} == {RuleEvidence}
    (operation,) = trace.operations
    assert [type(tool) for tool in operation.tools] == [CodeGenerator]
    assert [type(tool) for tool in trace.tools] == [CodeGenerator]


def test_links_into_an_ai_file_are_ai_evidence_and_the_run_calls_the_llm():
    trace = _build(ai_files={"bot.py"}, llm=LLM("gpt-5", "openai"))
    assert trace.validate()["success"]
    by_target = _evidence_by_target(trace)
    # The model -> file, element -> file and element -> region links of bot.py are AI; config.yaml's stay rules.
    assert by_target["bot.py"] == {AIEvidence}
    assert by_target["idle"] == {AIEvidence}
    assert by_target["config.yaml"] == {RuleEvidence}
    assert {e.algorithm for link in trace.trace_links for e in link.evidence if isinstance(e, AIEvidence)} == {"gpt-5"}
    (operation,) = trace.operations
    assert [(call.sequence_number, type(call.tool)) for call in operation.tool_calls] == [(0, CodeGenerator), (1, LLM)]
    assert [type(tool) for tool in trace.tools] == [CodeGenerator, LLM]


def test_an_unnamed_llm_is_recorded_when_none_is_given_and_an_unknown_path_changes_nothing():
    trace = _build(ai_files={"bot.py"})
    assert [(tool.name, tool.provider) for tool in trace.tools if isinstance(tool, LLM)] == [("llm", "unknown")]
    untouched = _build(ai_files={"other.py"})
    assert {type(e) for link in untouched.trace_links for e in link.evidence} == {RuleEvidence}
    assert [type(tool) for tool in untouched.tools] == [CodeGenerator]
