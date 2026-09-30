"""The user's request reaches every deciding stage whole.

Live failure: a 4,622-char spec ending "Frontend -> React" was
scaffolded with no frontend, because the generator selector saw only
``instructions[:500]``. Stack declarations conventionally come LAST, so a
head-clip is the worst shape for that decision. Three such clips were found
and fixed separately that day, each believed to be the last - that pattern
is the defect. Here the invariant enforces itself: every bound on the
request must be declared (``user_request(..., excerpt=, reason=)`` or a
``bounded:`` comment), and the two one-shot decision stages are shown the
tail of a long spec.
"""

import ast
import re
from pathlib import Path

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)
import besser.spec_driven_agent as engine_pkg
from besser.spec_driven_agent.pipeline.orchestrator import LLMOrchestrator

ENGINE = Path(engine_pkg.__file__).parent
# ENGINE was besser/generators/llm (2 hops to besser/); it is now
# besser/spec_driven_agent (1 hop). Anchor on the besser package instead --
# a stale hop count here pointed SERVICE at a path that does not exist, and
# the scan below then silently examined nothing.
import besser as _besser
_BESSER_PKG = Path(_besser.__file__).resolve().parent
SERVICE = (
    _BESSER_PKG / "utilities" / "web_modeling_editor" / "backend"
    / "services" / "spec_driven"
)

# A slice of the request, or the request handed to something that clips or
# truncates it. ``user_request(...)`` is deliberately NOT matched here: its
# bounds are checked by the AST test below, where the reason is required.
_BOUND = re.compile(
    r"\b_?instructions\b(?:\.strip\(\))?\s*\[\s*:"
    r"|\b\w*(?:clip|trunc)\w*\(\s*(?:self\.)?_?instructions\b"
)
_DECLARED = "bounded:"


def _engine_sources():
    for root in (ENGINE, SERVICE):
        for path in sorted(root.rglob("*.py")):
            if path.name != "user_request.py":
                yield path


def _simple_model() -> DomainModel:
    cls = Class(name="Order")
    cls.attributes = {Property(name="ref", type=PrimitiveDataType("str"))}
    return DomainModel(name="Shop", types={cls})


def _long_spec(total_chars: int, tail: str) -> str:
    """A realistic multi-paragraph request of about ``total_chars`` that
    ends with ``tail`` - the shape a head-clip loses."""
    body: list[str] = []
    i = 0
    while sum(len(p) + 2 for p in body) < total_chars - len(tail) - 2:
        body.append(
            f"Requirement {i}: the system records every order line with a "
            f"quantity, a unit price and a status that moves from draft to "
            f"confirmed to shipped, and rejects edits once shipped."
        )
        i += 1
    return "\n\n".join(body) + "\n\n" + tail


class TestInvariant:

    def test_every_bound_on_the_request_is_declared(self):
        """A raw slice of ``instructions`` (or handing it to a clipper) must
        carry a ``bounded:`` comment on that line or the one above, saying
        why the consumer may see less than the whole request."""
        undeclared: list[str] = []
        for path in _engine_sources():
            lines = path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                if not _BOUND.search(line):
                    continue
                context = line + (lines[i - 1] if i else "")
                if _DECLARED not in context:
                    undeclared.append(f"{path.relative_to(_BESSER_PKG.parent)}:{i + 1}: {line.strip()}")
        assert not undeclared, (
            "Undeclared bound on the user's request. Take it whole, or add a "
            "'# bounded: <why>' comment on/above the line:\n  " + "\n  ".join(undeclared)
        )

    def test_excerpts_state_their_reason(self):
        """``user_request(..., excerpt=N)`` is the one sanctioned way to
        shorten the request; every such call names its reason literally."""
        from besser.spec_driven_agent.planning.user_request import user_request

        assert user_request("whole text") == "whole text"
        assert user_request(None) == ""
        assert user_request("abc", excerpt=3, reason="test") == "abc"      # fits: no marker
        cut = user_request("abcdef", excerpt=3, reason="test")
        assert cut.startswith("abc") and "def" not in cut and "truncated" in cut

        missing: list[str] = []
        for path in _engine_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name != "user_request":
                    continue
                keywords = {kw.arg: kw.value for kw in node.keywords}
                if "excerpt" not in keywords:
                    continue
                reason = keywords.get("reason")
                if not (isinstance(reason, ast.Constant) and isinstance(reason.value, str) and reason.value.strip()):
                    missing.append(f"{path.name}:{node.lineno}")
        assert not missing, f"user_request(excerpt=...) without a literal reason: {missing}"


class TestDecisionStagesSeeTheWholeRequest:

    def _orchestrator(self, client, tmp_path):
        return LLMOrchestrator(
            llm_client=client, domain_model=_simple_model(), output_dir=str(tmp_path),
        )

    def test_generator_selector_sees_the_stack_clause_at_the_end_of_a_long_spec(self, tmp_path):
        """The live case: ~4.6k chars, stack named in the last line."""
        from besser.spec_driven_agent.providers.llm_client import UsageTracker
        seen: dict = {}

        class SelectorClient:
            model = "mock"
            usage = UsageTracker("mock")
            _client = object()          # the selector skips clients without one

            # No force_tool/model_override and no **kwargs: takes the text path.
            def chat(self, system, messages, tools):
                seen["prompt"] = messages[0]["content"]
                return {"content": [{"type": "text", "text": "generate_fastapi_backend"}]}

        spec = _long_spec(4_622, "Additionally, you must use: Frontend -> React")
        assert len(spec) >= 4_600
        chosen = self._orchestrator(SelectorClient(), tmp_path)._select_generator_with_llm(spec)
        assert chosen == "generate_fastapi_backend"     # the mocked path really ran
        assert "Frontend -> React" in seen["prompt"]

    def test_modify_class_inference_sees_past_the_first_thousand_chars(self, tmp_path):
        from besser.spec_driven_agent.providers.llm_client import UsageTracker
        seen: dict = {}

        class DeltaClient:
            model = "mock"
            usage = UsageTracker("mock")

            def chat(self, system, messages, tools, force_tool=None, model_override=None):
                seen["prompt"] = messages[0]["content"]
                return {"content": [{"type": "tool_use", "input": {"new_classes": []}}]}

        spec = _long_spec(1_600, "Finally, add a LoyaltyAccount entity with a points balance.")
        assert spec.index("LoyaltyAccount") > 1_000
        self._orchestrator(DeltaClient(), tmp_path)._request_model_deltas(spec)
        assert "LoyaltyAccount" in seen["prompt"]


class TestGapAnalyserInventionList:

    def test_the_condition_is_read_before_the_nouns(self):
        """Twelve vivid feature nouns followed by a trailing "if the user
        asked" had the analyser inventing authentication work for requests
        that never mentioned it. The gate must come first; the enumeration
        and the do-not-return-empty push stay."""
        from besser.spec_driven_agent.planning.gap_analyzer import _SYSTEM_PROMPT

        start = _SYSTEM_PROMPT.index("what the deterministic generator does NOT produce")
        bullet = _SYSTEM_PROMPT[start:_SYSTEM_PROMPT.index("\n", start)].lower()
        gate = bullet.index("if, and only if")
        for noun in ("authentication", "jwt", "payments", "email", "file upload"):
            assert noun in bullet, noun
            assert gate < bullet.index(noun), noun
        assert "empty array" in bullet
