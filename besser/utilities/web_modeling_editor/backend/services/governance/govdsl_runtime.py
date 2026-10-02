"""Parse a WME-authored Governance DSL snippet during Docker Compose generation."""
import io
import logging

from besser.utilities.web_modeling_editor.backend.services.exceptions import (
    ConfigurationError,
    GovernanceDslValidationError,
)

logger = logging.getLogger(__name__)

# Voting policy types resolved by the deterministic in-container tally. Mirrors
# docker_compose_generator._VOTING_POLICIES. Non-voting policies (leader/consensus/lazy)
# have NO vote, so their merge instruction must not ask the LLM to narrate one.
_VOTING_POLICY_TYPES = frozenset(("VotingPolicy", "MajorityPolicy", "AbsoluteMajorityPolicy"))

# Per-policy-type decision rule, in plain language for the LLM (the "concepts").
_POLICY_RULES = {
    "VotingPolicy": ("Treat each collaborator's reply as a weighted vote; select the "
                     "answer whose combined weight reaches the ratio threshold."),
    "MajorityPolicy": ("Select the answer supported by more than half of the "
                       "collaborators (ratio defaults to 0.5)."),
    "AbsoluteMajorityPolicy": ("Select the answer supported by more than half of ALL "
                               "eligible collaborators; abstentions count against it."),
    "LeaderDrivenPolicy": ("Defer to the lead collaborator's reply; only fall back to "
                           "the other replies if the leader gives none."),
    "ConsensusPolicy": ("Synthesise a single answer that ALL collaborators could agree "
                        "on; if they fundamentally conflict, say so explicitly."),
    "LazyConsensusPolicy": ("Accept the proposed answer unless a collaborator objects — "
                            "silence means assent."),
}


def _strip_comments(text: str) -> str:
    # The govdsl grammar has no LINE_COMMENT rule; WME emits `//` headers.
    # Strip them so the parse succeeds regardless of the upstream grammar fix.
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("//"))

def _parse_policies(text: str):
    """Run the published GovernanceDSL parser and return policy metamodel objects."""
    try:
        from antlr4 import CommonTokenStream, InputStream, ParseTreeWalker
        from governancedsl.grammar.govdslLexer import govdslLexer
        from governancedsl.grammar.govdslParser import govdslParser
        from governancedsl.grammar.PolicyCreationListener import PolicyCreationListener
        from governancedsl.grammar.govErrorListener import govErrorListener
    except ImportError as exc:
        raise ConfigurationError(
            "Governance DSL parser dependency is unavailable; install governancedsl==0.1.1."
        ) from exc

    lexer = govdslLexer(InputStream(text))
    lexer.removeErrorListeners()
    lexer_errors = govErrorListener(io.StringIO())
    lexer.addErrorListener(lexer_errors)

    parser = govdslParser(CommonTokenStream(lexer))
    parser.removeErrorListeners()
    parser_errors = govErrorListener(io.StringIO())
    parser.addErrorListener(parser_errors)

    tree = parser.governance()

    syntax_error = lexer_errors.error_message or parser_errors.error_message
    if syntax_error:
        raise GovernanceDslValidationError(
            f"Governance DSL syntax error: {syntax_error}"
        )

    try:
        listener = PolicyCreationListener()
        ParseTreeWalker().walk(listener, tree)
        policies = listener.get_policies()
        if not policies:
            raise ValueError("no policies parsed")
        return policies
    except Exception as exc:
        raise GovernanceDslValidationError(
            f"Governance DSL validation error: {exc}"
        ) from exc

def _summarize_policy(policy) -> dict:
    from governancedsl.metamodel.governance import Agent, Human, Role

    ptype = type(policy).__name__
    participants, requires_human = [], False
    for p in (getattr(policy, "participants", None) or []):
        if isinstance(p, Agent):
            kind = "agent"
        elif isinstance(p, Human):
            kind = "human"
        elif isinstance(p, Role):
            kind = "role"
        else:
            kind = "other"
        # A human can enter the policy two ways: as a non-(Agent) Individual (parsed as
        # Human) or as a Role standing in for human actors (the WME generator only ever
        # emits agents as `(Agent)`, so any Role/Human participant is a person). Either
        # way the merge point must defer the decision to a human.
        if kind != "agent":
            requires_human = True
        participants.append({
            "name": p.name,
            "kind": kind,
            "confidence": getattr(p, "confidence", None),
            "roles": sorted(r.name for r in (getattr(p, "roles", None) or [])),
        })
    return {
        "policy_type": ptype,
        "ratio": getattr(policy, "ratio", None),
        "decision_type": type(policy.decision_type).__name__ if getattr(policy, "decision_type", None) else None,
        "participants": participants,
        "requires_human": requires_human,
    }


def _build_instruction(summary: dict):
    """Return (human_summary, llm_instruction).

    `human_summary` is the policy facts shown to a person at the approval step.
    `llm_instruction` adds the merge preamble and the auditable-vote directive on top
    of those facts; it is the agent's system message only, NOT for human display.
    """
    rule = _POLICY_RULES.get(summary["policy_type"],
                             "Apply the stated governance policy to merge the replies.")
    names = ", ".join(p["name"] for p in summary["participants"]) or "(unspecified)"
    facts = [
        f"Policy type: {summary['policy_type']}.",
        f"Decision rule: {rule}",
        f"Participants (collaborators): {names}.",
    ]
    if summary["ratio"] is not None:
        facts.append(f"Ratio threshold: {summary['ratio']}.")
    if summary["decision_type"]:
        facts.append(f"Decision type: {summary['decision_type']}.")
    human_summary = "\n".join(facts)
    if summary["policy_type"] in _VOTING_POLICY_TYPES:
        # Voting policies are normally resolved by the deterministic in-container tally
        #; this directive only steers the LLM on the degraded fallback path
        # (no producer resolved to a service), where an auditable vote narrative is the
        # best available audit trail.
        directive = (
            "First output a 'Votes:' section so the decision is auditable: list each "
            "collaborator using the reply label you were given (e.g. an agent name with "
            "its #replica index) and the position/option its reply supports; add your own "
            "position, and the human's choice if one was provided. Then state the tally "
            "against the policy (and the ratio threshold, if any) and the resulting "
            "outcome. Finally, give the merged final answer.")
    else:
        # Non-voting policies (leader-driven / consensus / lazy-consensus) are NOT decided
        # by a vote. Asking for a 'Votes:'/tally section makes the LLM fabricate one (e.g. a
        # LeaderDriven merge inventing "3 Approve, 0 Disapprove"). Forbid the tally AND the
        # softer "the collaborators approved/agreed" narrative it falls back to — neither
        # event happened; the collaborators only returned candidate replies. Just merge.
        directive = (
            "Apply the decision rule above to the collaborators' replies and produce a "
            "single merged final answer. Output ONLY that answer. Do NOT invent a vote, "
            "tally, approval count, or outcome line, and do NOT claim that the collaborators "
            "approved, agreed on, endorsed, confirmed, or reviewed the answer — no such step "
            "happened. This policy is not decided by voting.")
    llm_instruction = "\n".join(
        ["You are the merge point of an agent swarm. Apply this governance policy to "
         "combine the collaborators' replies into one result."]
        + facts + [directive])
    return human_summary, llm_instruction


def summarize_governance(dsl_text):
    """Return a structured governance summary, or None only for empty DSL text."""
    if not dsl_text or not dsl_text.strip():
        return None

    policies = _parse_policies(_strip_comments(dsl_text))
    if len(policies) > 1:
        logger.warning(
            "[governance] %d policies parsed from one gateway's DSL; v1 wires only the first (%s)",
            len(policies),
            type(policies[0]).__name__,
        )

    summary = _summarize_policy(policies[0])
    human_summary, instruction = _build_instruction(summary)
    return {
        "instruction": instruction,
        "summary": human_summary,
        "requires_human": summary["requires_human"],
        "policy_type": summary["policy_type"],
        "participants": summary["participants"],
        "ratio": summary["ratio"],
        "raw": dsl_text,
    }
