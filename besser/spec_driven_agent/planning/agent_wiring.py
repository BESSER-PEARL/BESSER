"""The agent of a class-model + agent-model project must use the model's data.

Live (2026-10-30): "build my task assistant chatbot" shipped the BAF scaffold
alone; its add_task intent answered with canned text and never touched a task.
Phase 2 gets a checklist item to wire it, and the requirements judge checks it
in Phase 3 from a cited line of code, in any language and however the agent
reaches the data (models, an API, a helper module).
"""
from __future__ import annotations

AGENT_DATA_KIND = "agent_data"

AGENT_DOMAIN_TASK = (
    "Connect the {agent} agent to the domain model: every intent about the "
    "model's data (adding, listing or changing records) reads and writes "
    "that data through the app's model or persistence code, not a canned "
    "reply. Drop this item, with the reason, only if the agent has nothing "
    "to do with the model's data."
)

# Enough for the judge to tell data intents from small talk, short enough
# for one requirement line.
_MAX_INTENTS = 12
_EXAMPLES_PER_INTENT = 2
_MAX_EXAMPLE_CHARS = 60


def agent_data_requirement(agent_model, domain_model) -> dict | None:
    """The requirement the judge checks, or None without intents or domain types."""
    if agent_model is None or domain_model is None or not agent_model.intents:
        return None
    types = sorted(t.name for t in (*domain_model.get_classes(), *domain_model.get_enumerations()))
    if not types:
        return None
    records = ", ".join(types)
    intents = "; ".join(
        f"{intent.name} ({', '.join(repr(s[:_MAX_EXAMPLE_CHARS]) for s in intent.training_sentences[:_EXAMPLES_PER_INTENT])})"
        for intent in agent_model.intents[:_MAX_INTENTS]
    )
    return {
        "kind": AGENT_DATA_KIND,
        # Names the records, not "the app's data": the live judge counted a help
        # bot's opening hours as data and failed it for answering with a fixed reply.
        "text": (
            f"Every intent of the {agent_model.name} agent that creates, lists or changes "
            f"{records} records reads or writes them through the app's code (its models, "
            "database, API or a helper it calls), not with a fixed reply. An intent that is "
            f"not about {records} records (a greeting, help, a fixed fact such as opening "
            "hours) is rightly answered with a fixed reply. A reply that only names the "
            "records, such as 'Task added!', does not read or write them. Implemented when "
            f"every intent about {records} records reads or writes them, and also when no "
            "intent is about them. Evidence: the line that reads or writes the records; "
            "when no intent is about them, the line defining one intent, saying so in the "
            f"note. The intents: {intents}."
        ),
    }
