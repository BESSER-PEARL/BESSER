"""Does the agent of a class-model + agent-model project use the model's data?

Live (2026-10-30): "build my task assistant chatbot" shipped the BAF scaffold
alone; its add_task intent answered with canned text and never touched a task.
The orchestrator turns a negative answer here into a checklist item.
"""
from __future__ import annotations

import re

from besser.spec_driven_agent.planning.code_text import code_only
from besser.spec_driven_agent.planning.gap_analyzer import class_name_pattern
from besser.spec_driven_agent.planning.requirements_ledger import source_files

AGENT_DOMAIN_TASK = (
    "Connect the {agent} agent to the domain model: every intent about the "
    "model's data (adding, listing or changing records) reads and writes "
    "that data through the app's model or persistence code, not a canned "
    "reply. Drop this item, with the reason, only if the agent has nothing "
    "to do with the model's data."
)

# A file implementing this many of the agent's intents holds the agent, even
# without naming it (a web app's chat service rarely does).
_INTENTS_MARKING_AGENT_CODE = 2


def agent_uses_domain(output_dir: str, agent_model, domain_model) -> bool:
    """A source file holding the agent names a domain class or enumeration in code.

    The file holds the agent when it names the agent (or is named after it),
    imports BAF, or implements its intents. Names count in any casing
    (``TodoItem``, ``todo_item``): an agent with its own store uses table and
    field names, not the classes. Strings and comments do not count, so a reply
    such as "Task added!" or a training sentence is not a use. A generic
    enumeration name such as ``status`` can still match unrelated code; across
    14 live runs this test told every wired agent from the unwired one.
    """
    agent_name = agent_model.name.lower()
    intents = [intent.name.lower() for intent in agent_model.intents]
    needed = min(_INTENTS_MARKING_AGENT_CODE, len(intents))
    types = (*domain_model.get_classes(), *domain_model.get_enumerations())
    if not types:
        return False
    uses = re.compile(r"\b(?:" + "|".join(class_name_pattern(t.name) for t in types) + r")\b", re.IGNORECASE)
    for rel, path in source_files(output_dir).items():
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError:
            continue
        lower = text.lower()
        holds_agent = (agent_name in lower or agent_name in rel.lower() or "besser.agent" in lower
                       or (needed and sum(intent in lower for intent in intents) >= needed))
        if holds_agent and uses.search(code_only(text, path)):
            return True
    return False
