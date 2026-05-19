"""Stereotype-token helper shared by Component / Deployment converters.

Resolves WME ``stereotype`` strings into typed metamodel slots
(``AgentCategory`` / ``Locality`` / ``NodeKind`` / ``AgenticEdgeKind``) per
the hybrid profile of decision D3, with case-insensitive token matching,
guillemets stripping, empty-stereotype short-circuit, and silent consumption
of Apollon default tokens. See 02-... §3.3 / §3.4 / §3.5 and Appendix B.2 for
the resolution tables and nuances.

Two directions:

* ``parse_stereotype(raw, element_type)`` -- tokenise a WME stereotype string
  and apply tokens to the metamodel element passed in (mutates in place).
* ``format_stereotype(...)`` -- build a WME stereotype string from a
  metamodel element's typed slots + free-form ``stereotypes`` list.

Both honour the same token tables in
``backend.constants.constants`` so the directions stay symmetric.
"""

import re
from typing import Optional

from besser.BUML.metamodel.uml_component import (
    AgentCategory,
    AgenticEdgeKind,
    Component,
    Locality as ComponentLocality,
    Skill,
    Tool,
)
from besser.BUML.metamodel.uml_deployment import (
    Artifact,
    Locality as DeploymentLocality,
    Node,
    NodeKind,
)
from besser.utilities.web_modeling_editor.backend.constants.constants import (
    AGENT_CATEGORY_TOKENS,
    AGENTIC_EDGE_KIND_TOKENS,
    COMPONENT_DEFAULT_STEREOTYPES,
    DEPLOYMENT_DEFAULT_STEREOTYPES,
    HUMAN_ACTOR_TOKENS,
    LOCALITY_TOKENS,
    NODE_KIND_TOKENS,
)


# Separator pattern: comma OR whitespace runs. WME emits a single token in
# practice, but the metamodel allows multi-stereotype lists so split tolerantly.
_TOKEN_SPLIT = re.compile(r"[,\s]+")

# Permission-suffix grammar for AgenticEdge stereotypes (02-... §3.7 / Q1=a).
# Example: "delegates {permission: repo:merge:approve, repo:write}".
_PERMISSION_SUFFIX = re.compile(r"\{permission:\s*(?P<scopes>[^}]*)\}", re.IGNORECASE)


def normalise_token(token: str) -> str:
    """Lowercase, trim, strip guillemets and outer quotes from a raw token."""
    cleaned = token.strip().strip("«»").strip("\"'")
    return cleaned.lower()


def tokenise(raw: Optional[str]) -> list:
    """Split a raw stereotype string into normalised tokens.

    Empty / None / whitespace-only / lone guillemets all return ``[]``.
    """
    if not raw or not isinstance(raw, str):
        return []
    # Strip the permission suffix first; the caller handles that separately.
    suffix_free = _PERMISSION_SUFFIX.sub(" ", raw)
    parts = [normalise_token(p) for p in _TOKEN_SPLIT.split(suffix_free) if p.strip()]
    return [p for p in parts if p]


def extract_permission_scopes(raw: Optional[str]) -> list:
    """Return the list of permission-scope strings from a stereotype suffix.

    Parses one ``{permission: scope1, scope2, ...}`` suffix per stereotype
    string. Returns ``[]`` when the suffix is absent. Scopes are trimmed but
    case is preserved (permission scopes are typically ``repo:merge:approve``
    style and case-sensitive on the consuming side).
    """
    if not raw or not isinstance(raw, str):
        return []
    match = _PERMISSION_SUFFIX.search(raw)
    if not match:
        return []
    scopes_blob = match.group("scopes")
    return [s.strip() for s in scopes_blob.split(",") if s.strip()]


def parse_component_node_subtype(raw: Optional[str]) -> Optional[str]:
    """Return ``"Skill"`` / ``"Tool"`` if the stereotype string promotes
    a bare ``Component`` to a subclass, else ``None``.

    Looks for the *first* subtype token in the tokenised list — Apollon
    typically emits one stereotype, but defensively scan all.
    """
    for token in tokenise(raw):
        if token == "skill":
            return "Skill"
        if token == "tool":
            return "Tool"
    return None


def parse_agentic_edge_kind(raw: Optional[str]) -> Optional[AgenticEdgeKind]:
    """Return the ``AgenticEdgeKind`` matching the first agentic-kind token,
    or ``None`` if the stereotype carries no agentic-kind token."""
    for token in tokenise(raw):
        if token in AGENTIC_EDGE_KIND_TOKENS:
            return AgenticEdgeKind(token)
    return None


def apply_component_stereotype_tokens(component: Component, raw: Optional[str]) -> None:
    """Apply WME stereotype tokens to a Component (or Skill / Tool / Subsystem).

    Sets typed slots (``agent_category``, ``locality``, ``is_human``) per the
    token tables in 02-... §3.3. Unknown tokens land in
    ``component.stereotypes`` as free-form passthrough. Default tokens
    (``"component"`` / ``"subsystem"``) are consumed silently.

    No-op if ``raw`` is empty / missing.
    """
    tokens = tokenise(raw)
    if not tokens:
        return
    leftovers = []
    for token in tokens:
        if token in COMPONENT_DEFAULT_STEREOTYPES:
            continue
        if token in AGENT_CATEGORY_TOKENS:
            component.agent_category = AgentCategory(token)
            continue
        if token in LOCALITY_TOKENS:
            component.locality = ComponentLocality(token)
            continue
        if token in HUMAN_ACTOR_TOKENS:
            component.is_human = True
            continue
        if token in {"skill", "tool"}:
            # Subtype promotion is handled by the processor at construction
            # time (we can't re-class an already-instantiated object). Drop
            # the token here so it doesn't double-emit on round-trip.
            continue
        leftovers.append(token)
    if leftovers:
        # Preserve order while deduping against what's already there.
        existing = set(component.stereotypes)
        for token in leftovers:
            if token not in existing:
                component.stereotypes = component.stereotypes + [token]
                existing.add(token)


def apply_node_stereotype_tokens(node: Node, raw: Optional[str]) -> None:
    """Apply WME stereotype tokens to a Deployment Node.

    Sets ``kind`` (NodeKind) and ``locality`` per the token tables in
    02-... §3.3 / §3.5. Default ``"node"`` token maps to ``NodeKind.GENERIC``.
    """
    tokens = tokenise(raw)
    if not tokens:
        return
    leftovers = []
    for token in tokens:
        if token in DEPLOYMENT_DEFAULT_STEREOTYPES:
            node.kind = NodeKind.GENERIC
            continue
        if token in NODE_KIND_TOKENS:
            # NODE_KIND_TOKENS is the lowercased set including
            # "executionenvironment"; NodeKind enum values are the
            # camelCase-preserved strings (D5 — `executionEnvironment`).
            if token == "executionenvironment":
                node.kind = NodeKind.EXECUTION_ENVIRONMENT
            elif token == "device":
                node.kind = NodeKind.DEVICE
            else:
                node.kind = NodeKind.GENERIC
            continue
        if token in LOCALITY_TOKENS:
            node.locality = DeploymentLocality(token)
            continue
        leftovers.append(token)
    if leftovers:
        existing = set(node.stereotypes)
        for token in leftovers:
            if token not in existing:
                node.stereotypes = node.stereotypes + [token]
                existing.add(token)


def apply_artifact_stereotype_tokens(artifact: Artifact, raw: Optional[str]) -> None:
    """Apply WME stereotype tokens to a Deployment Artifact. Only locality is
    typed; everything else lands in ``stereotypes``."""
    tokens = tokenise(raw)
    if not tokens:
        return
    leftovers = []
    for token in tokens:
        if token in DEPLOYMENT_DEFAULT_STEREOTYPES:
            # "node" doesn't apply to artifacts but Apollon may emit it on
            # synthetic shapes — drop silently.
            continue
        if token in LOCALITY_TOKENS:
            artifact.locality = DeploymentLocality(token)
            continue
        leftovers.append(token)
    if leftovers:
        existing = set(artifact.stereotypes)
        for token in leftovers:
            if token not in existing:
                artifact.stereotypes = artifact.stereotypes + [token]
                existing.add(token)


def format_component_stereotype(component: Component) -> str:
    """Build the WME ``stereotype`` string for a Component / Skill / Tool /
    Subsystem from its typed slots + free-form ``stereotypes`` list.

    Emission order:
    1. Agent category (if not NONE).
    2. Locality (if not LOCAL — the default).
    3. ``"human"`` (if is_human).
    4. Free-form ``stereotypes`` extras, in the order stored.

    Returns ``""`` when nothing would be emitted, so the converter can skip
    the field entirely (compact round-trip with omitting-style real exports).
    """
    parts = []
    if component.agent_category is not AgentCategory.NONE:
        parts.append(component.agent_category.value)
    if component.locality is not ComponentLocality.LOCAL:
        parts.append(component.locality.value)
    if component.is_human:
        parts.append("human")
    for extra in component.stereotypes:
        if extra and extra not in parts:
            parts.append(extra)
    return " ".join(parts)


def format_component_subtype_stereotype(component: Component) -> str:
    """For a Skill / Tool / bare-Component round-trip: emit the subtype
    promotion token in front of the rest so the importer sees it.
    """
    base = format_component_stereotype(component)
    if isinstance(component, Skill):
        prefix = "skill"
    elif isinstance(component, Tool):
        prefix = "tool"
    else:
        return base
    if not base:
        return prefix
    if prefix in base.split():
        return base
    return f"{prefix} {base}"


def format_agentic_edge_stereotype(kind: AgenticEdgeKind, permissions: list,
                                   extras: list) -> str:
    """Build the WME ``stereotype`` string for an AgenticEdge.

    ``permissions`` is the list of ``Permission`` instances on the edge; the
    function emits the kind token plus a sorted ``{permission: ...}`` suffix
    (Q1=a, byte-stable). ``extras`` is the edge's free-form ``stereotypes``
    list.
    """
    parts = [kind.value]
    for extra in extras:
        if extra and extra not in parts:
            parts.append(extra)
    base = " ".join(parts)
    if permissions:
        scopes = sorted({p.scope for p in permissions if p.scope})
        if scopes:
            base = f"{base} {{permission: {', '.join(scopes)}}}"
    return base


def format_node_stereotype(node: Node) -> str:
    """Build the WME ``stereotype`` string for a Deployment Node.

    Emits ``kind.value`` plus locality (when not LOCAL) plus extras.
    """
    parts = [node.kind.value]
    if node.locality is not DeploymentLocality.LOCAL:
        parts.append(node.locality.value)
    for extra in node.stereotypes:
        if extra and extra not in parts:
            parts.append(extra)
    return " ".join(parts)


def format_artifact_stereotype(artifact: Artifact) -> str:
    """Build the WME ``stereotype`` string for a Deployment Artifact.

    Real exports emit no stereotype on bare artifacts; only locality / extras
    show up if set. Returns ``""`` for the all-defaults case so the caller
    can omit the field.
    """
    parts = []
    if artifact.locality is not DeploymentLocality.LOCAL:
        parts.append(artifact.locality.value)
    for extra in artifact.stereotypes:
        if extra and extra not in parts:
            parts.append(extra)
    return " ".join(parts)
