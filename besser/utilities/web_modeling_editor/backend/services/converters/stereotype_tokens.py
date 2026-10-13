"""Stereotype-token helper shared by Component / Deployment converters.

Resolves WME ``stereotype`` strings into typed metamodel slots
(``AgentCategory`` / ``Locality`` / ``NodeKind`` / ``AgenticEdgeKind``) with
case-insensitive token matching and guillemets stripping. Tokens without a
typed slot -- including the editor's defaults such as «component», «subsystem»
and «node» -- are kept in the element's free-form ``stereotypes`` list, so the
stereotype the editor wrote comes back unchanged.

Two directions:

* ``apply_*_stereotype_tokens(element, raw)`` -- tokenise a WME stereotype
  string and apply the tokens to the metamodel element (mutates in place).
* ``format_*_stereotype(element)`` -- build a WME stereotype string from the
  element's typed slots + free-form ``stereotypes`` list.

Both use the token tables in ``backend.constants.constants`` so the directions
stay symmetric.
"""

import re
from typing import Optional

from besser.BUML.metamodel.uml_component import (
    AgentCategory,
    AgenticComponent,
    AgenticEdgeKind,
    Component,
    Database,
    LLM,
    Locality as ComponentLocality,
    RAG,
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
    COMPONENT_SUBTYPE_TOKENS,
    LOCALITY_TOKENS,
    NODE_KIND_TOKENS,
)


# Separator pattern: comma OR whitespace runs. WME emits a single token in
# practice, but the metamodel allows multi-stereotype lists so split tolerantly.
_TOKEN_SPLIT = re.compile(r"[,\s]+")

# Permission suffix on AgenticEdge stereotypes, e.g.
# "delegates {permission: repo:merge:approve, repo:write}".
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

    Parses one ``{permission: scope1, scope2}`` suffix per stereotype
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
    """Return the BUML class name (``"Skill"`` / ``"Tool"`` / ``"LLM"`` /
    ``"Database"`` / ``"RAG"``) if the stereotype string promotes a bare
    ``Component`` to a subclass, else ``None``.

    Looks for the *first* subtype token in the tokenised list — the editor
    typically emits one stereotype, but defensively scan all.
    """
    for token in tokenise(raw):
        if token in COMPONENT_SUBTYPE_TOKENS:
            return COMPONENT_SUBTYPE_TOKENS[token]
    return None


def parse_agentic_edge_kind(raw: Optional[str]) -> Optional[AgenticEdgeKind]:
    """Return the ``AgenticEdgeKind`` matching the first agentic-kind token,
    or ``None`` if the stereotype carries no agentic-kind token."""
    for token in tokenise(raw):
        if token in AGENTIC_EDGE_KIND_TOKENS:
            return AgenticEdgeKind(token)
    return None


def agentic_edge_extra_tokens(raw: Optional[str], kind: AgenticEdgeKind) -> list:
    """Return the tokens of an AgenticEdge stereotype other than its kind token
    (the permission suffix is not a token)."""
    return [token for token in tokenise(raw) if token != kind.value]


def stereotype_has_agentic_tokens(raw: Optional[str]) -> bool:
    """True if the stereotype string carries any token that requires an
    ``AgenticComponent`` (an agent-category token).

    The processor uses this to decide whether to build an ``AgenticComponent``
    or a plain ``Component``. ``locality`` tokens are *not* agentic -- they set
    ``Component.locality`` and keep the element a plain ``Component``. A «human»
    stereotype is likewise *not* agentic: a human has no implementation and
    stays a plain ``Component``.
    """
    for token in tokenise(raw):
        if token in AGENT_CATEGORY_TOKENS:
            return True
    return False


def _keep_free_form(element, tokens: list) -> None:
    """Append ``tokens`` to ``element.stereotypes``, in order, without duplicates."""
    existing = list(element.stereotypes)
    for token in tokens:
        if token not in existing:
            existing.append(token)
    element.stereotypes = existing


def apply_component_stereotype_tokens(component: Component, raw: Optional[str]) -> None:
    """Apply WME stereotype tokens to a Component (or AgenticComponent / Skill /
    Tool / Subsystem).

    Sets ``locality`` on any ``Component`` and ``agent_category`` on an
    ``AgenticComponent`` (the processor builds an ``AgenticComponent`` when
    agent tokens are present). Subtype tokens (skill/tool/llm/db/rag) are
    consumed by the class choice. Every other token -- including «component» /
    «subsystem» and an agent token that reached a non-agentic element -- is kept
    in ``component.stereotypes``.

    No-op if ``raw`` is empty / missing.
    """
    leftovers = []
    for token in tokenise(raw):
        if token in AGENT_CATEGORY_TOKENS and isinstance(component, AgenticComponent):
            component.agent_category = AgentCategory(token)
        elif token in LOCALITY_TOKENS:
            component.locality = ComponentLocality(token)
        elif token in COMPONENT_SUBTYPE_TOKENS:
            # Subtype promotion happens at construction time (the processor
            # picks the class); the converter re-emits the token from the class.
            continue
        else:
            leftovers.append(token)
    _keep_free_form(component, leftovers)


def apply_node_stereotype_tokens(node: Node, raw: Optional[str]) -> None:
    """Apply WME stereotype tokens to a Deployment Node.

    «device» / «executionEnvironment» set ``kind``; locality tokens set
    ``locality``. Everything else, including a literal «node» (the default
    ``NodeKind.GENERIC`` has no token), is kept in ``node.stereotypes``.
    """
    leftovers = []
    for token in tokenise(raw):
        if token in NODE_KIND_TOKENS:
            node.kind = NODE_KIND_TOKENS[token]
        elif token in LOCALITY_TOKENS:
            node.locality = DeploymentLocality(token)
        else:
            leftovers.append(token)
    _keep_free_form(node, leftovers)


def apply_artifact_stereotype_tokens(artifact: Artifact, raw: Optional[str]) -> None:
    """Apply WME stereotype tokens to a Deployment Artifact. Only locality is
    typed; everything else lands in ``stereotypes``."""
    leftovers = []
    for token in tokenise(raw):
        if token in LOCALITY_TOKENS:
            artifact.locality = DeploymentLocality(token)
        else:
            leftovers.append(token)
    _keep_free_form(artifact, leftovers)


def _join_parts(typed: list, extras: list) -> str:
    parts = list(typed)
    for extra in extras:
        if extra and extra not in parts:
            parts.append(extra)
    return " ".join(parts)


def format_component_stereotype(component: Component) -> str:
    """Build the WME ``stereotype`` string for a Component / Skill / Tool /
    Subsystem from its typed slots + free-form ``stereotypes`` list.

    Emission order:
    1. Agent category (if not NONE).
    2. Locality (if not LOCAL — the default).
    3. Free-form ``stereotypes`` extras, in the order stored.

    Returns ``""`` when nothing would be emitted, so the converter can skip
    the field entirely.
    """
    typed = []
    if (isinstance(component, AgenticComponent)
            and component.agent_category is not AgentCategory.NONE):
        typed.append(component.agent_category.value)
    if component.locality is not ComponentLocality.LOCAL:
        typed.append(component.locality.value)
    return _join_parts(typed, component.stereotypes)


def format_component_subtype_stereotype(component: Component) -> str:
    """For a Skill / Tool / LLM / Database / RAG / bare-Component round-trip:
    emit the subtype promotion token in front of the rest so the importer
    sees it. LLM/Database/RAG subclass Tool, so they are checked first.
    """
    base = format_component_stereotype(component)
    if isinstance(component, Skill):
        prefix = "skill"
    elif isinstance(component, LLM):
        prefix = "llm"
    elif isinstance(component, Database):
        prefix = "db"
    elif isinstance(component, RAG):
        prefix = "rag"
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
    function emits the kind token plus a sorted (byte-stable)
    ``{permission: scope, ...}`` suffix. ``extras`` is the edge's free-form
    ``stereotypes`` list.
    """
    base = _join_parts([kind.value], extras)
    if permissions:
        scopes = sorted({p.scope for p in permissions if p.scope})
        if scopes:
            base = f"{base} {{permission: {', '.join(scopes)}}}"
    return base


def format_node_stereotype(node: Node) -> str:
    """Build the WME ``stereotype`` string for a Deployment Node.

    Emits the kind token (when not the default GENERIC), the locality (when
    not LOCAL), then the free-form extras.
    """
    typed = []
    if node.kind is not NodeKind.GENERIC:
        typed.append(node.kind.value)
    if node.locality is not DeploymentLocality.LOCAL:
        typed.append(node.locality.value)
    return _join_parts(typed, node.stereotypes)


def format_artifact_stereotype(artifact: Artifact) -> str:
    """Build the WME ``stereotype`` string for a Deployment Artifact.

    Real exports emit no stereotype on bare artifacts; only locality / extras
    show up if set. Returns ``""`` for the all-defaults case so the caller
    can omit the field.
    """
    typed = []
    if artifact.locality is not DeploymentLocality.LOCAL:
        typed.append(artifact.locality.value)
    return _join_parts(typed, artifact.stereotypes)
