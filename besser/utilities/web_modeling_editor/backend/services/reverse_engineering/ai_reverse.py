"""Propose a domain model for a repo the deterministic reader could not read, using an LLM.

``code_reverse.analyze_repository`` stays first: it is exact and free. When it
finds no data classes (a TypeScript, Java, Go, Ruby, ... repo, or SQL only),
this module sends a bounded digest of the repository to the server's keyless
model and asks for the same JSON system spec ``kg_to_buml`` uses, then converts
it with ``convert_spec_json_to_buml`` and ``process_class_diagram``.

Repository code is only read as text, never executed. Secret-looking files are
left out and the digest is passed through the spec-driven secret redaction.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from besser.BUML.metamodel.structural import DomainModel
from besser.spec_driven_agent.providers.llm_client import (
    FREE_TIER_PROVIDER,
    SPONSORED_PROVIDER,
    create_llm_client,
    free_tier_available,
    sponsored_tier_available,
)
from besser.utilities.kg_to_buml import clean_json_response, convert_spec_json_to_buml, parse_json_safely
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.class_diagram_processor import (
    process_class_diagram,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.utils import sanitize_name
from besser.utilities.web_modeling_editor.backend.services.reverse_engineering.code_reverse import (
    _MAX_FILE_BYTES,
    _SKIPPED_DIRS,
    CodeAnalysis,
)
from besser.utilities.web_modeling_editor.backend.services.spec_driven.secret_redaction import redact_text

logger = logging.getLogger(__name__)

AI_MODEL_SOURCE = "AI reading of the code"

# SQL migrations are often the only schema a non-Python repo has, so they are read here.
_DIGEST_SKIPPED_DIRS = _SKIPPED_DIRS - {"migrations"}
# Bounds the walk so a monorepo cannot stall the import.
_MAX_WALKED_FILES = 5000
# Enough paths for the model to see the repo's layout without flooding the prompt.
_MAX_TREE_ENTRIES = 300
# The README head usually names the domain; the rest is install notes.
_MAX_README_CHARS = 4000
# A few schema files describe the data; more mostly repeats it.
_MAX_DATA_FILES = 8
# One large file must not crowd out the others.
_MAX_CHARS_PER_FILE = 6000
# Keeps the prompt well inside a small open model's context window.
_MAX_DIGEST_CHARS = 40000
# Below this, a file excerpt is too short to be worth adding.
_MIN_EXCERPT_CHARS = 400
# Bytes read to decide whether a file describes data.
_SNIFF_BYTES = 65536
# A proposal larger than this is unreadable on the canvas and likely noise.
_MAX_CLASSES = 30
_MAX_ATTRIBUTES_PER_CLASS = 40
_MAX_ENUM_LITERALS = 50
# A 30-class spec fits in about 6k output tokens.
_MAX_OUTPUT_TOKENS = 8000
# Per HTTP request; the endpoint bounds the whole reading separately.
_REQUEST_TIMEOUT_SECONDS = 75.0
# Whole AI reading, enforced by the import endpoint.
AI_READ_TIMEOUT_SECONDS = 90.0

_README_NAMES = ("readme.md", "readme.rst", "readme.txt", "readme")
_DATA_DIRS = {"types", "models", "model", "entities", "entity", "domain", "schemas", "schema",
              "interfaces", "dto", "dtos", "beans", "modele", "modeles"}
# Languages whose plain classes in a model folder are the data (an MVC "Model/" of POJOs).
_CLASS_SOURCE_SUFFIXES = (".java", ".kt", ".cs", ".scala", ".swift", ".dart", ".ts", ".js", ".py", ".rb", ".php")
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".crt", ".jks", ".keystore")
_SECRET_WORDS = ("key", "secret", "credential", "password", "token")

_TYPE_ALIASES = {
    "str": "str", "string": "str", "text": "str", "varchar": "str", "char": "str", "uuid": "str",
    "email": "str", "url": "str",
    "int": "int", "integer": "int", "long": "int", "bigint": "int", "smallint": "int", "short": "int",
    "float": "float", "double": "float", "decimal": "float", "number": "float", "numeric": "float", "real": "float",
    "bool": "bool", "boolean": "bool",
    "date": "date", "datetime": "datetime", "timestamp": "datetime", "time": "time",
}
_RELATION_TYPES = {"association", "composition", "aggregation", "inheritance", "generalization"}
_MULTIPLICITY_RE = re.compile(r"^(\d+|\*)(\.\.(\d+|\*))?$")
_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class _DataFileRule:
    """A file kind that usually describes a repository's data, with the reason it is read."""
    reason: str
    priority: int  # lower is read first
    matches_path: Callable[[str], bool]
    content: Optional[re.Pattern] = None


def _name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _in_data_dir(path: str) -> bool:
    return bool(set(path.split("/")[:-1]) & _DATA_DIRS)


def _is_ts_types_file(path: str) -> bool:
    name = _name(path)
    return path.endswith((".ts", ".tsx")) and (
        _in_data_dir(path) or name in {"types.ts", "models.ts", "model.ts"}
        or name.endswith((".types.ts", ".model.ts", ".entity.ts"))
    )


_DATA_FILE_RULES = (
    _DataFileRule("Prisma schema", 0, lambda p: p.endswith(".prisma")),
    _DataFileRule("SQL tables", 0, lambda p: p.endswith(".sql"), re.compile(r"create\s+table", re.I)),
    _DataFileRule("Rails schema", 0, lambda p: p.endswith("db/schema.rb"), re.compile(r"create_table")),
    _DataFileRule("GraphQL schema", 1, lambda p: p.endswith((".graphql", ".gql")),
                  re.compile(r"^\s*type\s+\w+", re.M)),
    _DataFileRule("OpenAPI spec", 1, lambda p: _name(p).startswith(("openapi", "swagger"))
                  and p.endswith((".json", ".yaml", ".yml"))),
    _DataFileRule("JPA or Kotlin entity", 1, lambda p: p.endswith((".java", ".kt")),
                  re.compile(r"@Entity\b|\bdata\s+class\s")),
    _DataFileRule("Active Record model", 1, lambda p: p.endswith(".rb") and "/models/" in f"/{p}",
                  re.compile(r"<\s*(ApplicationRecord|ActiveRecord::Base)")),
    _DataFileRule("Eloquent model", 1, lambda p: p.endswith(".php"), re.compile(r"extends\s+Model\b")),
    _DataFileRule("Mongoose schema", 1, lambda p: p.endswith((".js", ".mjs", ".cjs", ".ts")),
                  re.compile(r"new\s+(mongoose\.)?Schema\s*\(|mongoose\.model\s*\(")),
    _DataFileRule("JSON Schema", 2, lambda p: p.endswith(".schema.json")),
    _DataFileRule("C# entity", 2, lambda p: p.endswith(".cs"), re.compile(r"\bDbSet<|\[Key\]|\{\s*get;\s*set;\s*\}")),
    _DataFileRule("TypeScript types", 2, _is_ts_types_file, re.compile(r"\b(interface|type)\s+[A-Z]\w*")),
    _DataFileRule("Go structs", 2, lambda p: p.endswith(".go"), re.compile(r'struct\s*\{[^}]*`[^`]*(json|gorm|db):"')),
    _DataFileRule("Classes in a model folder", 2,
                  lambda p: p.endswith(_CLASS_SOURCE_SUFFIXES) and _in_data_dir(p),
                  re.compile(r"\b(class|struct|record|interface)\s+[A-Z]\w*")),
    _DataFileRule("Python classes", 3, lambda p: _name(p) in {"models.py", "model.py", "entities.py", "schemas.py"},
                  re.compile(r"^class\s+\w+", re.M)),
)

_SYSTEM_PROMPT = (
    "You reverse-engineer the data model of a software repository into a UML class diagram. "
    "You read code as text and report only what it states. Reply with JSON only."
)

_USER_PROMPT = """Below is a digest of a code repository: its file tree, the head of its README and the
files most likely to describe its data. Propose the domain model the code stores and works with.

Rules:
- Classes are the stored or exchanged entities (tables, models, entities, records). Skip framework,
  controller, service, component and configuration classes.
- Attribute types: one of str, int, float, bool, date, datetime, time, or the name of an enumeration
  you declare. A reference to another class is a relationship, not an attribute.
- Relationships: type is "Association", "Composition", "Aggregation" or "Inheritance"; multiplicities
  are "1", "0..1", "*", "0..*" or "1..*"; name is the role the source uses to reach the target.
  For "Inheritance", source is the subclass and target its parent.
- At most {max_classes} classes. Names are identifiers without spaces.
- No methods.

Return ONLY a JSON object with this structure:
{{
  "systemName": "SystemName",
  "classes": [
    {{"className": "ClassName", "attributes": [{{"name": "attr", "type": "str"}}]}}
  ],
  "enumerations": [
    {{"name": "Status", "literals": ["OPEN", "CLOSED"]}}
  ],
  "relationships": [
    {{"type": "Association", "source": "ClassName1", "target": "ClassName2",
     "sourceMultiplicity": "1", "targetMultiplicity": "*", "name": "items"}}
  ]
}}

Repository digest:
{digest}
"""


@dataclass
class RepoDigest:
    """The text sent to the model and the repository files it quotes."""
    text: str
    files: list[str] = field(default_factory=list)


def ai_reading_available(provider: str = FREE_TIER_PROVIDER) -> bool:
    """True when the server offers ``provider``: the keyless model, or the sponsored one for demo tabs."""
    return sponsored_tier_available() if provider == SPONSORED_PROVIDER else free_tier_available()


def propose_model_with_ai(root: str, analysis: CodeAnalysis,
                          provider: str = FREE_TIER_PROVIDER) -> Optional[DomainModel]:
    """A model proposed by the LLM from the repo at ``root``, or None when it could not read one.

    ``provider`` is the server-paid tier to read with: ``free`` (keyless) or
    ``sponsored`` (the demo tier, only for a request that carried the demo secret).

    On success ``analysis`` is marked as an AI reading (``ai``, ``model_source``,
    ``source_files``, ``entities``). Failures are logged and return None.
    """
    digest = build_repo_digest(root)
    if not digest.files:
        logger.warning("AI code reading skipped for %s: no README or data-describing files", root)
        return None
    try:
        reply = _ask_model(_USER_PROMPT.format(max_classes=_MAX_CLASSES, digest=digest.text), provider)
    except Exception:
        logger.warning("AI code reading failed: the model call raised", exc_info=True)
        return None
    spec = _parse_reply(reply)
    if spec is None:
        logger.warning("AI code reading failed: the reply was not a JSON system spec")
        return None
    model = _spec_to_model(_normalise_spec(spec))
    if model is None:
        return None
    analysis.ai = True
    analysis.model_source = AI_MODEL_SOURCE
    analysis.source_files = digest.files
    analysis.entities = sorted(c.name for c in model.get_classes())
    return model


# --------------------------------------------------------------------------
# Digest
# --------------------------------------------------------------------------


def build_repo_digest(root: str) -> RepoDigest:
    """File tree, README head and the files most likely to describe data, within the char caps."""
    paths = _repo_files(root)
    parts = [f"## File tree ({min(len(paths), _MAX_TREE_ENTRIES)} of {len(paths)} files)",
             "\n".join(paths[:_MAX_TREE_ENTRIES])]
    files: list[str] = []

    readme = next((p for p in paths if "/" not in p and p.lower() in _README_NAMES), None)
    if readme:
        parts.append(f"## {readme} (head)\n{_read(root, readme, _MAX_README_CHARS)}")

    used = sum(len(p) for p in parts)
    for path, reason in _data_files(root, paths):
        remaining = _MAX_DIGEST_CHARS - used
        if remaining < _MIN_EXCERPT_CHARS or len(files) >= _MAX_DATA_FILES:
            break
        excerpt = _read(root, path, min(_MAX_CHARS_PER_FILE, remaining - 100))
        if not excerpt.strip():
            continue
        block = f"## File: {path} ({reason})\n{excerpt}"
        parts.append(block)
        files.append(path)
        used += len(block)

    text, _findings = redact_text("\n\n".join(parts))
    # The data files first: the editor names the first few as the source of the proposal.
    return RepoDigest(text=text[:_MAX_DIGEST_CHARS], files=files + ([readme] if readme else []))


def _is_secret_file(name: str) -> bool:
    lowered = name.lower()
    return (lowered.startswith(".env") or lowered.endswith(_SECRET_SUFFIXES)
            or any(word in lowered for word in _SECRET_WORDS))


def _repo_files(root: str) -> list[str]:
    """Repository-relative paths, sorted, without skipped dirs or secret-looking files."""
    found: list[str] = []
    for current, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _DIGEST_SKIPPED_DIRS and not d.startswith("."))
        for name in sorted(names):
            path = os.path.join(current, name)
            if _is_secret_file(name) or os.path.islink(path):
                continue
            found.append(os.path.relpath(path, root).replace(os.sep, "/"))
            if len(found) >= _MAX_WALKED_FILES:
                logger.warning("AI code reading listed only the first %d files of %s", _MAX_WALKED_FILES, root)
                return found
    return found


def _data_files(root: str, paths: list[str]) -> list[tuple[str, str]]:
    """``(path, reason)`` for files that describe data, most telling first."""
    ranked = []
    for path in paths:
        lowered = path.lower()
        rules = [r for r in _DATA_FILE_RULES if r.matches_path(lowered)]
        if not rules:
            continue
        try:
            if os.path.getsize(os.path.join(root, *path.split("/"))) > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        head = _read(root, path, _SNIFF_BYTES) if any(r.content for r in rules) else ""
        if "\x00" in head:
            continue
        rule = next((r for r in rules if r.content is None or r.content.search(head)), None)
        if rule is None:
            continue
        # Migrations restate the schema step by step; a schema file says it once.
        priority = rule.priority + (1 if "migration" in lowered else 0)
        ranked.append((priority, lowered.count("/"), path, rule.reason))
    ranked.sort()
    return [(path, reason) for _priority, _depth, path, reason in ranked]


def _read(root: str, path: str, limit: int) -> str:
    try:
        with open(os.path.join(root, *path.split("/")), "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(limit + 1)
    except OSError:
        return ""
    return text if len(text) <= limit else text[:limit] + "\n[... truncated]"


# --------------------------------------------------------------------------
# Model call and spec validation
# --------------------------------------------------------------------------


def _ask_model(prompt: str, provider: str = FREE_TIER_PROVIDER) -> str:
    """One completion from a server-paid tier; the reply text."""
    client = create_llm_client(provider=provider, max_tokens=_MAX_OUTPUT_TOKENS,
                               timeout=_REQUEST_TIMEOUT_SECONDS)
    response = client.chat(system=_SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}], tools=[])
    texts = []
    for block in response.get("content", []):
        kind = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if kind == "text":
            texts.append(getattr(block, "text", None) or (block.get("text", "") if isinstance(block, dict) else ""))
    return "".join(texts)


def _parse_reply(reply: str) -> Optional[dict]:
    spec = parse_json_safely(clean_json_response(reply or ""))
    if not isinstance(spec, dict):
        # Small models wrap the JSON in prose; take the outermost object.
        start, end = (reply or "").find("{"), (reply or "").rfind("}")
        spec = parse_json_safely(reply[start:end + 1]) if 0 <= start < end else None
    return spec if isinstance(spec, dict) and isinstance(spec.get("classes"), list) else None


def _identifier(value) -> Optional[str]:
    """A name valid for a B-UML element, or None."""
    if not isinstance(value, str):
        return None
    name = _IDENTIFIER_RE.sub("", sanitize_name(value.strip()))
    return name if name and not name[0].isdigit() else None


def _multiplicity(value, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text if _MULTIPLICITY_RE.fullmatch(text) else default


def _normalise_spec(spec: dict) -> dict:
    """The spec with invalid names, unknown types and dangling relationships removed, within the caps."""
    classes: dict[str, list] = {}
    for item in spec.get("classes") or []:
        name = _identifier(item.get("className") or item.get("name")) if isinstance(item, dict) else None
        if name and name not in classes and len(classes) < _MAX_CLASSES:
            classes[name] = item.get("attributes") or []
    if len(spec.get("classes") or []) > len(classes):
        logger.warning("AI code reading dropped %d invalid, duplicate or over-cap classes",
                       len(spec.get("classes") or []) - len(classes))

    enums: dict[str, list[str]] = {}
    for item in spec.get("enumerations") or []:
        name = _identifier(item.get("name")) if isinstance(item, dict) else None
        if not name or name in classes or name in enums:
            continue
        literals = list(dict.fromkeys(filter(None, map(_identifier, item.get("literals") or []))))
        if literals:
            enums[name] = literals[:_MAX_ENUM_LITERALS]

    out_classes = []
    for class_name, raw_attributes in classes.items():
        attributes, seen = [], set()
        for attribute in raw_attributes if isinstance(raw_attributes, list) else []:
            name = _identifier(attribute.get("name")) if isinstance(attribute, dict) else None
            if not name or name in seen or len(attributes) >= _MAX_ATTRIBUTES_PER_CLASS:
                continue
            raw_type = str(attribute.get("type") or "str").strip()
            if raw_type in classes:
                continue  # a reference to a class is a relationship
            kind = raw_type if raw_type in enums else _TYPE_ALIASES.get(raw_type.lower(), "str")
            seen.add(name)
            attributes.append({"name": name, "type": kind, "visibility": "public"})
        out_classes.append({"className": class_name, "attributes": attributes})

    relationships, pairs = [], set()
    for rel in spec.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        source = _identifier(rel.get("source") or rel.get("sourceClass"))
        target = _identifier(rel.get("target") or rel.get("targetClass"))
        kind = str(rel.get("type") or "Association").strip().lower()
        if source not in classes or target not in classes or kind not in _RELATION_TYPES:
            continue
        if kind in {"inheritance", "generalization"}:
            # A self-parent or a two-way pair is a cycle the metamodel rejects.
            if source == target or ("inherit", target, source) in pairs or ("inherit", source, target) in pairs:
                continue
            pairs.add(("inherit", source, target))
            relationships.append({"type": "Inheritance", "source": source, "target": target})
            continue
        relationships.append({
            "type": kind.capitalize(), "source": source, "target": target,
            "sourceMultiplicity": _multiplicity(rel.get("sourceMultiplicity"), "1"),
            "targetMultiplicity": _multiplicity(rel.get("targetMultiplicity"), "*"),
            "name": _identifier(rel.get("name")) or target[0].lower() + target[1:],
        })

    return {
        "systemName": _identifier(spec.get("systemName")) or "ImportedModel",
        "classes": out_classes,
        "enumerations": [{"name": name, "literals": literals} for name, literals in enums.items()],
        "relationships": relationships,
    }


def _spec_to_model(spec: dict) -> Optional[DomainModel]:
    """Convert the normalised spec; relationships the metamodel rejects are dropped, with a warning."""
    if not spec["classes"]:
        logger.warning("AI code reading found no classes in the model's reply")
        return None
    try:
        return process_class_diagram(_with_enumerations(spec))
    except Exception:
        if not spec["relationships"]:
            logger.warning("AI code reading failed: its classes could not be converted", exc_info=True)
            return None
        logger.warning("AI code reading: relationships rejected by the metamodel; keeping the classes",
                       exc_info=True)
    try:
        return process_class_diagram(_with_enumerations({**spec, "relationships": []}))
    except Exception:
        logger.warning("AI code reading failed: its classes could not be converted", exc_info=True)
        return None


def _with_enumerations(spec: dict) -> dict:
    """``convert_spec_json_to_buml``'s diagram plus the spec's enumerations, which that format lacks."""
    diagram = convert_spec_json_to_buml(spec, title=spec["systemName"])
    elements = diagram["model"]["elements"]
    for n, enum in enumerate(spec["enumerations"], start=1):
        literal_ids = []
        for m, literal in enumerate(enum["literals"], start=1):
            literal_id = f"enum-{n}-literal-{m}"
            elements[literal_id] = {"id": literal_id, "type": "ClassAttribute", "owner": f"enum-{n}", "name": literal}
            literal_ids.append(literal_id)
        elements[f"enum-{n}"] = {"id": f"enum-{n}", "type": "Enumeration", "name": enum["name"],
                                 "attributes": literal_ids, "methods": []}
    return diagram
