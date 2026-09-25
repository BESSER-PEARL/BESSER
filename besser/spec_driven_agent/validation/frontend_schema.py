"""Static checks between generated editable form metadata and FastAPI schemas.

No generated code is imported or executed. Only literal TableBlock options and
dataBinding objects, literal route paths and statically resolvable Pydantic
models are checked. Dynamic/ambiguous code is not proof of a mismatch. Display
columns remain display-only when a nonempty formColumns list is supplied. The
generated renderer otherwise uses those columns as editable form fields too.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from besser.spec_driven_agent.parsed_source import parse_source


_SKIP_DIRS = {"node_modules", "__pycache__", ".git", "venv", ".venv", "dist", "build"}
_FRONTEND_SUFFIXES = {".js", ".jsx", ".ts", ".tsx"}
_MAX_FILE_BYTES = 1_000_000
# Only a template literal may span lines. Without the newline bound on the
# quoted alternatives, two apostrophes in JSX prose would read as one string
# literal and hide every <TableBlock> between them.
_JS_NONCODE = re.compile(
    r'''"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`'''
    r'''|//[^\r\n]*|/\*[\s\S]*?\*/'''
)


@dataclass(frozen=True)
class _Model:
    fields: frozenset[str]
    allows_extra: bool = False


def _name(node: ast.AST | None) -> str:
    return getattr(node, "id", getattr(node, "attr", ""))


def _sources(workspace, changed_path=None, content=None):
    root = Path(workspace).resolve()
    overlay = None
    if changed_path is not None and content is not None:
        candidate = (root / changed_path).resolve()
        if candidate.is_relative_to(root):
            overlay = candidate
    for parent, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS
                         and not d.startswith(".besser")
                         and not (Path(parent) / d).is_symlink())
        for filename in sorted(files):
            path = Path(parent) / filename
            if path.suffix not in _FRONTEND_SUFFIXES | {".py"} or path.is_symlink():
                continue
            try:
                if not path.resolve().is_relative_to(root) or path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = content if overlay == path else path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeError):
                continue
            yield path.relative_to(root).as_posix(), text


def _request_schemas(sources):
    """Resolve literal route -> accepted field names without importing the app."""
    modules, classes = {}, {}
    for path, text in sources:
        if not path.endswith(".py"):
            continue
        try:
            tree = parse_source(text)
        except SyntaxError:
            continue
        imports = {alias.asname or alias.name: alias.name
                   for node in tree.body if isinstance(node, ast.ImportFrom)
                   for alias in node.names if alias.name != "*"}
        modules[path] = (tree, imports)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, []).append((path, node))

    cache = {}

    def resolve(name, active=frozenset()):
        if name in cache:
            return cache[name]
        if name in active or len(classes.get(name, [])) != 1:
            return None
        path, cls = classes[name][0]
        imports = modules[path][1]
        fields, model_found, extra = set(), False, False
        for base in cls.bases:
            base_name = imports.get(_name(base), _name(base))
            if base_name == "BaseModel":
                model_found = True
            elif base_name in {"ABC", "object"}:
                continue
            else:
                inherited = resolve(base_name, active | {name})
                if inherited is None:
                    return None
                model_found = True
                fields.update(inherited.fields)
                extra = extra or inherited.allows_extra
        if not model_found:
            return None
        for stmt in cls.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Before-model validators can accept arbitrary legacy names.
                if any(isinstance(d, ast.Call) and _name(d.func) == "model_validator"
                       and any(k.arg == "mode" and isinstance(k.value, ast.Constant)
                               and k.value.value == "before" for k in d.keywords)
                       for d in stmt.decorator_list):
                    return None
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                field = stmt.target.id
                if field.startswith("_") or field == "model_config":
                    continue
                annotation = stmt.annotation
                if isinstance(annotation, ast.Subscript) and _name(annotation.value) == "ClassVar":
                    continue
                fields.add(field)
                if isinstance(stmt.value, ast.Call) and _name(stmt.value.func) == "Field":
                    for keyword in stmt.value.keywords:
                        if keyword.arg in {"alias", "validation_alias"}:
                            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                                fields.add(keyword.value.value)
                            else:
                                return None  # AliasChoices/generators need runtime schema inspection.
            # Both Pydantic v2 ConfigDict/dict and v1 Config.extra.
            config_nodes = [stmt]
            if isinstance(stmt, ast.ClassDef) and stmt.name == "Config":
                config_nodes = stmt.body
            for config in config_nodes:
                if not isinstance(config, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = config.targets if isinstance(config, ast.Assign) else [config.target]
                names = {_name(target) for target in targets}
                value = config.value
                if names & {"alias_generator"}:
                    return None
                if "extra" in names and isinstance(value, ast.Constant):
                    extra = value.value == "allow"
                if "model_config" not in names:
                    continue
                pairs = ({k.arg: k.value for k in value.keywords} if isinstance(value, ast.Call)
                         else {k.value: v for k, v in zip(value.keys, value.values)
                               if isinstance(k, ast.Constant)} if isinstance(value, ast.Dict) else {})
                if "alias_generator" in pairs:
                    return None
                if "extra" in pairs and isinstance(pairs["extra"], ast.Constant):
                    extra = pairs["extra"].value == "allow"
        result = _Model(frozenset(fields), extra)
        cache[name] = result
        return result

    routes = {}
    for path, (tree, imports) in modules.items():
        routers = {}
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if imports.get(_name(call.func), _name(call.func)) not in {"APIRouter", "FastAPI"}:
                    continue
                prefix_node = next((k.value for k in call.keywords if k.arg == "prefix"), ast.Constant(""))
                if not isinstance(prefix_node, ast.Constant) or not isinstance(prefix_node.value, str):
                    continue
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        routers[target.id] = prefix_node.value
        for fn in tree.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            candidates = []
            for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
                annotation = arg.annotation
                if isinstance(annotation, ast.Subscript) and _name(annotation.value) == "Annotated":
                    annotation = annotation.slice.elts[0] if isinstance(annotation.slice, ast.Tuple) else None
                model_name = annotation.value if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str) else _name(annotation)
                model_name = imports.get(model_name, model_name)
                model = resolve(model_name)
                if model is not None:
                    candidates.append((model_name, model))
            if len(candidates) != 1:
                continue
            for decorator in fn.decorator_list:
                if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                        and decorator.func.attr in {"post", "put", "patch"}
                        and _name(decorator.func.value) in routers and decorator.args
                        and isinstance(decorator.args[0], ast.Constant)
                        and isinstance(decorator.args[0].value, str)):
                    continue
                route = routers[_name(decorator.func.value)] + decorator.args[0].value
                key = (decorator.func.attr, route.rstrip("/") or "/")
                routes.setdefault(key, []).append((path, *candidates[0]))
    return {key: matches[0] for key, matches in routes.items() if len(matches) == 1}


def _table_metadata(text):
    """Read generated JSX attributes using JSON decoding, not executable JS."""
    decoder = json.JSONDecoder()
    excluded = [(match.start(), match.end()) for match in _JS_NONCODE.finditer(text)]
    excluded_index = 0
    for match in re.finditer(r"<TableBlock\b", text):
        while excluded_index < len(excluded) and excluded[excluded_index][1] <= match.start():
            excluded_index += 1
        if excluded_index < len(excluded) and excluded[excluded_index][0] <= match.start():
            continue  # documentation/commented-out components are not controls.
        # Find this opening tag's end, respecting quoted values and JS braces.
        quote, escaped, depth, end = None, False, 0, match.end()
        for end in range(match.end(), len(text)):
            char = text[end]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in "\"'`":
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            elif char == ">" and depth == 0:
                break
        tag = text[match.end():end]
        values = {}
        for attribute in ("options", "dataBinding"):
            start = re.search(r"\b" + attribute + r"\s*=\s*\{\s*", tag)
            if not start:
                continue
            try:
                value, tail = decoder.raw_decode(tag, start.end())
                if isinstance(value, dict) and tag[tail:].lstrip().startswith("}"):
                    values[attribute] = value
            except ValueError:
                continue
        if len(values) == 2:
            yield text.count("\n", 0, match.start()) + 1, values["options"], values["dataBinding"]


def collect_frontend_schema_diagnostics(workspace, changed_path=None, content=None) -> list[dict]:
    """Return proven editable-field/request-schema mismatches for tool feedback.

    ``changed_path``/``content`` optionally overlay the current write. A Python
    schema change checks its frontend consumers too, not just the written file.
    """
    sources = list(_sources(workspace, changed_path, content))
    routes = _request_schemas(sources)
    findings = []
    for path, text in sources:
        if Path(path).suffix not in _FRONTEND_SUFFIXES:
            continue
        for line, options, binding in _table_metadata(text):
            endpoint = binding.get("endpoint")
            if not isinstance(endpoint, str) or not endpoint.startswith("/"):
                continue
            endpoint = endpoint.rstrip("/") or "/"
            actions = options.get("actionButtons")
            if actions is None:
                actions = options.get("action-buttons", False)
            if actions in (None, False, 0, ""):
                continue
            columns = options.get("formColumns")
            if columns is None:
                columns = options.get("form_columns")
            # Match normalizeOptionColumns + its display-column fallback. An
            # empty formColumns array is NOT a way to disable the edit form.
            columns = [column for column in columns if column] if isinstance(columns, list) else []
            if not columns:
                columns = options.get("columns")
            if not isinstance(columns, list):
                continue
            operations = []
            for (method, route), schema in routes.items():
                create = method == "post" and route == endpoint
                update = method in {"put", "patch"} and re.fullmatch(re.escape(endpoint) + r"/\{[^/{}]+\}", route)
                if create or update:
                    operations.append(("create" if create else "update", schema))
            for column in columns:
                if isinstance(column, str):
                    column = {"field": column}
                if not isinstance(column, dict):
                    continue
                # These are generated editable descriptors, not display columns.
                # The current TableBlock renderer does not consume readOnly or
                # editable flags here, so adding an ignored flag cannot fix it.
                field = column.get("path") if column.get("column_type") == "lookup" else column.get("field")
                if not isinstance(field, str) or not field:
                    continue
                refused = [(operation, schema) for operation, schema in operations
                           if not schema[2].allows_extra and field not in schema[2].fields]
                if not refused:
                    continue
                contracts = ", ".join(f"{operation} ({schema[1]} in {schema[0]})" for operation, schema in refused)
                requirement = "required editable" if column.get("required") is True else "editable"
                findings.append({
                    "source": "frontend-schema", "severity": "error",
                    "code": "form-field-not-accepted", "path": path, "line": line,
                    "message": (
                        f"{path}:{line}: {endpoint} form exposes {requirement} field '{field}', "
                        f"but its backend {contracts} does not declare that input. "
                        "Align editable formColumns with the accepted request schema; retain correct "
                        "display columns for server-calculated values. Do not reintroduce writable "
                        "server-owned fields just to satisfy the form."
                    ),
                })
    return findings


def collect_frontend_schema_issues(workspace) -> list[str]:
    """Blocker strings shared by validate_app and final validation."""
    return ["frontend contract: " + finding["message"]
            for finding in collect_frontend_schema_diagnostics(workspace)]
