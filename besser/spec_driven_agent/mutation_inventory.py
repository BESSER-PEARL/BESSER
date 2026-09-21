"""Bounded, source-only map of relationship-affecting FastAPI entry points.

This is planning/verification coverage, never evidence that a rule is enforced
or absent. No generated module is imported. Unresolved mounts and schemas stay
explicitly unknown instead of silently turning into an exact route contract.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


_SKIP = {"node_modules", "venv", "env", "dist", "build", "__pycache__", "tests", "test", "fixtures", "verification"}
_WRITES = {"post", "put", "patch", "delete"}


def _name(node):
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""


def _string(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _keyword(call, name):
    return next((item.value for item in call.keywords if item.arg == name), None)


def _value(statement):
    return statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None


def _targets(statement):
    if isinstance(statement, ast.Assign):
        return statement.targets
    return [statement.target] if isinstance(statement, ast.AnnAssign) else []


def _type_names(annotation, depth=0):
    result = {node.id for node in ast.walk(annotation) if isinstance(node, ast.Name)}
    if depth < 3:
        for node in ast.walk(annotation):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 2000:
                try:
                    result |= _type_names(ast.parse(node.value, mode="eval"), depth + 1)
                except SyntaxError:
                    pass
    return result


def _source_files(root):
    parsed, unknown = {}, []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in _SKIP and not name.startswith("."))
        for name in list(dirs):
            child = Path(directory, name)
            if child.is_symlink() or getattr(child, "is_junction", lambda: False)() or not child.resolve().is_relative_to(root.resolve()):
                dirs.remove(name)
                unknown.append(f"excluded linked source directory: {child.relative_to(root).as_posix()}")
        for name in sorted(files):
            if not name.endswith(".py") or name.startswith((".", "test_")) or name.endswith("_test.py"):
                continue
            path = Path(directory, name)
            relative = path.relative_to(root).as_posix()
            if len(parsed) >= 300:
                unknown.append(f"file limit: {relative}")
                continue
            try:
                if not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size > 500_000:
                    unknown.append(f"excluded/outsize source: {relative}")
                    continue
                parsed[path] = ast.parse(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError, SyntaxError) as exc:
                unknown.append(f"unreadable source: {relative} ({type(exc).__name__})")
    return parsed, unknown


def _bindings(tree):
    result = {}
    for statement in tree.body:
        value = _value(statement)
        if not isinstance(value, ast.Call) or _name(value.func) not in {"FastAPI", "APIRouter"}:
            continue
        prefix_node = _keyword(value, "prefix")
        prefix = "" if prefix_node is None else _string(prefix_node)
        if any(keyword.arg is None for keyword in value.keywords):
            prefix = None
        for target in _targets(statement):
            if isinstance(target, ast.Name):
                result[target.id] = (_name(value.func), prefix)
    return result


def _mounts(parsed, bindings, unknown):
    mounts = {}
    for path, tree in parsed.items():
        aliases = {}
        for statement in tree.body:
            if isinstance(statement, ast.ImportFrom):
                for alias in statement.names:
                    if alias.name == "*":
                        continue
                    module = (statement.module or "").split(".")[-1] if alias.name == "router" else alias.name
                    aliases[alias.asname or alias.name] = (module, alias.name == "router")
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    aliases[alias.asname or alias.name.split(".")[0]] = (alias.name.split(".")[-1], False)
        for statement in tree.body:
            call = statement.value if isinstance(statement, ast.Expr) else None
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or call.func.attr != "include_router":
                continue
            receiver = _name(call.func.value)
            target = call.args[0] if call.args else _keyword(call, "router")
            base = _name(target.value) if isinstance(target, ast.Attribute) else _name(target)
            module, direct = aliases.get(base, (None, False))
            candidates = [candidate for candidate in parsed if candidate.stem == module and candidate.is_relative_to(path.parent)]
            if len(candidates) != 1 or bindings[path].get(receiver, (None,))[0] != "FastAPI":
                unknown.append(f"unresolved router mount: {path.name}:{statement.lineno}")
                continue
            binding = "router" if direct else _name(target)
            prefix_node = _keyword(call, "prefix")
            prefix = "" if prefix_node is None else _string(prefix_node)
            if any(keyword.arg is None for keyword in call.keywords):
                prefix = None
            mounts.setdefault((candidates[0], binding), []).append(prefix)
    return mounts


def _input_shape(value, previous):
    """Describe supplied keys/types, not secrets or unobserved reference values."""
    from besser.spec_driven_agent.api_probe import _resolve_references

    shapes, keys, complete = {}, set(), True

    def visit(item, path="", depth=0):
        nonlocal complete
        if depth > 5 or len(shapes) >= 50:
            complete = False
            return
        if isinstance(item, str) and "{{" in item:
            try:
                item = _resolve_references(item, previous)
            except (ValueError, TypeError, KeyError, IndexError):
                shapes.setdefault(path or "$", set()).add("reference:UNKNOWN")
                complete = False
                return
        label = ("null" if item is None else "boolean" if isinstance(item, bool)
                 else "number" if isinstance(item, (int, float)) else "string" if isinstance(item, str)
                 else "object" if isinstance(item, dict) else "array" if isinstance(item, list) else "UNKNOWN")
        if isinstance(item, (str, dict, list)):
            label += ":nonempty" if item else ":empty"
        shapes.setdefault(path or "$", set()).add(label)
        if isinstance(item, dict):
            for key, child in item.items():
                keys.add(key)
                visit(child, f"{path}.{key}" if path else key, depth + 1)
        elif isinstance(item, list):
            for child in item[:20]:
                visit(child, path + "[]", depth + 1)
            complete &= len(item) <= 20

    visit(value)
    return shapes, keys, complete


def _observed_inputs(routes, relations, schema_fields, records, current_revision, *, max_chars=4500):
    """Join retained observations to static candidates without inventing coverage."""
    from besser.spec_driven_agent.api_probe import _NO_JSON

    lines = [
        "Observed/requested API inputs (planning only; NOT invariant coverage):",
        "Only a recorded HTTP status counts as an executed request. Current means matching source revision, not proof of a rule. Scenario names and passed assertions do not establish untested branches; stale/unknown-revision results and unexecuted requests are not current verification. Missing inputs below suggest specification-relevant probes, not a requirement to test every route.",
    ]
    observations, unmatched, omitted = {}, 0, False
    for number, record in enumerate(records, 1):
        if number > 10:
            omitted = True
            break
        if not isinstance(record, dict):
            unmatched += 1
            continue
        scenario, report = record.get("scenario") or {}, record.get("report") or {}
        if not isinstance(scenario, dict) or not isinstance(report, dict):
            unmatched += 1
            continue
        requests, responses = scenario.get("requests") or [], report.get("responses") or []
        if not isinstance(requests, list) or not isinstance(responses, list):
            unmatched += 1
            continue
        revision = record.get("revision")
        freshness = ("current" if current_revision is not None and revision == current_revision
                     else "stale" if revision is not None and current_revision is not None else "unknown-revision")
        by_index = {item["index"]: item for item in responses[:20]
                    if isinstance(item, dict) and type(item.get("index")) is int}
        previous = [by_index.get(index, {}).get("json", _NO_JSON)
                    if not by_index.get(index, {}).get("truncated") else _NO_JSON for index in range(min(len(requests), 20))]
        executed = sum(type(item.get("status")) is int and 100 <= item["status"] <= 599 for item in by_index.values())
        assertion_state = ("passed" if report.get("boot") == "ok" and report.get("status") == "passed"
                           and executed == len(requests) and executed and not report.get("assertion_failures")
                           and all(not item.get("failures") and not item.get("error") for item in by_index.values())
                           else "failed/unknown")
        identifier = json.dumps(str(record.get("scenario_id") or "unnamed")[:80], ensure_ascii=True)
        lines.append(f"- S{number} id={identifier}: {freshness}; executed={executed}/{len(requests)}; saved assertions={assertion_state}.")
        backend = report.get("backend") or scenario.get("backend")
        for index, request in enumerate(requests[:20]):
            if not isinstance(request, dict):
                unmatched += 1
                continue
            response = by_index.get(index, {})
            status = response.get("status")
            ran = type(status) is int and 100 <= status <= 599
            method = response.get("method") if ran else request.get("method")
            path = response.get("path") if ran else request.get("path")
            if method not in {"POST", "PUT", "PATCH", "DELETE"}:
                continue
            matches = []
            for route_index, (verb, route, site, _, _, resolved) in enumerate(routes, 1):
                if (not resolved or method != verb or not isinstance(path, str)
                        or not path.startswith("/") or path.startswith("//")):
                    continue
                if backend and not Path(site.rsplit(":", 1)[0]).is_relative_to(Path(backend)):
                    continue
                parts = re.split(r"(\{[^{}]+\})", route)
                pattern = "".join((".*" if part.endswith(":path}") else "[^/]+")
                                  if part.startswith("{") else re.escape(part) for part in parts)
                if re.fullmatch(pattern, urlsplit(path).path):
                    matches.append(route_index)
            if len(matches) != 1:
                unmatched += 1
                continue
            state = freshness if ran else "unexecuted"
            shape, keys, complete = _input_shape(request["json"], previous[:index]) if "json" in request else ({"$": {"no JSON"}}, set(), True)
            bucket = observations.setdefault(matches[0], [])
            bucket.append((state, status if ran else None, shape, keys, complete, f"S{number}[{index}]"))
        omitted |= len(requests) > 20 or len(responses) > 20
    if not observations:
        lines.append("- No executed or requested writes could be matched unambiguously; coverage is UNKNOWN.")

    def related_fields(route_index):
        _, _, _, schemas, affected, _ = routes[route_index - 1]
        roles = {role for owner, role, _, _ in relations if owner in affected}
        accepted = {key for name in schemas for key in (schema_fields(name) or {})}
        return roles & accepted

    def observation_priority(item):
        route_index, entries = item
        current = [entry for entry in entries if entry[0] == "current"]
        missing = related_fields(route_index) - {key for entry in current for key in entry[3]}
        # Concrete unexercised input branches should survive clipping ahead of
        # routine fixture creation; this is a planning hint, never a new gate.
        return (not (missing and current and all(entry[4] for entry in current)), route_index)

    for route_index, entries in sorted(observations.items(), key=observation_priority):
        verb, route, _, schemas, affected, _ = routes[route_index - 1]
        counts, outcomes, shapes = {}, {}, {}
        for state, status, shape, _, _, _ in entries:
            counts[state] = counts.get(state, 0) + 1
            if status is not None:
                outcomes[f"{state}:HTTP{status}"] = outcomes.get(f"{state}:HTTP{status}", 0) + 1
            if state == "current":
                for key, values in shape.items():
                    shapes.setdefault(key, set()).update(values)
        current = [entry for entry in entries if entry[0] == "current"]
        role_fields = related_fields(route_index)
        missing = sorted(role_fields - {key for entry in current for key in entry[3]})
        shape_order = sorted(shapes, key=lambda key: (not any(part.replace("[]", "") in role_fields for part in key.split(".")), key))
        body = ",".join(f"{json.dumps(key, ensure_ascii=True)}={'|'.join(sorted(shapes[key]))}" for key in shape_order)
        if len(body) > 550:
            body = body[:550] + "... [shape truncated]"
        line = f"- W{route_index} {verb} {route}: requests={counts}; outcomes={outcomes}; current supplied JSON={body or 'NONE observed'}"
        if missing and current and all(entry[4] for entry in current):
            line += f"; relation keys NOT supplied in {len(current)}/{len(current)} current bodies: {','.join(missing)}"
        if any(not entry[4] for entry in current):
            line += "; reference/shape UNKNOWN: no absence inference"
        requested_keys = sorted({key for entry in entries if entry[0] != "current" for key in entry[3]})
        if requested_keys:
            line += "; stale/unexecuted/unknown-revision requested keys=" + json.dumps(requested_keys, ensure_ascii=True)[:250]
        line += "; cases=" + ",".join(entry[5] for entry in entries[:8]) + (",..." if len(entries) > 8 else "")
        lines.append(line)
    lines.append(f"Unmatched/ambiguous write observations: {unmatched}. Zero observations is not proof of a defect.")
    if omitted:
        lines.append("Further scenarios/requests OMITTED by the 10-scenario/20-request bounds; unshown coverage is UNKNOWN.")
    result = "\n".join(lines)
    if len(result) > max_chars:
        note = "\n[API INPUT OBSERVATIONS TRUNCATED; unshown methods/fields are UNKNOWN, not verified.]"
        result = result[:max_chars - len(note)].rsplit("\n", 1)[0] + note
    return result


def build_mutation_manifest(output_dir: str, *, max_chars: int = 14_000,
                            scenario_records=None, current_revision: str | None = None) -> str:
    """Describe potential invariant entry points; unknowns are not passing checks.

    Optional retained scenario records add bounded input observations. Reports
    without a matching non-None revision remain historical/unknown, not proof.
    """
    if max_chars < 500:
        return "Mutation coverage OMITTED: character budget too small; no coverage verified."[:max(0, max_chars)]
    root = Path(output_dir).resolve()
    parsed, unknown = _source_files(root)
    classes, orm, relations = {}, {}, []
    for path, tree in parsed.items():
        for statement in tree.body:
            if isinstance(statement, ast.ClassDef):
                classes.setdefault(statement.name, []).append((path, statement))
                if any(isinstance(item, (ast.Assign, ast.AnnAssign))
                       and any(_name(target) == "__tablename__" for target in _targets(item)) for item in statement.body):
                    orm.setdefault(statement.name, []).append((path, statement))
                candidates = [(statement.name, item) for item in statement.body]
            else:
                candidates = [(None, statement)]
            for owner, item in candidates:
                value = _value(item)
                if not isinstance(value, ast.Call) or _name(value.func) != "relationship":
                    continue
                for target in _targets(item):
                    role = _name(target)
                    entity = owner or (_name(target.value) if isinstance(target, ast.Attribute) else None)
                    linked = _string(value.args[0]) if value.args else _string(_keyword(value, "argument"))
                    if not entity or not linked:
                        unknown.append(f"unresolved relationship: {path.name}:{item.lineno}")
                        continue
                    secondary = _keyword(value, "secondary")
                    native = _name(secondary.value) if isinstance(secondary, ast.Attribute) and secondary.attr == "__table__" else None
                    relations.append((entity, role, linked, native))
    if not relations:
        return (f"Mutation coverage UNKNOWN: no relationship map could be built; {len(unknown)} source items unresolved. "
                + "; ".join(unknown[:5]))[:max_chars] if unknown else ""

    def schema_fields(name, seen=frozenset()):
        matches = classes.get(name, [])
        if len(matches) != 1 or name in seen:
            return None
        _, declaration = matches[0]
        result = {}
        for parent in declaration.bases:
            parent_name = _name(parent)
            if parent_name in classes:
                inherited = schema_fields(parent_name, seen | {name})
                if inherited is None:
                    return None
                result.update(inherited)
        for item in declaration.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and not item.target.id.startswith("_"):
                annotation = ast.unparse(item.annotation)
                if not annotation.startswith("ClassVar["):
                    result[item.target.id] = annotation
        return result

    schema_names = {name for name, declarations in classes.items()
                    if name not in orm and any(any(_name(parent) == "BaseModel" for parent in node.bases)
                                              or name.endswith(("Create", "Update", "Patch")) for _, node in declarations)}
    # Include specialized schemas inheriting from a known input schema.
    for _ in range(len(classes)):
        added = {name for name, declarations in classes.items() if name not in orm
                 and any(any(_name(parent) in schema_names for parent in node.bases) for _, node in declarations)}
        if added <= schema_names:
            break
        schema_names |= added

    bindings = {path: _bindings(tree) for path, tree in parsed.items()}
    mounts = _mounts(parsed, bindings, unknown)
    routes, used_schemas = [], set()
    for path, tree in parsed.items():
        for function in tree.body:
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            schemas = {name for argument in function.args.posonlyargs + function.args.args + function.args.kwonlyargs
                       if argument.annotation for name in _type_names(argument.annotation) if name in schema_names}
            owners = {name.removesuffix("Create").removesuffix("Update").removesuffix("Patch") for name in schemas} & orm.keys()
            affected = owners | {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and node.id in orm}
            for _ in range(len(orm)):
                parents = {_name(parent) for name in affected for _, node in orm.get(name, []) for parent in node.bases} & orm.keys()
                if parents <= affected:
                    break
                affected |= parents
            for decorator in function.decorator_list:
                if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                    continue
                verb = decorator.func.attr.lower()
                methods = [verb] if verb in _WRITES else []
                if verb == "api_route":
                    declared = _keyword(decorator, "methods")
                    if isinstance(declared, (ast.List, ast.Tuple, ast.Set)) and all(_string(item) for item in declared.elts):
                        methods = sorted({_string(item).lower() for item in declared.elts} & _WRITES)
                    elif declared is not None:
                        unknown.append(f"dynamic api_route methods: {path.relative_to(root).as_posix()}:{function.lineno}")
                if not methods:
                    continue
                binding = _name(decorator.func.value)
                kind, router_prefix = bindings[path].get(binding, (None, None))
                route_node = decorator.args[0] if decorator.args else _keyword(decorator, "path")
                route = _string(route_node)
                if route is None:
                    unknown.append(f"dynamic write route: {path.relative_to(root).as_posix()}:{function.lineno}")
                    continue
                includes = [""] if kind == "FastAPI" else mounts.get((path, binding), [None])
                for include in includes:
                    resolved = router_prefix is not None and include is not None
                    full = "/" + "/".join(part.strip("/") for part in (include or "", router_prefix or "", route) if part.strip("/"))
                    if route.endswith("/") and not full.endswith("/"):
                        full += "/"
                    site = f"{path.relative_to(root).as_posix()}:{function.lineno}"
                    path_entities = {name for name in orm for segment in full.split("/")
                                     if segment.replace("_", "").lower() == name.replace("_", "").lower()}
                    routes.extend((method.upper(), full, site, schemas, affected | path_entities, resolved) for method in methods)
                    used_schemas |= schemas

    # Preserve exact link role names, including association-class endpoints.
    native_groups = {}
    for owner, role, target, native in relations:
        if native and native in orm:
            native_groups.setdefault((tuple(sorted((owner, target))), native), set()).add(f"{owner}.{role}->{target}")
    groups = dict(native_groups)
    for owner, role, target, native in relations:
        matching = [key for key in native_groups if (owner == key[1] and target in key[0])
                    or (target == key[1] and owner in key[0]) or (native == key[1] and owner in key[0] and target in key[0])]
        for key in matching or [(tuple(sorted((owner, target))), "")]:
            groups.setdefault(key, set()).add(f"{owner}.{role}->{target}")

    routes.sort(key=lambda item: (item[1], item[0], item[2]))
    lines = [
        "### Relationship mutation coverage (static candidates; NOT verification)",
        "For each specification invariant, use the groups below to inspect all relevant public writes, including reverse input fields, bulk, link/unlink and deletion. A candidate may only read data: inspect its handler/shared service before deciding it affects the rule.",
        "Verify specification-based valid, boundary and refused cases across affected paths with test_api; after refusal read both ends/link rows and check unchanged persisted state. Read derived values immediately after a successful mutation, BEFORE any recompute action. On reassignment check source AND destination aggregates. Preserve valid behavior; do not demand duplicate guards when a shared service/database constraint already enforces it.",
        "Roles below are actual ORM spellings, not guessed singular/plural names. Schema fields are accepted input candidates, not proof they are writable/effective. UNKNOWN/unshown paths remain unverified; do not invent new requirements or mark coverage complete from this inventory.",
        "Relationships / potentially affected write IDs:",
    ]
    for (ends, native), roles in sorted(groups.items()):
        entities = set(ends) | ({native} if native else set())
        identifiers = [f"W{index}" for index, route in enumerate(routes, 1) if entities & route[4]]
        label = " <-> ".join(ends) + (f" via {native}" if native else "")
        lines.append(f"- {label}: {', '.join(sorted(roles))}; writes={','.join(identifiers) or 'UNKNOWN (no static owner match)'}")
    if scenario_records is not None:
        records = scenario_records.values() if isinstance(scenario_records, dict) else scenario_records
        lines.append(_observed_inputs(routes, relations, schema_fields, records, current_revision))
    lines.append("Request shapes (inherited fields included):")
    pending = set(used_schemas)
    rendered = set()
    while pending:
        name = sorted(pending)[0]
        pending.remove(name)
        rendered.add(name)
        fields = schema_fields(name)
        if fields is None:
            lines.append(f"- {name}: UNKNOWN ambiguous/unresolved schema")
            continue
        lines.append(f"- {name}: " + ", ".join(f"{field}:{annotation}" for field, annotation in sorted(fields.items())))
        for annotation in fields.values():
            try:
                nested = _type_names(ast.parse(annotation, mode="eval"))
            except SyntaxError:
                nested = set()
            pending |= (nested & schema_names) - rendered
    lines.append("Write catalog (source references; full path only when mount resolved):")
    for index, (method, route, site, schemas, affected, resolved) in enumerate(routes, 1):
        flags = []
        if not resolved:
            flags.append("UNKNOWN mount: decorator/router-local path")
        if not schemas and method in {"POST", "PUT", "PATCH"}:
            flags.append("UNKNOWN input shape / no typed body")
        if not affected:
            flags.append("UNKNOWN affected entities")
        lines.append(f"- W{index} {method} {route} body={','.join(sorted(schemas)) or '-'} @{site}"
                     + (f" [{' ; '.join(flags)}]" if flags else ""))
    lines.append(f"Inventory: {len(routes)} write candidates; {len(groups)} relationship groups; {len(unknown)} unresolved/skipped source items.")
    lines.extend(f"- UNKNOWN {item}" for item in unknown[:15])
    if len(unknown) > 15:
        lines.append(f"- UNKNOWN {len(unknown) - 15} further source items omitted.")
    output = "\n".join(lines)
    if len(output) > max_chars:
        note = f"\n[MUTATION INVENTORY TRUNCATED: {len(routes)} write candidates total; omitted content is UNVERIFIED. Inspect source/OpenAPI before claiming coverage.]"
        output = output[:max_chars - len(note)].rsplit("\n", 1)[0] + note
    return output
