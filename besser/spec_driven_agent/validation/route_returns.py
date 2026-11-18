"""The routes a FastAPI module declares and what each handler returns, read from source.

Standard library only: ``.besser_probe.py`` carries a copy of this file, so the
workspace probe and the Phase 3 check describe a response the same way. Routes
declared with ``response_model=None`` (every generated one) publish no response
schema in the app's OpenAPI, so the handler's ``return`` statements are the only
statement of what it sends.
"""
import ast

ROUTE_METHODS = ("get", "post", "put", "patch", "delete")


def route_handlers(tree):
    """``(METHOD, path, function, decorator)`` per ``@<router>.<method>("<path>")``.

    The router must be an ``APIRouter(...)`` or ``FastAPI(...)`` assigned in the
    module; its literal ``prefix`` is part of the path.
    """
    prefixes = {}
    for statement in tree.body:
        if not (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
                and _name(statement.value.func) in ("APIRouter", "FastAPI")):
            continue
        prefix = next((k.value for k in statement.value.keywords if k.arg == "prefix"), ast.Constant(""))
        if isinstance(prefix, ast.Constant) and isinstance(prefix.value, str):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    prefixes[target.id] = prefix.value
    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in function.decorator_list:
            if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr in ROUTE_METHODS
                    and isinstance(decorator.func.value, ast.Name)
                    and decorator.func.value.id in prefixes
                    and decorator.args and isinstance(decorator.args[0], ast.Constant)
                    and isinstance(decorator.args[0].value, str)):
                path = prefixes[decorator.func.value.id] + decorator.args[0].value
                yield decorator.func.attr.upper(), path, function, decorator


def response_shape(function, decorator=None):
    """``("object", keys)`` when every ``return`` sends a dict literal with constant
    string keys, ``("list", None)`` when every one sends a list, else None.

    A declared ``response_model`` or ``response_class`` reshapes the value, so a
    route with one is None too.
    """
    if decorator is not None and any(
            keyword.arg in ("response_model", "response_class")
            and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)
            for keyword in decorator.keywords):
        return None
    nodes = list(_own_nodes(function))
    shapes = [_shape(node.value, nodes) for node in nodes if isinstance(node, ast.Return)]
    if not shapes or None in shapes or len({kind for kind, _ in shapes}) != 1:
        return None
    if shapes[0][0] == "list":
        return "list", None
    return "object", frozenset().union(*(keys for _, keys in shapes))


def _name(node):
    return getattr(node, "id", getattr(node, "attr", ""))


def _own_nodes(function):
    """Every node of ``function`` outside the functions and classes nested in it."""
    stack = list(function.body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))


def _literal_shape(value):
    if isinstance(value, ast.Dict):
        if all(isinstance(key, ast.Constant) and isinstance(key.value, str) for key in value.keys):
            return "object", frozenset(key.value for key in value.keys)
        return None
    if isinstance(value, (ast.List, ast.ListComp)):
        return "list", None
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr == "all" and not value.args):
        return "list", None  # a SQLAlchemy query's rows
    return None


def _shape(value, nodes):
    """The shape of a returned expression; a name counts only when every binding
    of it is a plain ``name = <literal>`` of one kind."""
    if not isinstance(value, ast.Name):
        return _literal_shape(value)
    name, stores, assigned, extra = value.id, 0, [], set()
    for node in nodes:
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store):
            stores += 1
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name) and target.id == name:
                    assigned.append(_literal_shape(node.value) if node.value is not None else None)
                elif isinstance(target, ast.Subscript) and _name(target.value) == name:
                    if not (isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str)):
                        return None
                    extra.add(target.slice.value)
        elif isinstance(node, ast.AugAssign) and _name(node.target) == name:
            return None
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and _name(node.func.value) == name and node.func.attr in ("update", "setdefault", "pop", "clear")):
            return None
    # A binding that is not a plain assignment (a loop, ``with ... as``, unpacking) is a store too.
    if not assigned or stores != len(assigned) or None in assigned or len({kind for kind, _ in assigned}) != 1:
        return None
    if assigned[0][0] == "list":
        return "list", None
    return "object", frozenset().union(extra, *(keys for _, keys in assigned))
