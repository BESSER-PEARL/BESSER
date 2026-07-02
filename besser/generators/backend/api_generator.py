"""Renders the modular FastAPI layer (``main_api.py`` + ``database.py`` +
``bal_stdlib.py`` + one ``routers/<class>.py`` per resource) for the
:class:`~besser.generators.backend.backend_generator.BackendGenerator`.

This used to be a single ~1,600-line ``main_api.py`` produced by
``RESTAPIGenerator`` (backend=True mode). That monolith is now split into a
slim ``main_api.py`` (app setup + router includes, still importable as
``main_api:app`` for uvicorn/Docker/deployment tooling that expects that
filename) plus one router module per B-UML class. The split is purely
structural: every endpoint, validation rule and code path is unchanged from
the previous monolith, just relocated.
"""

import os
import re
from typing import Dict, List, Tuple

from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.structural import DomainModel
from besser.BUML.notations.action_language.ActionLanguageASTBuilder import parse_bal
from besser.generators.action_language.RESTGenerator import bal_to_rest
from besser.generators.structural_utils import get_foreign_keys

_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Endpoint-name suffixes/prefixes that identify a function as belonging to a
# given class's router. Used by ``cross_router_calls`` to detect when a
# BAL/CODE method body (rendered to Python source before we ever see it)
# calls another class's endpoint function directly, so we can emit a local
# import for it instead of relying on a module-level import (which could
# deadlock on circular router imports when two classes reference each
# other).
_SIMPLE_FUNC_TEMPLATES = (
    "create_{c}",
    "bulk_create_{c}",
    "update_{c}",
    "delete_{c}",
    "bulk_delete_{c}",
    "get_{c}",
    "get_all_{c}",
    "get_count_{c}",
    "get_paginated_{c}",
    "search_{c}",
    "{c}_",  # class-level method endpoints are named "<class>_<method>"
)


def clean_method_name(name):
    """Extract just the method name without parameters."""
    if '(' in str(name):
        return str(name).split('(')[0].strip()
    return str(name).strip()


def cross_router_calls(method_code: str, current_class_name: str, class_names: List[str]) -> List[Tuple[str, str]]:
    """Find function calls in a rendered method body that target another
    class's router module.

    BAL-derived method bodies (see ``besser.generators.action_language.RESTGenerator``)
    can call another class's CRUD/relationship/method endpoint functions
    directly, e.g. ``await update_manager(...)`` or ``await create_book(...)``.
    When routes lived in a single ``main_api.py`` file those names were all in
    the same module namespace. Now that each class has its own router module,
    a call that targets a *different* class needs an explicit import. We
    detect it here (instead of importing every router into every other
    router at module scope) to avoid circular imports between routers that
    reference each other.

    Returns a sorted, de-duplicated list of ``(target_module, function_name)``
    tuples. Calls to functions on ``current_class_name`` itself are excluded
    since those are already defined in the same router module.
    """
    if not method_code:
        return []

    current_lower = current_class_name.lower()
    candidate_names = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", method_code))

    found = set()
    for target in class_names:
        target_lower = target.lower()
        if target_lower == current_lower:
            continue
        simple_names = {tmpl.format(c=target_lower) for tmpl in _SIMPLE_FUNC_TEMPLATES}
        for name in candidate_names:
            if (
                name in simple_names
                or name.startswith(f"execute_{target_lower}_")
                or name.endswith(f"_of_{target_lower}")
                or (name.startswith("add_") and name.endswith(f"_to_{target_lower}"))
                or (name.startswith("remove_") and name.endswith(f"_from_{target_lower}"))
            ):
                found.add((target_lower, name))

    return sorted(found)


def _make_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(_TEMPLATES_DIR),
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=['jinja2.ext.do'],
    )
    env.filters['clean_method_name'] = clean_method_name
    env.globals.update(parse_bal=parse_bal, bal_to_rest=bal_to_rest)
    return env


def generate_modular_api(
    model: DomainModel,
    http_methods: List[str],
    nested_creations: bool,
    port: int,
    output_dir: str,
) -> None:
    """Render ``main_api.py``, ``database.py``, ``bal_stdlib.py`` and one
    ``routers/<class>.py`` per class into ``output_dir``.

    ``main_api.py`` keeps its historical filename (and still exposes the
    module-level ``app`` object) so that existing tooling which shells out to
    ``uvicorn main_api:app`` (Docker images, the GitHub deployment service)
    keeps working unmodified; only its *contents* shrink to app setup plus
    ``include_router`` calls.
    """
    classes = model.classes_sorted_by_inheritance()
    class_names = [cls.name for cls in classes]
    fkeys: Dict[str, List[str]] = get_foreign_keys(model)

    env = _make_env()
    env.globals['cross_router_calls'] = (
        lambda method_code, current_class_name: cross_router_calls(method_code, current_class_name, class_names)
    )

    routers_dir = os.path.join(output_dir, "routers")
    os.makedirs(routers_dir, exist_ok=True)
    with open(os.path.join(routers_dir, "__init__.py"), mode="w", encoding="utf-8") as f:
        f.write("")

    # database.py: engine/session setup shared by main_api.py and every router.
    database_template = env.get_template("database.py.j2")
    with open(os.path.join(output_dir, "database.py"), mode="w", encoding="utf-8") as f:
        f.write(database_template.render(name=model.name))

    # bal_stdlib.py: BESSER Action Language standard-library helpers, shared
    # by any router whose class methods use them.
    bal_stdlib_template = env.get_template("bal_stdlib.py.j2")
    with open(os.path.join(output_dir, "bal_stdlib.py"), mode="w", encoding="utf-8") as f:
        f.write(bal_stdlib_template.render())

    # One router module per class.
    router_template = env.get_template("router.py.j2")
    for cls in classes:
        router_code = router_template.render(
            **{
                "class": cls,
                "classes": classes,
                "http_methods": http_methods,
                "nested_creations": nested_creations,
                "fkeys": fkeys,
                "model": model,
            }
        )
        router_path = os.path.join(routers_dir, f"{cls.name.lower()}.py")
        with open(router_path, mode="w", encoding="utf-8") as f:
            f.write(router_code)

    # main_api.py: slim app setup + router includes (keeps its historical
    # filename so `uvicorn main_api:app` / Docker / deployment tooling that
    # expects it keeps working).
    main_api_template = env.get_template("main_api.py.j2")
    main_api_code = main_api_template.render(
        name=model.name,
        model=model,
        classes=classes,
        http_methods=http_methods,
        nested_creations=nested_creations,
        port=port,
        fkeys=fkeys,
    )
    with open(os.path.join(output_dir, "main_api.py"), mode="w", encoding="utf-8") as f:
        f.write(main_api_code)

    print("Code generated in the location: " + os.path.join(output_dir, "main_api.py"))
