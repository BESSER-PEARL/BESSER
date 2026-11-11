"""The REST routes the backend generator writes for each class, without generating it.

``besser info`` lists them so that a coding agent can call the generated API without reading it.
They follow ``templates/router.py.j2`` and ``templates/router_methods.py.j2``, with every HTTP
method enabled (the generator's default); ``tests/generators/backend/test_endpoints.py`` checks
each one against the generated routers.
"""

from __future__ import annotations

from besser.BUML.metamodel.structural import Class
from besser.generators.backend.api_generator import clean_method_name

# The routes of every class beyond create, read, update and delete.
EXTRA_ROUTES = ("bulk", "count", "paginated", "search")


def _is_class_method(code: str) -> bool:
    """As ``router_methods.py.j2`` decides for a method written in Python: class-level only when its code
    has no ``self`` parameter. A method without code is an instance method."""
    return bool(code) and "(self" not in code


def class_endpoints(cls: Class) -> list[str]:
    """The method and path of each create, read, update, delete and method route of ``cls``."""
    path = f"/{cls.name.lower()}/"
    item = f"{path}{{{cls.name.lower()}_id}}/"
    routes = [f"POST {path}", f"GET {path}", f"GET {item}", f"PUT {item}", f"DELETE {item}"]
    for method in sorted(cls.methods, key=lambda method: method.name):
        target = path if _is_class_method(getattr(method, "code", "") or "") else item
        routes.append(f"POST {target}methods/{clean_method_name(method.name)}/")
    return routes
