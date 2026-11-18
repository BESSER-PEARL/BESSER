"""The REST routes the backend generator writes for each class, without generating it.

``besser info`` lists them so that a coding agent can call the generated API without reading it.
They follow the conditions of ``templates/router.py.j2`` and ``templates/router_methods.py.j2``,
with every HTTP method enabled (the generator's default): an association class is addressed by the
ids of its two ends and has no bulk routes, a read-only class has no update or delete routes, and
``/search/`` exists only when the class or a parent has an attribute of a primitive type.
``tests/generators/backend/test_endpoints.py`` checks each one against the generated routers.
The relationship routes under ``/x/{x_id}/<end>/`` are not listed.
"""

from __future__ import annotations

from besser.BUML.metamodel.structural import AssociationClass, Class, DomainModel
from besser.generators.backend.api_generator import clean_method_name
from besser.generators.structural_utils import get_pk_py_types

# The attribute types ``/search/`` filters on, as ``router.py.j2`` names them.
SEARCHABLE_TYPES = ("str", "string", "int", "integer", "float", "bool", "boolean")


def _is_class_method(code: str) -> bool:
    """As ``router_methods.py.j2`` decides for a method written in Python: class-level only when its code
    has no ``self`` parameter. A method without code is an instance method."""
    return bool(code) and "(self" not in code


def _is_searchable(cls: Class) -> bool:
    """Whether ``router.py.j2`` writes a ``/search/`` route: an own or inherited primitive attribute.
    The type is read leniently: ``besser info`` describes invalid models too."""
    return any(getattr(attribute.type, "name", None) in SEARCHABLE_TYPES
               for owner in (cls, *cls.all_parents()) for attribute in owner.attributes)


def _item_path(cls: Class, pk_types: dict[str, str]) -> str:
    """The path of one instance: its id, or for an association class the ids of its two ends, sorted by
    end name, with ``:int`` when the end's class has an integer primary key."""
    path = f"/{cls.name.lower()}/"
    if not isinstance(cls, AssociationClass):
        return f"{path}{{{cls.name.lower()}_id}}/"
    segments = [f"{{{end.name}_id{':int' if pk_types.get(getattr(end.type, 'name', None), 'int') == 'int' else ''}}}"
                for end in sorted(cls.association.ends, key=lambda end: end.name)]
    return f"{path}{'/'.join(segments)}/"


def class_endpoints(cls: Class, model: DomainModel) -> list[str]:
    """The method and path of each create, read, update, delete, bulk, count, paginated, search and
    method route the backend generator writes for ``cls`` of ``model``."""
    path = f"/{cls.name.lower()}/"
    item = _item_path(cls, get_pk_py_types(model))
    routes = [f"POST {path}", f"GET {path}", f"GET {item}"]
    if not cls.is_read_only:
        routes += [f"PUT {item}", f"DELETE {item}"]
    if not isinstance(cls, AssociationClass):
        routes += [f"POST {path}bulk/", f"DELETE {path}bulk/"]
    routes += [f"GET {path}count/", f"GET {path}paginated/"]
    if _is_searchable(cls):
        routes.append(f"GET {path}search/")
    for method in sorted(cls.methods, key=lambda method: method.name):
        target = path if _is_class_method(getattr(method, "code", "") or "") else item
        routes.append(f"POST {target}methods/{clean_method_name(method.name)}/")
    return routes

