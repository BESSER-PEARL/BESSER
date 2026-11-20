"""The generated FastAPI backend's API as data.

``build_api_contract`` lists every route ``generate_modular_api`` renders (``router.py.j2``,
``router_methods.py.j2`` and ``main_api.py.j2``) with what it reads and returns: path and query
parameters, the body keys the ``<Class>Create`` schema accepts, and the JSON each route sends
back, including its wrappers. ``BackendGenerator`` writes it as ``api_contract.json`` and
``API.md``; the React generator renders its typed API client from the same dictionary, so the
contract and the client cannot disagree.

Types are named as in the model: ``int``, ``float``, ``str``, ``bool``, ``date``, ``datetime``,
``time``, ``timedelta``, ``any``, an enumeration name (listed under ``enums``) or a class name
(a record, as listed under ``entities``). The routes are the templates' own rules restated;
``tests/generators/backend/test_api_contract.py`` checks them against the routes, request
schemas and columns of generated apps.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from besser.BUML.metamodel.structural import AssociationClass, Class, DomainModel, Enumeration, Property
from besser.generators.backend.api_generator import clean_method_name, get_pk_names, method_route_code
from besser.generators.backend.nn_methods import collect_nn_modules
from besser.generators.structural_utils import (
    get_deferred_fk_associations,
    get_foreign_keys,
    get_pk_py_types,
    is_secret_attribute,
    is_server_owned_attribute,
    is_shadowed_by_association_end,
)
from besser.utilities.utils import sort_by_timestamp

#: Version of the contract's own layout; bumped when a key changes meaning.
API_CONTRACT_VERSION = 1

#: Environment variable the generated frontend reads the backend's address from, and its default.
BASE_URL_ENV = "VITE_API_URL"
DEFAULT_BASE_URL = "http://localhost:8000"

ALL_HTTP_METHODS = ("GET", "POST", "PUT", "DELETE")

# Attribute types the /search/ route filters on (router.py.j2, ns_search).
_SEARCH_TYPES = {"str": "str", "string": "str", "int": "int", "integer": "int", "float": "float",
                 "bool": "bool", "boolean": "bool"}
# Method parameter types read as plain values (router_methods.py.j2); any other is a class.
_PLAIN_PARAM_TYPES = {"str", "int", "float", "bool", "date", "datetime", "time", "any"}

ERROR_SHAPES = [
    {"status": "4xx/5xx raised by a route",
     "body": {"error": "str", "message": "str | object", "detail": "str | object"}},
    {"status": "422 request validation",
     "body": {"detail": "[{loc: (str | int)[], msg: str, type: str}]"}},
    {"status": "500 unhandled exception", "body": "text: Internal Server Error"},
]


def _field(name: str, type_name: str, *, many: bool = False, nullable: bool = False,
           required: Optional[bool] = None, link: Optional[str] = None) -> dict:
    field = {"name": name, "type": type_name, "many": many, "nullable": nullable}
    if required is not None:
        field["required"] = required
    if link:
        field["link"] = link
    return field


def _sorted_ends(association) -> list:
    return sorted(association.ends, key=lambda end: end.name)


class _Relations:
    """A class's associations as ``router.py.j2`` classifies them (its ``ns`` lists)."""

    def __init__(self, cls: Class, fkeys: dict, assoc_by_association: dict):
        self.one = []            # end_name_one: (target, min, end name, association name)
        self.many = []           # end_name_multiple: (near class, target, near end, far end, table, far min, far max, near min)
        self.one_to_many = []    # (target, near end, far end, near min, self association)
        self.links = []          # assoc_links: (near class, target, near end, far end, association class, far min, far max, near min)
        self.reverse_one = []    # (end holding the column, end without it, class)
        for owner in [cls] + sorted(cls.all_parents(), key=lambda c: c.name):
            for association in sorted(owner.associations, key=lambda a: a.name):
                if len(association.ends) == 2:
                    self._add(owner, association, fkeys, assoc_by_association)
        targets = [x[1] for x in self.many]
        # Key of an N:M end in "<name>_ids": the target class, or the end when that is ambiguous.
        self.many_name = {
            x[3]: x[3] if x[1] == cls.name or targets.count(x[1]) > 1 else x[1].lower() for x in self.many
        }

    def _add(self, owner: Class, association, fkeys: dict, assoc_by_association: dict) -> None:
        ends = _sorted_ends(association)
        if ends[0].type.name == ends[1].type.name:
            far_first = ends[0].multiplicity.max > 1 or (
                ends[1].multiplicity.max <= 1 and fkeys[association.name][1] == ends[0].name)
            end1, end2 = (ends[1], ends[0]) if far_first else (ends[0], ends[1])
        else:
            end1 = next(end for end in ends if end.type.name == owner.name)
            end2 = next(end for end in ends if end.type.name != owner.name)
        if not end2.is_navigable:
            return
        name1, name2 = end1.type.name, end2.type.name
        max1, max2 = end1.multiplicity.max, end2.multiplicity.max
        if association.name in assoc_by_association:
            self.links.append((name1, name2, end1.name, end2.name, assoc_by_association[association.name],
                               end2.multiplicity.min, max2, end1.multiplicity.min))
        elif max1 > 1 and max2 > 1:
            table = association.name.lower()
            self.many.append((name1, name2, end1.name, end2.name, table, end2.multiplicity.min, max2,
                              end1.multiplicity.min))
            if name1 == name2 and end1.is_navigable:
                self.many.append((name1, name2, end2.name, end1.name, table, end1.multiplicity.min, max1,
                                  end2.multiplicity.min))
        elif max1 == 1 and max2 > 1:
            self.one_to_many.append((name2, end1.name, end2.name, end1.multiplicity.min, name1 == name2))
        elif max1 > 1 and max2 == 1:
            self.one.append((name2, end2.multiplicity.min, end2.name, association.name))
        elif max1 == 1 and max2 == 1:
            self.one.append((name2, end2.multiplicity.min, end2.name, association.name))
            if name1 == name2 and end1.is_navigable:
                self.reverse_one.append((end2.name, end1.name, name1))

    @property
    def wraps_writes(self) -> bool:
        """POST and PUT answer ``{<class>: record, <end>_ids: [...]}`` instead of the record."""
        return bool(self.many or self.one_to_many or self.links)


class _ContractBuilder:
    def __init__(self, model: DomainModel, http_methods):
        self.model = model
        self.http_methods = set(http_methods or ALL_HTTP_METHODS)
        self.classes = sorted(model.get_classes(), key=lambda c: c.name)
        self.fkeys = get_foreign_keys(model)
        self.deferred = get_deferred_fk_associations(model)
        self.pk_types = get_pk_py_types(model)
        self.assoc_classes = {cls.name: cls for cls in self.classes if isinstance(cls, AssociationClass)}
        self.assoc_by_association = {cls.association.name: cls.name for cls in self.assoc_classes.values()}
        self.pk_names = get_pk_names(model)
        # Classes whose key is one of their attributes; the others get a surrogate `id` column.
        self.declared_ids = {cls.name for cls in self.classes
                             if any(a.is_id or a.name == "id" for a in cls.attributes)}
        self.concrete_parents = {
            cls.name for cls in self.classes
            if cls.is_abstract and not cls.parents() and not cls.association_ends()
        }
        self.referenced_links = {
            end.type.name for association in model.associations for end in association.ends
            if end.type.name in self.assoc_classes and end.type.association is not association
        }
        self.nn_modules = collect_nn_modules(model)

    # -- shared facts -------------------------------------------------------------------------

    def pk_type(self, class_name: str) -> str:
        return self.pk_types.get(class_name, "int")

    # -- records as the backend returns them (SQLAlchemy columns) -------------------------------

    def _own_columns(self, cls: Class) -> List[dict]:
        columns = []
        if cls.name in self.assoc_classes:
            if cls.name not in self.declared_ids and cls.name in self.referenced_links:
                columns.append(_field("id", "int"))
            for end in _sorted_ends(cls.association):
                columns.append(_field(f"{end.name}_id", self.pk_type(end.type.name)))
            for end in sorted(cls.association_ends(), key=lambda e: e.name):
                association = end.owner
                if (len(association.ends) == 2 and association.name != cls.association.name
                        and self.fkeys.get(association.name) == [cls.name, end.name]):
                    columns.append(_field(f"{end.name}_id", self.pk_type(end.type.name),
                                          nullable=end.multiplicity.min == 0))
            columns.extend(self._attribute_columns(cls))
            return columns
        table_parents = [p for p in cls.parents() if p.name not in self.concrete_parents]
        if table_parents:
            columns.append(_field("id", self.pk_type(sorted(table_parents, key=lambda p: p.name)[0].name)))
        elif cls.name not in self.declared_ids:
            columns.append(_field("id", "int"))
        columns.extend(self._attribute_columns(cls))
        for end in sorted(cls.association_ends(), key=lambda e: e.name):
            association = end.owner
            if (len(association.ends) == 2 and association.name not in self.assoc_by_association
                    and self.fkeys.get(association.name) == [cls.name, end.name]):
                columns.append(_field(f"{end.name}_id", self.pk_type(end.type.name),
                                      nullable=association.name in self.deferred or end.multiplicity.min == 0))
        if cls.specializations() and cls.name not in self.concrete_parents:
            columns.append(_field("type_spec", "str"))
        return columns

    def _attribute_columns(self, cls: Class) -> List[dict]:
        return [
            _field(attribute.name, attribute.type.name, nullable=attribute.is_optional and not attribute.is_id)
            for attribute in sorted(cls.attributes, key=lambda a: a.name)
            if not is_shadowed_by_association_end(attribute)
        ]

    def read_fields(self, cls: Class) -> List[dict]:
        """The keys of a record: its own columns and those of the classes it inherits from."""
        fields: Dict[str, dict] = {}
        for owner in [cls] + sorted(cls.all_parents(), key=lambda c: c.name):
            for column in self._own_columns(owner):
                fields.setdefault(column["name"], column)
        return list(fields.values())

    # -- request bodies (the <Class>Create schema) ----------------------------------------------

    def write_fields(self, cls: Class) -> List[dict]:
        """The keys ``<Class>Create`` accepts, the parents' first (pydantic inheritance)."""
        fields: Dict[str, dict] = {}
        for owner in sorted(cls.all_parents(), key=lambda c: c.name) + [cls]:
            for field in self._own_write_fields(owner):
                fields[field["name"]] = field
        return list(fields.values())

    def _own_write_fields(self, cls: Class) -> List[dict]:
        fields = []
        inherited = {attribute.name for attribute in cls.inherited_attributes()} if cls.parents() else set()
        for attribute in sorted(cls.attributes, key=lambda a: a.name):
            if (attribute.name in inherited or is_server_owned_attribute(attribute)
                    or is_shadowed_by_association_end(attribute)):
                continue
            optional = attribute.is_optional and not attribute.is_id
            fields.append(_field(attribute.name, attribute.type.name, nullable=optional,
                                 required=not optional and attribute.default_value is None))
        if cls.name in self.assoc_classes:
            for end in _sorted_ends(cls.association):
                fields.append(_field(end.name, self.pk_type(end.type.name), required=True))
        for association in sorted(cls.associations, key=lambda a: a.name):
            if len(association.ends) == 2:
                fields.extend(self._association_write_fields(cls, association))
        return fields

    def _association_write_fields(self, cls: Class, association) -> List[dict]:
        ends = _sorted_ends(association)
        link_class = self.assoc_by_association.get(association.name)
        if link_class:
            fields = []
            for end in ends:
                if end.type.name == cls.name:
                    continue
                pk = self.pk_type(end.type.name)
                if end.multiplicity.max > 1:
                    fields.append(_field(end.name, pk, many=True, link=link_class,
                                         nullable=end.multiplicity.min == 0,
                                         required=end.multiplicity.min > 0))
                else:
                    fields.append(_field(end.name, pk, link=link_class, nullable=True, required=False))
            return fields
        has_fk = self.fkeys.get(association.name, [""])[0] == cls.name
        if ends[0].type.name == ends[1].type.name:
            pairs = [(ends[(i + 1) % 2], ends[i]) for i in range(2)]
            deferred_matters = False
        else:
            near = next((end for end in ends if end.type.name == cls.name), None)
            far = next((end for end in ends if end.type.name != cls.name), None)
            if near is None or far is None or not far.is_navigable:
                return []
            pairs = [(near, far)]
            deferred_matters = True
        fields = []
        for near, far in pairs:
            pk = self.pk_type(far.type.name)
            to_many, from_many = far.multiplicity.max > 1, near.multiplicity.max > 1
            if from_many and to_many:
                fields.append(_field(far.name, pk, many=True, nullable=far.multiplicity.min == 0,
                                     required=far.multiplicity.min > 0))
            elif from_many and has_fk:
                fields.append(_field(far.name, pk, nullable=far.multiplicity.min == 0,
                                     required=far.multiplicity.min > 0))
            elif to_many:
                fields.append(_field(far.name, pk, many=True, nullable=True, required=False))
            elif not from_many and (has_fk or not deferred_matters):
                required = far.multiplicity.min > 0 and not (deferred_matters and association.name in self.deferred)
                fields.append(_field(far.name, pk, nullable=not required, required=required))
        return fields

    # -- routes ---------------------------------------------------------------------------------

    def entity(self, cls: Class, relations: _Relations) -> dict:
        is_link = cls.name in self.assoc_classes
        if is_link:
            key = [{"name": f"{end.name}_id", "type": self.pk_type(end.type.name)}
                   for end in _sorted_ends(cls.association)]
        else:
            key = [{"name": f"{cls.name.lower()}_id", "type": self.pk_type(cls.name)}]
        return {
            "name": cls.name,
            "path": f"/{cls.name.lower()}/",
            "primary_key": self.pk_names[cls.name],
            "id_type": self.pk_type(cls.name),
            "key": key,
            "abstract": cls.is_abstract,
            "read_only": cls.is_read_only,
            "association_class": is_link,
            "read": self.read_fields(cls),
            "write": self.write_fields(cls),
            "detailed": self._detailed_fields(relations),
            "record_ids": [] if is_link else self._record_ids(relations),
        }

    def _detailed_fields(self, relations: _Relations) -> List[dict]:
        """Keys ``GET /<class>/?detailed=true`` adds to each record (none when it adds nothing)."""
        if not (relations.one or relations.many or relations.one_to_many or relations.links):
            return []
        fields = [_field(x[2], x[0], nullable=True) for x in relations.one]
        fields += [_field(x[1], x[0], nullable=True) for x in relations.one_to_many if x[4]]
        fields += [_field(x[1], x[2], nullable=True) for x in relations.reverse_one]
        fields += [_field(x[3], x[1], many=True) for x in relations.many]
        fields += [_field(x[2], x[0], many=True) for x in relations.one_to_many]
        for x in relations.links:
            fields.append(_field(f"{x[3]}_links", x[4], many=True))
            fields.append(_field(x[3], x[1], many=True))
        unique: Dict[str, dict] = {}
        for field in fields:
            unique[field["name"]] = field
        return list(unique.values())

    def _record_ids(self, relations: _Relations) -> List[dict]:
        """The ``<end>_ids`` lists that GET by id (and wrapped writes) send next to the record."""
        ids = [_field(f"{relations.many_name[x[3]]}_ids", self.pk_type(x[1]), many=True) for x in relations.many]
        ids += [_field(f"{x[3]}_ids", self.pk_type(x[1]), many=True) for x in relations.links]
        ids += [_field(f"{x[2]}_ids", self.pk_type(x[0]), many=True) for x in relations.one_to_many]
        return ids

    def routes(self, cls: Class, relations: _Relations) -> List[dict]:
        name, lower = cls.name, cls.name.lower()
        base = f"/{lower}/"
        is_link = name in self.assoc_classes
        if is_link:
            key_params = [{"name": f"{end.name}_id", "type": self.pk_type(end.type.name)}
                          for end in _sorted_ends(cls.association)]
            record_path = base + "".join("{" + p["name"] + "}/" for p in key_params)
        else:
            key_params = [{"name": f"{lower}_id", "type": self.pk_type(name)}]
            record_path = f"{base}{{{lower}_id}}/"
        wrapped = {"kind": "wrapped_record", "entity": name, "key": lower}
        flat = {"kind": "record", "entity": name}
        write_response = wrapped if relations.wraps_writes and not is_link else flat
        body = {"kind": "create", "entity": name}
        routes = []

        def add(op, method, path, response, *, path_params=(), query=(), request=None, group="crud", extra=None):
            route = {"entity": name, "group": group, "op": op, "method": method, "path": path,
                     "path_params": list(path_params), "query": list(query), "body": request,
                     "response": response}
            route.update(extra or {})
            routes.append(route)

        if "GET" in self.http_methods:
            list_query = [{"name": "detailed", "type": "bool", "default": False}] if self._detailed_fields(relations) else []
            add("list", "GET", base, {"kind": "records", "entity": name,
                                      "detailed": "records_detailed" if list_query else None},
                query=list_query)
            add("count", "GET", f"{base}count/", {"kind": "count"})
            add("page", "GET", f"{base}paginated/", {"kind": "page", "entity": name},
                query=[{"name": "skip", "type": "int", "default": 0},
                       {"name": "limit", "type": "int", "default": 100},
                       {"name": "detailed", "type": "bool", "default": False}])
            search = self._search_params(cls)
            if search:
                add("search", "GET", f"{base}search/", {"kind": "records", "entity": name},
                    query=[{"name": attribute.name, "type": _SEARCH_TYPES[attribute.type.name], "default": None,
                            "match": "contains" if _SEARCH_TYPES[attribute.type.name] == "str" else "equals"}
                           for attribute in search])
            add("get", "GET", record_path, flat if is_link else wrapped, path_params=key_params)
        if "POST" in self.http_methods and not cls.is_abstract:
            add("create", "POST", base, write_response, request=body)
        if "POST" in self.http_methods and not is_link:
            if not cls.is_abstract:
                add("bulkCreate", "POST", f"{base}bulk/", {"kind": "bulk_create", "id_type": self.pk_type(name)},
                    request={"kind": "create_list", "entity": name})
            add("bulkDelete", "DELETE", f"{base}bulk/", {"kind": "bulk_delete", "id_type": self.pk_type(name)},
                request={"kind": "id_list", "type": self.pk_type(name)})
        if cls.is_read_only:
            return routes
        if "PUT" in self.http_methods:
            add("update", "PUT", record_path, write_response, path_params=key_params, request=body)
        if "DELETE" in self.http_methods:
            add("delete", "DELETE", record_path, flat, path_params=key_params)
        own_id = [{"name": f"{lower}_id", "type": self.pk_type(name)}]
        for x in relations.many:
            related = f"related_{x[1].lower()}_id" if lower == x[1].lower() else f"{x[1].lower()}_id"
            item_path = f"{base}{{{lower}_id}}/{x[3]}/{{{related}}}/"
            params = own_id + [{"name": related, "type": self.pk_type(x[1])}]
            extra = {"end": x[3], "target": x[1]}
            add("add", "POST", item_path, {"kind": "message"}, path_params=params, group="relation", extra=extra)
            add("remove", "DELETE", item_path, {"kind": "message"}, path_params=params, group="relation", extra=extra)
            add("list", "GET", f"{base}{{{lower}_id}}/{x[3]}/",
                {"kind": "related", "entity": x[1], "key": x[3], "owner_key": f"{lower}_id"},
                path_params=own_id, group="relation", extra=extra)
        for x in relations.links:
            related = f"related_{x[1].lower()}_id" if lower == x[1].lower() else f"{x[1].lower()}_id"
            item_path = f"{base}{{{lower}_id}}/{x[3]}/{{{related}}}/"
            params = own_id + [{"name": related, "type": self.pk_type(x[1])}]
            extra = {"end": x[3], "target": x[1], "link": x[4]}
            add("add", "POST", item_path, {"kind": "message"}, path_params=params, group="relation",
                request={"kind": "link_attributes", "entity": x[4]}, extra=extra)
            add("remove", "DELETE", item_path, {"kind": "message"}, path_params=params, group="relation", extra=extra)
            add("list", "GET", f"{base}{{{lower}_id}}/{x[3]}/",
                {"kind": "related", "entity": x[1], "key": x[3], "owner_key": f"{lower}_id",
                 "links_key": f"{x[3]}_links", "link_entity": x[4]},
                path_params=own_id, group="relation", extra=extra)
        for x in relations.one_to_many:
            add("list", "GET", f"{base}{{{lower}_id}}/{x[2]}/",
                {"kind": "related", "entity": x[0], "key": x[2], "owner_key": f"{lower}_id"},
                path_params=own_id, group="relation", extra={"end": x[2], "target": x[0]})
        return routes

    def _search_params(self, cls: Class) -> List[Property]:
        params = []
        for owner in sorted(cls.all_parents(), key=lambda c: c.name) + [cls]:
            for attribute in sorted(owner.attributes, key=lambda a: a.name):
                if (attribute.type.name in _SEARCH_TYPES and not is_secret_attribute(attribute)
                        and not attribute.is_derived and not is_shadowed_by_association_end(attribute)):
                    params.append(attribute)
        return params

    def method_routes(self, cls: Class) -> List[dict]:
        if "POST" not in self.http_methods:
            return []
        name, lower = cls.name, cls.name.lower()
        routes = []
        for method in sorted(getattr(cls, "methods", None) or (), key=lambda m: m.name):
            method_name = clean_method_name(method.name)
            code = method_route_code(self.model, cls, method, self.nn_modules)
            class_level = bool(code) and "(self" not in code
            params = [self._method_param(parameter) for parameter in sort_by_timestamp(method.parameters)]
            route = {"entity": name, "group": "method", "op": method_name, "method": "POST",
                     "query": [], "parameters": params,
                     "implemented": bool(code), "class_level": class_level}
            if class_level:
                route.update(path=f"/{lower}/methods/{method_name}/", path_params=[],
                             body={"kind": "method_params"} if params else None,
                             response={"kind": "method_result", "result": "any", "owner_key": "class"})
            else:
                route.update(path=f"/{lower}/{{{lower}_id}}/methods/{method_name}/",
                             path_params=[{"name": f"{lower}_id", "type": self.pk_type(name)}],
                             body={"kind": "method_params"},
                             response={"kind": "method_result", "result": "str", "owner_key": f"{lower}_id"})
            routes.append(route)
        return routes

    def _method_param(self, parameter) -> dict:
        type_name = parameter.type.name
        param = {"name": parameter.name, "type": type_name,
                 "required": parameter.default_value is None}
        if parameter.default_value is not None:
            param["default"] = str(parameter.default_value)
        if type_name not in _PLAIN_PARAM_TYPES:
            if isinstance(parameter.type, Enumeration):
                param["enum"] = True
            else:
                # The route loads the record by this id (get_<class>), so callers send the id.
                param["type"] = self.pk_type(type_name)
                param["record"] = type_name
        return param

    def build(self) -> dict:
        entities, routes = [], []
        for cls in self.classes:
            relations = _Relations(cls, self.fkeys, self.assoc_by_association)
            entities.append(self.entity(cls, relations))
            routes.extend(self.routes(cls, relations))
            routes.extend(self.method_routes(cls))
        routes.extend(self.system_routes())
        return {
            "contract_version": API_CONTRACT_VERSION,
            "model": self.model.name,
            "base_url": {"env": BASE_URL_ENV, "default": DEFAULT_BASE_URL},
            "trailing_slash": True,
            "errors": ERROR_SHAPES,
            "enums": {
                enum.name: [literal.name for literal in sort_by_timestamp(enum.literals)]
                for enum in sorted(self.model.get_enumerations(), key=lambda e: e.name)
            },
            "links": {
                cls.name: {
                    "target": sorted({self.pk_type(end.type.name) for end in cls.association.ends}),
                    "attributes": [
                        _field(attribute.name, attribute.type.name, nullable=attribute.is_optional,
                               required=not attribute.is_optional)
                        for attribute in sort_by_timestamp(cls.attributes)
                        if not is_server_owned_attribute(attribute)
                    ],
                }
                for cls in self.assoc_classes.values()
            },
            "entities": entities,
            "routes": routes,
        }

    def system_routes(self) -> List[dict]:
        return [
            {"entity": None, "group": "system", "op": op, "method": "GET", "path": path, "path_params": [],
             "query": [], "body": None, "response": {"kind": kind}}
            for op, path, kind in (("root", "/", "info"), ("health", "/health", "health"),
                                   ("statistics", "/statistics", "statistics"))
        ]


def build_api_contract(model: DomainModel, http_methods=None) -> dict:
    """The routes ``BackendGenerator`` generates for ``model``, with their parameters and shapes.

    Args:
        model: The domain model the backend is generated from.
        http_methods: The HTTP methods the backend was generated with (default: all four).
    """
    return _ContractBuilder(model, http_methods).build()


# -- API.md ---------------------------------------------------------------------------------------

def _value_type(field: dict, enums: dict) -> str:
    type_name = field["type"]
    if type_name in enums:
        type_name = " | ".join(f'"{literal}"' for literal in enums[type_name])
        if field.get("many"):
            type_name = f"({type_name})"
    if field.get("link"):
        type_name = f"{type_name} or {field['link']} link"
    return type_name + ("[]" if field.get("many") else "")


def _body_keys(fields: List[dict], enums: dict) -> str:
    parts = []
    for field in fields:
        flags = "required" if field.get("required") else "optional"
        if field.get("nullable"):
            flags += ", nullable"
        parts.append(f"`{field['name']}` ({_value_type(field, enums)}, {flags})")
    return ", ".join(parts) or "none"


def _record_keys(fields: List[dict], enums: dict) -> str:
    return ", ".join(
        f"`{field['name']}` {_value_type(field, enums)}{' | null' if field.get('nullable') else ''}"
        for field in fields
    )


def _response_text(response: dict, entities: dict) -> str:
    kind = response["kind"]
    entity = response.get("entity")
    if kind == "records":
        text = f"`{entity}[]` (a JSON array)"
        if response.get("detailed"):
            text += "; with `?detailed=true` each record also carries its relations (see the class)"
        return text
    if kind == "record":
        return f"`{entity}` record"
    if kind == "wrapped_record":
        ids = "".join(f', "{f["name"]}": {f["type"]}[]' for f in entities[entity]["record_ids"])
        return f'`{{"{response["key"]}": {entity}{ids}}}` (the record is under `"{response["key"]}"`)'
    if kind == "count":
        return '`{"count": int}`'
    if kind == "page":
        return (f'`{{"total": int, "skip": int, "limit": int, "data": {entity}[]}}`; with `detailed=true` and '
                f"relations each item of `data` is shaped like GET by id")
    if kind == "related":
        text = (f'`{{"{response["owner_key"]}": id, "{response["key"]}_count": int, '
                f'"{response["key"]}": {entity}[]')
        if response.get("links_key"):
            text += f', "{response["links_key"]}": {response["link_entity"]}[]'
        return text + "}`"
    if kind == "message":
        return '`{"message": str}`'
    if kind == "bulk_create":
        return f'`{{"created_count": int, "created_ids": {response["id_type"]}[], "message": str}}`'
    if kind == "bulk_delete":
        return f'`{{"deleted_count": int, "not_found": {response["id_type"]}[], "message": str}}`'
    if kind == "method_result":
        result = "str | null (the return value converted with str())" if response["result"] == "str" else "any"
        return (f'`{{"{response["owner_key"]}": ..., "method": str, "status": "executed", '
                f'"result": ..., "output": str | null}}`; `result` is {result}')
    return {"info": '`{"name": str, "version": str, "status": str}`',
            "health": '`{"status": str, "timestamp": str, "database": str}`',
            "statistics": '`{"<class>_count": int, ..., "total_entities": int}`'}[kind]


def _body_text(body: Optional[dict], route: dict, contract: dict) -> str:
    if body is None:
        return ""
    kind = body["kind"]
    if kind == "create":
        return f"body `{body['entity']}Create`"
    if kind == "create_list":
        return f"body `{body['entity']}Create[]` (a JSON array)"
    if kind == "id_list":
        return f"body `{body['type']}[]` (a JSON array of ids)"
    if kind == "link_attributes":
        attributes = contract["links"][body["entity"]]["attributes"]
        return f"body (optional) {_body_keys(attributes, contract['enums'])}"
    params = ", ".join(
        f"`{p['name']}` ({p.get('record', '') + ' id' if p.get('record') else _value_type(p, contract['enums'])}"
        f"{'' if p['required'] else ', default ' + p['default']})"
        for p in route["parameters"]
    )
    return f'body `{{"params": {{...}}}}` with {params}'


def render_api_markdown(contract: dict) -> str:
    """``API.md``: the contract as a compact reference for people and coding agents."""
    enums = contract["enums"]
    entities = {entity["name"]: entity for entity in contract["entities"]}
    base = contract["base_url"]
    lines = [
        f"# {contract['model']} API",
        "",
        "Generated from the domain model, like `api_contract.json` (the same facts, machine-readable) and the",
        "frontend's typed client `src/api/generated/client.ts`. Regenerating the backend rewrites this file.",
        "",
        f"- Base URL: `{base['env']}` (default `{base['default']}`). Paths end with `/` except `/health` and"
        " `/statistics`.",
        "- A relation is WRITTEN by its end name holding ids (`project: 3`, `tags: [1, 2]`) and READ back as a",
        "  foreign-key column (`project_id`) on the record that holds it; lists of related records come from the",
        "  relation routes or `?detailed=true`.",
        "- Errors: a route's own error is `{\"error\": str, \"message\": str | object, \"detail\": str | object}`"
        " (show `detail`);",
        "  request validation (422) is `{\"detail\": [{\"loc\": [..., field], \"msg\": str, \"type\": str}]}`;",
        "  an unexpected failure is a plain-text 500 `Internal Server Error`.",
        "",
    ]
    if enums:
        lines.append("## Enumerations")
        lines.append("")
        lines.extend(f"- `{name}`: " + " | ".join(f'"{literal}"' for literal in literals)
                     for name, literals in enums.items())
        lines.append("")
    routes_by_entity: Dict[Optional[str], List[dict]] = {}
    for route in contract["routes"]:
        routes_by_entity.setdefault(route["entity"], []).append(route)
    for entity in contract["entities"]:
        name = entity["name"]
        key = ", ".join(f"`{part['name']}` ({part['type']})" for part in entity["key"])
        lines += [f"## {name}", ""]
        notes = [n for n, on in (("abstract: no create", entity["abstract"]),
                                 ("read-only: no update, delete or relation routes", entity["read_only"]),
                                 ("association class: addressed by both ends", entity["association_class"])) if on]
        lines.append(f"- Primary key `{entity['primary_key']}` ({entity['id_type']}); path parameter {key}."
                     + (f" {'; '.join(notes).capitalize()}." if notes else ""))
        lines.append(f"- Record (what reads return): {_record_keys(entity['read'], enums)}")
        lines.append(f"- `{name}Create` (POST and PUT body; PUT replaces every field): "
                     f"{_body_keys(entity['write'], enums)}")
        if entity["detailed"]:
            lines.append(f"- `?detailed=true` adds: {_record_keys(entity['detailed'], enums)}")
        lines.append("")
        for route in routes_by_entity.get(name, []):
            lines.append(_route_line(route, contract, entities))
        lines.append("")
    lines += ["## System", ""]
    lines += [_route_line(route, contract, entities) for route in routes_by_entity.get(None, [])]
    return "\n".join(lines) + "\n"


def _route_line(route: dict, contract: dict, entities: dict) -> str:
    parts = [f"- `{route['method']} {route['path']}`"]
    query = ", ".join(
        f"`{q['name']}` {q['type']}" + (f" = {_json_literal(q['default'])}" if q.get("default") is not None else "")
        + (" (substring match)" if q.get("match") == "contains" else "")
        for q in route["query"]
    )
    if query:
        parts.append(f"query {query}")
    body = _body_text(route["body"], route, contract)
    if body:
        parts.append(body)
    if route["group"] == "method" and not route["implemented"]:
        parts.append("answers 501 until implemented")
    return "; ".join(parts) + f" -> {_response_text(route['response'], entities)}"


def _json_literal(value) -> str:
    return "true" if value is True else "false" if value is False else str(value)
