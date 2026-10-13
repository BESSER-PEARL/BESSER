"""Writes the JPA entity and enumeration classes of a generated Spring Boot project."""

from collections import defaultdict
from typing import Any, List, Tuple

from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DateTimeType,
    DateType,
    DomainModel,
    Enumeration,
    Property,
    StringType,
    TimeDeltaType,
    TimeType,
)
from besser.generators.spring._sub_generator import SpringSubGenerator, build_environment
from besser.generators.spring.java_types import (
    java_type_for,
    java_type_import,
    java_visibility,
    is_many,
    pluralize,
    to_java_accessor_suffix,
    to_java_class_name,
    to_java_field_name,
    to_snake_case,
    validate_java_package,
)

#: An association seen from one of its two sides: (association, source end, target end).
AssociationView = Tuple[BinaryAssociation, Property, Property]

#: Java types Hibernate can auto-generate an identity value for.
GENERATED_ID_TYPES = ("Integer", "Long")


class SpringEntityGenerator(SpringSubGenerator):
    """Generates the ``entity`` package of a Spring Boot project (internal helper)."""

    def __init__(self, model: DomainModel,
                 output_dir: str = "./generated/entity",
                 package_name: str = "com.example.entity"):
        super().__init__(model, output_dir)

        self.package_name: str = validate_java_package(package_name)
        self.association_views: dict[str, List[AssociationView]] = self._build_association_views(model)
        self.relation_owners: dict[BinaryAssociation, str] = self._get_relation_owner_map(model)

    def generate(self):
        for enum in sorted(self.enumerations, key=lambda e: e.name):
            self._generate_enum_file(enum)

        for cls in self.classes:
            self._generate_class_file(cls, self.association_views.get(cls.name, []))

    # ------------------------------------------------------------------
    # Associations
    # ------------------------------------------------------------------

    @staticmethod
    def _build_association_views(model: DomainModel) -> dict[str, List[AssociationView]]:
        """Index every navigable association side by the class that owns the field.

        Each binary association yields up to two *directed* views. Handling the
        two ends separately is what makes self-associations come out right: both
        views land in the same class, but with the two distinct role names
        instead of the same one twice.
        """
        views: defaultdict[str, List[AssociationView]] = defaultdict(list)

        for assoc in sorted(model.associations, key=lambda a: a.name):
            ends = sorted(assoc.ends, key=lambda end: (end.name, end.type.name))
            if len(ends) != 2:
                continue
            first, second = ends
            # A field only exists on a side when the opposite end is navigable.
            if second.is_navigable:
                views[first.type.name].append((assoc, first, second))
            if first.is_navigable:
                views[second.type.name].append((assoc, second, first))

        for class_name in views:
            views[class_name].sort(key=lambda view: (view[0].name, view[2].name, view[1].name))

        return dict(views)

    @staticmethod
    def _get_relation_owner_map(model: DomainModel) -> dict[BinaryAssociation, str]:
        """Pick the owning side of the associations where JPA leaves a choice.

        One-to-one and many-to-many bidirectional associations have no natural
        owner, so one has to be elected: without it both sides would emit a
        ``@JoinTable``/``@JoinColumn`` and neither a ``mappedBy``. The end whose
        role name sorts first wins, which keeps the choice stable across runs and
        works for self-associations too (where both ends share a type name but
        never a role name).
        """
        owners: dict[BinaryAssociation, str] = {}

        for assoc in model.associations:
            ends = sorted(assoc.ends, key=lambda end: (end.name, end.type.name))
            if len(ends) != 2:
                continue
            first, second = ends
            first_many = is_many(first.multiplicity)
            second_many = is_many(second.multiplicity)
            ambiguous = (first_many and second_many) or (not first_many and not second_many)
            if ambiguous and first.is_navigable and second.is_navigable:
                owners[assoc] = first.name

        return owners

    def _prepare_relations(self, views: List[AssociationView]) -> List[dict]:
        relations: List[dict] = []

        for assoc, source, target in views:
            bidirectional: bool = source.is_navigable and target.is_navigable
            source_many: bool = is_many(source.multiplicity)
            target_many: bool = is_many(target.multiplicity)

            relation: str | None = None
            mapped_by: str | None = None
            owning: bool = True
            # The foreign key lives in the table of whichever side holds the
            # single-valued end, so the column is named after the other one.
            join_column: str = f"{to_snake_case(target.type.name)}_id"

            if not source_many and not target_many:
                relation = "OneToOne"
                if bidirectional and self.relation_owners.get(assoc) != source.name:
                    owning = False
                    mapped_by = to_java_field_name(source.name)
            elif not source_many and target_many:
                relation = "OneToMany"
                # A unidirectional @OneToMany puts the foreign key in the target
                # table, pointing back at the source.
                join_column = f"{to_snake_case(source.type.name)}_id"
                if bidirectional:
                    owning = False
                    mapped_by = to_java_field_name(source.name)
            elif source_many and not target_many:
                relation = "ManyToOne"
            else:
                relation = "ManyToMany"
                if bidirectional and self.relation_owners.get(assoc) != source.name:
                    owning = False
                    mapped_by = to_java_field_name(source.name)

            target_type: str = to_java_class_name(target.type.name)
            join_table: str = to_snake_case(assoc.name)
            source_column: str = f"{to_snake_case(source.type.name)}_id"
            target_column: str = f"{to_snake_case(target.type.name)}_id"

            annotations: List[str] = [
                f"@{relation}" + (f'(mappedBy = "{mapped_by}")' if mapped_by else "")
            ]
            if relation == "ManyToOne" or (owning and relation in ("OneToOne", "OneToMany")):
                annotations.append(f'@JoinColumn(name = "{join_column}")')
            elif relation == "ManyToMany" and owning:
                annotations.append(
                    f'@JoinTable(name = "{join_table}",\n'
                    f'        joinColumns = @JoinColumn(name = "{source_column}"),\n'
                    f'        inverseJoinColumns = @JoinColumn(name = "{target_column}"))'
                )

            relations.append({
                "assoc": join_table,
                "source_property": to_java_field_name(source.name),
                "target_property": to_java_field_name(target.name),
                "source_column": source_column,
                "target_column": target_column,
                "annotations": annotations,
                "relation": relation,
                "mapped_by": mapped_by,
                "join_column": join_column,
                "owning": owning,
                "is_list": target_many,
                "type": f"List<{target_type}>" if target_many else target_type,
                "accessor": to_java_accessor_suffix(target.name),
            })

        return relations

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------

    def _generate_class_file(self, cls: Class, views: List[AssociationView]):
        class_name: str = to_java_class_name(cls.name)
        env = build_environment(trim_blocks=True, lstrip_blocks=True)
        entity_template = env.get_template("entity.java.j2")

        relations: List[dict] = self._prepare_relations(views)
        attributes: List[dict] = sorted(
            self._prepare_attributes(cls),
            key=lambda a: (not a["is_id"], not a["is_enum"], not a["is_list"], a["name"]),
        )
        imports: set[str] = self._get_mapping_imports(cls, attributes).union(
            self._get_specific_imports_for_class(cls, relations)
        )
        parents = sorted(cls.parents(), key=lambda parent: parent.name)

        context = {
            "class_name": class_name,
            "package_name": self.package_name,
            "imports": sorted(imports),
            "is_abstract": cls.is_abstract,
            "table_name": pluralize(to_snake_case(cls.name)),
            "parent": to_java_class_name(parents[0].name) if parents else None,
            "attributes": attributes,
            "methods": self._prepare_methods(cls),
            "relations": relations,
        }

        self.write(f"{class_name}.java", entity_template.render(**context))

    def _prepare_attributes(self, cls: Class) -> List[dict]:
        attributes: List[dict] = []

        for attr in cls.attributes:
            is_enum: bool = isinstance(attr.type, Enumeration) or any(
                attr.type.name == enum.name for enum in self.enumerations
            )
            is_list: bool = is_many(attr.multiplicity)
            attr_type: str = java_type_for(attr.type, self.model_type_names)
            if is_list:
                attr_type = f"List<{attr_type}>"

            attributes.append({
                "is_id": attr.is_id,
                # Hibernate can only auto-generate identifiers of an integral
                # type; a natural key such as an ISBN is supplied by the caller.
                "is_generated": attr.is_id and attr_type in GENERATED_ID_TYPES,
                "column_name": to_snake_case(attr.name),
                "is_optional": attr.is_optional,
                "is_enum": is_enum,
                "is_list": is_list,
                # Rendered as a prefix so that package-private access (which has
                # no Java keyword) does not leave a stray space in the declaration.
                "visibility": f"{java_visibility(attr.visibility)} ".lstrip(),
                "type": attr_type,
                "name": to_java_field_name(attr.name),
                "accessor": to_java_accessor_suffix(attr.name),
                "default_value": self._prepare_default_value(attr, is_enum),
            })

        return attributes

    @staticmethod
    def _prepare_default_value(attr: Property, is_enum: bool) -> Any:
        if not attr.default_value:
            return ""

        type_name = attr.type.name
        if type_name == DateType.name:
            return (f"LocalDate.of({attr.default_value['year']}, {attr.default_value['month']}, "
                    f"{attr.default_value['day']})")
        if type_name == DateTimeType.name:
            return (f"LocalDateTime.of({attr.default_value['year']}, {attr.default_value['month']}, "
                    f"{attr.default_value['day']}, {attr.default_value['hour']}, "
                    f"{attr.default_value['minute']}, {attr.default_value['second']})")
        if type_name == TimeType.name:
            return (f"LocalTime.of({attr.default_value['hour']}, {attr.default_value['minute']}, "
                    f"{attr.default_value['second']})")
        if type_name == TimeDeltaType.name:
            return f"Duration.ofSeconds({attr.default_value})"
        if type_name == StringType.name:
            escaped = str(attr.default_value).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if is_enum:
            return f"{to_java_class_name(attr.type.name)}.{to_java_field_name(attr.default_value)}"
        return attr.default_value

    def _prepare_methods(self, cls: Class) -> List[dict]:
        methods: List[dict] = []

        for method in sorted(cls.methods, key=lambda m: m.name):
            parameters = [
                {
                    "name": to_java_field_name(parameter.name),
                    "type": java_type_for(parameter.type, self.model_type_names),
                }
                for parameter in method.parameters
            ]

            methods.append({
                "name": to_java_field_name(method.name),
                "visibility": java_visibility(method.visibility) or "public",
                "return_type": java_type_for(method.type, self.model_type_names) if method.type else "void",
                "code": method.code,
                "parameters": parameters,
            })

        return methods

    # ------------------------------------------------------------------
    # Enumerations
    # ------------------------------------------------------------------

    def _generate_enum_file(self, enum: Enumeration):
        enum_name: str = to_java_class_name(enum.name)
        env = build_environment()
        enum_template = env.get_template("enum.java.j2")

        context = {
            "package_name": self.package_name,
            "name": enum_name,
            # ``literals`` is a set: sorting keeps consecutive runs identical.
            "literals": sorted(
                (to_java_field_name(literal.name) for literal in enum.literals)
            ),
        }

        self.write(f"{enum_name}.java", enum_template.render(**context))

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    @staticmethod
    def _get_mapping_imports(cls: Class, attributes: List[dict]) -> set[str]:
        """The ``jakarta.persistence`` imports the class mapping actually uses."""
        if cls.is_abstract:
            imports = {"jakarta.persistence.MappedSuperclass"}
        else:
            imports = {"jakarta.persistence.Entity", "jakarta.persistence.Table"}

        if any(attr["is_id"] for attr in attributes):
            imports.add("jakarta.persistence.Id")
        if any(attr["is_generated"] for attr in attributes):
            imports.add("jakarta.persistence.GeneratedValue")
            imports.add("jakarta.persistence.GenerationType")
        if any(not attr["is_enum"] for attr in attributes):
            imports.add("jakarta.persistence.Column")

        return imports

    def _get_specific_imports_for_class(self, cls: Class, relations: List[dict]) -> set[str]:
        imports: set[str] = set()

        if any(isinstance(attr.type, Enumeration) or
               any(attr.type.name == enum.name for enum in self.enumerations)
               for attr in cls.attributes):
            imports.add("jakarta.persistence.Enumerated")
            imports.add("jakarta.persistence.EnumType")

        for attr in cls.attributes:
            type_import = java_type_import(java_type_for(attr.type, self.model_type_names))
            if type_import:
                imports.add(type_import)

            if is_many(attr.multiplicity):
                imports.add("java.util.List")
                imports.add("java.util.ArrayList")
                if attr.default_value:
                    imports.add("java.util.Arrays")

        for method in cls.methods:
            if method.type:
                type_import = java_type_import(java_type_for(method.type, self.model_type_names))
                if type_import:
                    imports.add(type_import)
            for parameter in method.parameters:
                type_import = java_type_import(java_type_for(parameter.type, self.model_type_names))
                if type_import:
                    imports.add(type_import)

        for relation in relations:
            if relation["is_list"]:
                imports.add("java.util.List")
                imports.add("java.util.ArrayList")
            if relation["relation"] == "OneToOne":
                imports.add("jakarta.persistence.OneToOne")
                if relation["owning"]:
                    imports.add("jakarta.persistence.JoinColumn")
            if relation["relation"] == "OneToMany":
                imports.add("jakarta.persistence.OneToMany")
                if relation["owning"]:
                    imports.add("jakarta.persistence.JoinColumn")
            if relation["relation"] == "ManyToOne":
                imports.add("jakarta.persistence.ManyToOne")
                imports.add("jakarta.persistence.JoinColumn")
            if relation["relation"] == "ManyToMany":
                imports.add("jakarta.persistence.ManyToMany")
                if relation["owning"]:
                    imports.add("jakarta.persistence.JoinTable")
                    imports.add("jakarta.persistence.JoinColumn")

        return imports
