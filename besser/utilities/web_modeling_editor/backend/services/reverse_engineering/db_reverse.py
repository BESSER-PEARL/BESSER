"""Reverse-engineer an external relational database into a BUML DomainModel.

Uses SQLAlchemy's metadata reflection (``inspect()``) — never reads user data,
only schema metadata (tables, columns, primary keys, foreign keys).
"""

from __future__ import annotations

import re
from typing import Iterable

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL, make_url
from sqlalchemy.exc import SQLAlchemyError
import sqlalchemy.types as sa_types

from besser.BUML.metamodel.structural.structural import (
    BinaryAssociation,
    BooleanType,
    Class,
    DateType,
    DateTimeType,
    DomainModel,
    FloatType,
    IntegerType,
    Multiplicity,
    Property,
    StringType,
    TimeType,
    UNLIMITED_MAX_MULTIPLICITY,
)

__all__ = [
    "SUPPORTED_DIALECTS",
    "MAX_TABLES",
    "MAX_COLUMNS_PER_TABLE",
    "build_engine",
    "list_schemas_and_tables",
    "db_to_domain_model",
]

SUPPORTED_DIALECTS = {"postgresql", "mysql", "sqlite", "mssql", "oracle"}

# Default driver per dialect — keeps URLs consistent and matches the drivers
# already declared in requirements.txt.
_DEFAULT_DRIVERS = {
    "postgresql": "psycopg2",
    "mysql": "pymysql",
    "sqlite": "pysqlite",  # built-in
    "mssql": "pyodbc",
    "oracle": "oracledb",
}

_CONNECT_TIMEOUT_SECONDS = 5

# Resource caps to bound work for huge databases.
MAX_TABLES = 500
MAX_COLUMNS_PER_TABLE = 200


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------

def build_engine(
    *,
    dialect: str | None = None,
    host: str | None = None,
    port: int | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str | None = None,
    raw_url: str | None = None,
    options: dict[str, str] | None = None,
) -> Engine:
    """Build a SQLAlchemy ``Engine`` from structured params or a raw URL.

    ``raw_url`` takes precedence when non-empty. Raises ``ValueError`` for
    invalid input.
    """
    if raw_url:
        try:
            url_obj = make_url(raw_url.strip())
        except Exception as exc:
            raise ValueError(f"Invalid SQLAlchemy URL: {exc}") from exc
        backend = (url_obj.get_backend_name() or "").lower()
        if backend not in SUPPORTED_DIALECTS:
            raise ValueError(
                f"Unsupported dialect '{backend}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_DIALECTS))}."
            )
    else:
        if not dialect:
            raise ValueError("dialect is required when no raw_url is provided.")
        dialect = dialect.lower()
        if dialect not in SUPPORTED_DIALECTS:
            raise ValueError(
                f"Unsupported dialect '{dialect}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_DIALECTS))}."
            )
        if not database:
            raise ValueError("database is required.")

        drivername = (
            f"{dialect}+{_DEFAULT_DRIVERS[dialect]}"
            if dialect != "sqlite"
            else "sqlite"
        )
        url_obj = URL.create(
            drivername=drivername,
            username=username or None,
            password=password or None,
            host=host if dialect != "sqlite" else None,
            port=port if dialect != "sqlite" else None,
            database=database,
            query=dict(options or {}),
        )

    connect_args = _connect_args_for(url_obj)

    try:
        return create_engine(
            url_obj,
            connect_args=connect_args,
            pool_pre_ping=True,
            future=True,
        )
    except SQLAlchemyError as exc:
        raise ValueError(f"Could not create engine: {exc}") from exc


def _connect_args_for(url_obj: URL) -> dict:
    """Return per-dialect connect_args that enforce a short connect timeout."""
    backend = (url_obj.get_backend_name() or "").lower()
    if backend == "postgresql":
        return {"connect_timeout": _CONNECT_TIMEOUT_SECONDS}
    if backend == "mysql":
        return {"connect_timeout": _CONNECT_TIMEOUT_SECONDS}
    if backend == "mssql":
        return {"timeout": _CONNECT_TIMEOUT_SECONDS}
    if backend == "oracle":
        # oracledb uses tcp_connect_timeout (seconds)
        return {"tcp_connect_timeout": _CONNECT_TIMEOUT_SECONDS}
    # sqlite: file-based, no network timeout to set
    return {}


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def list_schemas_and_tables(engine: Engine) -> dict[str, list[str]]:
    """Return ``{schema: [table_names]}`` for every accessible schema.

    Schemas the connected user cannot read are silently skipped. SQLite and
    other schema-less backends collapse into a single ``"default"`` bucket.
    """
    try:
        inspector = inspect(engine)
    except SQLAlchemyError as exc:
        raise ValueError(f"Could not connect to database: {exc}") from exc

    try:
        schemas = inspector.get_schema_names() or []
    except SQLAlchemyError:
        schemas = []

    if not schemas:
        # SQLite returns an empty list; use a single bucket.
        try:
            return {"default": sorted(inspector.get_table_names() or [])}
        except SQLAlchemyError as exc:
            raise ValueError(f"Could not list tables: {exc}") from exc

    out: dict[str, list[str]] = {}
    for schema in schemas:
        try:
            tables = inspector.get_table_names(schema=schema) or []
        except SQLAlchemyError:
            # Permission denied or similar — skip this schema.
            continue
        if tables:
            out[schema] = sorted(tables)
    return out


# ---------------------------------------------------------------------------
# Domain model construction
# ---------------------------------------------------------------------------

def db_to_domain_model(
    engine: Engine,
    selection: dict[str, Iterable[str]] | None = None,
    *,
    model_name: str = "ClassDiagram",
) -> tuple[DomainModel, list[str]]:
    """Reverse-engineer the selected tables into a BUML ``DomainModel``.

    Args:
        engine: A live SQLAlchemy engine.
        selection: ``{schema: [table_names]}``. Pass ``None`` to import every
            accessible table (subject to ``MAX_TABLES``).
        model_name: Name for the resulting ``DomainModel``.

    Returns:
        A tuple of ``(domain_model, warnings)``. Warnings collect non-fatal
        issues (skipped composite FKs, truncated columns, unresolved targets).
    """
    try:
        inspector = inspect(engine)
    except SQLAlchemyError as exc:
        raise ValueError(f"Could not connect to database: {exc}") from exc

    if selection is None:
        selection = list_schemas_and_tables(engine)

    # Normalize selection: accept list/tuple values, drop empty schemas.
    norm_selection: dict[str, list[str]] = {}
    for schema, tables in selection.items():
        tables_list = [t for t in tables if t]
        if tables_list:
            norm_selection[schema] = tables_list

    total_tables = sum(len(t) for t in norm_selection.values())
    if total_tables == 0:
        raise ValueError("No tables selected for import.")
    if total_tables > MAX_TABLES:
        raise ValueError(
            f"Too many tables selected ({total_tables}). "
            f"Maximum is {MAX_TABLES}."
        )

    warnings: list[str] = []

    # ---- Pass 1: detect class-name collisions across schemas ------------
    table_to_schemas: dict[str, list[str]] = {}
    for schema, tables in norm_selection.items():
        for table in tables:
            table_to_schemas.setdefault(table, []).append(schema)
    needs_schema_prefix = {
        table for table, schemas in table_to_schemas.items() if len(schemas) > 1
    }

    # ---- Pass 2: prefetch metadata for every selected table -------------
    columns_by_table: dict[tuple[str, str], list[dict]] = {}
    pks_by_table: dict[tuple[str, str], set[str]] = {}
    fks_by_table: dict[tuple[str, str], list[dict]] = {}

    for schema, tables in norm_selection.items():
        sa_schema = _to_sa_schema(schema)
        for table in tables:
            key = (schema, table)
            try:
                cols = inspector.get_columns(table, schema=sa_schema) or []
            except SQLAlchemyError as exc:
                warnings.append(f"Skipped {schema}.{table}: {exc}")
                cols = []
            if len(cols) > MAX_COLUMNS_PER_TABLE:
                warnings.append(
                    f"{schema}.{table}: truncated columns from {len(cols)} "
                    f"to {MAX_COLUMNS_PER_TABLE}."
                )
                cols = cols[:MAX_COLUMNS_PER_TABLE]
            columns_by_table[key] = cols

            try:
                pk_info = inspector.get_pk_constraint(table, schema=sa_schema) or {}
            except SQLAlchemyError:
                pk_info = {}
            pks_by_table[key] = set(pk_info.get("constrained_columns") or [])

            try:
                fks = inspector.get_foreign_keys(table, schema=sa_schema) or []
            except SQLAlchemyError:
                fks = []
            fks_by_table[key] = fks

    selected_keys = set(columns_by_table.keys())

    # ---- Pass 3: identify pure bridge (junction) tables -----------------
    # bridges maps bridge-key -> (target_a_key, target_b_key, fk_a, fk_b)
    bridges: dict[tuple[str, str], tuple[tuple[str, str], tuple[str, str], dict, dict]] = {}
    for key in selected_keys:
        bridge_info = _detect_bridge_table(
            key,
            columns_by_table[key],
            pks_by_table[key],
            fks_by_table[key],
            selected_keys,
        )
        if bridge_info is not None:
            bridges[key] = bridge_info

    # ---- Pass 4: build classes + attributes for non-bridge tables -------
    classes: dict[tuple[str, str], Class] = {}
    for key in selected_keys:
        if key in bridges:
            continue
        schema, table = key
        class_name = _safe_class_name(
            table,
            schema if table in needs_schema_prefix else None,
        )
        cls = Class(class_name)
        classes[key] = cls

        cols = columns_by_table[key]
        pk_cols = pks_by_table[key]
        fks = fks_by_table[key]
        single_fk_columns = {
            fk["constrained_columns"][0]
            for fk in fks
            if len(fk.get("constrained_columns") or []) == 1
        }

        for col in cols:
            cname = col["name"]
            if cname in single_fk_columns:
                continue
            buml_type = _sa_type_to_buml(col.get("type"))
            is_id = cname in pk_cols
            nullable = bool(col.get("nullable", True))
            prop = Property(
                name=cname,
                type=buml_type,
                owner=cls,
                is_id=is_id,
                is_optional=(nullable and not is_id),
            )
            cls.add_attribute(prop)

    # ---- Pass 5: build associations from foreign keys (non-bridges) -----
    associations: set[BinaryAssociation] = set()
    for key, fks in fks_by_table.items():
        if key in bridges:
            continue
        if key not in classes:
            continue
        schema, table = key
        owner_cls = classes[key]
        col_meta_by_name = {c["name"]: c for c in columns_by_table.get(key, [])}

        for fk in fks:
            constrained = fk.get("constrained_columns") or []
            referred_table = fk.get("referred_table")
            referred_schema = fk.get("referred_schema") or schema
            if not referred_table:
                continue
            if len(constrained) != 1:
                warnings.append(
                    f"{schema}.{table}: skipped composite foreign key "
                    f"on columns {constrained}."
                )
                continue

            target_key = _resolve_target(
                classes, referred_schema, referred_table, schema
            )
            if target_key is None:
                warnings.append(
                    f"{schema}.{table}.{constrained[0]} → "
                    f"{referred_schema}.{referred_table}: target not in selection."
                )
                continue
            target_cls = classes[target_key]

            fk_col_meta = col_meta_by_name.get(constrained[0], {})
            nullable = bool(fk_col_meta.get("nullable", True))
            src_min = 0 if nullable else 1

            fk_role = _camel_case(target_cls.name)
            ref_role = _pluralize(owner_cls.name[0].lower() + owner_cls.name[1:])
            end_fk = Property(
                name=fk_role,
                type=target_cls,
                owner=None,
                multiplicity=Multiplicity(src_min, 1),
            )
            end_target = Property(
                name=ref_role,
                type=owner_cls,
                owner=None,
                multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY),
            )
            assoc_name = (
                f"{owner_cls.name}_{constrained[0]}_to_{target_cls.name}"
            )
            associations.add(
                BinaryAssociation(assoc_name, ends={end_fk, end_target})
            )

    # ---- Pass 6: collapse pure bridge tables into M:N associations ------
    for bridge_key, (target_a_key, target_b_key, _fk_a, _fk_b) in bridges.items():
        cls_a = classes.get(target_a_key)
        cls_b = classes.get(target_b_key)
        if cls_a is None or cls_b is None:
            # Target was itself a bridge (degenerate) — skip; should not happen
            # because bridges' targets are required to be in selected_keys, but
            # be defensive in case of nested junctions.
            warnings.append(
                f"{bridge_key[0]}.{bridge_key[1]}: junction target not modeled "
                "as a class; bridge skipped."
            )
            continue

        bridge_table_name = bridge_key[1]
        role_a = _pluralize(_camel_case(cls_a.name))
        role_b = _pluralize(_camel_case(cls_b.name))
        if cls_a is cls_b:
            # Self-many-to-many: differentiate role names so they don't collide.
            role_b = role_b + "_2"
        end_a = Property(
            name=role_a,
            type=cls_a,
            owner=None,
            multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY),
        )
        end_b = Property(
            name=role_b,
            type=cls_b,
            owner=None,
            multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY),
        )
        assoc_name = _safe_class_name(bridge_table_name) or (
            f"{cls_a.name}_{cls_b.name}"
        )
        associations.add(BinaryAssociation(assoc_name, ends={end_a, end_b}))

    domain_model = DomainModel(
        model_name,
        types=set(classes.values()),
        associations=associations,
    )
    return domain_model, warnings


def _detect_bridge_table(
    table_key: tuple[str, str],
    cols: list[dict],
    pk_cols: set[str],
    fks: list[dict],
    selected_keys: set[tuple[str, str]],
) -> tuple[tuple[str, str], tuple[str, str], dict, dict] | None:
    """Return (target_a_key, target_b_key, fk_a, fk_b) if ``table`` is a pure
    junction table, else ``None``.

    Pure bridge criteria (deliberately strict — anything richer is an
    association class and must remain a regular Class):
      - Exactly 2 single-column foreign keys.
      - The two FK columns are distinct.
      - The composite primary key equals the set of those 2 FK columns.
      - The table has no other columns (no payload, no audit fields).
      - Both FK targets are present in the import selection.
    """
    if len(fks) != 2:
        return None
    constrained_lists = [fk.get("constrained_columns") or [] for fk in fks]
    if any(len(cl) != 1 for cl in constrained_lists):
        return None
    fk_col_names = {cl[0] for cl in constrained_lists}
    if len(fk_col_names) != 2:
        return None
    if pk_cols != fk_col_names:
        return None
    col_names = {c["name"] for c in cols}
    if col_names != fk_col_names:
        return None

    schema = table_key[0]
    targets: list[tuple[str, str]] = []
    for fk in fks:
        referred_table = fk.get("referred_table")
        referred_schema = fk.get("referred_schema") or schema
        if not referred_table:
            return None
        candidate = None
        for s in (referred_schema, schema, "default", None):
            if (s, referred_table) in selected_keys:
                candidate = (s, referred_table)
                break
        if candidate is None:
            return None
        targets.append(candidate)
    return targets[0], targets[1], fks[0], fks[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_sa_schema(schema: str | None) -> str | None:
    """Translate the API's ``"default"`` bucket back to ``None`` for SA."""
    if schema in (None, "", "default"):
        return None
    return schema


def _resolve_target(
    classes: dict[tuple[str, str], Class],
    referred_schema: str | None,
    referred_table: str,
    fallback_schema: str,
) -> tuple[str, str] | None:
    """Find the (schema, table) key in ``classes`` for a FK target."""
    for candidate in (referred_schema, fallback_schema, "default", None):
        if (candidate, referred_table) in classes:
            return (candidate, referred_table)
    return None


_SA_TYPE_TO_BUML = (
    # Order matters: check more specific types first.
    (sa_types.Boolean, BooleanType),
    (sa_types.Float, FloatType),
    (sa_types.Numeric, FloatType),
    (sa_types.Integer, IntegerType),
    (sa_types.DateTime, DateTimeType),
    (sa_types.Date, DateType),
    (sa_types.Time, TimeType),
    (sa_types.String, StringType),
    (sa_types.Text, StringType),
)


def _sa_type_to_buml(sa_type):
    """Map a SQLAlchemy type instance/class to a BUML primitive."""
    if sa_type is None:
        return StringType
    for sa_cls, buml_t in _SA_TYPE_TO_BUML:
        try:
            if isinstance(sa_type, sa_cls):
                return buml_t
        except TypeError:
            # sa_type is itself a class, not an instance.
            try:
                if issubclass(sa_type, sa_cls):
                    return buml_t
            except TypeError:
                pass
    return StringType


_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_]+")


def _safe_class_name(table: str, schema: str | None = None) -> str:
    """Produce a Python-identifier-safe class name from ``[schema.]table``."""
    base = f"{schema}_{table}" if schema and schema != "default" else table
    cleaned = _INVALID_NAME_CHARS.sub("_", base).strip("_")
    if not cleaned:
        cleaned = "Table"
    if cleaned[0].isdigit():
        cleaned = f"T_{cleaned}"
    return cleaned


def _camel_case(name: str) -> str:
    parts = re.split(r"[_\-\s]+", name)
    if not parts:
        return name
    return parts[0][:1].lower() + parts[0][1:] + "".join(
        p[:1].upper() + p[1:] for p in parts[1:] if p
    )


def _pluralize(name: str) -> str:
    if not name:
        return name
    if name.endswith("s"):
        return name
    return name + "s"
