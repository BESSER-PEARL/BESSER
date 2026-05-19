import os
import re
from collections import deque
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

from besser.BUML.metamodel.structural import (
    DomainModel, Class, Enumeration, AssociationClass,
)
from besser.generators import GeneratorInterface
from besser.generators.structural_utils import get_foreign_keys
from besser.utilities.utils import sort_by_timestamp


class SupabaseGenerator(GeneratorInterface):
    """
    Generates Supabase-flavored Postgres DDL for a BUML DomainModel.

    Conventions:
      - Every attribute with ``is_id=True`` becomes ``UUID PRIMARY KEY DEFAULT
        gen_random_uuid()`` -- except on the user-root class, where the id
        instead ``REFERENCES auth.users(id) ON DELETE CASCADE`` (no default).
      - FK columns targeting a class with an ``is_id`` PK are emitted as UUID.
      - The user-root class (default: the one named "User") is materialized as
        ``public.<name>`` mirroring ``auth.users`` via a ``handle_new_user``
        trigger.
      - Any class reachable from the user-root via associations gets a
        denormalized ``user_id UUID`` column referencing ``public.<user_root>``,
        and RLS policies keyed on ``auth.uid() = user_id``. Association FKs
        pointing AT the user-root are suppressed to avoid duplicate columns.
      - All identifiers are double-quoted in DDL and escaped (``"`` doubled to
        ``""``) before reaching the template, so a malicious model name cannot
        break out of the quoted identifier and inject DDL.
      - Enum literal values are escaped (``'`` doubled to ``''``) for the
        single-quoted ``CREATE TYPE ... AS ENUM`` string.
    """

    POSTGRES_TYPES = {
        "str": "TEXT",
        "int": "INTEGER",
        "float": "DOUBLE PRECISION",
        "bool": "BOOLEAN",
        "date": "DATE",
        "time": "TIME",
        "datetime": "TIMESTAMPTZ",
        "timedelta": "INTERVAL",
        "any": "JSONB",
    }

    def __init__(self, model: DomainModel, output_dir: str = None, user_root: str = "User"):
        """
        Args:
            model: Source BUML domain model.
            output_dir: Where to write the generated ``.sql`` file. Defaults
                to ``./output``.
            user_root: Name of the class that mirrors ``auth.users``. Pass
                ``None`` to skip auth integration entirely.
        """
        super().__init__(model, output_dir)
        self.user_root_name = user_root

    # ---------------- Sanitization helpers ----------------
    #
    # Every name flowing into the SQL template passes through one of these
    # helpers in the generator (not the template), so the template can render
    # values verbatim without re-escaping. ``_safe_ident`` is the boundary for
    # any value that appears inside a double-quoted identifier ("..."), and
    # ``_safe_string`` is the boundary for any value that appears inside a
    # single-quoted string literal ('...'). NUL / CR / LF are rejected outright
    # -- they cannot appear in valid identifiers or single-line literals, and
    # would almost certainly be an injection or a corrupted model.

    @staticmethod
    def _reject_control_chars(value: str, kind: str) -> str:
        if any(c in value for c in ("\0", "\n", "\r")):
            raise ValueError(
                f"Invalid character (NUL/CR/LF) in {kind}: {value!r}"
            )
        return value

    @classmethod
    def _safe_ident(cls, name: str) -> str:
        """Escape ``name`` for inclusion inside a double-quoted Postgres identifier."""
        return cls._reject_control_chars(name, "identifier").replace('"', '""')

    @classmethod
    def _safe_string(cls, value: str) -> str:
        """Escape ``value`` for inclusion inside a single-quoted Postgres string literal."""
        return cls._reject_control_chars(value, "string literal").replace("'", "''")

    # ---------------- Model introspection ----------------

    def _find_user_root(self):
        """Return the Class named ``self.user_root_name``, or None."""
        if self.user_root_name is None:
            return None
        for cls in self.model.get_classes():
            if cls.name == self.user_root_name:
                return cls
        return None

    def _per_user_classes(self, user_root):
        """Classes reachable from user_root via associations (excluding user_root itself)."""
        if user_root is None:
            return set()
        reachable = set()
        seen = {user_root}
        queue = deque([user_root])
        while queue:
            cur = queue.popleft()
            for end in cur.association_ends():
                neighbor = end.type
                if isinstance(neighbor, Class) and neighbor not in seen:
                    seen.add(neighbor)
                    reachable.add(neighbor)
                    queue.append(neighbor)
        return reachable

    def _pg_type(self, attr):
        """Map a BUML attribute to its Postgres type (already safe-quoted for enums)."""
        if isinstance(attr.type, Enumeration):
            return f'"{self._safe_ident(attr.type.name.lower())}"'
        return self.POSTGRES_TYPES.get(attr.type.name, "TEXT")

    def _classes_for_emit(self, user_root):
        """
        Regular classes, user-root first so inline FKs to it resolve.
        Association classes are skipped (not yet supported).
        """
        classes = [
            c for c in self.model.classes_sorted_by_inheritance()
            if not isinstance(c, AssociationClass)
        ]
        if user_root is not None and user_root in classes:
            classes.remove(user_root)
            classes.insert(0, user_root)
        return classes

    def _table_name(self, cls):
        """Pre-escaped, lowercased class name, ready to drop inside ``"..."``."""
        return self._safe_ident(cls.name.lower())

    # ---------------- Column / table builders ----------------

    def _build_table_columns(self, cls, user_root, per_user):
        """Return list of column-definition strings for a class."""
        is_user_root = (user_root is not None and cls is user_root)
        cols = []
        emitted_pk = False

        for attr in sort_by_timestamp(cls.attributes):
            attr_name = self._safe_ident(attr.name)
            if attr.is_id:
                if is_user_root:
                    cols.append(
                        f'"{attr_name}" UUID PRIMARY KEY '
                        f'REFERENCES auth.users(id) ON DELETE CASCADE'
                    )
                else:
                    cols.append(
                        f'"{attr_name}" UUID PRIMARY KEY DEFAULT gen_random_uuid()'
                    )
                emitted_pk = True
            else:
                null_clause = "" if attr.is_optional else " NOT NULL"
                cols.append(f'"{attr_name}" {self._pg_type(attr)}{null_clause}')

        # The user-root MUST have an id column referencing auth.users. If the
        # modeler forgot is_id on every attribute, force-emit the canonical id
        # so the auth integration doesn't silently break.
        if is_user_root and not emitted_pk:
            cols.insert(
                0,
                '"id" UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE',
            )

        # Denormalized user_id for per-user classes (skip the user-root itself)
        if cls in per_user:
            user_tbl = self._table_name(user_root)
            cols.append(
                f'"user_id" UUID NOT NULL REFERENCES public."{user_tbl}"(id) ON DELETE CASCADE'
            )

        return cols

    def _junction_tables(self, user_root, per_user):
        """
        Yield junction-table specs for each binary M:N association.

        A junction has two FK columns (one per endpoint) forming a composite
        PK. If either endpoint is the user-root or per-user, the junction is
        also per-user and gets a denormalized ``user_id`` (unless one of the
        FK columns is already named ``user_id``, in which case it does
        double duty).
        """
        for assoc in self.model.associations:
            if len(assoc.ends) != 2:
                continue
            ends = list(assoc.ends)
            if not all(e.multiplicity.max > 1 for e in ends):
                continue  # not M:N

            cols = []
            fk_col_names = []
            for end in ends:
                target_tbl = self._table_name(end.type)
                col_name = self._safe_ident(f"{end.name}_id")
                fk_col_names.append(col_name)
                cols.append(
                    f'"{col_name}" UUID NOT NULL '
                    f'REFERENCES public."{target_tbl}"(id) ON DELETE CASCADE'
                )

            is_per_user = any(
                (user_root is not None and e.type is user_root) or e.type in per_user
                for e in ends
            )

            # Denormalize user_id unless an FK column is already named user_id
            if is_per_user and "user_id" not in fk_col_names:
                user_tbl = self._table_name(user_root)
                cols.append(
                    f'"user_id" UUID NOT NULL '
                    f'REFERENCES public."{user_tbl}"(id) ON DELETE CASCADE'
                )

            # Composite PK on the two FK columns (last so constraints follow columns)
            pk_cols = ", ".join(f'"{c}"' for c in fk_col_names)
            cols.append(f"PRIMARY KEY ({pk_cols})")

            yield {
                "cls": None,
                "name": self._safe_ident(assoc.name.lower()),
                "columns": cols,
                "is_user_root": False,
                "is_per_user": is_per_user,
                "fk_col_names": fk_col_names,
            }

    def _association_fks(self, user_root, fkeys):
        """
        Yield ``(table, column, target_table, nullable)`` for each association
        FK to emit via ALTER TABLE. Suppresses FKs pointing AT the user-root
        because the denormalized ``user_id`` already covers that relationship.
        """
        for assoc in self.model.associations:
            if len(assoc.ends) != 2:
                continue
            entry = fkeys.get(assoc.name)
            if not entry:
                continue
            fk_class_name, fk_end_name = entry
            target_end = next(
                (e for e in assoc.ends if e.name == fk_end_name), None
            )
            if target_end is None:
                continue
            target_cls = target_end.type
            if user_root is not None and target_cls is user_root:
                continue
            yield (
                self._safe_ident(fk_class_name.lower()),
                self._safe_ident(f"{fk_end_name}_id"),
                self._table_name(target_cls),
                target_end.multiplicity.min == 0,
            )

    def _build_safe_enums(self):
        """Pre-sanitize enums for the template. Skips enums with no literals."""
        safe = []
        for enum in self.model.get_enumerations():
            if not enum.literals:
                continue
            safe.append({
                "name": self._safe_ident(enum.name.lower()),
                "literals": [self._safe_string(lit.name) for lit in enum.literals],
            })
        return safe

    # ---------------- Render ----------------

    def generate(self) -> None:
        """Render the SQL file to ``output_dir``. Filename is a Supabase migration-style ``<timestamp>_<slug>.sql``."""
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        env = Environment(
            loader=FileSystemLoader(templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        template = env.get_template("supabase_template.sql.j2")

        # Validate model name now so injection attempts via the comment header
        # surface as a clean ValueError rather than a corrupted SQL file.
        self._reject_control_chars(self.model.name, "model name")

        user_root = self._find_user_root()
        per_user = self._per_user_classes(user_root)
        fkeys = get_foreign_keys(self.model)

        classes = self._classes_for_emit(user_root)
        tables = []
        for cls in classes:
            tables.append({
                "cls": cls,
                "name": self._table_name(cls),
                "columns": self._build_table_columns(cls, user_root, per_user),
                "is_user_root": user_root is not None and cls is user_root,
                "is_per_user": cls in per_user,
            })
        tables.extend(self._junction_tables(user_root, per_user))

        fk_alters = list(self._association_fks(user_root, fkeys))
        safe_enums = self._build_safe_enums()

        # Migration-ready filename: <YYYYMMDDHHMMSS>_<model_slug>.sql is the
        # naming convention expected by `supabase db reset` (which applies
        # files in supabase/migrations/ in filename order). Works equally well
        # for paste-into-SQL-editor use.
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "_", self.model.name.lower()).strip("_") or "model"
        filename = f"{timestamp}_{slug}.sql"

        rendered = template.render(
            model=self.model,
            tables=tables,
            user_root=user_root,
            user_root_tbl=self._table_name(user_root) if user_root else None,
            fk_alters=fk_alters,
            enums=safe_enums,
            filename=filename,
        )

        file_path = self.build_generation_path(file_name=filename)
        with open(file_path, mode="w", encoding="utf-8") as f:
            f.write(rendered)
        print(f"Supabase SQL generated at: {file_path}")
