"""
Tests for the Supabase generator.

The generator writes a timestamped filename so each test resolves the
output via glob rather than a fixed name.
"""
import glob
import os

import pytest

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Enumeration, EnumerationLiteral,
    Multiplicity, Property, StringType,
)
from besser.generators.supabase import SupabaseGenerator


def _read_generated_sql(output_dir: str) -> str:
    matches = glob.glob(os.path.join(output_dir, "*.sql"))
    assert len(matches) == 1, f"Expected one .sql file in {output_dir}, got {matches}"
    with open(matches[0], encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Library / Book / Author -- shared fixture (no is_id, no user-root)
# ---------------------------------------------------------------------------

def test_library_model_emits_tables_and_mn_junction(library_book_author_model, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    SupabaseGenerator(
        model=library_book_author_model, output_dir=str(out), user_root="User",
    ).generate()
    sql = _read_generated_sql(str(out))

    for table in ("library", "book", "author"):
        assert f'CREATE TABLE IF NOT EXISTS public."{table}"' in sql

    # Author <-->> Book is M:N -> junction with composite PK and both FK columns.
    assert 'CREATE TABLE IF NOT EXISTS public."book_author"' in sql
    assert 'REFERENCES public."book"(id) ON DELETE CASCADE' in sql
    assert 'REFERENCES public."author"(id) ON DELETE CASCADE' in sql
    assert "PRIMARY KEY (" in sql

    # No user-root in the fixture, so no auth integration and no RLS.
    assert "handle_new_user" not in sql
    assert "ENABLE ROW LEVEL SECURITY" not in sql


# ---------------------------------------------------------------------------
# User-rooted model -- exercises auth.users mirror, denormalization, RLS
# ---------------------------------------------------------------------------

@pytest.fixture
def user_rooted_model():
    """User --< Note (Note transitively per-user)."""
    user = Class(name="User", attributes={
        Property(name="id", type=StringType, is_id=True),
    })
    note = Class(name="Note", attributes={
        Property(name="id", type=StringType, is_id=True),
        Property(name="title", type=StringType),
        Property(name="body", type=StringType, is_optional=True),
    })

    user_note = BinaryAssociation(
        name="user_note",
        ends={
            Property(name="user", type=user, multiplicity=Multiplicity(1, 1)),
            Property(name="notes", type=note, multiplicity=Multiplicity(0, "*")),
        },
    )

    return DomainModel(name="UserRootedDemo", types={user, note}, associations={user_note})


def test_user_root_mirrors_auth_users(user_rooted_model, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    SupabaseGenerator(
        model=user_rooted_model, output_dir=str(out), user_root="User",
    ).generate()
    sql = _read_generated_sql(str(out))

    assert 'CREATE TABLE IF NOT EXISTS public."user"' in sql
    assert "REFERENCES auth.users(id) ON DELETE CASCADE" in sql

    assert "CREATE OR REPLACE FUNCTION public.handle_new_user" in sql
    assert "SECURITY DEFINER" in sql
    assert "SET search_path = ''" in sql
    assert "ON CONFLICT (id) DO NOTHING" in sql
    assert "AFTER INSERT ON auth.users" in sql


def test_per_user_table_gets_denormalized_user_id_and_rls(user_rooted_model, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    SupabaseGenerator(
        model=user_rooted_model, output_dir=str(out), user_root="User",
    ).generate()
    sql = _read_generated_sql(str(out))

    assert 'CREATE TABLE IF NOT EXISTS public."note"' in sql
    assert '"user_id" UUID NOT NULL REFERENCES public."user"(id) ON DELETE CASCADE' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_note_user_id" ON public."note"("user_id")' in sql

    # RLS policies follow Supabase best practices.
    assert 'ALTER TABLE public."note" ENABLE ROW LEVEL SECURITY' in sql
    assert "TO authenticated" in sql
    assert '(SELECT auth.uid()) = "user_id"' in sql
    assert 'WITH CHECK ((SELECT auth.uid()) = "user_id")' in sql

    # Idempotency: every CREATE POLICY is preceded by DROP POLICY IF EXISTS.
    assert sql.count("DROP POLICY IF EXISTS") >= 4
    assert sql.count("CREATE POLICY") >= 4


def test_is_id_yields_uuid_pk_with_gen_random_uuid(user_rooted_model, tmp_path):
    """Non-user-root classes with is_id get UUID PRIMARY KEY DEFAULT gen_random_uuid()."""
    out = tmp_path / "out"
    out.mkdir()
    SupabaseGenerator(
        model=user_rooted_model, output_dir=str(out), user_root="User",
    ).generate()
    sql = _read_generated_sql(str(out))
    assert "UUID PRIMARY KEY DEFAULT gen_random_uuid()" in sql


# ---------------------------------------------------------------------------
# Sanitization -- the security blocker
#
# The metamodel rejects names with spaces / hyphens but does NOT reject
# double-quote, single-quote, or semicolon, so a malicious model can still
# carry an injection payload past construction. The generator must
# defang it before emission.
# ---------------------------------------------------------------------------

def test_double_quote_in_class_name_is_escaped_not_injected(tmp_path):
    evil = Class(name='evil";DROP_TABLE_x;SELECT_1', attributes={
        Property(name="id", type=StringType, is_id=True),
    })
    model = DomainModel(name="EvilDemo", types={evil}, associations=set())

    out = tmp_path / "out"
    out.mkdir()
    SupabaseGenerator(model=model, output_dir=str(out), user_root=None).generate()
    sql = _read_generated_sql(str(out))

    # The embedded " is doubled so the identifier still terminates correctly.
    # Table name is lowercased by _table_name(), but the escape rule is the same.
    assert 'CREATE TABLE IF NOT EXISTS public."evil"";drop_table_x;select_1"' in sql
    # No raw DROP statement leaked outside the identifier.
    for line in sql.splitlines():
        stripped = line.strip().upper()
        assert not stripped.startswith("DROP TABLE")


def test_single_quote_in_enum_literal_is_escaped_not_injected(tmp_path):
    """Enum literal containing ' must be doubled to '' so it stays inside the string."""
    evil_enum = Enumeration(name="Status", literals={
        EnumerationLiteral(name="ok"),
        EnumerationLiteral(name="bad');DROP_TABLE_x;SELECT_1"),
    })
    cls = Class(name="Thing", attributes={
        Property(name="id", type=StringType, is_id=True),
        Property(name="status", type=evil_enum),
    })
    model = DomainModel(name="EnumEvil", types={cls, evil_enum}, associations=set())

    out = tmp_path / "out"
    out.mkdir()
    SupabaseGenerator(model=model, output_dir=str(out), user_root=None).generate()
    sql = _read_generated_sql(str(out))

    assert "'bad'');DROP_TABLE_x;SELECT_1'" in sql
    for line in sql.splitlines():
        stripped = line.strip().upper()
        assert not stripped.startswith("DROP TABLE")


def test_rejects_newline_in_identifier(tmp_path):
    """NUL/CR/LF surface a clean ValueError instead of corrupting the output."""
    bad = Class(name="Foo\nbar", attributes={
        Property(name="id", type=StringType, is_id=True),
    })
    model = DomainModel(name="BadName", types={bad}, associations=set())

    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="Invalid character"):
        SupabaseGenerator(model=model, output_dir=str(out), user_root=None).generate()


# ---------------------------------------------------------------------------
# Filename / determinism guard
# ---------------------------------------------------------------------------

def test_filename_is_migration_style(user_rooted_model, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    SupabaseGenerator(
        model=user_rooted_model, output_dir=str(out), user_root="User",
    ).generate()

    matches = glob.glob(os.path.join(str(out), "*.sql"))
    assert len(matches) == 1
    name = os.path.basename(matches[0])
    # <YYYYMMDDHHMMSS>_<slug>.sql
    assert name.endswith("_userrooteddemo.sql")
    timestamp = name.split("_", 1)[0]
    assert timestamp.isdigit() and len(timestamp) == 14
