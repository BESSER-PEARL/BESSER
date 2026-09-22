Supabase Schema Generator
=========================

.. warning::

   **Experimental.** This generator is in preview. It targets the most common
   Supabase patterns (UUID primary keys, ``auth.users`` mirroring, per-user
   RLS) but has known gaps -- see :ref:`supabase-limitations` below.

The Supabase generator produces a single migration-style
``<YYYYMMDDHHMMSS>_<model>.sql`` file you can either paste into the Supabase
SQL Editor or drop straight into ``supabase/migrations/`` for the CLI. It
emits Postgres DDL plus the Supabase-specific pieces: ``auth.users``
cascading, a ``handle_new_user`` trigger, denormalized ``user_id`` columns
on per-user tables, explicit ``GRANT``\ s, and Row Level Security policies
that follow Supabase's documented best practices.

Quick start
-----------

.. code-block:: python

    from besser.generators.supabase import SupabaseGenerator

    generator = SupabaseGenerator(
        model=my_model,
        output_dir="output",
        user_root="User",  # class name that maps to auth.users
    )
    generator.generate()

The generator writes a migration-ready filename like
``20260511132351_supplementassistant.sql``. You have two ways to apply it:

**Cloud (Supabase dashboard):**
Open https://supabase.com/dashboard -> your project ->
**SQL Editor** -> **New query** -> paste the file -> **Run**.

**Local (with the Supabase CLI):**

.. code-block:: bash

    # one-time setup, in your project root:
    supabase init                       # creates supabase/
    supabase start                      # spins up local Postgres/Auth/Studio via Docker

    # apply the generated schema:
    mv output/<generated-file>.sql supabase/migrations/
    supabase db reset                   # wipes local DB, replays all migrations

    # later, push the same schema to your cloud project:
    supabase link --project-ref <your-project-ref>
    supabase db push

Requires `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_
and the `Supabase CLI <https://supabase.com/docs/guides/cli/getting-started>`_
(``npm install -g supabase`` or ``scoop install supabase``).

The same SQL works in both places; only the apply mechanism differs. The
generated file's header comment summarizes both paths so you don't need to
keep this doc open.

Parameters
----------

- ``model``: The input B-UML structural model.
- ``output_dir``: Optional output directory (default: ``output/`` in the current directory).
- ``user_root``: Name of the class that should mirror ``auth.users`` (default:
  ``"User"``). Pass ``None`` to skip auth integration entirely.

What gets generated
-------------------

Two terms used throughout:

- **user-root**: the class that mirrors ``auth.users`` (set via the
  ``user_root`` parameter; default name ``"User"``).
- **per-user table**: any class reachable from the user-root by following
  associations. The generator denormalizes a ``user_id`` column onto each one
  so a single RLS predicate (``user_id = auth.uid()``) covers it.

**For the user-root class:**

- ``id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE`` (no
  ``gen_random_uuid()`` default -- the id comes from ``auth.users``). If the
  modeler forgot to mark any attribute with ``is_id``, this column is injected
  automatically so the auth integration doesn't silently break.
- A ``handle_new_user()`` trigger function with ``SECURITY DEFINER`` and
  ``SET search_path = ''`` (defense in depth against search-path injection).
  The insert uses ``ON CONFLICT (id) DO NOTHING`` so re-runs and admin
  pre-seeding don't break signup.
- RLS enabled with all four policies (``SELECT`` / ``INSERT`` / ``UPDATE`` /
  ``DELETE``), each ``TO authenticated`` and keyed on
  ``(SELECT auth.uid()) = id``.

**For every other class:**

- ``id UUID PRIMARY KEY DEFAULT gen_random_uuid()`` for any attribute with
  ``is_id=True``.
- A denormalized ``user_id UUID NOT NULL REFERENCES public.<user_root>(id) ON
  DELETE CASCADE`` column *if* the class is reachable from the user-root via
  associations.
- An index on ``user_id``.
- RLS enabled with all four policies, each ``TO authenticated`` and keyed on
  ``(SELECT auth.uid()) = user_id``. ``UPDATE`` policies include a ``WITH
  CHECK`` clause so authenticated users can't transfer rows to another user by
  rewriting ``user_id``.

**For each 1:1 / 1:N association:**

- FK column added via ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``
  (so table order doesn't matter), plus a ``CREATE INDEX IF NOT EXISTS`` for
  the FK column. FKs pointing **at** the user-root are suppressed -- the
  denormalized ``user_id`` covers them.

**For each M:N association:**

- A junction table named after the association, with two UUID FK columns
  (one per endpoint, derived from the end names) and a composite
  ``PRIMARY KEY`` over them. Both FKs use ``ON DELETE CASCADE``. If either
  endpoint is per-user, the junction itself becomes per-user: it gets a
  denormalized ``user_id`` column, an index, and the full RLS policy set
  (unless one of the FK columns is already named ``user_id``, in which case
  it does double duty).

**For each enumeration:**

- A ``CREATE TYPE ... AS ENUM`` wrapped in a ``DO $$ ... EXCEPTION WHEN
  duplicate_object THEN NULL; END $$`` block (idempotent). Enumerations with
  no literals are skipped (an empty ``ENUM ()`` is not valid Postgres).

**Grants:**

- A single combined ``GRANT SELECT, INSERT, UPDATE, DELETE`` on all generated
  tables ``TO authenticated``, plus ``GRANT USAGE ON SCHEMA public``. Supabase
  ships sensible default privileges, but emitting them explicitly makes the
  script work on non-Supabase Postgres or projects where defaults were reset.

**Idempotency:**

- ``CREATE TABLE IF NOT EXISTS``, ``CREATE INDEX IF NOT EXISTS``,
  ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``, ``CREATE OR REPLACE FUNCTION``,
  ``DROP TRIGGER IF EXISTS; CREATE TRIGGER``, and ``DROP POLICY IF EXISTS;
  CREATE POLICY`` -- so the script is safe to re-run on an existing schema.
  Note however that ``CREATE TABLE IF NOT EXISTS`` does **not** add new
  columns to an existing table; if you change attributes on a class, drop the
  table first or write a manual migration.

Sample output
-------------

For the simple ``User --< IntakeSchedule`` model, the generator emits roughly
the following (abbreviated for brevity):

.. code-block:: sql

    -- user-root mirrors auth.users (no gen_random_uuid default).
    -- Note: keep email/password/phone OUT of this table -- they live in
    -- auth.users. See Known limitations #3.
    CREATE TABLE IF NOT EXISTS public."user" (
        "id" UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
        "display_name" TEXT NOT NULL
    );

    -- per-user table, with denormalized user_id and its index
    CREATE TABLE IF NOT EXISTS public."intakeschedule" (
        "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        "time" TIMESTAMPTZ NOT NULL,
        "user_id" UUID NOT NULL REFERENCES public."user"(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS "idx_intakeschedule_user_id"
        ON public."intakeschedule"("user_id");

    -- mirror trigger: SECURITY DEFINER + empty search_path + ON CONFLICT
    CREATE OR REPLACE FUNCTION public.handle_new_user()
    RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER SET search_path = ''
    AS $$
    BEGIN
        INSERT INTO public."user" (id) VALUES (NEW.id)
        ON CONFLICT (id) DO NOTHING;
        RETURN NEW;
    END;
    $$;
    DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
    CREATE TRIGGER on_auth_user_created AFTER INSERT ON auth.users
        FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

    -- one RLS policy of four for intakeschedule
    ALTER TABLE public."intakeschedule" ENABLE ROW LEVEL SECURITY;
    DROP POLICY IF EXISTS "intakeschedule_update_own" ON public."intakeschedule";
    CREATE POLICY "intakeschedule_update_own" ON public."intakeschedule"
        AS PERMISSIVE FOR UPDATE
        TO authenticated
        USING ((SELECT auth.uid()) = "user_id")
        WITH CHECK ((SELECT auth.uid()) = "user_id");

The three Supabase-specific patterns to notice:

- ``(SELECT auth.uid())`` wraps the auth function so Postgres evaluates it
  once per query instead of once per row (Supabase RLS performance
  guidance).
- ``TO authenticated`` pins each policy to the authenticated role; without
  it, the policy applies to every role including ``anon``.
- ``WITH CHECK`` on ``UPDATE`` policies prevents authenticated users from
  rewriting ``user_id`` to transfer rows to a different user.

Output structure
----------------

A single ``<timestamp>_<slug>.sql`` file with sections in this order:

1. Header comment + ``pgcrypto`` note
2. Enumerations (skipped if none)
3. Tables (user-root first so inline FKs to it resolve), each followed by its
   ``user_id`` index when applicable
4. ``handle_new_user`` function + trigger on ``auth.users``
5. Association FKs via ``ALTER TABLE``, each followed by an index on the FK
   column
6. ``GRANT`` block
7. ``ENABLE ROW LEVEL SECURITY`` + policies (user-root first, then per-user
   tables)

Conventions
-----------

- All identifiers are double-quoted in DDL to bypass Postgres reserved words
  (``user``, ``time``, ``order``, etc.). Embedded double-quotes in names are
  doubled (``"`` → ``""``) and enum literal single-quotes are doubled
  (``'`` → ``''``) before reaching the template, so a malicious or malformed
  model name cannot break out of the quoted identifier or string literal.
  Names containing ``NUL`` / ``CR`` / ``LF`` are rejected with ``ValueError``.
- Class names become table names via simple lowercasing (``IntakeSchedule`` ->
  ``intakeschedule``). No automatic pluralization or snake_case conversion --
  pick the names you want in the model.
- Postgres type mapping: ``str`` → ``TEXT``, ``int`` → ``INTEGER``,
  ``float`` → ``DOUBLE PRECISION``, ``bool`` → ``BOOLEAN``, ``date`` →
  ``DATE``, ``time`` → ``TIME``, ``datetime`` → ``TIMESTAMPTZ``,
  ``timedelta`` → ``INTERVAL``, ``any`` → ``JSONB``. Enumerations are
  emitted as Postgres ``ENUM`` types.

.. _supabase-limitations:

Known limitations
-----------------

1. **All ``is_id`` attributes become UUID.** There's no per-attribute opt-out.
   This is intentional for Supabase but means integer-id lookup tables get
   UUIDs anyway.

2. **Reachable means per-user.** Any class reachable from the user-root via
   associations gets a ``user_id`` and per-user RLS. This catches genuinely
   per-user data but also incorrectly tags shared catalog data (e.g., a global
   ``Ingredient`` table linked into the graph) as per-user.

   Worked example for the associations
   ``User --< IntakeSchedule --< Supplement``:

   - ``IntakeSchedule`` is directly associated with ``User`` -> gets
     ``user_id``.
   - ``Supplement`` is associated only with ``IntakeSchedule``, but
     ``IntakeSchedule`` is associated with ``User``, so ``Supplement`` is
     transitively reachable -> also gets ``user_id``.
   - A ``Country`` class with no associations into this graph is **not**
     reachable -> no ``user_id``, no RLS enabled. Supabase will warn that
     ``Country`` is exposed without RLS; either accept it (read-only lookup
     data) or add policies manually.

3. **Auth-owned attributes aren't filtered.** If your user-root class has
   ``email`` / ``password`` / ``phone`` attributes, they are emitted as columns
   on ``public.<user_root>``. Supabase's convention is to keep those in
   ``auth.users`` only. Remove the auth-owned attributes from your user-root
   class before generating.

4. **Non-per-user tables don't get RLS enabled.** Supabase will warn that the
   table is exposed without RLS. Either accept this for genuinely public data,
   move the table out of the per-user graph, or add RLS manually.

5. **Association classes are silently dropped.** A class attached to an
   M:N association (carrying extra attributes on the relationship itself) is
   not yet supported -- only the plain junction is emitted. No warning is
   logged. Plain binary 1:1, 1:N, and M:N associations are all supported.

6. **Inheritance is flattened.** No polymorphic discriminator column;
   parent and child classes become independent tables.

7. **No ``created_at`` / ``updated_at`` columns.** Add them as attributes on
   your model if you want them; the generator doesn't inject audit columns
   automatically.

8. **No Realtime publication.** To enable change subscriptions, run
   ``ALTER PUBLICATION supabase_realtime ADD TABLE public."<table>";``
   manually after generation.

9. **``ON DELETE CASCADE`` on every FK.** Ownership cascades are correct;
   non-ownership cascades (e.g., deleting a parent record wipes children
   that merely reference it) may be more aggressive than intended.

Re-running on an existing schema
--------------------------------

The script is idempotent for schema *structure* but not for column changes.
If you modify a class's attributes, ``CREATE TABLE IF NOT EXISTS`` won't add
the new columns. The cleanest re-run is to drop the schema first:

.. code-block:: sql

    -- Replace with your actual table list
    DROP TABLE IF EXISTS
        public."user", public."ingredient", public."intakeschedule",
        public."ingredientcontent", public."supplement", public."dosagereference"
        CASCADE;
    DROP FUNCTION IF EXISTS public.handle_new_user() CASCADE;
    DROP TYPE IF EXISTS "unit";

The ``CASCADE`` on ``handle_new_user`` also removes its trigger on
``auth.users``, releasing the lock that can make full-script re-runs slow.

Related
-------

- :doc:`sql` -- generic SQL generator (multiple dialects, no auth/RLS).
- :doc:`alchemy` -- SQLAlchemy ORM generator (used as the Postgres compile
  path by ``SQLGenerator``).
