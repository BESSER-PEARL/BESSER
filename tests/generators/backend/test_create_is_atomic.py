"""A rejected create must leave nothing behind.

The create endpoint used to ``commit()`` the new row and only then validate
the related ids, with no rollback anywhere in the function. The client got a
400 and the half-created row stayed in the database — verified live on
2026-09-18: posting a booking naming a BookingRoom id that does not exist
returned 400 and left one orphan booking.

The bulk endpoint in the same template already did it correctly
(``flush()  # Get ID without committing``); the single-create path now does
too: flush to obtain the id, commit once at the end.
"""
import re

import pytest

from besser.generators.backend import BackendGenerator


@pytest.fixture
def booking_router(tmp_path, library_book_author_model):
    BackendGenerator(library_book_author_model, output_dir=str(tmp_path),
                     http_methods=["GET", "POST", "PUT", "DELETE"]).generate()
    routers = list(tmp_path.rglob("routers/*.py"))
    assert routers, "no routers generated"
    return {p.name: p.read_text(encoding="utf-8") for p in routers}


def _create_bodies(src):
    """Every create_<entity> function body in a router module."""
    for m in re.finditer(r"\nasync def create_\w+\(.*?(?=\n@router\.|\Z)", src, re.S):
        yield m.group(0)


def test_no_raise_after_a_commit_inside_create(booking_router):
    for name, src in booking_router.items():
        for body in _create_bodies(src):
            commit = body.find("database.commit()")
            if commit == -1:
                continue
            after = body[commit:]
            assert "raise HTTPException" not in after, (
                f"{name}: create() raises after committing — the rejected row "
                f"stays in the database")


def test_create_uses_flush_to_obtain_the_id(booking_router):
    """flush() makes the id available without making the row durable."""
    seen_flush = False
    for src in booking_router.values():
        for body in _create_bodies(src):
            if "database.flush()" in body:
                seen_flush = True
    assert seen_flush, "no create() uses flush(); ids are still obtained by committing"


def test_create_still_commits_before_returning(booking_router):
    """Atomic must not mean 'never persisted'."""
    for name, src in booking_router.items():
        for body in _create_bodies(src):
            if "database.flush()" not in body:
                continue
            assert "database.commit()" in body, f"{name}: create() never commits"
            assert body.index("database.commit()") > body.index("database.flush()"), (
                f"{name}: the commit must come after the flush, at the end")


def test_generated_routers_are_valid_python(booking_router):
    for name, src in booking_router.items():
        compile(src, name, "exec")
