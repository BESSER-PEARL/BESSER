"""``DjangoGenerator.generate`` must not destroy user content, and must not
report success when it produced nothing.

Two defects, both found in the v8.0.0 release review:

* ``project_dir`` is just ``<output_dir>/<project_name>``, and the generator
  ``rmtree``'d it unconditionally "for idempotency" — so pointing the generator
  at a directory that already held hand-written code under that name deleted it
  with no warning and no confirmation.
* ``generate()`` caught every exception, printed it, and returned normally, so
  a failed generation was indistinguishable from a successful one. The web
  editor's ``/generate-output`` packaged the result and reported success.
"""
import pytest

from besser.BUML.metamodel.structural import (
    Class, DomainModel, PrimitiveDataType, Property,
)


def _model():
    attr = Property(name="title", type=PrimitiveDataType("str"))
    attr.default_value = "ok"
    book = Class(name="Book")
    book.attributes = {
        Property(name="id", type=PrimitiveDataType("int"), is_id=True), attr,
    }
    return DomainModel(name="M", types={book})

def test_django_refuses_to_delete_unrelated_content(tmp_path):
    """``project_dir`` is just ``<output_dir>/<project_name>``, so a user
    directory that happens to share the name used to be rmtree'd silently."""
    import os

    from besser.generators.django import DjangoGenerator

    model = _model()
    gen = DjangoGenerator(model=model, project_name="myproject",
                          app_name="app", output_dir=str(tmp_path))
    victim = tmp_path / "myproject"
    victim.mkdir()
    (victim / "important.py").write_text("# a year of work", encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        gen.generate()

    assert (victim / "important.py").read_text() == "# a year of work"


def test_django_still_replaces_its_own_output(tmp_path, monkeypatch):
    """A previously generated project (manage.py present) is still cleared,
    so regeneration into the same directory stays idempotent."""
    import os
    import subprocess

    from besser.generators.django import DjangoGenerator

    model = _model()
    gen = DjangoGenerator(model=model, project_name="myproject",
                          app_name="app", output_dir=str(tmp_path))
    stale = tmp_path / "myproject"
    stale.mkdir()
    (stale / "manage.py").write_text("# from a previous run", encoding="utf-8")

    # Stop at the first subprocess: the removal has already happened by then.
    def _stop(*a, **kw):
        raise RuntimeError("stop-after-cleanup")

    monkeypatch.setattr(subprocess, "run", _stop)
    with pytest.raises(Exception):
        gen.generate()

    assert not stale.exists(), "a generated project should still be replaced"
