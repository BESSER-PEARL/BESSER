"""A domain-bound GUI DataBinding must survive export -> re-import.

``gui_model_builder`` used to emit this for a binding with a domain concept::

    _dm_ref = globals().get('domain_model')
    if _dm_ref is not None:
        _dc = _dm_ref.get_class_by_name("Book")
        if _dc:
            layer.field = next((a for a in _dc.attributes if a.name == "title"), None)

Four separate things there are refused by the safe BUML loader: ``globals()``,
a top-level ``if``, a generator expression, and ``_``-prefixed names. So a GUI
model with such a binding could be exported and never loaded back -- and no
test covered it, because the converter suite never round-tripped one.

The fix is not to widen the loader for four more constructs. It is to stop
emitting inline logic: the builder now writes ONE call to ``bind_domain_field``,
which the GUI converter injects into ``allowed_names``, wrapped in the
``try: ... except NameError: pass`` guard the loader already accepts.
"""
import pytest

from besser.utilities.buml_code_builder.common import bind_domain_field
from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json import (
    _safe_buml_loader as loader,
)


class _Attr:
    def __init__(self, name):
        self.name = name


class _Cls:
    attributes = (_Attr("title"), _Attr("pages"))


class _Model:
    def get_class_by_name(self, name):
        return _Cls() if name == "Book" else None


class _Holder:
    field = None


# What the builder emits today.
EMITTED = (
    'try:\n'
    '    layer_binding.field = bind_domain_field(domain_model, "Book", "title")\n'
    'except NameError:\n'
    '    pass\n'
)

# What it emitted before, kept so the loader is never quietly widened for it.
OLD_INLINE_FORM = (
    "_dm_ref = globals().get('domain_model')\n"
    'if _dm_ref is not None:\n'
    '    _dc = _dm_ref.get_class_by_name("Book")\n'
)


# --------------------------------------------------------------------------- #
# The helper
# --------------------------------------------------------------------------- #
def test_the_helper_resolves_a_field():
    assert bind_domain_field(_Model(), "Book", "title").name == "title"


@pytest.mark.parametrize("model, cls, field", [
    (_Model(), "Book", "nope"),      # field absent
    (_Model(), "Missing", "title"),  # class absent
    (None, "Book", "title"),         # no domain model at all
])
def test_the_helper_degrades_to_none(model, cls, field):
    """A GUI model is often exported without its domain model; an unbound
    binding must not break the import."""
    assert bind_domain_field(model, cls, field) is None


# --------------------------------------------------------------------------- #
# The loader
# --------------------------------------------------------------------------- #
def test_the_emitted_binding_loads_and_resolves():
    holder = _Holder()
    loader.safe_load_buml(EMITTED, allowed_names={
        "bind_domain_field": bind_domain_field,
        "domain_model": _Model(),
        "layer_binding": holder,
    })
    assert holder.field is not None and holder.field.name == "title"


def test_a_standalone_gui_export_still_loads():
    """No `domain_model` in scope: the NameError guard swallows it and the
    import succeeds with the binding left unset."""
    holder = _Holder()
    loader.safe_load_buml(EMITTED, allowed_names={
        "bind_domain_field": bind_domain_field,
        "layer_binding": holder,
    })
    assert holder.field is None


def test_the_old_inline_form_is_still_refused():
    """The loader must not have been widened to accept what we stopped
    emitting -- globals(), top-level if, generator expressions and underscore
    names all stay out."""
    with pytest.raises(loader.SafeBumlLoaderError):
        loader.safe_load_buml(OLD_INLINE_FORM, allowed_names={
            "bind_domain_field": bind_domain_field,
        })


# --------------------------------------------------------------------------- #
# The wiring
# --------------------------------------------------------------------------- #
def test_the_builder_emits_the_helper_for_the_field_binding():
    import inspect

    from besser.utilities.buml_code_builder import gui_model_builder

    src = inspect.getsource(gui_model_builder)
    assert "bind_domain_field" in src, "the builder must emit the helper call"


@pytest.mark.xfail(
    reason=(
        "KNOWN GAP: two emit sites still write globals() into generated BUML -- "
        "the DataBinding constructor block and the dataSourceClass block. Only "
        "the per-field binding has been converted. Those two gate a multi-"
        "statement body on `if <class>:` and one builds a set(...) generator "
        "expression, so they need the surrounding logic moved into helpers, not "
        "just the lookup swapped. Until then a GUI model using them still "
        "cannot be re-imported."
    ),
    strict=True,
)
def test_no_emit_site_writes_globals_into_generated_buml():
    import inspect

    from besser.utilities.buml_code_builder import gui_model_builder

    src = inspect.getsource(gui_model_builder)
    marker = "domain_model_ref = globals().get("
    remaining = src.count(marker)
    assert remaining == 0, (
        f"{remaining} emit site(s) still write globals() into generated BUML; "
        "the safe loader refuses it, so those models cannot round-trip"
    )


def test_the_converter_injects_the_helper():
    """The emitted call is only loadable if the converter allows the name."""
    import inspect

    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json import (
        gui_diagram_converter,
    )

    src = inspect.getsource(gui_diagram_converter)
    assert '"bind_domain_field": bind_domain_field' in src
