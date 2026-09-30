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


# --------------------------------------------------------------------------- #
# The DataBinding constructor and dataSourceClass blocks, end to end
# --------------------------------------------------------------------------- #
def _bound_gui_and_domain():
    from besser.BUML.metamodel.gui import (
        DataList, DataSourceElement, GUIModel, Module, Screen,
    )
    from besser.BUML.metamodel.gui.binding import DataBinding
    from besser.BUML.metamodel.gui.dashboard import BarChart
    from besser.BUML.metamodel.structural import Class, DomainModel, IntegerType, Property, StringType

    book = Class(name="Book")
    title = Property(name="title", type=StringType)
    pages = Property(name="pages", type=IntegerType)
    book.attributes = {title, pages}
    domain_model = DomainModel(name="library", types={book})

    chart = BarChart(name="pages_chart")
    chart.data_binding = DataBinding(
        domain_concept=book, name="pages_binding", label_field=title, data_field=pages,
    )
    source = DataSourceElement(
        name="books", dataSourceClass=book, fields={title, pages}, label_field=title, value_field=pages,
    )
    books = DataList(name="book_list", description="", list_sources={source})
    screen = Screen(
        name="home", description="", view_elements={chart, books},
        is_main_page=True, route_path="/", screen_size="Medium",
    )
    gui_model = GUIModel(
        name="ui", package="p", versionCode="1", versionName="1", description="",
        modules={Module(name="Main", screens={screen})},
    )
    return gui_model, domain_model


def _gui_code(tmp_path, gui_model, domain_model=None):
    from besser.utilities.buml_code_builder.gui_model_builder import gui_model_to_code

    path = tmp_path / "gui.py"
    gui_model_to_code(gui_model, str(path), domain_model=domain_model)
    return path.read_text(encoding="utf-8")


def _components(gui_model):
    screen = next(iter(next(iter(gui_model.modules)).screens))
    return {element.name: element for element in screen.view_elements}


def test_domain_bound_chart_and_data_source_reimport_with_their_bindings(tmp_path):
    """The loader refused both blocks outright (top-level ``if``); now they load bound."""
    from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
        _parse_gui_model,
    )

    gui_model, domain_model = _bound_gui_and_domain()
    domain_path = tmp_path / "domain.py"
    domain_model_to_code(domain_model, str(domain_path))

    loaded = _parse_gui_model(
        _gui_code(tmp_path, gui_model), context_code=domain_path.read_text(encoding="utf-8"),
    )

    components = _components(loaded)
    binding = components["pages_chart"].data_binding
    assert binding is not None and binding.domain_concept.name == "Book"
    assert binding.name == "pages_binding"
    assert binding.label_field.name == "title" and binding.data_field.name == "pages"
    source = next(iter(components["book_list"].list_sources))
    assert source.dataSourceClass.name == "Book"
    assert {field.name for field in source.fields} == {"title", "pages"}
    assert source.label_field.name == "title" and source.value_field.name == "pages"


def test_domain_bound_gui_without_its_domain_model_keeps_the_names(tmp_path):
    """Loaded on its own, the binding is unresolved but the field names survive."""
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.gui_diagram_converter import (
        _parse_gui_model,
    )

    gui_model, _ = _bound_gui_and_domain()
    loaded = _parse_gui_model(_gui_code(tmp_path, gui_model))

    components = _components(loaded)
    assert components["pages_chart"].data_binding is None
    source = next(iter(components["book_list"].list_sources))
    assert source.dataSourceClass is None
    assert sorted(source.field_names) == ["pages", "title"]


def test_the_emitted_file_still_runs_as_plain_python(tmp_path):
    """Executed directly (not through the loader), the helpers come from its imports."""
    gui_model, domain_model = _bound_gui_and_domain()
    namespace = {}
    exec(_gui_code(tmp_path, gui_model, domain_model=domain_model), namespace)  # noqa: S102

    components = _components(namespace["gui_model"])
    assert components["pages_chart"].data_binding.domain_concept.name == "Book"
    source = next(iter(components["book_list"].list_sources))
    assert {field.name for field in source.fields} == {"title", "pages"}
