"""Scaffold map, Django family: anchors over REAL DjangoGenerator output.

The map is computed from the files on disk, so these tests generate the
project into a temporary workspace exactly as the agent's ``_gen_django``
does (``myproject``/``myapp`` under ``<workspace>/django``) and assert anchors
by id, never by line number. DjangoGenerator writes ``admin.py`` in hash-seed
order and takes ``settings.py``/``urls.py`` from the host's ``django-admin``,
so nothing here depends on the order or quoting of those lines either.

Two generated variants are covered: the admin-only scaffold (no GUI model, the
usual Phase 1 output) and the GUI variant of a model with an OCL constraint,
whose ``models.py``, ``views.py``, ``forms.py`` and ``urls.py`` do not parse
and are read with the extractor's line scanner. Hand-written projects cover
the pitfalls, DRF, several roots and malformed files.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from besser.BUML.metamodel.gui import DataList, DataSourceElement, GUIModel, Module, Screen
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    BooleanType,
    Class,
    Constraint,
    DateType,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    FloatType,
    Generalization,
    IntegerType,
    Method,
    Multiplicity,
    Parameter,
    Property,
    StringType,
)
from besser.spec_driven_agent.planning.scaffold_map import build_workspace_map, locate
from besser.spec_driven_agent.planning.scaffold_map import django as django_map
from besser.spec_driven_agent.planning.scaffold_map.model import (
    KIND_BLOCK,
    KIND_CLASS,
    KIND_FILE,
    KIND_FUNCTION,
    KIND_INSERT,
    KIND_ROUTE,
    KIND_STUB,
)
from besser.spec_driven_agent.planning.scaffold_map.render import render_overview
from besser.spec_driven_agent.planning.scaffold_map.source import SourceIndex

CLASS_NAMES = ["Person", "Library", "Book", "Author"]
ROOT = "django/myproject"
APP = f"{ROOT}/myapp"


# ---------------------------------------------------------------------------
# Fixtures: a library model with an enum, a generalization, two associations,
# a CODE method with a parameter and a body-less method (NotImplementedError).
# ---------------------------------------------------------------------------

def _library_model(with_ocl: bool, string_pk: bool = False) -> DomainModel:
    genre = Enumeration(name="Genre", literals={
        EnumerationLiteral(name="FICTION"), EnumerationLiteral(name="SCIENCE"),
    })
    person = Class(name="Person", attributes={
        Property(name="name", type=StringType), Property(name="email", type=StringType),
    })
    library = Class(name="Library", attributes={
        Property(name="name", type=StringType), Property(name="address", type=StringType),
    })
    book = Class(name="Book", attributes={
        Property(name="title", type=StringType), Property(name="pages", type=IntegerType),
        Property(name="release", type=DateType), Property(name="genre", type=genre),
        Property(name="available", type=BooleanType), Property(name="price", type=FloatType),
    })
    author = Class(name="Author", attributes={Property(name="nationality", type=StringType)})
    if string_pk:       # a CharField primary key: the GUI views still look rows up by id=
        library.add_attribute(Property(name="code", type=StringType, is_id=True))
    book.methods = {Method(
        name="apply_discount", parameters=[Parameter(name="percent", type=FloatType)], type=FloatType,
        code="def apply_discount(self, percent: float) -> float:\n"
             "    self.price = self.price * (1 - percent / 100)\n"
             "    return self.price\n")}
    library.methods = {Method(name="count_books", type=IntegerType)}
    located_in = Property(name="locatedIn", type=library, multiplicity=Multiplicity(1, 1))
    has = Property(name="has", type=book, multiplicity=Multiplicity(0, "*"))
    written_by = Property(name="writtenBy", type=author, multiplicity=Multiplicity(1, "*"))
    publishes = Property(name="publishes", type=book, multiplicity=Multiplicity(0, "*"))
    constraints = set()
    if with_ocl:
        constraints.add(Constraint(name="pages_positive", context=book,
                                   expression="context Book inv pagesPositive: self.pages > 0", language="OCL"))
    return DomainModel(
        name="Library_Model",
        types={genre, person, library, book, author},
        associations={
            BinaryAssociation(name="lib_book", ends={located_in, has}),
            BinaryAssociation(name="book_author", ends={written_by, publishes}),
        },
        generalizations={Generalization(general=person, specific=author)},
        constraints=constraints,
    )


def _gui_model(model: DomainModel) -> GUIModel:
    classes = {c.name: c for c in model.get_classes()}
    screens = {Screen(name="Home", description="", x_dpi="", y_dpi="", screen_size="Small",
                      view_elements=set(), is_main_page=True)}
    for name in ("Book", "Library", "Author"):
        source = DataSourceElement(name=f"{name}DS", dataSourceClass=classes[name],
                                   fields=list(classes[name].attributes))
        screens.add(Screen(name=f"{name}Screen", description="", x_dpi="", y_dpi="", screen_size="Small",
                           view_elements={DataList(name=f"{name}List", description="", list_sources={source})}))
    return GUIModel(name="app", package="com.example.app", versionCode="1", versionName="1.0",
                    description="Library app", screenCompatibility=True,
                    modules={Module(name="main", screens=screens)})


def _generate(model: DomainModel, gui_model, workspace: Path) -> None:
    """Exactly how tool_executor._gen_django calls the generator (Phase 1 passes {} args)."""
    from besser.generators.django import DjangoGenerator

    bin_dir = os.path.dirname(sys.executable)
    if shutil.which("django-admin", path=bin_dir + os.pathsep + os.environ.get("PATH", "")) is None:
        pytest.skip("django-admin is not installed")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PATH", bin_dir + os.pathsep + os.environ.get("PATH", ""))
        DjangoGenerator(model=model, project_name="myproject", app_name="myapp", gui_model=gui_model,
                        output_dir=str(workspace / "django")).generate()
    assert (workspace / ROOT / "manage.py").is_file()


@pytest.fixture(scope="module")
def admin_only(tmp_path_factory) -> Path:
    """The usual Phase 1 output: no GUI model, models.py parses."""
    ws = tmp_path_factory.mktemp("django_admin_only")
    _generate(_library_model(with_ocl=False), None, ws)
    return ws


@pytest.fixture(scope="module")
def gui_ocl(tmp_path_factory) -> Path:
    """GUI variant of a model with an OCL constraint: models/views/forms/urls do not parse."""
    ws = tmp_path_factory.mktemp("django_gui_ocl")
    model = _library_model(with_ocl=True, string_pk=True)
    _generate(model, _gui_model(model), ws)
    return ws


@pytest.fixture
def workspace(admin_only, tmp_path) -> Path:
    ws = tmp_path / "ws"
    shutil.copytree(admin_only, ws)
    return ws


def _maps(ws: Path, class_names=CLASS_NAMES):
    return django_map.extract(SourceIndex(str(ws)), list(class_names))


def _only(ws: Path, class_names=CLASS_NAMES):
    maps = _maps(ws, class_names)
    assert len(maps) == 1, [m.root for m in maps]
    return maps[0]


def _anchors(smap) -> dict:
    return {a.id: a for a in smap.all_anchors()}


def _assert_quotes_valid(ws: Path, anchors) -> None:
    for anchor in anchors:
        lines = (ws / anchor.file).read_text(encoding="utf-8").splitlines()
        assert 1 <= anchor.start_line <= anchor.end_line <= max(1, len(lines)), anchor
        if not anchor.quote:
            continue
        assert lines.count(anchor.quote) == 1, (anchor.id, anchor.quote)
        # modify_file matches old_text as a substring: unique in the whole text.
        assert (ws / anchor.file).read_text(encoding="utf-8").count(anchor.quote) == 1, (anchor.id, anchor.quote)
        at = lines.index(anchor.quote) + 1
        assert anchor.start_line <= at <= anchor.end_line, (anchor.id, at, anchor.start_line, anchor.end_line)
        if anchor.kind == KIND_INSERT:
            assert anchor.start_line == anchor.end_line == at, anchor.id


def _write(root: Path, files: dict) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")


_MANAGE = '''
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "{pkg}.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
'''


# ---------------------------------------------------------------------------
# The generated admin-only scaffold
# ---------------------------------------------------------------------------

def test_admin_only_scaffold_is_mapped_per_entity(admin_only):
    smap = _only(admin_only)
    assert smap.family == "django"
    assert smap.root == ROOT
    assert "admin-only" in smap.title
    assert sorted(smap.entities) == sorted(CLASS_NAMES)
    anchors = _anchors(smap)

    # Project level: settings lists, the project URLconf, requirements, per-app insertion points.
    for anchor_id in ("manage", "settings", "installed_apps", "installed_apps_end", "middleware", "middleware_end",
                      "databases", "settings_end", "urls_import", "urlpatterns", "urlpatterns_end",
                      "requirements_end", "models_end:myapp", "admin_end:myapp", "views_end:myapp"):
        assert f"django:{anchor_id}" in anchors, anchor_id
    assert anchors["django:installed_apps"].kind == KIND_BLOCK
    assert "myapp" in anchors["django:installed_apps"].label
    assert anchors["django:installed_apps_end"].kind == KIND_INSERT
    assert anchors["django:installed_apps_end"].quote == "    'myapp',"
    assert anchors["django:urlpatterns"].kind == KIND_BLOCK
    assert anchors["django:urlpatterns_end"].quote == "]"
    assert "BEFORE" in anchors["django:urlpatterns_end"].hint
    assert "NOT imported" in anchors["django:urls_import"].hint        # startproject imports path only
    assert "NO trailing newline" in anchors["django:requirements_end"].hint
    assert anchors["django:views_end:myapp"].quote == "# Create your views here."
    assert anchors["django:manage"].kind == KIND_FILE
    # The no-GUI scaffold has no app urls.py.
    assert not [i for i in anchors if i.startswith("django:app_urlpatterns")]

    # Per entity: model class, field insertion point, admin registration, stubs.
    book = {a.id: a for a in smap.entities["Book"].anchors}
    assert {"django:model:book", "django:add_field:book", "django:admin:book", "django:admin_class:book",
            "django:method:book:apply_discount"} <= set(book)
    assert book["django:model:book"].kind == KIND_CLASS
    assert book["django:model:book"].quote == "class Book(models.Model):"
    assert "FK locatedIn->Library" in book["django:model:book"].label
    assert "M2M writtenBy->Author" in book["django:model:book"].label
    add_field = book["django:add_field:book"]
    assert add_field.kind == KIND_INSERT and "AFTER" in add_field.hint and "makemigrations" in add_field.hint
    assert book["django:admin:book"].quote == "admin.site.register(Book, BookAdmin)"
    assert book["django:admin_class:book"].quote == "class BookAdmin(admin.ModelAdmin):"
    assert book["django:method:book:apply_discount"].kind == KIND_FUNCTION
    assert [a.id for a in smap.entities["Book"].anchors][0] == "django:model:book"     # definition first

    stub = anchors["django:stub:library:count_books"]
    assert stub.kind == KIND_STUB and stub.status == "NotImplementedError stub"
    assert stub.quote.strip().startswith("raise NotImplementedError(")
    assert "library_count_books_action" in stub.hint            # the admin action that calls it
    assert "inherits Person" in anchors["django:model:author"].hint
    assert anchors["django:admin:person"].quote == "admin.site.register(Person)"
    assert "register(Person, PersonAdmin)" in anchors["django:admin:person"].hint
    assert anchors["django:enum:Genre"].kind == KIND_CLASS

    assert all(not e.missing for e in smap.entities.values())
    assert any("NO urls.py" in line for line in smap.summary)
    assert any("admin/ only" in line for line in smap.summary)


def test_every_quote_is_unique_and_inside_its_span(admin_only, gui_ocl):
    for ws in (admin_only, gui_ocl):
        smap = _only(ws)
        _assert_quotes_valid(ws, smap.all_anchors())
        ids = [a.id for a in smap.all_anchors()]
        assert len(ids) == len(set(ids))
        assert not [i for i in ids if re.search(r":\d+$", i)]      # no collision suffix on generated code


def test_admin_action_without_the_method_argument_is_a_pitfall(admin_only):
    smap = _only(admin_only)
    warning = next(w for w in smap.warnings if "book_apply_discount_action" in w)
    assert "obj.apply_discount()" in warning and "percent" in warning
    assert f"{APP}/admin.py" in warning
    assert len(smap.warnings) == 1


def test_recipes_point_at_existing_anchors(admin_only):
    smap = _only(admin_only)
    ids = set(_anchors(smap))
    recipes = {r.intent: " ".join(r.steps) for r in smap.recipes}
    assert {"Add a field", "Implement a model method", "Add a page or view", "Add a REST API",
            "Customise the admin", "Add a new model", "Run migrations", "Change settings"} <= set(recipes)
    for text in recipes.values():
        assert not re.search(r"\bline \d", text)
        for ref in re.findall(r"`(django[^`]*)`", text):
            if "<" in ref:
                pattern = re.escape(ref).replace("<entity>", "[a-z]+").replace("<method>", r"\w+")
                assert any(re.fullmatch(pattern, i) for i in ids), ref
            else:
                assert ref in ids, ref
    # No app urls.py and no include(): the page recipe says to create and include it.
    page = recipes["Add a page or view"]
    assert f"CREATE {APP}/urls.py" in page and "include('myapp.urls')" in page and "`django:urls_import`" in page
    assert "makemigrations myapp" in recipes["Run migrations"] and "none ship" in recipes["Run migrations"]


def test_overview_and_locate_end_to_end(admin_only):
    model = _library_model(with_ocl=False)
    ws_map = build_workspace_map(str(admin_only), model)
    django_maps = [m for m in ws_map.maps if m.family == "django"]
    assert [m.root for m in django_maps] == [ROOT]
    assert not ws_map.broken

    overview = render_overview(ws_map)
    assert "Django project myproject" in overview
    assert "`django:add_field:<entity>`" in overview
    assert not re.search(r"\blines? \d", overview)                 # line-free

    found = locate(str(admin_only), entity="Book", domain_model=model)
    assert "[django:model:book]" in found["locations"]
    assert "read_file offset=" in found["locations"]
    one = locate(str(admin_only), query="django:installed_apps_end", domain_model=model)
    assert one["count"] == 1 and "'myapp'," in one["locations"]
    first = locate(str(admin_only), domain_model=model)
    assert {"Person", "Library", "Book", "Author"} <= set(first["entities"])
    assert any("book_apply_discount_action" in p for p in first["pitfalls"])


def test_anchors_follow_edits(workspace):
    before = _anchors(_only(workspace))["django:model:book"]
    models = workspace / APP / "models.py"
    models.write_text("# a\n# b\n# c\n" + models.read_text(encoding="utf-8"), encoding="utf-8")
    after = _anchors(_only(workspace))["django:model:book"]
    assert after.start_line == before.start_line + 3 and after.quote == before.quote


def test_class_in_model_but_not_in_code_is_missing(admin_only, workspace):
    smap = _only(admin_only, CLASS_NAMES + ["Loan"])
    assert smap.entities["Loan"].missing == ["model class", "admin registration"]
    assert not smap.entities["Loan"].anchors
    overview = render_overview(build_workspace_map(str(admin_only), None))
    assert "MISSING" not in overview                               # no domain model: nothing is claimed

    admin = workspace / APP / "admin.py"
    admin.write_text(re.sub(r"(?m)^admin\.site\.register\(Person\)\n", "", admin.read_text(encoding="utf-8")),
                     encoding="utf-8")
    smap = _only(workspace)
    assert smap.entities["Person"].missing == ["admin registration"]
    assert "django:admin:person" not in _anchors(smap)


def test_no_class_names_maps_models_by_their_code_names(admin_only):
    smap = _only(admin_only, [])
    assert sorted(smap.entities) == sorted(CLASS_NAMES)
    assert all(not e.missing for e in smap.entities.values())


def test_map_is_deterministic(admin_only, gui_ocl):
    for ws in (admin_only, gui_ocl):
        first = [dataclasses.asdict(m) for m in _maps(ws)]
        second = [dataclasses.asdict(m) for m in _maps(ws)]
        assert first == second


# ---------------------------------------------------------------------------
# The generated GUI variant (with an OCL constraint): four files do not parse
# ---------------------------------------------------------------------------

def test_gui_variant_with_unparseable_files_is_still_mapped(gui_ocl):
    index = SourceIndex(str(gui_ocl))
    smap = django_map.extract(index, CLASS_NAMES)[0]
    assert {f"{APP}/{name}" for name in ("models.py", "views.py", "forms.py", "urls.py")} <= set(index.syntax_errors)
    assert "server-rendered views" in smap.title
    anchors = _anchors(smap)

    # models.py does not parse: the line scanner still finds every model, its fields and methods.
    assert sorted(smap.entities) == sorted(CLASS_NAMES)
    model = anchors["django:model:book"]
    assert "does not parse" in model.hint and model.quote == "class Book(models.Model):"
    assert "FK locatedIn->Library" in model.label
    assert anchors["django:clean:book"].kind == KIND_FUNCTION
    assert anchors["django:stub:library:count_books"].kind == KIND_STUB
    assert anchors["django:add_field:book"].kind == KIND_INSERT

    # The GUI pieces per entity: ModelForm, views, url patterns, templates.
    book = {a.id for a in smap.entities["Book"].anchors}
    assert {"django:form:book:BookForm", "django:view:book:create_book", "django:view:book:edit_book",
            "django:view:book:delete_book", "django:view:book:book_list", "django:url:book:create_book",
            "django:url:book:book_list", "django:template:book:book", "django:template:book:book_list",
            "django:template:book:book_form"} <= book
    url = anchors["django:url:book:edit_book"]
    assert url.kind == KIND_ROUTE and "book/<int:book_id>/edit/" in url.label
    assert "'book/<int:book_id>/edit/'" in url.quote
    assert "extends book.html" in anchors["django:template:book:book_list"].label
    assert "django:view:home" in anchors and "django:url:home" in anchors      # not an entity's

    # The app URLconf is unterminated: the block says so, the insert point follows the last entry.
    block = anchors["django:app_urlpatterns:myapp"]
    assert "never closed" in block.hint
    assert anchors["django:app_urlpatterns_end:myapp"].kind == KIND_INSERT
    # The project urls include the app: no "never included" or "unrouted view" pitfall.
    assert "include(myapp.urls)" in anchors["django:urlpatterns"].label
    assert not [w for w in smap.warnings if "include" in w or "no route" in w]
    recipes = {r.intent: " ".join(r.steps) for r in smap.recipes}
    assert "`django:app_urlpatterns_end:myapp`" in recipes["Add a page or view"]


def test_overview_lists_the_broken_files_first(gui_ocl):
    overview = render_overview(build_workspace_map(str(gui_ocl), _library_model(with_ocl=True)))
    assert overview.startswith("Files that do not parse")
    assert f"{APP}/models.py" in overview.split("###", 1)[0]


# ---------------------------------------------------------------------------
# Hand-written projects
# ---------------------------------------------------------------------------

_PITFALL_PROJECT = {
    "site/manage.py": _MANAGE.format(pkg="site_cfg"),
    "site/site_cfg/__init__.py": "",
    "site/site_cfg/settings.py": '''
        INSTALLED_APPS = [
            "django.contrib.admin",
            "django.contrib.auth",
            "shop.apps.ShopConfig",
        ]
        MIDDLEWARE = [
            "corsheaders.middleware.CorsMiddleware",
            "django.middleware.common.CommonMiddleware",
        ]
        ROOT_URLCONF = "site_cfg.urls"
        DATABASES = {"default": {"ENGINE": "django.db.backends.postgresql"}}
    ''',
    "site/site_cfg/urls.py": '''
        from django.contrib import admin
        from django.urls import path

        urlpatterns = [
            path("admin/", admin.site.urls),
            path("blog/", include("blog.urls")),
        ]
    ''',
    "site/shop/__init__.py": "",
    "site/shop/apps.py": '''
        from django.apps import AppConfig


        class ShopConfig(AppConfig):
            name = "shop"
    ''',
    "site/shop/models.py": '''
        from django.db import models


        class Product(models.Model):
            name = models.CharField(max_length=100)
            category = models.ForeignKey("Category", on_delete=models.CASCADE)
            owner = models.ForeignKey("auth.User", on_delete=models.CASCADE)

            def price_with_tax(self, rate):
                raise NotImplementedError


        class Order(models.Model):
            product = models.ForeignKey(Product)
            quantity = models.IntegerField(default=1)

            class Meta:
                ordering = ["-id"]
    ''',
    "site/shop/admin.py": '''
        from django.contrib import admin

        from .models import Order, Product

        admin.site.register(Product)


        @admin.register(Product)
        class ProductAdmin(admin.ModelAdmin):
            list_display = ["name"]
    ''',
    "site/shop/views.py": '''
        from django.http import JsonResponse

        from rest_framework.decorators import api_view


        def product_list(request):
            return JsonResponse({"items": []})


        def order_detail(request, order_id):
            return JsonResponse({"id": order_id})
    ''',
    "site/shop/urls.py": '''
        from django.urls import path

        from . import views

        urlpatterns = [
            path("products/", views.product_list, name="product_list"),
        ]
    ''',
    "site/blog/__init__.py": "",
    "site/blog/models.py": '''
        from django.db import models


        class Post(models.Model):
            title = models.CharField(max_length=200)
    ''',
    "site/requirements.txt": "Django==5.1.5\n",
}


def test_pitfalls_found_in_hand_written_code(tmp_path, monkeypatch):
    _write(tmp_path, _PITFALL_PROJECT)
    monkeypatch.setattr(django_map, "_MAX_WARNINGS", 50)
    smap = _only(tmp_path, ["Product", "Order"])
    assert smap.root == "site" and smap.title.startswith("Django project site_cfg")
    text = "\n".join(smap.warnings)
    for expected in (
        "site/shop/admin.py: Product is registered twice (admin.site.register and @admin.register)",
        "Product.category references 'Category'",
        "Order.product is a ForeignKey without on_delete",
        "site/site_cfg/urls.py calls include() without importing it",
        "site/site_cfg/urls.py includes 'blog.urls', which does not exist",
        "site/shop/urls.py is never include()d",
        "site/shop/views.py: order_detail has no route",
        "site/blog has models but 'blog' is not in INSTALLED_APPS",
        "'corsheaders' is used in site/site_cfg/settings.py but is not in INSTALLED_APPS",
        "'rest_framework' is used in site/shop/views.py",
        "site/requirements.txt lacks djangorestframework",
    ):
        assert expected in text, expected
    assert "auth.User" not in text                                   # another app's model is not checked
    assert "product_list" not in text                                # routed
    anchors = _anchors(smap)
    # Apps come from INSTALLED_APPS ("shop.apps.ShopConfig") and from apps.py/models.py siblings.
    assert {"django:models_end:shop", "django:models_end:blog", "django:app_urlpatterns:shop",
            "django:app_urlpatterns_end:shop", "django:model:post"} <= set(anchors)
    assert anchors["django:stub:product:price_with_tax"].status == "NotImplementedError stub"
    assert anchors["django:admin_class:product"].entity == "Product"
    assert anchors["django:databases"].label == "DATABASES (postgresql)"
    assert smap.entities["Order"].missing == ["admin registration"]
    assert any("NOT in INSTALLED_APPS" in line for line in smap.summary)
    _assert_quotes_valid(tmp_path, smap.all_anchors())


def test_drf_api_and_class_based_views(tmp_path):
    _write(tmp_path, {
        "api/manage.py": _MANAGE.format(pkg="conf"),
        "api/conf/settings.py": """
            INSTALLED_APPS = ['django.contrib.admin', 'rest_framework', 'books']
            ROOT_URLCONF = 'conf.urls'
        """,
        "api/conf/urls.py": """
            from django.urls import include, path

            urlpatterns = [
                path('api/', include('books.urls')),
            ]
        """,
        "api/books/models.py": """
            from django.db import models


            class Book(models.Model):
                title = models.CharField(max_length=100)
        """,
        "api/books/serializers.py": """
            from rest_framework import serializers

            from .models import Book


            class BookSerializer(serializers.ModelSerializer):
                class Meta:
                    model = Book
                    fields = ['id', 'title']
        """,
        "api/books/api.py": """
            from rest_framework import viewsets

            from .models import Book
            from .serializers import BookSerializer


            class BookViewSet(viewsets.ModelViewSet):
                queryset = Book.objects.all()
                serializer_class = BookSerializer
        """,
        "api/books/views.py": """
            from django.views.generic import ListView

            from .models import Book


            class ShelfView(ListView):
                model = Book
        """,
        "api/books/urls.py": """
            from django.urls import include, path
            from rest_framework.routers import DefaultRouter

            from . import api, views

            router = DefaultRouter()
            router.register(r'books', api.BookViewSet)

            urlpatterns = [
                path('shelf/', views.ShelfView.as_view(), name='shelf'),
                path('', include(router.urls)),
            ]
        """,
    })
    smap = _only(tmp_path, ["Book"])
    assert "DRF API" in smap.title
    book = {a.id: a for a in smap.entities["Book"].anchors}
    assert {"django:model:book", "django:serializer:book:BookSerializer", "django:view:book:BookViewSet",
            "django:view:book:ShelfView", "django:url:book:BookViewSet", "django:url:book:shelf"} <= set(book)
    assert book["django:view:book:ShelfView"].kind == KIND_CLASS
    assert book["django:url:book:BookViewSet"].label.startswith("router 'books'")
    assert not smap.warnings                                        # everything is routed and included
    recipes = {r.intent for r in smap.recipes}
    assert "Add a REST endpoint" in recipes and "Add a REST API" not in recipes
    _assert_quotes_valid(tmp_path, smap.all_anchors())


def test_several_roots_get_unique_slugged_ids(tmp_path):
    for root, model in (("one", "Ticket"), ("two", "Invoice")):
        _write(tmp_path, {
            f"{root}/manage.py": _MANAGE.format(pkg="proj"),
            f"{root}/proj/settings.py": "INSTALLED_APPS = ['core']\nROOT_URLCONF = 'proj.urls'\n",
            f"{root}/proj/urls.py": "from django.urls import path\nurlpatterns = []\n",
            f"{root}/core/models.py": f"""
                from django.db import models


                class {model}(models.Model):
                    title = models.CharField(max_length=10)
            """,
        })
    maps = _maps(tmp_path, ["Ticket", "Invoice"])
    assert [m.root for m in maps] == ["one", "two"]
    ids = [a.id for m in maps for a in m.all_anchors()]
    assert len(ids) == len(set(ids))
    assert "django@one:model:ticket" in ids and "django@two:model:invoice" in ids
    # A class modelled in the other project is not reported missing here.
    assert sorted(maps[0].entities) == ["Ticket"] and sorted(maps[1].entities) == ["Invoice"]


def test_malformed_files_never_raise(tmp_path):
    _write(tmp_path, {
        "proj/manage.py": _MANAGE.format(pkg="cfg"),
        "proj/cfg/settings.py": "INSTALLED_APPS = [\n    'core',\n",            # unterminated
        "proj/cfg/urls.py": "urlpatterns = [path('x/', views.x\n",
        "proj/core/models.py": """
            from django.db import models
            from django.core.exceptions import ValidationError

            class Ticket(models.Model):
                title = models.CharField(max_length=10
                status = models.CharField(max_length=10)

                def clean(self):
                    if errors:
                        raise ValidationError(errors)class Comment(Ticket):
                body = models.TextField()

            class Broken(models.Model:
                x = 1
        """,
        "proj/core/admin.py": "from django.contrib import admin\nadmin.site.register(Ticket\n",
        "proj/core/views.py": "def (:\n",
    })
    (tmp_path / "proj/core/forms.py").write_bytes(b"\xff\xfe\x00 not utf-8")
    index = SourceIndex(str(tmp_path))
    maps = django_map.extract(index, ["Ticket", "Comment", "Broken"])
    smap = maps[0]
    anchors = _anchors(smap)
    assert "django:model:ticket" in anchors and "django:add_field:ticket" in anchors
    # A class glued to the previous statement by the generator's newline stripping is still found.
    comment = anchors["django:model:comment"]
    assert comment.quote.endswith("raise ValidationError(errors)class Comment(Ticket):")
    assert "inherits Ticket" in comment.hint
    # The header that does not even match is not invented, and not reported missing either.
    assert smap.entities["Broken"].missing == ["admin registration"]
    assert "proj/core/models.py" in index.syntax_errors
    _assert_quotes_valid(tmp_path, smap.all_anchors())
    # The whole workspace map still builds and renders.
    assert render_overview(build_workspace_map(str(tmp_path)))


def test_manage_py_without_django_is_not_a_root(tmp_path):
    _write(tmp_path, {"tool/manage.py": "print('not a web framework')\n",
                      "tool/app/models.py": "from django.db import models\n"})
    assert _maps(tmp_path) == []


# ---------------------------------------------------------------------------
# Review fixes: overview budget, wrapped views, unregister, split settings,
# list insertion points, recipes, entity names, inheritance, string keys
# ---------------------------------------------------------------------------

_SHOP = ["Customer", "Address", "Order", "OrderItem", "Product", "Category", "Supplier", "Warehouse",
         "Inventory", "Payment", "Shipment", "Review", "Coupon", "Cart", "CartItem"]


def _shop_model() -> DomainModel:
    """15 classes, 16 associations, six of them with a parameterised method (one admin action each)."""
    classes = {name: Class(name=name, attributes={
        Property(name="name", type=StringType), Property(name="code", type=StringType),
        Property(name="qty", type=IntegerType), Property(name="active", type=BooleanType),
    }) for name in _SHOP}
    for name in ("Order", "Cart", "Product", "Coupon", "Payment", "Shipment"):
        classes[name].methods = {Method(name="apply", parameters=[Parameter(name="amount", type=FloatType)],
                                        type=FloatType), Method(name="total", type=FloatType)}
    pairs = [("Customer", "Address"), ("Customer", "Order"), ("Order", "OrderItem"), ("Product", "OrderItem"),
             ("Category", "Product"), ("Supplier", "Product"), ("Warehouse", "Inventory"), ("Product", "Inventory"),
             ("Order", "Payment"), ("Order", "Shipment"), ("Product", "Review"), ("Customer", "Review"),
             ("Customer", "Cart"), ("Cart", "CartItem"), ("Product", "CartItem")]
    associations = set()
    for one, many in pairs:
        associations.add(BinaryAssociation(name=f"{one}_{many}", ends={
            Property(name=one.lower() + "Ref", type=classes[one], multiplicity=Multiplicity(1, 1)),
            Property(name=many.lower() + "s", type=classes[many], multiplicity=Multiplicity(0, "*"))}))
    associations.add(BinaryAssociation(name="order_coupon", ends={
        Property(name="coupons", type=classes["Coupon"], multiplicity=Multiplicity(0, "*")),
        Property(name="orders", type=classes["Order"], multiplicity=Multiplicity(0, "*"))}))
    return DomainModel(name="Shop", types=set(classes.values()), associations=associations)


def test_overview_of_a_15_class_model_keeps_its_entities_within_budget(tmp_path):
    model = _shop_model()
    _generate(model, None, tmp_path)
    ws_map = build_workspace_map(str(tmp_path), model)
    smap = next(m for m in ws_map.maps if m.family == "django")
    # One line for the six actions that miss the method's argument, not one line each.
    action_warnings = [w for w in smap.warnings if "admin action" in w]
    assert len(action_warnings) == 1 and "6 admin actions" in action_warnings[0], smap.warnings
    assert "apply(amount)" in action_warnings[0] and "+3" in action_warnings[0]

    overview = render_overview(ws_map)
    assert len(overview) <= 4000
    assert "Per entity:" in overview
    assert "- OrderItem: model (" in overview                     # full entity lines, not file lists
    assert "6 admin actions" in overview and "Where new code goes:" in overview


@pytest.mark.parametrize("parses", [True, False])
def test_views_wrapped_in_urls_py_count_as_routed(tmp_path, parses):
    urls = '''
        from django.contrib.auth.decorators import login_required
        from django.urls import path
        from django.views.decorators.cache import cache_page
        from django.views.decorators.csrf import csrf_exempt

        from . import views

        urlpatterns = [
            path("dashboard/", login_required(views.dashboard), name="dashboard"),
            path("stats/", cache_page(60)(views.book_stats), name="stats"),
            path("about/", csrf_exempt(views.AboutView.as_view()), name="about"),
            path("loose/", views.loose),
    '''
    _write(tmp_path, {
        "p/manage.py": _MANAGE.format(pkg="cfg"),
        "p/cfg/settings.py": "INSTALLED_APPS = ['django.contrib.auth', 'lib']\nROOT_URLCONF = 'cfg.urls'\n",
        "p/cfg/urls.py": "from django.urls import include, path\n\n"
                         "urlpatterns = [\n    path('', include('lib.urls')),\n]\n",
        "p/lib/models.py": "from django.db import models\n\n\nclass Book(models.Model):\n"
                           "    title = models.CharField()\n",
        "p/lib/views.py": '''
            from django.shortcuts import render
            from django.views.generic import TemplateView


            def dashboard(request):
                return render(request, "dashboard.html")


            def book_stats(request):
                return render(request, "stats.html")


            def loose(request):
                return render(request, "loose.html")


            def orphan(request):
                return render(request, "orphan.html")


            class AboutView(TemplateView):
                template_name = "about.html"
        ''',
        # The unparseable variant never closes the list: the line scanner reads the entries.
        "p/lib/urls.py": urls + ("        ]\n" if parses else ""),
    })
    index = SourceIndex(str(tmp_path))
    smap = django_map.extract(index, ["Book"])[0]
    assert ("p/lib/urls.py" in index.syntax_errors) is not parses
    unrouted = [w for w in smap.warnings if "no route" in w]
    assert unrouted == ["p/lib/views.py: orphan has no route in any urls.py."], smap.warnings
    anchors = _anchors(smap)
    assert anchors["django:url:dashboard"].label.startswith("'dashboard/' -> views.dashboard")
    assert "-> views.AboutView" in anchors["django:url:about"].label
    assert anchors["django:url:book:stats"].entity == "Book"
    _assert_quotes_valid(tmp_path, smap.all_anchors())


def test_unregister_between_two_registrations_is_not_a_clash(workspace):
    admin = workspace / APP / "admin.py"
    text = admin.read_text(encoding="utf-8")
    assert "admin.site.register(Person)\n" in text
    clash = text + "\n\n@admin.register(Person)\nclass PersonAdmin(admin.ModelAdmin):\n    list_display = ['name']\n"
    admin.write_text(clash, encoding="utf-8")
    assert any("Person is registered twice" in w for w in _only(workspace).warnings)

    admin.write_text(text + "\nadmin.site.unregister(Person)\n" + clash[len(text):], encoding="utf-8")
    smap = _only(workspace)
    assert not [w for w in smap.warnings if "registered twice" in w], smap.warnings
    assert "django:admin_class:person" in _anchors(smap)

    # Unregistered and never registered again: the admin registration is missing.
    admin.write_text(text + "\nadmin.site.unregister(Person)\n", encoding="utf-8")
    assert _only(workspace).entities["Person"].missing == ["admin registration"]


def test_installed_apps_split_over_lists_and_modules(tmp_path):
    common = {
        "cfg/urls.py": "from django.urls import path\n\nurlpatterns = []\n",
        "shop/__init__.py": "",
        "shop/models.py": "from django.db import models\n\n\nclass Product(models.Model):\n"
                          "    name = models.CharField()\n",
        "shop/views.py": '''
            from rest_framework.decorators import api_view
        ''',
    }
    # 1. INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS
    split = tmp_path / "split"
    _write(split, {f"p/{rel}": text for rel, text in common.items()} | {
        "p/manage.py": _MANAGE.format(pkg="cfg"),
        "p/cfg/settings.py": '''
            DJANGO_APPS = [
                "django.contrib.admin",
                "django.contrib.auth",
            ]
            THIRD_PARTY_APPS = ["rest_framework"]
            LOCAL_APPS = [
                "shop",
            ]
            INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS
            ROOT_URLCONF = "cfg.urls"
        ''',
    })
    smap = _only(split, ["Product"])
    assert not [w for w in smap.warnings if "INSTALLED_APPS" in w], smap.warnings
    anchors = _anchors(smap)
    label = anchors["django:installed_apps"].label
    assert "shop" in label and "rest_framework" in label
    end = anchors["django:installed_apps_end"]
    assert end.quote == '    "shop",' and "LOCAL_APPS" in end.hint
    assert not any("NOT in INSTALLED_APPS" in line for line in smap.summary)

    # 2. A settings package: manage.py names local.py, which star-imports base.py and extends the list.
    package = tmp_path / "package"
    _write(package, {f"p/{rel}": text for rel, text in common.items()} | {
        "p/manage.py": _MANAGE.format(pkg="cfg").replace("cfg.settings", "cfg.settings.local"),
        "p/cfg/settings/__init__.py": "",
        "p/cfg/settings/base.py": '''
            INSTALLED_APPS = [
                "django.contrib.admin",
                "shop",
            ]
            ROOT_URLCONF = "cfg.urls"
        ''',
        "p/cfg/settings/local.py": '''
            from .base import *  # noqa

            DEBUG = True
            INSTALLED_APPS += ["rest_framework"]
        ''',
    })
    smap = _only(package, ["Product"])
    assert not [w for w in smap.warnings if "INSTALLED_APPS" in w], smap.warnings
    anchors = _anchors(smap)
    assert anchors["django:installed_apps"].file == "p/cfg/settings/base.py"
    assert anchors["django:installed_apps_end"].quote == '    "shop",'
    assert "p/cfg/settings/local.py" in anchors["django:settings"].label
    assert "rest_framework" in anchors["django:installed_apps"].label
    assert anchors["django:urlpatterns"].file == "p/cfg/urls.py"          # ROOT_URLCONF read from base.py
    _assert_quotes_valid(package, smap.all_anchors())

    # 3. Not readable statically: nothing is claimed about what it lacks.
    dynamic = tmp_path / "dynamic"
    _write(dynamic, {f"p/{rel}": text for rel, text in common.items()} | {
        "p/manage.py": _MANAGE.format(pkg="cfg"),
        "p/cfg/settings.py": "import os\n\nINSTALLED_APPS = os.environ['APPS'].split(',')\nROOT_URLCONF = 'cfg.urls'\n",
    })
    smap = _only(dynamic, ["Product"])
    assert not [w for w in smap.warnings if "INSTALLED_APPS" in w], smap.warnings
    assert not any("NOT in INSTALLED_APPS" in line for line in smap.summary)
    assert "django:installed_apps_end" not in _anchors(smap)


def test_list_insert_points_stay_inside_the_list(workspace):
    settings = workspace / ROOT / "myproject" / "settings.py"
    text = settings.read_text(encoding="utf-8")
    # startproject double-quotes settings.py only when black is on PATH.
    last_middleware = re.search(r"XFrameOptionsMiddleware(['\"]),\n\]", text)
    assert "    'myapp',\n]" in text and last_middleware
    q = last_middleware.group(1)
    # The closing bracket on the last entry's line: the new entry goes BEFORE that entry.
    settings.write_text(text.replace("    'myapp',\n]", "    'myapp']").replace(
        last_middleware.group(0), f"XFrameOptionsMiddleware{q}]"), encoding="utf-8")
    anchors = _anchors(_only(workspace))
    apps, middleware = anchors["django:installed_apps_end"], anchors["django:middleware_end"]
    assert apps.quote == "    'myapp']" and "BEFORE" in apps.hint and "end the new entry with a comma" in apps.hint
    assert middleware.quote.endswith(f"XFrameOptionsMiddleware{q}]") and "BEFORE" in middleware.hint
    _assert_quotes_valid(workspace, [apps, middleware])

    # No trailing comma on the last entry: AFTER it, with a warning about string concatenation.
    settings.write_text(text.replace("    'myapp',\n]", "    'myapp'\n]"), encoding="utf-8")
    apps = _anchors(_only(workspace))["django:installed_apps_end"]
    assert apps.quote == "    'myapp'" and "AFTER" in apps.hint and "no trailing comma" in apps.hint

    # A single-line list cannot take an entry on a line of its own: no insertion point at all.
    settings.write_text(text.replace("MIDDLEWARE = [", "MIDDLEWARE_UNUSED = [").replace(
        "ROOT_URLCONF", "MIDDLEWARE = ['django.middleware.common.CommonMiddleware']\nROOT_URLCONF"),
        encoding="utf-8")
    anchors = _anchors(_only(workspace))
    assert "django:middleware" in anchors and "django:middleware_end" not in anchors


def test_recipes_follow_the_code_without_stubs(workspace):
    models = workspace / APP / "models.py"
    text = models.read_text(encoding="utf-8")
    assert "raise NotImplementedError" in text
    models.write_text(re.sub(r"raise NotImplementedError\([^)]*\)", "return None", text), encoding="utf-8")
    smap = _only(workspace)
    ids = set(_anchors(smap))
    assert not [i for i in ids if i.startswith("django:stub:")]
    recipes = {r.intent: " ".join(r.steps) for r in smap.recipes}
    assert "`django:method:<entity>:<method>`" in recipes["Implement a model method"]
    for text in recipes.values():
        for ref in re.findall(r"`(django[^`]*)`", text):
            pattern = re.escape(ref).replace("<entity>", "[a-z]+").replace("<method>", r"\w+")
            assert any(re.fullmatch(pattern, i) for i in ids), ref
    # TEMPLATES has APP_DIRS=True and no DIRS: only an app's templates/ directory is searched.
    assert f"{APP}/templates/registration/login.html" in recipes["Add auth"]


def test_multi_word_entities_win_over_their_first_word(tmp_path):
    _write(tmp_path, {
        "p/manage.py": _MANAGE.format(pkg="cfg"),
        "p/cfg/settings.py": "INSTALLED_APPS = ['shop']\nROOT_URLCONF = 'cfg.urls'\n",
        "p/cfg/urls.py": "from django.urls import path\n\nurlpatterns = []\n",
        "p/shop/models.py": '''
            from django.db import models


            class Order(models.Model):
                total = models.FloatField()


            class OrderItem(models.Model):
                qty = models.IntegerField()


            class Cart(models.Model):
                total = models.FloatField()


            class CartItem(models.Model):
                qty = models.IntegerField()
        ''',
        "p/shop/views.py": '''
            from django.shortcuts import render


            def order_item_list(request):
                return render(request, "order_item_list.html")


            def cart_item_detail(request, pk):
                return render(request, "cart_item_detail.html")


            def order_list(request):
                return render(request, "order_list.html")


            def create_cartItem(request):
                return render(request, "x.html")
        ''',
    })
    smap = _only(tmp_path, ["Order", "OrderItem", "Cart", "CartItem"])
    entity = {a.id.rsplit(":", 1)[-1]: a.entity for a in smap.all_anchors() if a.id.startswith("django:view:")}
    assert entity == {"order_item_list": "OrderItem", "cart_item_detail": "CartItem", "order_list": "Order",
                      "create_cartItem": "CartItem"}
    found = locate(str(tmp_path), entity="OrderItem", domain_model=None)
    assert "order_item_list" in found["locations"]


def test_project_package_with_views_is_not_described_as_an_app(workspace):
    project = workspace / ROOT / "myproject"
    (project / "views.py").write_text(
        "from django.http import HttpResponse\n\n\ndef home(request):\n    return HttpResponse('hi')\n",
        encoding="utf-8")
    urls = project / "urls.py"
    text = urls.read_text(encoding="utf-8")
    urls.write_text(text.replace("from django.urls import path", "from django.urls import path\n\nfrom . import views")
                    .replace("urlpatterns = [", "urlpatterns = [\n    path('', views.home),"), encoding="utf-8")
    smap = _only(workspace)
    line = next(line for line in smap.summary if "(django/myproject/myproject)" in line)
    assert line.startswith("project package myproject") and "the project urls.py" in line
    assert "NO urls.py" not in line and "migrations" not in line
    assert not [w for w in smap.warnings if "no route" in w]


def test_abstract_and_proxy_parents(tmp_path):
    _write(tmp_path, {
        "p/manage.py": _MANAGE.format(pkg="cfg"),
        "p/cfg/settings.py": "INSTALLED_APPS = ['shop']\nROOT_URLCONF = 'cfg.urls'\n",
        "p/shop/models.py": '''
            from django.db import models


            class TimeStamped(models.Model):
                created = models.DateTimeField(auto_now_add=True)

                class Meta:
                    abstract = True


            class Product(TimeStamped):
                name = models.CharField(max_length=100)


            class Special(Product):
                class Meta:
                    proxy = True


            class Gadget(Product):
                volts = models.IntegerField()
        ''',
    })
    smap = _only(tmp_path, ["Product", "Special", "Gadget"])
    anchors = _anchors(smap)
    assert "extends abstract TimeStamped" in anchors["django:model:product"].hint
    assert "multi-table" not in anchors["django:model:product"].hint
    assert "proxy of Product: no fields allowed" in anchors["django:model:special"].hint
    assert "django:add_field:special" not in anchors                     # models.E017
    assert "multi-table" in anchors["django:model:gadget"].hint
    assert "extends abstract TimeStamped" in smap.entities["Product"].summary[0]
    assert "proxy of Product" in smap.entities["Special"].summary[0]


def test_string_primary_key_behind_int_routes_and_id_lookups(gui_ocl):
    smap = _only(gui_ocl)
    assert "code" in _anchors(smap)["django:model:library"].label
    warning = next(w for w in smap.warnings if "look rows up by id=" in w)
    assert warning.startswith(f"{APP}/views.py: ") and "Library.code" in warning
    assert "edit_library" in warning and "delete_library" in warning
    assert "use pk= and <str:...> instead of <int:...>" in warning
    assert "Book" not in warning                                          # integer primary keys are fine


# ---------------------------------------------------------------------------
# Review fixes, round 2: form helpers are not views, the page recipe names a
# base template that exists, migrations, and no regex backtracks badly
# ---------------------------------------------------------------------------

def _recipes(smap) -> dict:
    return {r.intent: " ".join(r.steps) for r in smap.recipes}


@pytest.fixture(scope="module")
def gui_home(tmp_path_factory) -> Path:
    """GUI variant whose only screen is Home: views.py holds home() and the generator's form helpers."""
    ws = tmp_path_factory.mktemp("django_gui_home")
    home = Screen(name="Home", description="", x_dpi="", y_dpi="", screen_size="Small", view_elements=set(),
                  is_main_page=True)
    gui = GUIModel(name="app", package="com.example.app", versionCode="1", versionName="1.0",
                   description="Library app", screenCompatibility=True, modules={Module(name="main", screens={home})})
    _generate(_library_model(with_ocl=False), gui, ws)
    return ws


def test_form_helpers_of_the_gui_scaffold_are_not_unrouted_views(gui_home):
    views = (gui_home / APP / "views.py").read_text(encoding="utf-8")
    # Helpers the entity views call: a route to one raises TypeError (form_class, success_message missing).
    assert "def handle_create(request, form_class, success_message):" in views
    assert "def handle_edit(request, form_class, instance, success_message, redirect_url):" in views
    smap = _only(gui_home)
    assert not [w for w in smap.warnings if "no route" in w], smap.warnings


@pytest.mark.parametrize("parses", [True, False])
def test_only_view_signatures_are_reported_unrouted(tmp_path, parses):
    _write(tmp_path, {
        "p/manage.py": _MANAGE.format(pkg="cfg"),
        "p/cfg/settings.py": "INSTALLED_APPS = ['lib']\nROOT_URLCONF = 'cfg.urls'\n",
        "p/cfg/urls.py": "from django.urls import include, path\n\n"
                         "urlpatterns = [\n    path('', include('lib.urls')),\n]\n",
        "p/lib/models.py": "from django.db import models\n\n\nclass Book(models.Model):\n"
                           "    title = models.CharField()\n",
        "p/lib/urls.py": '''
            from django.urls import path

            from . import views

            urlpatterns = [
                path("", views.home),
                path("year/<int:year>/", views.by_year),
            ]
        ''',
        "p/lib/views.py": '''
            from django.shortcuts import render


            def handle_create(request, form_class, success_message):
                return render(request, "form.html")


            def handle_edit(request, form_class, instance, success_message, redirect_url):
                return render(request, "form.html")


            def paginate(request, queryset, per_page=20):
                return queryset


            def book_stats():
                return {}


            def home(request):
                return render(request, "home.html")


            def by_year(request, year):
                return render(request, "year.html")


            def book_detail(request, book_id):
                return render(request, "book.html")


            def archive(request, year, *, page=1):
                return render(request, "archive.html")


            def post_by_slug(request, slug):
                return render(request, "post.html")
        ''',
    })
    if not parses:      # read by the line scanner
        with (tmp_path / "p/lib/views.py").open("a", encoding="utf-8") as handle:
            handle.write("\n\nbroken = (\n")
    index = SourceIndex(str(tmp_path))
    smap = django_map.extract(index, ["Book"])[0]
    assert ("p/lib/views.py" in index.syntax_errors) is not parses
    # Only request + URL kwargs (book_id, slug, a <int:year> some route passes) make a view.
    unrouted = [w for w in smap.warnings if "no route" in w]
    assert unrouted == ["p/lib/views.py: book_detail, archive, post_by_slug have no route in any urls.py."], \
        smap.warnings


def test_page_recipe_extends_only_a_base_template_that_exists(gui_ocl, gui_home):
    template = pytest.importorskip("django.template")
    # One base per entity (book.html, author.html...), each defining {% block content %}.
    page = _recipes(_only(gui_ocl))["Add a page or view"]
    found = re.search(r"extending its entity's base, e\.g\. '([^']+)' \(\{% block (\w+) %\}\)", page)
    assert found, page
    name, block = found.groups()
    base = (gui_ocl / APP / "templates" / name).read_text(encoding="utf-8")
    assert f"{{% block {block} %}}" in base and "{% extends" not in base
    # Following the recipe renders: the name resolves and the block is filled.
    engine = template.Engine(dirs=[str(gui_ocl / APP / "templates")])
    page_text = f"{{% extends '{name}' %}}{{% block {block} %}}<h1>About</h1>{{% endblock %}}"
    assert "<h1>About</h1>" in engine.from_string(page_text).render(template.Context({}))

    # Only home.html, which defines no block: nothing to extend, so nothing is named.
    page = _recipes(_only(gui_home))["Add a page or view"]
    assert "standalone template" in page and "extending" not in page and "<entity>" not in page
    assert sorted(os.listdir(gui_home / APP / "templates")) == ["home.html"]


def test_page_recipe_names_the_base_the_template_loader_finds(tmp_path):
    settings = '''
        from pathlib import Path

        BASE_DIR = Path(__file__).resolve().parent.parent
        INSTALLED_APPS = ["shop"]
        ROOT_URLCONF = "cfg.urls"
        TEMPLATES = [
            {
                "BACKEND": "django.template.backends.django.DjangoTemplates",
                "DIRS": [],
                "APP_DIRS": True,
            },
        ]
    '''
    _write(tmp_path, {
        "p/manage.py": _MANAGE.format(pkg="cfg"),
        "p/cfg/settings.py": settings,
        "p/cfg/urls.py": "from django.urls import path\n\nurlpatterns = []\n",
        "p/shop/models.py": "from django.db import models\n\n\nclass Product(models.Model):\n"
                            "    name = models.CharField()\n",
        "p/shop/templates/shop/base.html": "<html><body>{% block body %}{% endblock %}</body></html>\n",
        "p/shop/templates/shop/list.html": "{% extends 'shop/base.html' %}\n{% block body %}x{% endblock %}\n",
        # Project-level: searched only when TEMPLATES DIRS names a directory.
        "p/templates/layout.html": "<main>{% block content %}{% endblock %}</main>\n",
    })
    page = _recipes(_only(tmp_path, ["Product"]))["Add a page or view"]
    assert "extending 'shop/base.html' ({% block body %})" in page, page
    assert "layout.html" not in page

    path = tmp_path / "p/cfg/settings.py"
    path.write_text(path.read_text(encoding="utf-8").replace('"DIRS": []', '"DIRS": [BASE_DIR / "templates"]'),
                    encoding="utf-8")
    page = _recipes(_only(tmp_path, ["Product"]))["Add a page or view"]
    assert "extending 'layout.html' ({% block content %})" in page, page


def test_migrations_are_never_hand_written_when_none_ship(workspace):
    smap = _only(workspace)
    recipes = _recipes(smap)
    run = recipes["Run migrations"]
    # Phase 2 may have no run_command: a hand-written AddField with no 0001_initial under it breaks migrate.
    assert run.startswith("Never hand-write a migration: none ship"), run
    assert "With run_command: `python manage.py makemigrations myapp` then `migrate`" in run
    for intent in ("Add a field", "Add a new model"):
        assert "see Run migrations" in recipes[intent] and "makemigrations" not in recipes[intent]
    anchors = _anchors(smap)
    for anchor_id in ("django:add_field:book", "django:models_end:myapp"):
        assert "`makemigrations myapp` only with run_command, never hand-written" in anchors[anchor_id].hint

    # Migrations ship: a model change needs a matching one.
    (workspace / APP / "migrations" / "0001_initial.py").write_text(
        "from django.db import migrations\n\n\nclass Migration(migrations.Migration):\n    operations = []\n",
        encoding="utf-8")
    smap = _only(workspace)
    run = _recipes(smap)["Run migrations"]
    assert f"Migrations ship in {APP}/migrations/, so a model change needs a matching one." in run
    assert "without it, write the next migration file by hand" in run and "Never hand-write" not in run
    assert "a matching one, by `makemigrations myapp` with run_command" in _anchors(smap)["django:add_field:book"].hint

    # The migration is left out of the index by its file cap: it is still on disk, so still shipped.
    shallow = sum(1 for f in SourceIndex(str(workspace)).files if f.count("/") <= 3)
    index = SourceIndex(str(workspace), max_files=shallow)
    assert f"{APP}/migrations/0001_initial.py" not in index.files
    run = _recipes(django_map.extract(index, CLASS_NAMES)[0])["Run migrations"]
    assert "Migrations ship in" in run, run


def test_include_of_a_module_left_out_of_the_index_is_not_missing(tmp_path):
    _write(tmp_path, {
        "p/manage.py": _MANAGE.format(pkg="cfg"),
        "p/cfg/settings.py": "INSTALLED_APPS = ['lib']\nROOT_URLCONF = 'cfg.urls'\n",
        "p/cfg/urls.py": "from django.urls import include, path\n\n"
                         "urlpatterns = [\n    path('', include('lib.web.urls')),\n]\n",
        "p/lib/models.py": "from django.db import models\n\n\nclass Book(models.Model):\n"
                           "    title = models.CharField()\n",
        "p/lib/web/urls.py": "from django.urls import path\n\nurlpatterns = []\n",
    })
    shallow = sum(1 for f in SourceIndex(str(tmp_path)).files if f.count("/") <= 2)
    index = SourceIndex(str(tmp_path), max_files=shallow)
    assert "p/lib/web/urls.py" not in index.files
    smap = django_map.extract(index, ["Book"])[0]
    assert not [w for w in smap.warnings if "does not exist" in w], smap.warnings
    (tmp_path / "p/lib/web/urls.py").unlink()
    smap = _only(tmp_path, ["Book"])
    assert any("includes 'lib.web.urls', which does not exist" in w for w in smap.warnings), smap.warnings


_N = 20_000
# Inputs that made a pattern of this module backtrack quadratically (two quantifiers over the same
# characters, ``^\s*`` running over every blank line below it), and generic runs of its literals.
_PATHOLOGICAL = [
    "class a" + " " * _N + "!", "a) ->" + " " * _N + "!", ")->" * (_N // 3), "\n" * _N + "x",
    "include(" + " " * _N + "!", "from " + " " * _N + "!", "from " + "." * _N + "!", " " * _N + "!",
    "\t" * _N + "!", "a" * _N + "!", "a." * (_N // 2), "a(" * (_N // 2), "(" * _N, "'" + "a" * _N,
    "{% block " + " " * _N, "<" + "a" * _N, "@a" + " " * _N + "!", "a = " + " " * _N + "!",
    "'DIRS': [" + " " * _N, "def a(" + "a, " * (_N // 3),
]


def test_no_regex_backtracks_catastrophically():
    import time

    patterns = {name: value for name, value in vars(django_map).items() if isinstance(value, re.Pattern)}
    assert len(patterns) >= 25
    slow = []
    for name, pattern in sorted(patterns.items()):
        for text in _PATHOLOGICAL:
            start = time.perf_counter()
            pattern.match(text)
            for _ in pattern.finditer(text):
                pass
            elapsed = time.perf_counter() - start
            if elapsed > 0.2:
                slow.append((name, text[:12], round(elapsed, 2)))
                break       # one slow input condemns the pattern; the rest would only take longer
    assert not slow, slow


def test_pathological_files_are_mapped_quickly(tmp_path):
    import time

    lines = _N
    _write(tmp_path, {
        "p/manage.py": _MANAGE.format(pkg="cfg"),
        "p/cfg/urls.py": "from django.urls import path\n\nurlpatterns = []\n",
        "p/lib/models.py": "from django.db import models\n\n\nclass Book(models.Model):\n"
                           "    title = models.CharField()\n",
    })
    # Unparseable, so read as text: INSTALLED_APPS, then a long run of blank lines.
    (tmp_path / "p/cfg/settings.py").write_text(
        "INSTALLED_APPS = ['lib']\nROOT_URLCONF = 'cfg.urls'\n" + "\n" * lines + "x = (\n", encoding="utf-8")
    (tmp_path / "p/lib/views.py").write_text(
        "from django.shortcuts import render\n" + "\n" * lines
        + "class a" + " " * lines + "!\n" + "def a(request) ->" + " " * lines + "!\n", encoding="utf-8")
    (tmp_path / "p/lib/urls.py").write_text(
        "from . import views\nurlpatterns = [\n    path('x/', include(" + " " * lines + "!\n", encoding="utf-8")
    start = time.perf_counter()
    smap = _only(tmp_path, ["Book"])
    assert time.perf_counter() - start < 1.5
    assert "django:model:book" in _anchors(smap)
