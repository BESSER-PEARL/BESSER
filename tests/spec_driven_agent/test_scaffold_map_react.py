"""Scaffold map, React family: anchors over REAL ReactGenerator / WebAppGenerator output.

The map is computed from the files on disk, so the GUI scaffolds are generated
into temporary workspaces exactly as ``ToolExecutor._gen_web_app`` and
``_gen_react`` call the generators. Hand-written Vite apps cover the lenient
path (a frontend the model authored on the class-diagram path). Anchors are
asserted by id, never by absolute line numbers.
"""

from __future__ import annotations

import dataclasses
import importlib
import re
import shutil
import sys
import time
import types
from pathlib import Path

import pytest

from besser.BUML.metamodel.gui import (
    Button,
    ButtonActionType,
    ButtonType,
    DataBinding,
    Form,
    GUIModel,
    InputField,
    InputFieldType,
    Menu,
    MenuItem,
    Module,
    Screen,
    Text,
)
from besser.BUML.metamodel.gui.dashboard import BarChart, MetricCard, Series, Table
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DateType,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    FloatType,
    IntegerType,
    Method,
    Multiplicity,
    Parameter,
    Property,
    StringType,
)
from besser.spec_driven_agent.planning.scaffold_map import react as react_map
from besser.spec_driven_agent.planning.scaffold_map.model import (
    KIND_BLOCK,
    KIND_FUNCTION,
    KIND_INSERT,
    KIND_STUB,
    WorkspaceMap,
)
from besser.spec_driven_agent.planning.scaffold_map.render import render_overview
from besser.spec_driven_agent.planning.scaffold_map.source import SourceIndex

CLASSES = ["Library", "Book", "Author"]
FE = "web_app/frontend"


# ---------------------------------------------------------------------------
# Models and generated workspaces
# ---------------------------------------------------------------------------

def _models(extra: bool = False):
    """Library/Book/Author with an enum, two associations and a Book method, plus a 4-screen GUI.

    ``extra`` adds Publisher and Shelf (5 table screens) and a second unbound form:
    an ordinary model whose overview must still fit the shared prompt budget.
    """
    genre = Enumeration(name="Genre", literals={
        EnumerationLiteral(name="Fiction"), EnumerationLiteral(name="Science")})
    library, book, author = Class(name="Library"), Class(name="Book"), Class(name="Author")
    library.attributes = {Property(name="name", type=StringType), Property(name="address", type=StringType)}
    title, pages = Property(name="title", type=StringType), Property(name="pages", type=IntegerType)
    book.attributes = {title, pages, Property(name="release", type=DateType),
                       Property(name="genre", type=genre), Property(name="price", type=FloatType)}
    author.attributes = {Property(name="name", type=StringType), Property(name="email", type=StringType)}
    discount = Method(name="apply_discount", parameters=[Parameter(name="percent", type=IntegerType)],
                      type=FloatType,
                      code="def apply_discount(self, percent: int) -> float:\n"
                           "    self.price = self.price * (1 - percent / 100)\n    return self.price\n")
    book.methods = {discount}
    classes = [book, author, library]
    if extra:
        publisher, shelf = Class(name="Publisher"), Class(name="Shelf")
        publisher.attributes = {Property(name="name", type=StringType), Property(name="country", type=StringType)}
        shelf.attributes = {Property(name="code", type=StringType), Property(name="floor", type=IntegerType)}
        classes += [publisher, shelf]
    domain = DomainModel(name="Library_Model", types={*classes, genre}, associations={
        BinaryAssociation(name="lib_book", ends={
            Property(name="locatedIn", type=library, multiplicity=Multiplicity(1, 1)),
            Property(name="has", type=book, multiplicity=Multiplicity(0, "*"))}),
        BinaryAssociation(name="book_author", ends={
            Property(name="writtenBy", type=author, multiplicity=Multiplicity(1, "*")),
            Property(name="publishes", type=book, multiplicity=Multiplicity(0, "*"))})})

    screens, tables = [], {}
    for idx, cls in enumerate(classes):
        table = Table(name=f"{cls.name.lower()}_table", title=f"{cls.name} list",
                      data_binding=DataBinding(domain_concept=cls), component_id=f"{cls.name.lower()}-table",
                      action_buttons=True, show_pagination=True)
        tables[cls.name] = table
        screens.append(Screen(name=cls.name, description=f"{cls.name} page",
                              view_elements={Text(name=f"{cls.name.lower()}_title", content=f"{cls.name}s"), table},
                              route_path=f"/{cls.name.lower()}", is_main_page=idx == 0))
    screens[0].view_elements = set(screens[0].view_elements) | {
        Button(name="discount_btn", description="", label="Apply discount", buttonType=ButtonType.RaisedButton,
               actionType=ButtonActionType.RunMethod, method_btn=discount, is_instance_method=True,
               instance_source=tables["Book"]),
        Button(name="new_book_btn", description="", label="New book", buttonType=ButtonType.RaisedButton,
               actionType=ButtonActionType.Create, entity_class=book),
    }
    dashboard = Screen(name="Dashboard", description="Dashboard", route_path="/dashboard", view_elements={
        Menu(name="main_menu", description="nav", menuItems={MenuItem(label="Books", url="/book")}),
        BarChart(name="pages_chart", title="Pages per book", series=[
            Series(name="pages", label="Pages", styling=None,
                   data_binding=DataBinding(domain_concept=book, label_field=title, data_field=pages))]),
        MetricCard(name="book_count", metric_title="Books",
                   data_binding=DataBinding(domain_concept=book, data_field=pages)),
        Form(name="contact_form", description="contact", title="Contact", inputFields={
            InputField(name="email", description="", field_type=InputFieldType.Email, label="Email")}),
    })
    if extra:
        dashboard.view_elements = set(dashboard.view_elements) | {
            Form(name="feedback_form", description="feedback", title="Feedback", inputFields={
                InputField(name="message", description="", field_type=InputFieldType.Text, label="Message")})}
    screens.append(dashboard)
    gui = GUIModel(name="LibraryApp", package="com.lib", versionCode="1", versionName="1.0",
                   modules={Module(name="AppModule", screens=set(screens))}, description="Library app")
    return domain, gui


@pytest.fixture(scope="module")
def models():
    return _models()


@pytest.fixture(scope="module")
def web_app(tmp_path_factory, models) -> Path:
    """A workspace holding generate_web_app's output (== ToolExecutor._gen_web_app)."""
    from besser.generators.web_app import WebAppGenerator

    workspace = tmp_path_factory.mktemp("react_web_app")
    domain, gui = models
    WebAppGenerator(model=domain, gui_model=gui, output_dir=str(workspace / "web_app"),
                    agent_model=None, agent_config=None).generate()
    assert (workspace / FE / "src" / "App.tsx").is_file()
    return workspace


@pytest.fixture(scope="module")
def web_map(web_app):
    maps = react_map.extract(SourceIndex(str(web_app)), CLASSES)
    assert len(maps) == 1
    return maps[0]


def _anchors(smap) -> dict:
    return {a.id: a for a in smap.all_anchors()}


def _assert_quotes_valid(workspace: Path, anchors) -> None:
    """Every quote is one verbatim line, unique in its file, inside its range (or the insertion line)."""
    for anchor in anchors:
        lines = (workspace / anchor.file).read_text(encoding="utf-8").splitlines()
        assert 1 <= anchor.start_line <= anchor.end_line <= max(1, len(lines)), anchor
        if not anchor.quote:
            continue
        assert lines.count(anchor.quote) == 1, anchor
        # modify_file matches old_text as a substring: unique in the whole text.
        assert (workspace / anchor.file).read_text(encoding="utf-8").count(anchor.quote) == 1, (anchor.id, anchor.quote)
        line = lines.index(anchor.quote) + 1
        if anchor.kind == KIND_INSERT:
            assert line == anchor.start_line == anchor.end_line, anchor
        else:
            assert anchor.start_line <= line <= anchor.end_line, anchor


_NAMED_ID_RE = re.compile(r"\breact(?:@\w+)?:[\w.#:-]*[\w#]")


def _assert_named_ids_exist(maps) -> None:
    """Every anchor id a recipe or a warning names was emitted (no dangling ids)."""
    ids = {a.id for m in maps for a in m.all_anchors()}
    for smap in maps:
        for text in smap.warnings + [" ".join(r.steps) for r in smap.recipes]:
            for named in _NAMED_ID_RE.findall(text):
                assert named in ids, (named, text)


def _section(overview: str, title: str) -> str:
    """One map's section of a rendered overview."""
    start = overview.index(f"### {title}")
    end = overview.find("\n### ", start + 4)
    return overview[start:end if end >= 0 else len(overview)]


def _write(root: Path, files: dict[str, str | bytes]) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# generate_web_app: route table, pages, blocks
# ---------------------------------------------------------------------------

def test_web_app_frontend_is_one_gui_scaffold_root(web_map):
    assert web_map.family == "react"
    assert web_map.root == FE
    assert "GUI scaffold" in web_map.title
    assert all(a.id.startswith("react:") for a in web_map.all_anchors())


def test_route_table_entry_and_insertion_points(web_app, web_map):
    anchors = _anchors(web_map)
    routes = anchors["react:routes"]
    assert routes.kind == KIND_BLOCK and routes.file == f"{FE}/src/App.tsx"
    assert routes.quote.strip() == "<Routes>"
    app_lines = (web_app / routes.file).read_text(encoding="utf-8").splitlines()
    assert app_lines[routes.end_line - 1].strip() == "</Routes>"

    # New routes go before the '/' redirect, which precedes the '*' catch-all.
    insert = anchors["react:route_insert"]
    assert insert.kind == KIND_INSERT and insert.start_line == insert.end_line
    assert insert.quote.strip().startswith('<Route path="/" element={<Navigate')
    assert "BEFORE" in insert.hint
    assert routes.start_line < insert.start_line < routes.end_line

    imports = anchors["react:page_imports"]
    assert imports.kind == KIND_INSERT and imports.quote.startswith("import ") and "./pages/" in imports.quote
    assert "AFTER" in imports.hint and "./pages/NewPage" in imports.hint

    entry = anchors["react:entry"]
    assert entry.file == f"{FE}/src/index.tsx"
    assert entry.quote.strip() == "<App />"
    assert "BrowserRouter" in entry.label

    routes_line = next(line for line in web_map.summary if line.startswith("routes:"))
    for fragment in ("/book BookPage", "/author AuthorPage", "/library LibraryPage", "/dashboard Dashboard",
                     "/ -> /book", "* -> /book"):
        assert fragment in routes_line


def test_routes_resolve_to_page_files_linked_through_bound_blocks(web_map):
    anchors = _anchors(web_map)
    page = anchors["react:page:BookPage"]
    assert page.kind == KIND_FUNCTION
    assert page.file == f"{FE}/src/pages/BookPage.tsx"
    assert page.quote == "const BookPage: React.FC = () => {"
    assert "/book" in page.label
    assert 'new JSX goes right after the root line `<div id="Book">`' in page.hint
    # A page bound to one entity belongs to that entity's map, not the app-level list.
    assert page.entity == "Book"
    assert page in web_map.entities["Book"].anchors
    assert page not in web_map.anchors
    assert anchors["react:page:AuthorPage"].entity == "Author"
    assert anchors["react:page:LibraryPage"].entity == "Library"


def test_bound_runtime_blocks_carry_entity_and_detail(web_map):
    book = web_map.entities["Book"]
    by_id = {a.id: a for a in book.anchors}
    table = by_id["react:block:book:BookPage.book-table"]
    assert table.kind == KIND_BLOCK and table.label.startswith("TableBlock book-table")
    assert table.quote.strip().startswith('<TableBlock id="book-table"')
    assert "never in the runtime component" in table.hint
    assert table.label.endswith("endpoint /book/")
    method = by_id["react:block:book:BookPage.discount_btn"]
    assert method.label.startswith("MethodButton apply_discount")
    assert method.label.endswith("endpoint /book/{book_id}/methods/apply_discount/")
    assert by_id["react:block:book:BookPage.new_book_btn"].label.startswith("CrudButton create")
    # The dashboard's chart and metric card read Book: the dashboard is linked to Book too.
    assert "react:block:book:Dashboard.pages_chart" in by_id
    assert "react:block:book:Dashboard.book_count" in by_id
    assert "BookPage.tsx /book (TableBlock, MethodButton apply_discount, CrudButton create)" in book.summary
    # BookPage's lookup columns name Library and Author: that must not bind the page to them.
    for name in ("Library", "Author"):
        assert all("BookPage" not in a.id for a in web_map.entities[name].anchors)


def test_react_never_reports_missing_pieces(web_app):
    # Publisher is in the model but has no screen: pages follow the GUI, so nothing is "missing".
    maps = react_map.extract(SourceIndex(str(web_app)), CLASSES + ["Publisher"])
    assert all(not entity.missing for m in maps for entity in m.entities.values())
    assert "Publisher" not in maps[0].entities


def test_unbound_form_is_a_stub_and_a_pitfall(web_map):
    stub = _anchors(web_map)["react:form:Dashboard.contact_form"]
    assert stub.kind == KIND_STUB and stub.status == "no-op onSubmit"
    assert stub.file == f"{FE}/src/pages/Dashboard.tsx"
    assert "e.preventDefault()" in stub.quote
    assert any("Dashboard.tsx" in w and "preventDefault" in w and "Phase 3" in w for w in web_map.warnings)


def test_pristine_runtime_api_base_is_a_fact_not_a_pitfall(web_map):
    # The runtime blocks each declare the base: that is the scaffold's convention, in files
    # nobody edits, so it is stated in the summary and the recipe, never as a pitfall.
    assert not any(a.id in ("react:api_client", "react:api_base") for a in web_map.anchors)
    assert not any("API base" in w for w in web_map.warnings)
    data = next(line for line in web_map.summary if line.startswith("data:"))
    assert "VITE_API_URL || http://localhost:8000" in data and "no shared client" in data
    call = " ".join(next(r for r in web_map.recipes if r.intent == "Call a backend endpoint").steps)
    assert 'import.meta.env.VITE_API_URL || "http://localhost:8000"' in call
    assert f"{FE}/src/api/client.ts" in call


def test_entity_order_does_not_depend_on_class_name_order(web_app, web_map):
    # The model's classes come from a set: the map must not follow their (hash) order.
    for names in (list(reversed(CLASSES)), ["Book", "Library", "Author"]):
        other = react_map.extract(SourceIndex(str(web_app)), names)[0]
        assert dataclasses.asdict(other) == dataclasses.asdict(web_map)
    assert list(web_map.entities) == sorted(web_map.entities)


def test_runtime_files_are_do_not_edit_and_exist(web_app, web_map):
    assert f"{FE}/src/components/runtime/*" in web_map.do_not_edit
    assert f"{FE}/src/components/table/*" in web_map.do_not_edit
    assert f"{FE}/src/components/MethodButton.tsx" in web_map.do_not_edit
    assert f"{FE}/src/contexts/TableContext.tsx" in web_map.do_not_edit
    for entry in web_map.do_not_edit:
        path = web_app / entry.rstrip("*").rstrip("/")
        assert path.exists(), entry
    # Nothing in the runtime is mapped as a page or a block.
    assert not any("/components/" in a.file for a in web_map.all_anchors())


def test_config_theme_and_recipes(web_map):
    anchors = _anchors(web_map)
    assert anchors["react:theme"].quote == '@import "tailwindcss";'
    assert "Tailwind v4" in anchors["react:theme"].label
    assert anchors["react:dockerfile"].quote == "RUN npm run build"
    assert anchors["react:package_json"].quote.strip() == '"dependencies": {'
    assert "dev port 3000" in anchors["react:vite_config"].label
    intents = [r.intent for r in web_map.recipes]
    assert intents == ["Add a page", "Add a field to a table or form", "Call a backend endpoint",
                       "Add auth / login", "Change the theme"]
    add_page = " ".join(web_map.recipes[0].steps)
    assert f"{FE}/src/pages/<Name>.tsx" in add_page
    assert "react:page_imports" in add_page and "react:route_insert" in add_page
    assert "no global nav" in add_page
    theme = " ".join(web_map.recipes[-1].steps)
    assert "src/App.css is imported nowhere" in theme      # dead CSS the scaffold ships
    for recipe in web_map.recipes:
        assert not re.search(r"\bline \d+|\.(?:tsx?|jsx?|css|json):\d+", " ".join(recipe.steps)), recipe


def test_every_quote_is_unique_and_in_range(web_app, web_map):
    _assert_quotes_valid(web_app, web_map.all_anchors())
    _assert_named_ids_exist([web_map])
    ids = [a.id for a in web_map.all_anchors()]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"react:[a-z_]+(:[\w.#-]+)*", i) for i in ids), ids


def test_map_is_deterministic(web_app, tmp_path):
    first = react_map.extract(SourceIndex(str(web_app)), CLASSES)
    shutil.copytree(web_app / "web_app", tmp_path / "web_app")
    second = react_map.extract(SourceIndex(str(tmp_path)), CLASSES)
    assert [dataclasses.asdict(m) for m in first] == [dataclasses.asdict(m) for m in second]


# ---------------------------------------------------------------------------
# generate_react next to generate_web_app: two roots, ids stay unique
# ---------------------------------------------------------------------------

def test_react_generator_root_next_to_web_app(web_app, models, tmp_path):
    from besser.generators.react import ReactGenerator

    shutil.copytree(web_app / "web_app", tmp_path / "web_app")
    domain, gui = models
    ReactGenerator(model=domain, gui_model=gui, output_dir=str(tmp_path / "react")).generate()

    maps = react_map.extract(SourceIndex(str(tmp_path)), CLASSES)
    assert [m.root for m in maps] == ["react", FE]
    standalone, web = maps
    assert "react@react:routes" in _anchors(standalone)
    assert "react@web_app_frontend:routes" in _anchors(web)
    ids = [a.id for m in maps for a in m.all_anchors()]
    assert len(ids) == len(set(ids))
    # Byte-identical scaffolds except the web_app Dockerfile.
    assert "react@react:dockerfile" not in _anchors(standalone)
    assert "react@web_app_frontend:dockerfile" in _anchors(web)
    assert any("react@react:route_insert" in step for step in standalone.recipes[0].steps)
    _assert_quotes_valid(tmp_path, [a for m in maps for a in m.all_anchors()])
    _assert_named_ids_exist(maps)


def test_text_only_gui_flags_the_runtime_file_with_a_missing_import(models, tmp_path):
    """No Table in the GUI: TableBlock.tsx ships without table/TableComponent (a verified build break)."""
    from besser.generators.react import ReactGenerator

    domain, _ = models
    screen = Screen(name="Dashboard", description="Main dashboard", is_main_page=True,
                    view_elements={Text(name="t", content="Welcome")})
    gui = GUIModel(name="E", package="", versionCode="1", versionName="1", description="",
                   modules={Module(name="M", screens={screen})})
    ReactGenerator(model=domain, gui_model=gui, output_dir=str(tmp_path / "react")).generate()

    smap = react_map.extract(SourceIndex(str(tmp_path)), CLASSES)[0]
    assert any(w.startswith("react/src/components/runtime/TableBlock.tsx imports '../table/TableComponent'")
               and "delete it" in w for w in smap.warnings)
    page = _anchors(smap)["react:page:MainDashboard"]
    assert page.entity is None and page in smap.anchors
    assert not smap.entities
    _assert_quotes_valid(tmp_path, smap.all_anchors())


# ---------------------------------------------------------------------------
# Lenient: frontends the model wrote on the class-diagram path
# ---------------------------------------------------------------------------

_JSX_APP = {
    "backend/main_api.py": "app = None\n",
    "frontend/package.json": '{"dependencies": {"react": "^18.2.0", "react-dom": "^18.2.0", '
                             '"react-router-dom": "^6.20.0"}}\n',
    "frontend/index.html": '<div id="root"></div><script type="module" src="/src/main.jsx"></script>\n',
    "frontend/src/main.jsx": (
        "import React from 'react';\n"
        "import ReactDOM from 'react-dom/client';\n"
        "import { BrowserRouter } from 'react-router-dom';\n"
        "import App from './App';\n"
        "import './index.css';\n\n"
        "ReactDOM.createRoot(document.getElementById('root')).render(\n"
        "  <BrowserRouter>\n"
        "    <App />\n"
        "  </BrowserRouter>\n"
        ");\n"),
    "frontend/src/index.css": "body { margin: 0; }\n",
    "frontend/src/App.jsx": (
        "import { Routes, Route, NavLink } from 'react-router-dom';\n"
        "import Home from './pages/Home';\n"
        "import BookList from './pages/BookList';\n"
        "import AuthorList from './pages/AuthorList';\n\n"
        "// <Route path=\"/commented\" element={<Nope />} /> is not a route\n"
        "export default function App() {\n"
        "  return (\n"
        "    <div>\n"
        "      <nav><NavLink to=\"/books\">Books</NavLink></nav>\n"
        "      <Routes>\n"
        "        <Route path=\"/\" element={<Home />} />\n"
        "        <Route path=\"/books\" element={<BookList />} />\n"
        "        <Route path=\"/authors\" element={<AuthorList />} />\n"
        "        <Route path=\"*\" element={<p>Not found</p>} />\n"
        "      </Routes>\n"
        "    </div>\n"
        "  );\n"
        "}\n"),
    "frontend/src/api.js": (
        "export const API_BASE = 'http://localhost:8000';\n\n"
        "export async function listBooks() {\n"
        "  const res = await fetch(`${API_BASE}/book/`);\n"
        "  return res.json();\n"
        "}\n"),
    "frontend/src/pages/Home.jsx": (
        "function Home() {\n"
        "  return <h1>Don't panic: it's {\"fine\"}</h1>;\n"
        "}\n\n"
        "export default Home;\n"),
    "frontend/src/pages/BookList.jsx": (
        "import { useEffect, useState } from 'react';\n"
        "import { listBooks } from '../api';\n"
        "import EntityTable from '../components/EntityTable';\n\n"
        "export default function BookList() {\n"
        "  const [books, setBooks] = useState([]);\n"
        "  useEffect(() => { listBooks().then(setBooks); }, []);\n"
        "  return (\n"
        "    <section id=\"books\">\n"
        "      <EntityTable entity=\"book\" endpoint=\"/book/\" rows={books} />\n"
        "    </section>\n"
        "  );\n"
        "}\n"),
    "frontend/src/components/EntityTable.jsx": (
        "export default function EntityTable({ rows }) {\n"
        "  return <table>{rows.map((r) => <tr key={r.id}><td>{r.id}</td></tr>)}</table>;\n"
        "}\n"),
    "frontend/src/pages/AuthorList.jsx": (
        "const AuthorList = () => {\n"
        "  return (\n"
        "    <form id=\"author-form\" onSubmit={(e) => e.preventDefault()}>\n"
        "      <input name=\"name\" />\n"
        "    </form>\n"
        "  );\n"
        "};\n\n"
        "export default AuthorList;\n"),
    "frontend/src/pages/Orphan.jsx": "export default function Orphan() {\n  return <div>orphan</div>;\n}\n",
}


def test_lenient_jsx_router_app(tmp_path):
    workspace = _write(tmp_path, _JSX_APP)
    maps = react_map.extract(SourceIndex(str(workspace)), ["Book", "Author"])
    assert len(maps) == 1
    smap = maps[0]
    anchors = _anchors(smap)
    assert smap.root == "frontend" and smap.title == "React frontend"
    assert not smap.do_not_edit                       # no runtime convention to protect
    assert anchors["react:entry"].file == "frontend/src/main.jsx"   # from index.html's script
    assert anchors["react:route_insert"].quote.strip() == '<Route path="*" element={<p>Not found</p>} />'
    assert anchors["react:page:Home"].quote == "function Home() {"
    assert anchors["react:page:Home"].end_line == 3   # the apostrophe in JSX text does not derail the span
    # A model-written component with literal entity/endpoint props binds its page; the
    # model's spelling wins over the code's ('book').
    book_page = anchors["react:page:BookList"]
    assert book_page.entity == "Book" and book_page in smap.entities["Book"].anchors
    assert "react:block:book:BookList.EntityTable1" in anchors
    # The shared API module is the place for new calls.
    client = anchors["react:api_client"]
    assert client.file == "frontend/src/api.js"
    assert client.quote == "export const API_BASE = 'http://localhost:8000';"
    assert "listBooks" in client.label
    assert any("nav in src/App.jsx" in line for line in smap.summary)   # summary paths are root-relative
    assert "link it from frontend/src/App.jsx" in " ".join(smap.recipes[0].steps)
    assert any("Orphan.jsx has no <Route>" in w for w in smap.warnings)
    assert any("AuthorList.jsx: form author-form" in w for w in smap.warnings)
    assert not any("/commented" in line for line in smap.summary)
    _assert_quotes_valid(workspace, smap.all_anchors())
    _assert_named_ids_exist([smap])


def test_lenient_object_router_without_app_file(tmp_path):
    workspace = _write(tmp_path, {
        "web/package.json": '{"dependencies": {"react": "18", "react-dom": "18"}}',
        "web/src/main.tsx": (
            'import ReactDOM from "react-dom/client";\n'
            'import { createBrowserRouter, RouterProvider, Navigate } from "react-router-dom";\n'
            'import Root from "./routes/Root";\n'
            'import Books from "./routes/Books";\n\n'
            "const router = createBrowserRouter([\n"
            "  {\n"
            '    path: "/",\n'
            "    element: <Root />,\n"
            "    children: [\n"
            "      { index: true, element: <Books /> },\n"
            '      { path: "books/:id", element: <Books /> },\n'
            "    ],\n"
            "  },\n"
            '  { path: "*", element: <Navigate to="/" /> },\n'
            "]);\n\n"
            'ReactDOM.createRoot(document.getElementById("root")!).render(<RouterProvider router={router} />);\n'),
        "web/src/routes/Root.tsx": (
            'import { Outlet, Link } from "react-router-dom";\n\n'
            "export default function Root() {\n"
            "  return (\n"
            '    <div><nav><Link to="/">Books</Link></nav><Outlet /></div>\n'
            "  );\n"
            "}\n"),
        "web/src/routes/Books.tsx": (
            'import axios from "axios";\n\n'
            'const api = axios.create({ baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000" });\n\n'
            "export default function Books() {\n"
            '  return <ul id="books"></ul>;\n'
            "}\n"),
    })
    smap = react_map.extract(SourceIndex(str(workspace)), [])[0]
    anchors = _anchors(smap)
    assert anchors["react:routes"].quote == "const router = createBrowserRouter(["
    insert = anchors["react:route_insert"]
    assert insert.quote.strip() == '{ path: "*", element: <Navigate to="/" /> },'
    # A JS array of route objects: the hint and the recipe never prescribe a JSX <Route> line there.
    assert '{ path: "/x", element: <Page /> },' in insert.hint and "<Route" not in insert.hint
    assert "add its route object at react:route_insert" in smap.recipes[0].steps[1]
    assert anchors["react:page:Books"].file == "web/src/routes/Books.tsx"
    # A page's private axios instance is not a shared client; its base is still mapped.
    assert "react:api_client" not in anchors
    assert anchors["react:api_base"].file == "web/src/routes/Books.tsx"
    assert any("pages: src/routes/*" in line for line in smap.summary)
    assert "web/src/routes/<Name>.tsx" in smap.recipes[0].steps[0]
    # The index route shares '/' with its layout parent: not a duplicate path.
    assert not any("declares path" in w for w in smap.warnings)
    assert not smap.entities                          # never 'missing', even with no class names
    # The child route joins its parent's path.
    assert any("/books/:id Books" in line for line in smap.summary if line.startswith("routes:"))
    _assert_quotes_valid(workspace, smap.all_anchors())
    _assert_named_ids_exist([smap])


def test_malformed_and_unreadable_files_do_not_raise(tmp_path):
    workspace = _write(tmp_path, {
        "app/package.json": '{ "dependencies": { "react": "^19" BROKEN JSON',
        "app/src/App.tsx": (
            'import React from "react";\n'
            'import Foo from "./pages/Foo"\n'
            "function App() {\n"
            "  return (\n"
            "    <Routes>\n"
            '      <Route path="/a" element={<Foo />\n'
            "      <Route path=\n"
            "    </Routes\n"
            '  {{{ ))) ]]] "unterminated\n'
            "  `template ${\n"),
        "app/src/index.tsx": 'import App from "./App";\nReactDOM.createRoot(x).render(<App />\n',
        "app/src/pages/Broken.tsx": "export default function Broken( {\n  return <div id='x' ;\n",
        "app/src/pages/Empty.tsx": "",
        "app/src/binary.tsx": b"\xff\xfe\x00not utf-8 \xc3\x28",
    })
    maps = react_map.extract(SourceIndex(str(workspace)), ["Foo"])
    assert len(maps) == 1                              # detected through the entry's React import
    smap = maps[0]
    assert "react:routes" in _anchors(smap)
    assert any("imports './pages/Foo', which does not exist" in w for w in smap.warnings)
    _assert_quotes_valid(workspace, smap.all_anchors())
    # The truncated <Routes> has no insertion line: the recipe falls back to the table itself.
    assert "react:route_insert" not in _anchors(smap)
    assert "add its <Route> at react:routes" in " ".join(smap.recipes[0].steps)
    _assert_named_ids_exist([smap])
    assert render_overview(WorkspaceMap(maps=maps))


def _jsx_shell(root: str, app: str) -> dict[str, str]:
    """package.json, index.html and a main.jsx rendering ``App`` inside a BrowserRouter."""
    return {
        f"{root}/package.json": '{"dependencies": {"react": "^18.2.0", "react-dom": "^18.2.0", '
                                '"react-router-dom": "^6.20.0"}}\n',
        f"{root}/index.html": '<div id="root"></div><script type="module" src="/src/main.jsx"></script>\n',
        f"{root}/src/main.jsx": (
            "import ReactDOM from 'react-dom/client';\n"
            "import { BrowserRouter } from 'react-router-dom';\n"
            "import App from './App';\n\n"
            "ReactDOM.createRoot(document.getElementById('root')).render(\n"
            "  <BrowserRouter>\n"
            "    <App />\n"
            "  </BrowserRouter>\n"
            ");\n"),
        f"{root}/src/App.jsx": app,
    }


_EDITED_APP = {
    # What Phase 2 typically leaves: auth guards, a NavBar component, named exports, local API bases.
    **_jsx_shell("fe", (
        "import { Routes, Route, Navigate } from 'react-router-dom';\n"
        "import NavBar from './components/NavBar';\n"
        "import RequireAuth from './auth/RequireAuth';\n"
        "import Home from './pages/Home';\n"
        "import Login from './pages/Login';\n"
        "import { BookList } from './pages/books/List';\n"
        "import { ReportsPage } from './pages/ReportsPage';\n\n"
        "const isLoggedIn = () => Boolean(localStorage.getItem('token'));\n\n"
        "export default function App() {\n"
        "  return (\n"
        "    <div>\n"
        "      <NavBar />\n"
        "      <Routes>\n"
        "        <Route path=\"/\" element={<Home />} />\n"
        "        <Route path=\"/login\" element={<Login />} />\n"
        "        <Route path=\"/reports\"\n"
        "          element={isLoggedIn() ? <ReportsPage /> : <Navigate to=\"/login\" replace />} />\n"
        "        <Route element={<RequireAuth />}>\n"
        "          <Route path=\"/books\" element={<BookList />} />\n"
        "        </Route>\n"
        "      </Routes>\n"
        "    </div>\n"
        "  );\n"
        "}\n")),
    "fe/src/components/NavBar.jsx": (
        "import { NavLink } from 'react-router-dom';\n\n"
        "export default function NavBar() {\n"
        "  return <nav><NavLink to=\"/books\">Books</NavLink></nav>;\n"
        "}\n"),
    "fe/src/auth/RequireAuth.jsx": (
        "import { Navigate, Outlet } from 'react-router-dom';\n\n"
        "export default function RequireAuth() {\n"
        "  return localStorage.getItem('token') ? <Outlet /> : <Navigate to=\"/login\" replace />;\n"
        "}\n"),
    "fe/src/pages/Home.jsx": "export default function Home() {\n  return <h1>Home</h1>;\n}\n",
    "fe/src/pages/Login.jsx": (
        "const API_BASE = 'http://localhost:8000';\n\n"
        "export default function Login() {\n"
        "  return <button onClick={() => fetch(`${API_BASE}/token`)}>Log in</button>;\n"
        "}\n"),
    "fe/src/pages/books/List.jsx": (
        "const COLUMNS = ['title', 'pages'];\n"
        "const PAGE_SIZE = 20;\n\n"
        "export function BookList() {\n"
        "  return (\n"
        "    <table id=\"books\">\n"
        "      <thead><tr>{COLUMNS.map((c) => <th key={c}>{c}</th>)}</tr></thead>\n"
        "    </table>\n"
        "  );\n"
        "}\n"),
    "fe/src/pages/ReportsPage.jsx": (
        "const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';\n\n"
        "export const ReportsPage = () => {\n"
        "  return <div id=\"reports\">{API_BASE}</div>;\n"
        "};\n"),
    # Date helpers in src/lib: not an API client.
    "fe/src/lib/format.js": "export function formatDate(d) {\n  return new Date(d).toLocaleDateString();\n}\n",
}


def test_edited_app_named_exports_guards_layouts_and_nav(tmp_path):
    workspace = _write(tmp_path, _EDITED_APP)
    smap = react_map.extract(SourceIndex(str(workspace)), ["Book"])[0]
    anchors = _anchors(smap)
    # A named export below constants is the page, not the first capitalised constant.
    book_list = anchors["react:page:BookList"]
    assert book_list.file == "fe/src/pages/books/List.jsx" and book_list.quote == "export function BookList() {"
    assert "/books" in book_list.label and book_list.start_line > 1
    assert not any(":page:COLUMNS" in i or ":page:API_BASE" in i for i in anchors)
    assert "/reports" in anchors["react:page:ReportsPage"].label
    # `cond ? <Page /> : <Navigate />` guards a page; it is not a redirect.
    routes = next(line for line in smap.summary if line.startswith("routes:"))
    assert "/reports ReportsPage (guarded)" in routes and "/reports -> " not in routes
    # A pathless guard route is a layout, never a page "at ''" or an index route.
    assert "(layout) RequireAuth" in routes and "(index)" not in routes and "  " not in routes
    assert not any("RequireAuth" in i for i in anchors)
    assert not any("has no <Route>" in w for w in smap.warnings)
    # The nav lives in a component App.jsx renders, one import away.
    assert any("nav in src/components/NavBar.jsx" in line for line in smap.summary)
    assert "link it from fe/src/components/NavBar.jsx" in " ".join(smap.recipes[0].steps)
    # src/lib/format.js talks no HTTP; two editable pages declare their own base: that is the pitfall.
    assert "react:api_client" not in anchors
    assert any(w.startswith("2 editable files declare their own API base (fe/src/pages/Login.jsx, "
                            "fe/src/pages/ReportsPage.jsx)") for w in smap.warnings)
    _assert_quotes_valid(workspace, smap.all_anchors())
    _assert_named_ids_exist([smap])


_NESTED_OBJECT_ROUTER = {
    "frontend/package.json": '{"dependencies": {"react": "^18.2.0", "react-dom": "^18.2.0"}}',
    "frontend/src/main.jsx": (
        'import ReactDOM from "react-dom/client";\n'
        'import { createBrowserRouter, RouterProvider } from "react-router-dom";\n'
        'import Layout from "./pages/Layout";\n'
        'import BookList from "./pages/BookList";\n'
        'import BookNew from "./pages/BookNew";\n'
        'import AuthorList from "./pages/AuthorList";\n'
        'import AuthorNew from "./pages/AuthorNew";\n\n'
        "const router = createBrowserRouter([\n"
        "  {\n"
        '    path: "/",\n'
        "    element: <Layout />,\n"
        "    children: [\n"
        "      { index: true, element: <BookList /> },\n"
        "      {\n"
        '        path: "books",\n'
        "        children: [\n"
        "          { index: true, element: <BookList /> },\n"
        '          { path: "new", element: <BookNew /> },\n'
        "        ],\n"
        "      },\n"
        "      {\n"
        '        path: "authors",\n'
        "        children: [\n"
        "          { index: true, element: <AuthorList /> },\n"
        '          { path: "new", element: <AuthorNew /> },\n'
        "        ],\n"
        "      },\n"
        "    ],\n"
        "  },\n"
        "]);\n\n"
        'ReactDOM.createRoot(document.getElementById("root")).render(<RouterProvider router={router} />);\n'),
    "frontend/src/pages/Layout.jsx": (
        'import { Outlet, Link } from "react-router-dom";\n'
        "export default function Layout() {\n"
        '  return <div><nav><Link to="/books">Books</Link></nav><Outlet /></div>;\n'
        "}\n"),
    **{f"frontend/src/pages/{name}.jsx": (f"export default function {name}() {{\n"
                                          f"  return <div id=\"{name}\" />;\n}}\n")
       for name in ("BookList", "BookNew", "AuthorList", "AuthorNew")},
}


def test_nested_object_routes_join_their_parents_paths(tmp_path):
    workspace = _write(tmp_path, _NESTED_OBJECT_ROUTER)
    smap = react_map.extract(SourceIndex(str(workspace)), ["Book", "Author"])[0]
    anchors = _anchors(smap)
    # '/books/new' and '/authors/new', not two '/new' routes.
    assert not any("declares path" in w for w in smap.warnings)
    assert "/books/new" in anchors["react:page:BookNew"].label
    assert "/authors/new" in anchors["react:page:AuthorNew"].label
    assert anchors["react:page:BookList"].label.startswith("page BookList at /, /books")
    routes = next(line for line in smap.summary if line.startswith("routes:"))
    assert "/books/new BookNew" in routes and "(inline JSX)" not in routes   # grouping routes are not listed
    # No one-line catch-all: new route objects go before the array's closing ']'.
    insert = anchors["react:route_insert"]
    assert insert.quote == "]);" and "{ path:" in insert.hint and "<Route" not in insert.hint
    assert any("nav in src/pages/Layout.jsx" in line for line in smap.summary)
    _assert_quotes_valid(workspace, smap.all_anchors())
    _assert_named_ids_exist([smap])


def test_use_routes_with_an_array_declared_apart(tmp_path):
    workspace = _write(tmp_path, {
        **_jsx_shell("fe", (
            'import { useRoutes } from "react-router-dom";\n'
            'import Home from "./pages/Home";\n'
            'import Books from "./pages/Books";\n\n'
            "const routes = [\n"
            '  { path: "/", element: <Home /> },\n'
            '  { path: "/books", element: <Books /> },\n'
            '  { path: "*", element: <Home /> },\n'
            "];\n\n"
            "export default function App() {\n"
            "  return useRoutes(routes);\n"
            "}\n")),
        "fe/src/pages/Home.jsx": "export default function Home() {\n  return <h1>Home</h1>;\n}\n",
        "fe/src/pages/Books.jsx": "export default function Books() {\n  return <ul />;\n}\n",
    })
    smap = react_map.extract(SourceIndex(str(workspace)), [])[0]
    anchors = _anchors(smap)
    assert anchors["react:routes"].quote == "const routes = ["
    assert anchors["react:route_insert"].quote.strip() == '{ path: "*", element: <Home /> },'
    assert "/books" in anchors["react:page:Books"].label
    assert not any("has no <Route>" in w for w in smap.warnings)
    _assert_quotes_valid(workspace, smap.all_anchors())
    _assert_named_ids_exist([smap])


def test_roots_whose_slugs_coincide_keep_distinct_ids(tmp_path):
    app = ("import { Routes, Route } from 'react-router-dom';\n"
           "import Home from './pages/Home';\n\n"
           "export default function App() {\n"
           "  return (\n"
           "    <Routes>\n"
           "      <Route path=\"/\" element={<Home />} />\n"
           "    </Routes>\n"
           "  );\n"
           "}\n")
    home = "export default function Home() {\n  return <h1>Home</h1>;\n}\n"
    workspace = _write(tmp_path, {**_jsx_shell("front/end", app), **_jsx_shell("front_end", app),
                                  "front/end/src/pages/Home.jsx": home, "front_end/src/pages/Home.jsx": home})
    maps = react_map.extract(SourceIndex(str(workspace)), [])
    assert [m.root for m in maps] == ["front/end", "front_end"]
    ids = [a.id for m in maps for a in m.all_anchors()]
    assert len(ids) == len(set(ids))
    assert "react@front_end:routes" in _anchors(maps[0]) and "react@front_end_2:routes" in _anchors(maps[1])
    # Each root's recipes name that root's own anchors.
    assert "react@front_end_2:route_insert" in " ".join(maps[1].recipes[0].steps)
    _assert_named_ids_exist(maps)


def test_workspace_without_a_frontend_maps_nothing(tmp_path):
    workspace = _write(tmp_path, {
        "backend/main_api.py": "from fastapi import FastAPI\napp = FastAPI()\n",
        "vue/package.json": '{"dependencies": {"vue": "^3.4.0"}}',
        "vue/src/main.js": "import { createApp } from 'vue';\n",
        "tool/package.json": '{"dependencies": {"react": "^18"}}',   # React, but no src/ entry
    })
    maps = react_map.extract(SourceIndex(str(workspace)), ["Book"])
    assert maps == []
    assert render_overview(WorkspaceMap(maps=maps)) == ""


# ---------------------------------------------------------------------------
# Bounded work: the map is rebuilt on every locate_code call and overview
# ---------------------------------------------------------------------------

_PATHOLOGICAL = 20_000
# Long blank runs (comments are blanked to spaces), blank lines, comment blocks, unclosed brackets,
# comments that never close and strings that never close but hold escaped quotes.
_FILLS = (" " * _PATHOLOGICAL, "\n" * _PATHOLOGICAL, " \n" * (_PATHOLOGICAL // 2), "// note\n" * (_PATHOLOGICAL // 8),
          "{" * _PATHOLOGICAL, "(" * _PATHOLOGICAL, "a" * _PATHOLOGICAL, "a " * (_PATHOLOGICAL // 2),
          "/* " * (_PATHOLOGICAL // 3), '"\\' * (_PATHOLOGICAL // 2), "'\\" * (_PATHOLOGICAL // 2), "`\\" * (_PATHOLOGICAL // 2))
# The literal heads the module's patterns start from, so each one is driven into its backtracking.
_HEADS = ("", "onSubmit={", "onSubmit={async (e) => {", "export function ", "export default ", "export const X = ",
          "const API = ", "return ", "=> ", "import {", "<Route ", "<link ", "<script ", "@import ",
          "process.env.VITE_API", "path: ")
# Matched at one offset only (``pattern.match(text, pos)``): a leading \s* is linear there.
_MATCH_ONLY = {"_INITIALISER_RE", "_ROUTER_ARG_RE", "_FALLBACK_RE"}
# Only ever run through a helper that prepares its input.
_USED_THROUGH = {"_COMMENT_OR_STRING": react_map._code_view}


def test_every_pattern_is_linear_on_pathological_input():
    """Each module pattern finishes in under 0.2 s on 20k characters (the no-op onSubmit check took minutes)."""
    slow = []
    for name, value in sorted(vars(react_map).items()):
        for pattern in (value if isinstance(value, tuple) else (value,)):
            if not isinstance(pattern, re.Pattern):
                continue
            for head in _HEADS:
                for fill in _FILLS:
                    text = head + fill + "x"
                    started = time.perf_counter()
                    if name in _MATCH_ONLY:
                        pattern.match(text, len(head))
                    elif name in _USED_THROUGH:
                        _USED_THROUGH[name](text)
                    else:
                        for _ in pattern.finditer(text):
                            pass
                    elapsed = time.perf_counter() - started
                    if elapsed > 0.2:
                        slow.append((name, head, fill[:8], round(elapsed, 2)))
    assert not slow, slow


def _form_page(handler_comment_lines: int, body: str) -> dict[str, str]:
    comment = "\n".join(f"        // Step {n}: post the form state to /booking/, then refresh the list afterwards."
                        for n in range(handler_comment_lines))
    return {
        "fe/package.json": '{"dependencies": {"react": "^18", "react-router-dom": "^6"}}',
        "fe/src/App.tsx": (
            'import { Routes, Route } from "react-router-dom";\n'
            'import Booking from "./pages/Booking";\n\n'
            "export default function App() {\n"
            "  return (\n"
            "    <Routes>\n"
            '      <Route path="/" element={<Booking />} />\n'
            "    </Routes>\n"
            "  );\n"
            "}\n"),
        "fe/src/pages/Booking.tsx": (
            'import React, { useState } from "react";\n\n'
            "const Booking: React.FC = () => {\n"
            '  const [title, setTitle] = useState("");\n'
            "  return (\n"
            '    <form id="booking"\n'
            "      onSubmit={\n"
            f"{comment}\n"
            "        async (e) => {\n"
            "          e.preventDefault();\n"
            f"{body}"
            "        }\n"
            "      }\n"
            "    >\n"
            "      <input value={title} onChange={(e) => setTitle(e.target.value)} />\n"
            "    </form>\n"
            "  );\n"
            "};\n\n"
            "export default Booking;\n"),
    }


@pytest.mark.parametrize("working", [True, False])
def test_a_commented_on_submit_handler_maps_fast(tmp_path, working):
    """SAFE-1: comments are blanked to spaces, and the old no-op pattern, failing on a working handler
    after a 20-line comment, backtracked over that blank run for minutes on every locate_code call."""
    body = '          await fetch("/booking/", { method: "POST" });\n' if working else ""
    workspace = _write(tmp_path, _form_page(20, body))
    started = time.perf_counter()
    smap = react_map.extract(SourceIndex(str(workspace)), [])[0]
    assert time.perf_counter() - started < 2.0
    stub = _anchors(smap).get("react:form:Booking.booking")
    if working:
        assert stub is None and not any("preventDefault" in w for w in smap.warnings)
    else:
        assert stub.kind == KIND_STUB and stub.status == "no-op onSubmit"
        assert any("form booking in Booking only calls preventDefault" in w for w in smap.warnings)
    _assert_quotes_valid(workspace, smap.all_anchors())


@pytest.mark.parametrize("handler", [
    "() => {}", "(e) => { e.preventDefault(); }", "async (e) => { e.preventDefault() }", "e => {}",
    "e => e.preventDefault()", "(event) => event.preventDefault()",
])
def test_no_op_submit_shapes_are_still_recognised(handler):
    attrs = f" onSubmit={{{handler}}}"
    assert any(rx.search(attrs) for rx in react_map._NOOP_SUBMIT_RES), handler


@pytest.mark.parametrize("handler", [
    "handleSubmit", "async (e) => { e.preventDefault(); await save(); }", "(e) => save(e)",
])
def test_working_submit_handlers_are_not_stubs(handler):
    attrs = f" onSubmit={{{handler}}}"
    assert not any(rx.search(attrs) for rx in react_map._NOOP_SUBMIT_RES), handler


def test_tags_that_never_close_do_not_make_the_tag_scan_quadratic(tmp_path):
    """Each tag is scanned to its closing '>': when none comes (a write cut off mid-JSX, an unclosed '{'),
    every tag scanned to the end of the file. A broken page now costs a budget linear in its size."""
    tags = "\n".join(f'      <Card title="item {n}" entity="Book" onClick={{() => {{' for n in range(1500))
    workspace = _write(tmp_path, {
        "fe/package.json": '{"dependencies": {"react": "^18"}}',
        "fe/src/App.tsx": "export default function App() {\n  return <p />;\n}\n",
        "fe/src/pages/Broken.tsx": (
            "export default function Broken() {\n"
            "  return (\n"
            "    <div>\n"
            f"{tags}\n"),
    })
    started = time.perf_counter()
    maps = react_map.extract(SourceIndex(str(workspace)), ["Book"])
    assert time.perf_counter() - started < 3.0
    assert "react:page:Broken" in _anchors(maps[0])     # mapped, with fewer blocks, never an error


def test_a_capped_index_never_claims_that_files_on_disk_are_missing(tmp_path):
    """SAFE-2: the index leaves files out past its cap; the map must not call them missing or unreachable."""
    workspace = _write(tmp_path, {
        "fe/package.json": '{"dependencies": {"react": "^18", "react-router-dom": "^6"}}',
        "fe/src/main.tsx": "import App from './App';\nimport './index.css';\nReactDOM.createRoot(x).render(<App />);\n",
        "fe/src/index.css": "body { margin: 0; }\n",
        "fe/src/App.tsx": (
            "import logo from './logo.svg';\n"                       # on disk, never mapped (an image)
            "import Home from './pages/Home';\n"
            "import Deep from './pages/admin/reports/Deep';\n"      # on disk, left out by the cap
            "import Gone from './pages/Gone';\n"                    # really missing
            "export default function App() {\n"
            "  return (\n"
            "    <Routes>\n"
            '      <Route path="/" element={<Home />} />\n'
            '      <Route path="/deep" element={<Deep />} />\n'
            '      <Route path="/gone" element={<Gone />} />\n'
            "    </Routes>\n"
            "  );\n"
            "}\n"),
        "fe/src/logo.svg": "<svg/>",
        "fe/src/pages/Home.tsx": "export default function Home() {\n  return <img src={logo} />;\n}\n",
        "fe/src/pages/admin/reports/Deep.tsx": (
            "import Widget from '../../../components/deep/x/Widget';\n"
            "export default function Deep() {\n  return <Widget />;\n}\n"),
        "fe/src/components/deep/x/Widget.tsx": (
            "import '../../../theme/extra.css';\n"
            "import Listed from '../../../pages/Listed';\n"
            "export default function Widget() {\n  return <Listed />;\n}\n"),
        "fe/src/theme/extra.css": "a { color: red; }\n",
        "fe/src/pages/Listed.tsx": "export default function Listed() {\n  return <p />;\n}\n",
    })
    index = SourceIndex(str(workspace), max_files=7)
    assert index.capped and "fe/src/pages/admin/reports/Deep.tsx" not in index.files
    assert "fe/src/components/deep/x/Widget.tsx" not in index.files
    smap = react_map.extract(index, [])[0]
    text = " ".join(smap.warnings + [s for r in smap.recipes for s in r.steps])
    assert "'./pages/Gone', which does not exist" in text      # confirmed on disk: still reported
    assert "Deep" not in text and "logo.svg" not in text
    # Its only importer was never read: neither unreachable nor imported nowhere.
    assert "Listed" not in text and "extra.css" not in text

    # The same workspace uncapped: the facts the map can check are unchanged.
    full = react_map.extract(SourceIndex(str(workspace)), [])[0]
    text = " ".join(full.warnings + [s for r in full.recipes for s in r.steps])
    assert "'./pages/Gone', which does not exist" in text
    assert "Deep" not in text and "logo.svg" not in text and "Listed" not in text


def test_line_numbers_follow_newlines_only(tmp_path):
    """SourceIndex.lines splits on '\\n' only: a U+2028 or a form feed must not shift the map's lines."""
    workspace = _write(tmp_path, {
        "fe/package.json": '{"dependencies": {"react": "^18", "react-router-dom": "^6"}}',
        "fe/src/App.tsx": (
            'import { Routes, Route } from "react-router-dom";\n'
            'import Home from "./pages/Home";\n'
            "// a separator and a form feed\f in a comment\n"
            'const TITLE = "line separator";\n'
            "export default function App() {\n"
            "  return (\n"
            "    <Routes>\n"
            '      <Route path="/" element={<Home />} />\n'
            '      <Route path="*" element={<Home />} />\n'
            "    </Routes>\n"
            "  );\n"
            "}\n"),
        "fe/src/pages/Home.tsx": "export default function Home() {\n  return <p>home</p>;\n}\n",
    })
    index = SourceIndex(str(workspace))
    smap = react_map.extract(index, [])[0]
    anchors = _anchors(smap)
    lines = index.lines("fe/src/App.tsx")
    assert lines[anchors["react:routes"].start_line - 1].strip() == "<Routes>"
    insert = anchors["react:route_insert"]
    assert lines[insert.start_line - 1] == insert.quote == '      <Route path="*" element={<Home />} />'
    assert lines[anchors["react:page_imports"].start_line - 1] == 'import Home from "./pages/Home";'


# ---------------------------------------------------------------------------
# End to end through the package entry points
# ---------------------------------------------------------------------------

@pytest.fixture
def sibling_extractors(monkeypatch):
    """Let build_workspace_map run even before the other families' extractors exist."""
    import besser.spec_driven_agent.planning.scaffold_map as package

    for name in ("fastapi", "django", "generic"):
        full = f"{package.__name__}.{name}"
        try:
            importlib.import_module(full)
        except ImportError:
            stub = types.ModuleType(full)
            stub.extract = lambda index, class_names, **kwargs: []
            monkeypatch.setitem(sys.modules, full, stub)
            monkeypatch.setattr(package, name, stub, raising=False)


def test_end_to_end_build_overview_and_locate(web_app, models, sibling_extractors):
    from besser.spec_driven_agent.planning.scaffold_map import build_workspace_map, locate, scaffold_overview

    domain, _ = models
    ws = build_workspace_map(str(web_app), domain)
    react_maps = [m for m in ws.maps if m.family == "react"]
    assert [m.root for m in react_maps] == [FE]

    overview = scaffold_overview(str(web_app), domain)
    assert len(overview) <= 4000
    assert not re.search(r"\bline \d+|\.tsx:\d+", overview)
    # The React section keeps its content next to the backend's, not just its title.
    react = _section(overview, "React + Vite frontend (GUI scaffold, runtime blocks)")
    assert "Where new code goes:" in react and "- Add a page:" in react and "react:route_insert" in react

    # The backend's Book anchors come first; narrowing to the frontend root shows the page.
    by_entity = locate(str(web_app), entity="Book", path=FE, domain_model=domain)
    assert "[react:page:BookPage]" in by_entity["locations"]
    assert "[react:block:book:BookPage.book-table]" in by_entity["locations"]
    assert "quote: const BookPage: React.FC = () => {" in by_entity["locations"]
    assert "read_file offset=" in by_entity["locations"]

    by_query = locate(str(web_app), query="react:route_insert")
    assert by_query["count"] == 1
    assert '<Route path="/" element={<Navigate' in by_query["locations"]

    by_path = locate(str(web_app), path=f"{FE}/src/pages/Dashboard.tsx")
    assert "react:form:Dashboard.contact_form" in by_path["locations"]
    assert "(no-op onSubmit)" in by_path["locations"]

    first = locate(str(web_app))
    assert "Book" in first["entities"]
    assert any("preventDefault" in p for p in first["pitfalls"])


def test_end_to_end_overview_of_a_five_class_web_app(tmp_path, sibling_extractors):
    """Five table screens and a dashboard with two unbound forms: an ordinary model, inside the budget."""
    from besser.generators.web_app import WebAppGenerator
    from besser.spec_driven_agent.planning.scaffold_map import scaffold_overview

    domain, gui = _models(extra=True)
    WebAppGenerator(model=domain, gui_model=gui, output_dir=str(tmp_path / "web_app"),
                    agent_model=None, agent_config=None).generate()
    overview = scaffold_overview(str(tmp_path), domain)
    assert len(overview) <= 4000
    react = _section(overview, "React + Vite frontend (GUI scaffold, runtime blocks)")
    assert "Where new code goes:" in react and "- Add a page:" in react
    # Both no-op forms fold into one pitfall line; the runtime's API base copies are no pitfall.
    assert react.count("preventDefault") == 1 and "2 forms only call preventDefault" in react
    assert "API base" not in react
