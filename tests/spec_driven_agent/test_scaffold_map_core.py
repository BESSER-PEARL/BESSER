"""The scaffold map's core: file index, rendering, matching, ownership and caching.

Extractor behaviour is covered per family (test_scaffold_map_<family>.py); these
tests pin what every family relies on.
"""
from __future__ import annotations

import os
import re
import time

import pytest

from besser.generators.backend import BackendGenerator
from besser.spec_driven_agent.planning import scaffold_map
from besser.spec_driven_agent.planning.scaffold_map import locate, scaffold_overview
from besser.spec_driven_agent.planning.scaffold_map.model import Anchor, EntityMap, Recipe, ScaffoldMap, WorkspaceMap
from besser.spec_driven_agent.planning.scaffold_map.render import _match_query, render_locate, render_overview
from besser.spec_driven_agent.planning.scaffold_map.source import SourceIndex
from besser.spec_driven_agent.state.agent_trace import ATTRIBUTION_FILENAME, record_generated_files

_LINE_NUMBER = re.compile(r"\blines? \d|\.\w+:\d|@\d")


@pytest.fixture
def backend(tmp_path, library_book_author_model):
    BackendGenerator(model=library_book_author_model, output_dir=str(tmp_path / "backend")).generate()
    return tmp_path


def _age(path, seconds: float = 10.0) -> None:
    """Back-date every file so the map cache treats the tree as settled."""
    past = time.time() - seconds
    for root, _dirs, files in os.walk(path):
        for name in files:
            os.utime(os.path.join(root, name), (past, past))


# -- file index ----------------------------------------------------------------


def test_assets_do_not_use_up_the_file_cap(tmp_path):
    (tmp_path / "assets").mkdir()
    for n in range(50):
        (tmp_path / "assets" / f"img{n}.png").write_bytes(b"\x89PNG")
    (tmp_path / "app.py").write_text("def main():\n    pass\n", encoding="utf-8")
    index = SourceIndex(str(tmp_path), max_files=5)
    assert index.files == ["app.py"]
    assert index.exists("assets/img0.png")          # unmapped is not absent


def test_the_cap_keeps_the_shallowest_files_and_says_so(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    for n in range(10):
        (deep / f"m{n}.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    index = SourceIndex(str(tmp_path), max_files=3)
    assert "main.py" in index.files and len(index.files) == 3
    assert index.capped and "3 files mapped; not mapped: 8 under a/" == index.notes[0]


def test_a_dataset_cannot_hide_the_app(backend):
    data = backend / "data" / "train"
    data.mkdir(parents=True)
    for n in range(3000):
        (data / f"sample_{n}.json").write_text("{}", encoding="utf-8")
    index = SourceIndex(str(backend))
    assert "backend/main_api.py" in index.files and index.capped
    assert "under data/" in index.notes[0]
    assert locate(str(backend), query="fastapi:include_routers")["count"] >= 1


def test_hidden_and_dependency_directories_are_skipped(tmp_path):
    for folder in (".yarn/cache", "node_modules/pkg", ".venv/lib"):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "x.js").write_text("export const x = 1;\n", encoding="utf-8")
    (tmp_path / "src.js").write_text("export const y = 1;\n", encoding="utf-8")
    assert SourceIndex(str(tmp_path)).files == ["src.js"]


def test_lines_split_like_read_file_and_a_bom_is_not_code(tmp_path):
    (tmp_path / "a.py").write_text("﻿x = ' '\r\ny = 2\n", encoding="utf-8", newline="")
    index = SourceIndex(str(tmp_path))
    assert index.lines("a.py") ==["x = ' '", "y = 2"]
    assert index.python_ast("a.py") is not None and not index.syntax_errors


def test_a_quote_must_be_unique_as_text_not_only_as_a_line(tmp_path):
    (tmp_path / "a.py").write_text("    # Book.lend\n    # Book.lend: load the instance\nz = 1\n", encoding="utf-8")
    index = SourceIndex(str(tmp_path))
    assert index.unique_quote("a.py", 1) == ""            # also inside line 2
    assert index.unique_quote("a.py", 2) == "    # Book.lend: load the instance"


# -- locate --------------------------------------------------------------------


def test_locate_quotes_verbatim_with_indentation(backend, library_book_author_model):
    answer = locate(str(backend), entity="Book", domain_model=library_book_author_model)
    quotes = re.findall(r"^  quote: (.*)$", answer["locations"], re.M)
    assert any(q.startswith("    ") for q in quotes)     # an indented line keeps its indentation
    text = "".join((backend / p).read_text(encoding="utf-8") for p in ("backend/routers/book.py",))
    assert any(text.count(q) == 1 for q in quotes)


@pytest.mark.parametrize("spelling", ["backend/routers/book.py", "./backend/routers/book.py",
                                      "backend//routers/book.py", "backend/x/../routers/book.py",
                                      "backend\\routers\\book.py"])
def test_locate_accepts_the_path_spellings_read_file_accepts(backend, spelling):
    assert "backend/routers/book.py" in locate(str(backend), path=spelling)["locations"]


def test_locate_accepts_an_absolute_path_inside_the_workspace(backend):
    absolute = str(backend / "backend" / "routers" / "book.py")
    assert "backend/routers/book.py" in locate(str(backend), path=absolute)["locations"]


def test_blank_filters_are_no_filters(backend):
    assert locate(str(backend), entity=" ", path="", query="  ") == locate(str(backend))


def _anchor(id_, label, entity=None):
    return Anchor(id=id_, file="f.py", kind="function", label=label, start_line=1, end_line=1, entity=entity)


def test_query_words_match_whole_words_and_rank_partial_matches():
    anchors = [_anchor("fastapi:route:author:get_author", "GET /author/ -> get_author", "Author"),
               _anchor("fastapi:include_routers", "include_router calls"),
               _anchor("fastapi:stub:book:lend", "POST /book/{book_id}/methods/lend/ -> execute_book_lend", "Book")]
    assert _match_query(anchors, "auth")[0] == []                        # not a prefix of 'author'
    assert _match_query(anchors, "fastapi:include")[0] == [anchors[1]]   # id prefix
    found, mode = _match_query(anchors, "where do I implement Book.lend")
    assert found[0] is anchors[2] and mode == "all words"
    found, mode = _match_query(anchors, "lend a book to the library")
    assert found[0] is anchors[2] and mode == "ranked by matched words"


def test_a_malformed_attribution_file_costs_only_the_owner_labels(backend):
    (backend / ATTRIBUTION_FILENAME).write_text('{"contributors": [{"kind": "ai"}], "files": '
                                                 '{"backend/main_api.py": {"hashes": [1, 2], "runs": [[0, 2]]}}}',
                                                 encoding="utf-8")
    answer = locate(str(backend), path="backend/main_api.py")
    assert answer["locations"] and "owner:" not in answer["locations"]


def test_owner_labels_follow_the_attribution(backend):
    record_generated_files(str(backend), ["backend/routers/book.py"], "backend")
    router = backend / "backend" / "routers" / "book.py"
    router.write_text(router.read_text(encoding="utf-8") + "\n\ndef added():\n    return 1\n", encoding="utf-8")
    answer = locate(str(backend), path="backend/routers/book.py")
    assert "owner: generator" in answer["locations"]


def test_the_broken_list_is_capped(tmp_path):
    for n in range(15):
        (tmp_path / f"bad{n:02}.py").write_text("def broken(:\n", encoding="utf-8")
    answer = locate(str(tmp_path))
    assert len(answer["broken"]) == 11 and answer["broken"][-1] == "… 5 more"


# -- overview --------------------------------------------------------------------


def _big_map(n_entities: int, root: str) -> ScaffoldMap:
    entities = {f"E{i:02}": EntityMap(name=f"E{i:02}", summary=[f"routes {root}/routers/e{i:02}.py: list, read"])
                for i in range(n_entities)}
    return ScaffoldMap(family="fastapi", root=root, title="FastAPI backend", entities=entities,
                       recipes=[Recipe("Add an endpoint", ["Use `fastapi:insert_static:<entity>` in the router."] * 8)],
                       warnings=["w" * 150] * 6)


@pytest.mark.parametrize("budget", [300, 1500, 4000])
def test_the_overview_never_exceeds_its_budget_and_has_no_line_numbers(budget):
    ws = WorkspaceMap(maps=[_big_map(30, "backend"), _big_map(12, "web_app/backend")],
                      broken=["a.py: SyntaxError line 4: closing parenthesis ')' does not match '{' on line 368"])
    text = render_overview(ws, max_chars=budget)
    assert len(text) <= budget
    assert not _LINE_NUMBER.search(text)
    assert "locate_code" in text


def test_a_large_map_cannot_starve_a_small_one():
    small = ScaffoldMap(family="react", root="frontend", title="React frontend",
                        recipes=[Recipe("Add a page", ["New page; add its <Route> at `react:route_insert`."])])
    text = render_overview(WorkspaceMap(maps=[_big_map(60, "backend"), small]), max_chars=2500)
    assert "`react:route_insert`" in text


def test_clipped_recipes_never_cut_an_anchor_id():
    text = render_overview(WorkspaceMap(maps=[_big_map(60, "backend")]), max_chars=1200)
    for line in text.splitlines():
        assert line.count("`") % 2 == 0, line


def test_a_root_path_returns_that_map_not_the_maps_nested_below_it():
    outer = ScaffoldMap(family="generic", root="web_app", title="Project files",
                        anchors=[Anchor("generic:compose", "web_app/docker-compose.yml", "file", "compose", 1, 1)])
    inner = ScaffoldMap(family="fastapi", root="web_app/backend", title="FastAPI",
                        anchors=[Anchor("fastapi:app", "web_app/backend/main_api.py", "block", "app", 1, 1)])
    answer = render_locate(WorkspaceMap(maps=[outer, inner]), path="web_app")
    assert "generic:compose" in answer["locations"] and "fastapi:app" not in answer["locations"]


# -- caching ---------------------------------------------------------------------


def test_the_map_is_reused_until_a_file_changes(backend, monkeypatch):
    _age(backend)
    builds = []
    real_build = scaffold_map._build
    monkeypatch.setattr(scaffold_map, "_build", lambda index, names: builds.append(1) or real_build(index, names))
    first = scaffold_overview(str(backend))
    assert scaffold_overview(str(backend)) == first and len(builds) == 1

    router = backend / "backend" / "routers" / "book.py"
    router.write_text(router.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    scaffold_overview(str(backend))
    assert len(builds) == 2


def test_a_file_changed_just_now_always_forces_a_rebuild(backend, monkeypatch):
    builds = []
    real_build = scaffold_map._build
    monkeypatch.setattr(scaffold_map, "_build", lambda index, names: builds.append(1) or real_build(index, names))
    scaffold_overview(str(backend))      # files were written moments ago: not settled
    scaffold_overview(str(backend))
    assert len(builds) == 2


def test_an_empty_workspace_points_at_the_generators(tmp_path):
    # Eval evidence: "no scaffold structure recognised" made luna hand-write the
    # whole backend (2 of 3 runs) instead of calling generate_fastapi_backend.
    answer = locate(str(tmp_path))
    assert answer["count"] == 0
    assert "generate_" in answer["locations"]
