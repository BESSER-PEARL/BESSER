"""Regressions an independent review found in the TypeScript/class create-shape scanner.

- A generic create helper whose path is `/api/${resource}/` was read as the
  literal path `/api//`, so the page's `create('bookings', ...)` argument was
  dropped and Booking lost its create.
- An uncalled reference, or a call on any identifier once the page instantiates
  a client class, matched inside comments and strings (`// create lives
  elsewhere`, `const mode = 'create'`) and on `Object.create(null)`.
- The scanner rescanned to the end of the file for every candidate when
  brackets do not balance, which made 176 KB of JSX text take ~9 s.
"""
from __future__ import annotations

import time

import pytest

from besser.spec_driven_agent.validation.acceptance import (
    _direct_create_calls,
    _hook_post_paths,
    _post_helper_paths,
    build_acceptance_matrix,
)

# Each input took well under 1 s before the scanner; 2 s leaves room for a slow CI box.
_TIME_BOUND_SECONDS = 2.0


class _Class:
    def __init__(self, name: str):
        self.name = name


class _Model:
    def __init__(self, *names: str):
        self._names = names

    def get_classes(self) -> list[_Class]:
        return [_Class(n) for n in self._names]


def _creates(tmp_path, files: dict[str, str], *names: str) -> dict[str, bool]:
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    matrix = build_acceptance_matrix(str(tmp_path), _Model(*names), endpoint_manifest="x")
    return {n: matrix[n]["create"] for n in names}


_GENERIC_PAGE = ("import { create } from '../api';\nexport default function BookingPage() {\n"
                 "  const save = (form) => create('bookings', form);\n"
                 "  return <form onSubmit={save}>Booking</form>;\n}\n")


@pytest.mark.parametrize("client", [
    "export const create = (resource, data) => fetch(`/api/${resource}/`, "
    "{ method: 'POST', body: JSON.stringify(data) });\n",
    "export async function create(resource, data) {\n  return axios.post(`${BASE_URL}/api/v1/${resource}`, data);\n}\n",
])
def test_a_generic_helper_with_a_literal_prefix_keeps_the_callers_entity(tmp_path, client):
    files = {"frontend/src/api.js": client, "frontend/src/pages/BookingPage.jsx": _GENERIC_PAGE}
    assert _creates(tmp_path, files, "Booking", "Room") == {"Booking": True, "Room": False}


def test_a_slot_in_a_path_segment_is_a_computed_path():
    assert _post_helper_paths("export const create = (r, d) => fetch(`/api/${r}/`, { method: 'POST' });\n") == {
        "create": None}


_CLIENT_FN = ("export const list = (entity) => fetch(`${BASE}/${entity}/`);\n"
              "export const create = (entity, data) => fetch(`${BASE}/${entity}/`, "
              "{ method: 'POST', body: JSON.stringify(data) });\n")
_CLIENT_CLASS = ("export class ApiClient {\n  list(entity) { return fetch(`${BASE}/${entity}/`) }\n"
                 "  create(entity, data) { return fetch(`${BASE}/${entity}/`, "
                 "{ method: 'POST', body: JSON.stringify(data) }) }\n}\n")


@pytest.mark.parametrize("client,page", [
    (_CLIENT_CLASS, "import { ApiClient } from '../api';\nconst client = new ApiClient();\n"
                    "const empty = Object.create(null);\n"
                    "export default function RoomList() { return <ul>{client.list('rooms')}</ul>; }\n"),
    (_CLIENT_FN, "import { list, create } from '../api';\n// read-only page: create lives elsewhere\n"
                 "export default function RoomList() { return <ul>{list('rooms')}</ul>; }\n"),
    (_CLIENT_FN, "import { list, create } from '../api';\nexport default function RoomList() {\n"
                 "  const mode = 'create'; return <ul title={mode}>{list('rooms')}</ul>; }\n"),
    (_CLIENT_FN, "import { list, create } from '../api';\nexport default function RoomList() {\n"
                 "  return <ul>{list('rooms')}<small>No rooms yet - create one first.</small></ul>; }\n"),
], ids=["object-create", "comment", "string", "jsx-text"])
def test_a_mention_that_is_not_a_create_is_no_create(tmp_path, client, page):
    files = {"src/api.js": client, "src/pages/RoomList.jsx": page}
    assert _creates(tmp_path, files, "Room") == {"Room": False}


@pytest.mark.parametrize("page", [
    "const client = new ApiClient();\n"
    "export default function RoomList() { const add = (d) => client.create('rooms', d); }\n",
    "export default function RoomList() { const add = (d) => new ApiClient().create('rooms', d); }\n",
])
def test_a_create_on_the_instance_of_the_imported_client_class_still_counts(tmp_path, page):
    files = {"src/api.js": _CLIENT_CLASS, "src/pages/RoomList.jsx": "import { ApiClient } from '../api';\n" + page}
    assert _creates(tmp_path, files, "Room") == {"Room": True}


@pytest.mark.parametrize("source", [
    "const r = /[{(]/g; function save(d) { return api.post('/booking', d) }\n<p>a { b ( c</p>\n" * 2000,
    "f(" * 20000 + "api.post('/x', d)" + ")" * 20000,
    "x = " + "".join(f"c{i} ? v{i} : " for i in range(10000)) + "0; api.post('/x', d)",
], ids=["unbalanced-jsx-text", "deep-nesting", "one-line-ternary-chain"])
def test_the_scanner_is_linear_on_hostile_input(source):
    started = time.perf_counter()
    _post_helper_paths(source)
    _hook_post_paths(source)
    _direct_create_calls(source)
    assert time.perf_counter() - started < _TIME_BOUND_SECONDS
