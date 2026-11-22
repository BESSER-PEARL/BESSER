"""Create paths of JavaScript and TypeScript shapes the acceptance matrix missed.

Each read "no frontend create path found" for an entity the page can create:
a method on a client class the page instantiates, a create helper handed over
as a callback (react-query's ``mutationFn``, an ``onSubmit`` prop), and helper
definitions whose head the old single regex could not read (generics, a typed
const, nested parentheses in the parameters, a return type on a method).
``declare function`` has no body and is not a helper. A POST to
``API + '/booking/'`` names Booking only, not every entity the file mentions.
"""
from __future__ import annotations

import pytest

from besser.spec_driven_agent.validation.acceptance import (
    _hook_post_paths,
    _post_helper_paths,
    build_acceptance_matrix,
)


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


_FETCH = "fetch('/booking/', { method: 'POST', body: JSON.stringify(data) })"


@pytest.mark.parametrize("definition", [
    f"export async function createBooking<T>(data: T) {{\n  return {_FETCH};\n}}\n",
    f"export const createBooking = async <T,>(data: T) => {{\n  return {_FETCH};\n}};\n",
    f"export const createBooking = async <T>(data: T): Promise<T> => {{\n  return {_FETCH};\n}};\n",
    f"export function createBooking(data, onDone = () => {{}}) {{\n  return {_FETCH}.then(onDone);\n}}\n",
    f"export const createBooking = (data: Booking, onDone: (b: Booking) => void) => {_FETCH}.then(onDone);\n",
    "type Handler = (d: unknown) => Promise<Response>;\n"
    f"export const createBooking: Handler = async (data) => {{\n  return {_FETCH};\n}};\n",
    f"export const createBooking = data => {_FETCH};\n",
    "export const createBooking = (b: Booking) => request<Booking>('POST', '/booking/', b);\n",
])
def test_a_typescript_helper_head_is_read(definition):
    assert _post_helper_paths(definition) == {"createBooking": ["/booking/"]}


@pytest.mark.parametrize("client", [
    f"class Api {{\n  async createBooking(data: Booking): Promise<Booking> {{\n"
    f"    const r = await {_FETCH};\n    return r.json();\n  }}\n}}\nexport const api = new Api();\n",
    f"class Api {{\n  createBooking = async (data: Booking) => {{\n    return {_FETCH};\n  }};\n}}\n"
    "export const api = new Api();\n",
])
def test_a_client_class_method_is_a_helper(tmp_path, client):
    page = ("import { api } from '../api';\n"
            "export default function BookingPage() { const save = () => api.createBooking(form); }\n")
    assert _creates(tmp_path, {"src/api.ts": client, "src/pages/BookingPage.tsx": page}, "Booking") == {
        "Booking": True}


def test_a_method_of_a_client_class_the_page_instantiates_is_a_create(tmp_path):
    client = f"export class Api {{\n  createBooking(data) {{\n    return {_FETCH};\n  }}\n}}\n"
    page = ("import { Api } from '../api';\nexport default function BookingPage() {\n"
            "  const api = new Api();\n  const save = () => api.createBooking(form);\n}\n")
    assert _creates(tmp_path, {"src/api.js": client, "src/pages/BookingPage.jsx": page}, "Booking") == {
        "Booking": True}


def test_an_instantiated_client_whose_creates_are_never_called_is_no_create(tmp_path):
    client = (f"export class Api {{\n  createBooking(data) {{\n    return {_FETCH};\n  }}\n"
              "  listBookings() { return fetch('/booking/'); }\n}\n")
    page = ("import { Api } from '../api';\nexport default function BookingPage() {\n"
            "  const api = new Api();\n  api.listBookings();\n}\n")
    assert _creates(tmp_path, {"src/api.js": client, "src/pages/BookingPage.jsx": page}, "Booking") == {
        "Booking": False}


@pytest.mark.parametrize("use", [
    "const m = useMutation({ mutationFn: api.createBooking });",
    "return <BookingForm onSubmit={api.createBooking} />;",
])
def test_a_create_helper_passed_as_a_callback_is_a_create(tmp_path, use):
    client = f"export const api = {{\n  createBooking: (data) => {_FETCH},\n}};\n"
    page = "import { api } from '../api';\nexport default function BookingPage() {\n  " + use + "\n}\n"
    assert _creates(tmp_path, {"src/api.js": client, "src/pages/BookingPage.jsx": page}, "Booking") == {
        "Booking": True}


def test_a_create_helper_only_imported_is_no_create(tmp_path):
    client = (f"export const createBooking = (data) => {_FETCH};\n"
              "export const listBookings = () => fetch('/booking/');\n")
    page = ("import { createBooking, listBookings } from '../api';\n"
            "export default function BookingPage() { listBookings(); }\n")
    assert _creates(tmp_path, {"src/api.js": client, "src/pages/BookingPage.jsx": page}, "Booking") == {
        "Booking": False}


def test_a_declare_function_has_no_body():
    source = ("declare function useBookings(): void;\n"
              "export const api = { create: async (d) => fetch('/booking/', { method: 'POST' }) };\n")
    assert _hook_post_paths(source) == {}
    assert _post_helper_paths("declare function createBooking(d: Booking): Promise<void>;\n"
                              "fetch('/booking/', { method: 'POST' });\n") == {}


@pytest.mark.parametrize("call", [
    "axios.post(API + '/booking/', b)",
    "axios.post(`${API}/booking/`, b)",
    "fetch(API_BASE + '/booking/', { method: 'POST', body: b })",
    "fetch(`${API_BASE}/booking/`, { method: 'POST', body: b })",
    "request('POST', API + '/booking/', b)",
])
def test_a_post_to_a_concatenated_path_creates_only_the_entity_it_names(tmp_path, call):
    page = ("import axios from 'axios';\nconst API = '/api';\n"
            "export default function Dashboard() {\n  const save = (b) => " + call + ";\n"
            "  return <div>rooms, guests and bookings</div>;\n}\n")
    assert _creates(tmp_path, {"src/Dashboard.jsx": page}, "Booking", "Room", "Guest") == {
        "Booking": True, "Room": False, "Guest": False}


@pytest.mark.parametrize("call", [
    "axios.post(`${API}/${entity}/`, b)",
    "axios.post(API + '/' + entity + '/', b)",
    "axios.post(url, b)",
])
def test_a_post_to_a_computed_path_still_names_no_entity(tmp_path, call):
    page = ("import axios from 'axios';\n"
            "export default function BookingPage({ entity }) {\n  const save = (b) => " + call + ";\n}\n")
    assert _creates(tmp_path, {"src/pages/BookingPage.jsx": page}, "Booking") == {"Booking": True}
