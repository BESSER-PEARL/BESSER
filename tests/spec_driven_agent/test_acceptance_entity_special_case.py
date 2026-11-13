"""A shared form that special-cases one entity still creates every entity it serves.

Live evidence (Sonnet 5.5, a 7-class hotel app): each per-entity page rendered one
shared ``EntityPage`` that posts through ``api.create(config.key, ...)``. The form
also checked ``config.key === 'booking'`` to validate dates, so it was read as
Booking's own page and the other six entities were reported "no frontend create path".
"""
import os

from besser.spec_driven_agent.validation.acceptance import build_acceptance_matrix


class _Class:
    def __init__(self, name):
        self.name = name


class _Model:
    def __init__(self, *names):
        self._names = names

    def get_classes(self):
        return [_Class(n) for n in self._names]


API = """\
async function request(method, path, body) {
  const res = await fetch(`${BASE}${path}`, { method, body: JSON.stringify(body) });
  return res.json();
}
export const api = {
  list: (entity) => request('GET', `/${entity}/`),
  create: (entity, payload) => request('POST', `/${entity}/`, payload),
};
"""

ENTITIES = """\
export const ENTITIES = {
  hotel: { key: 'hotel', title: 'Hotels', fields: [{ name: 'name', type: 'text' }] },
  booking: { key: 'booking', title: 'Bookings', fields: [{ name: 'hotel', type: 'ref', ref: 'hotel' }] },
  guest: { key: 'guest', title: 'Guests', fields: [{ name: 'email', type: 'email' }] },
};
export const ENTITY_ORDER = ['hotel', 'booking', 'guest'];
"""

ENTITY_PAGE = """\
import { api } from '../api.js';
import { ENTITIES } from '../entities.js';
export default function EntityPage({ config, mode }) {
  const onSubmit = async () => {
    if (config.key === 'booking' && values.checkOut < values.checkIn) return;
    await api.create(config.key, payload);
  };
  return <form onSubmit={onSubmit}>{config.title}</form>;
}
"""


def _page(name, key):
    return ("import EntityPage from '../components/EntityPage.jsx';\n"
            "import { ENTITIES } from '../entities.js';\n"
            f"export default function {name}List({{ mode = 'list' }}) {{\n"
            f"  return <EntityPage config={{ENTITIES.{key}}} mode={{mode}} />;\n}}\n")


def _creates(tmp_path, entities=ENTITIES, page=ENTITY_PAGE):
    files = {
        "frontend/src/api.js": API,
        "frontend/src/entities.js": entities,
        "frontend/src/components/EntityPage.jsx": page,
        "frontend/src/pages/HotelList.jsx": _page("Hotel", "hotel"),
        "frontend/src/pages/BookingList.jsx": _page("Booking", "booking"),
        "frontend/src/pages/GuestList.jsx": _page("Guest", "guest"),
    }
    for rel, text in files.items():
        path = os.path.join(str(tmp_path), rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    matrix = build_acceptance_matrix(str(tmp_path), _Model("Hotel", "Booking", "Guest"), endpoint_manifest="x")
    return {n: matrix[n]["create"] for n in ("Hotel", "Booking", "Guest")}


def test_a_shared_form_that_checks_one_entity_creates_them_all(tmp_path):
    assert _creates(tmp_path) == {"Hotel": True, "Booking": True, "Guest": True}


def test_an_entity_the_shared_form_is_not_configured_for_has_no_create(tmp_path):
    without_guest = ENTITIES.replace(
        "  guest: { key: 'guest', title: 'Guests', fields: [{ name: 'email', type: 'email' }] },\n", "").replace(
        ", 'guest']", "]")
    assert _creates(tmp_path, without_guest) == {"Hotel": True, "Booking": True, "Guest": False}


# The branch the form runs for one entity calls a helper named after it.
BRANCH_PAGE = ENTITY_PAGE.replace(
    "    if (config.key === 'booking' && values.checkOut < values.checkIn) return;\n",
    "    if (config.key === 'booking') {\n"
    "      const nights = await api.bookingNights(values);\n"
    "      if (nights < 1) return;\n"
    "    }\n")


def test_a_shared_form_whose_branch_for_one_entity_names_it_creates_them_all(tmp_path):
    """Sonnet 5.5, 13-class wastewater app: the shared EntityList ran
    `if (entity === 'project') { await api.projectPlants(row.id) }` on edit, so the
    other twelve entities were reported "no frontend create path"."""
    assert _creates(tmp_path, page=BRANCH_PAGE) == {"Hotel": True, "Booking": True, "Guest": True}


def test_a_branch_on_something_else_still_names_the_entity(tmp_path):
    page = BRANCH_PAGE.replace("config.key === 'booking'", "config.mode === 'wizard'")
    assert _creates(tmp_path, page=page) == {"Hotel": False, "Booking": True, "Guest": False}


def test_a_shared_form_with_an_entity_branch_and_no_create_creates_nothing(tmp_path):
    page = BRANCH_PAGE.replace("await api.create(config.key, payload);", "")
    assert _creates(tmp_path, page=page) == {"Hotel": False, "Booking": False, "Guest": False}
