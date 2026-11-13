"""Create paths of real generated frontends that the acceptance matrix missed.

Live evidence (Sonnet 5.5, a 3-class todo app and a 13-class dashboard, 13 saved
runs): 121 of 149 entity cells read "no frontend create path found" although
every entity had a working create form, and the few cells that read "create"
did so through coincidences such as "running" (Run) or "username" (User).
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


def _creates(tmp_path, files, *names):
    for rel, text in files.items():
        path = os.path.join(str(tmp_path), rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    matrix = build_acceptance_matrix(str(tmp_path), _Model(*names), endpoint_manifest="x")
    return {n: matrix[n]["create"] for n in names}


# The method is the helper's first argument and fetch takes it as `{ method }`,
# so neither `.post(` nor `method: 'POST'` appears anywhere.
METHOD_ARG_API = """
async function request(method, path, body) {
  const res = await fetch(`${BASE}${path}`, { method, body: JSON.stringify(body) });
  return res.json();
}
export const api = {
  listTags: () => request('GET', '/tag/'),
  createTag: (d) => request('POST', '/tag/', d),
  listItems: () => request('GET', '/todoitem/?detailed=true'),
  createItem: (d) => request('POST', '/todoitem/', d),
};
"""
TAG_PAGE = ("import { api } from '../api.js';\n"
            "export default function TagList() { const save = () => api.createTag({ name }); }\n")
ITEM_PAGE = ("import { api } from '../api.js';\n"
             "export default function TodoItemList() {\n"
             "  api.listTags();\n  const save = () => api.createItem(payload);\n}\n")


def test_a_client_taking_the_method_as_an_argument_resolves(tmp_path):
    files = {"frontend/src/api.js": METHOD_ARG_API, "frontend/src/pages/TagList.jsx": TAG_PAGE,
             "frontend/src/pages/TodoItemList.jsx": ITEM_PAGE}
    assert _creates(tmp_path, files, "Tag", "TodoItem") == {"Tag": True, "TodoItem": True}


def test_a_helper_posting_to_a_literal_path_creates_only_that_entity(tmp_path):
    """TodoItemList lists tags and creates an item: that is not Tag's create."""
    files = {"frontend/src/api.js": METHOD_ARG_API.replace("  createTag: (d) => request('POST', '/tag/', d),\n", ""),
             "frontend/src/pages/TagList.jsx": TAG_PAGE.replace("api.createTag({ name })", "null"),
             "frontend/src/pages/TodoItemList.jsx": ITEM_PAGE}
    assert _creates(tmp_path, files, "Tag", "TodoItem") == {"Tag": False, "TodoItem": True}


def test_a_one_expression_arrow_helper_resolves(tmp_path):
    files = {
        "frontend/src/api.js": ("export const api = { post: (p, b) => request('POST', p, b) };\n"
                                "export const createEntity = (route, body) => api.post(`/${route}/`, body)\n"),
        "frontend/src/pages/VirusPage.jsx": ("import { createEntity } from '../api.js';\n"
                                             "const save = () => createEntity('virus', body);\n"),
    }
    assert _creates(tmp_path, files, "Virus") == {"Virus": True}


# Entities declared only in a config module; one generic form POSTs to `meta.path`.
CONFIG_APP = {
    "frontend/src/api.js": ("export async function request(path, { method = 'GET', body } = {}) {\n"
                            "  const res = await fetch(path, { method, body });\n"
                            "  if (!res.ok) throw new Error('Is the backend running?');\n}\n"
                            "export const api = {\n"
                            "  login: (username, password) => request('/auth/login', { method: 'POST', body: {} }),\n"
                            "  metrics: () => request('/dashboard/metrics'),\n"
                            "  create: (entity, payload) => request(`/${entity}/`, { method: 'POST', body: payload }),\n"
                            "};\n"),
    "frontend/src/entities.js": ("export const RunStatus = ['RUNNING', 'DONE'];\n"
                                 "export const ENTITIES = {\n"
                                 "  virus: { label: 'Viruses', fields: [{ name: 'name', type: 'text' }] },\n"
                                 "  detection: {\n    label: 'Detections',\n    fields: [\n"
                                 "      { name: 'virus', type: 'ref', ref: 'virus' },\n"
                                 "      { name: 'run', type: 'ref', ref: 'run' },\n    ],\n  },\n"
                                 "  datasetversion: { label: 'Dataset Version', fields: [] },\n"
                                 "};\n"),
    "frontend/src/pages/EntityForm.jsx": ("import { api } from '../api.js';\nimport { ENTITIES } from '../entities.js';\n"
                                          "export default function EntityForm({ entity }) {\n"
                                          "  const meta = ENTITIES[entity];\n"
                                          "  const save = () => api.create(entity, payload);\n}\n"),
    "frontend/src/App.jsx": ("import EntityForm from './pages/EntityForm.jsx';\nimport { useAuth } from './auth.jsx';\n"
                             "export default function App() {\n  const { user } = useAuth();\n"
                             "  return <EntityForm entity={key} />;\n}\n"),
    "frontend/src/auth.jsx": ("import { api } from './api.js';\n"
                              "export function useAuth() { const [user, setUser] = useState(null);\n"
                              "  const login = (u, p) => api.login(u, p).then(setUser); }\n"),
}


def test_a_config_driven_generic_form_creates_the_entities_its_config_declares(tmp_path):
    assert _creates(tmp_path, CONFIG_APP, "Virus", "Detection", "DatasetVersion") == {
        "Virus": True, "Detection": True, "DatasetVersion": True}


def test_references_and_coincidences_are_not_a_create(tmp_path):
    """Run is only a `ref`, an enum and "running"; Dataset only half a label;
    Metric only a dashboard read; User only the signed-in user and a login."""
    assert _creates(tmp_path, CONFIG_APP, "Virus", "Run", "Dataset", "Metric", "User") == {
        "Virus": True, "Run": False, "Dataset": False, "Metric": False, "User": False}


def test_create_helpers_nobody_calls_are_not_a_create(tmp_path):
    """Every page's create call removed from a saved todo app: the client still
    defines createTag/createItem, and its own POSTs were counted as the creates."""
    files = {"frontend/src/api.js": METHOD_ARG_API,
             "frontend/src/pages/TagList.jsx": TAG_PAGE.replace("api.createTag({ name })", "api.listTags()"),
             "frontend/src/pages/TodoItemList.jsx": ITEM_PAGE.replace("api.createItem(payload)", "null")}
    assert _creates(tmp_path, files, "Tag", "TodoItem") == {"Tag": False, "TodoItem": False}


# The saved dashboard's client: a generic create helper over `api.post`.
GENERIC_CLIENT_APP = {
    "frontend/src/api.js": ("async function request(method, path, body) {\n"
                            "  const res = await fetch(`${BASE}${path}`, { method, body: JSON.stringify(body) });\n"
                            "  return res.json();\n}\n"
                            "export const api = {\n  get: (path) => request('GET', path),\n"
                            "  post: (path, body) => request('POST', path, body),\n"
                            "  put: (path, body) => request('PUT', path, body),\n};\n"
                            "export const listEntities = (route) => api.get(`/${route}/`)\n"
                            "export const createEntity = (route, body) => api.post(`/${route}/`, body)\n"
                            "export const getMetrics = () => api.get('/api/metrics')\n"),
    "frontend/src/entities.js": ("export const ENTITIES = {\n  virus: { label: 'Viruses', fields: [] },\n"
                                 "  metric: { label: 'Metrics', fields: [] },\n};\n"),
    "frontend/src/pages/Dashboard.jsx": ("import { getMetrics } from '../api.js';\n"
                                         "export default function Dashboard() { getMetrics(); }\n"),
    "frontend/src/pages/EntityPage.jsx": ("import { ENTITIES } from '../entities.js';\n"
                                          "import { createEntity, listEntities } from '../api.js';\n"
                                          "export default function EntityPage({ route }) {\n"
                                          "  const meta = ENTITIES[route];\n  listEntities(route);\n"
                                          "  const save = () => createEntity(route, body);\n}\n"),
    "frontend/src/App.jsx": ("import EntityPage from './pages/EntityPage.jsx';\n"
                             "export default function App() { return <EntityPage route={key} />; }\n"),
}


def test_a_generic_create_helper_switched_to_put_is_not_a_create(tmp_path):
    """The page calls createEntity, which no longer POSTs; the client's own
    `post` helper, which nothing calls, still counted for Metric, which the
    client names in `getMetrics`."""
    assert _creates(tmp_path, GENERIC_CLIENT_APP, "Virus", "Metric") == {"Virus": True, "Metric": True}
    files = dict(GENERIC_CLIENT_APP)
    files["frontend/src/api.js"] = files["frontend/src/api.js"].replace(
        "createEntity = (route, body) => api.post(", "createEntity = (route, body) => api.put(")
    assert _creates(tmp_path, files, "Virus", "Metric") == {"Virus": False, "Metric": False}


def test_a_page_routed_without_jsx_keeps_its_own_create(tmp_path):
    """A router config imports the page as a value, never rendering `<TagList`:
    the page is not an api client, so the POST it issues itself still counts."""
    files = {
        "frontend/src/pages/TagList.jsx": ("export default function TagList() {\n"
                                           "  const save = () => fetch('/tag/', { method: 'POST', body });\n}\n"),
        "frontend/src/router.js": ("import TagList from './pages/TagList.jsx';\n"
                                   "export const routes = [{ path: '/tags', Component: TagList }];\n"),
    }
    assert _creates(tmp_path, files, "Tag") == {"Tag": True}


def test_a_context_provider_that_posts_is_not_an_api_client(tmp_path):
    """Pages call `useData()` from a provider that renders JSX and issues the POSTs;
    treating every called-into module as an api client dropped those creates."""
    files = {
        "frontend/src/data/DataContext.jsx": (
            "import { createContext, useContext } from 'react';\nconst Ctx = createContext(null);\n"
            "export function DataProvider({ children }) {\n"
            "  const createBooking = (b) => fetch('/booking/', { method: 'POST', body: JSON.stringify(b) });\n"
            "  const createRoom = (r) => fetch('/room/', { method: 'POST', body: JSON.stringify(r) });\n"
            "  return <Ctx.Provider value={{ createBooking, createRoom }}>{children}</Ctx.Provider>;\n}\n"
            "export const useData = () => useContext(Ctx);\n"),
        "frontend/src/pages/BookingPage.jsx": (
            "import { useData } from '../data/DataContext';\nexport default function BookingPage() {\n"
            "  const { createBooking } = useData();\n"
            "  return <form onSubmit={() => createBooking({})}><h1>Bookings</h1></form>;\n}\n"),
        "frontend/src/pages/RoomPage.jsx": (
            "import { useData } from '../data/DataContext';\nexport default function RoomPage() {\n"
            "  const { createRoom } = useData();\n"
            "  return <form onSubmit={() => createRoom({})}><h1>Rooms</h1></form>;\n}\n"),
    }
    assert _creates(tmp_path, files, "Booking", "Room") == {"Booking": True, "Room": True}


def test_a_page_that_posts_and_exports_a_helper_keeps_its_own_create(tmp_path):
    """Another file reads the page's field list and label helper; the page is
    still a component, so the POST it issues itself counts."""
    files = {
        "frontend/src/pages/BookingPage.jsx": (
            "export const BOOKING_FIELDS = ['guest', 'checkIn'];\nexport default function BookingPage() {\n"
            "  const save = (b) => fetch('/booking/', { method: 'POST', body: JSON.stringify(b) });\n"
            "  return <form onSubmit={() => save({})}><h1>Bookings</h1></form>;\n}\n"),
        "frontend/src/pages/RoomPage.jsx": (
            "export function roomLabel(r) { return `Room ${r.number}`; }\nexport default function RoomPage() {\n"
            "  const save = (r) => fetch('/room/', { method: 'POST', body: JSON.stringify(r) });\n"
            "  return <Form onSubmit={() => save({})} />;\n}\n"),
        "frontend/src/pages/Summary.jsx": (
            "import { BOOKING_FIELDS } from './BookingPage';\nimport { roomLabel } from './RoomPage';\n"
            "export default function Summary({ rooms }) {\n"
            "  return <div>{BOOKING_FIELDS.join(', ')} {rooms.map((r) => roomLabel(r))}</div>;\n}\n"),
    }
    assert _creates(tmp_path, files, "Booking", "Room") == {"Booking": True, "Room": True}


def test_a_typescript_client_with_generics_is_still_an_api_client(tmp_path):
    """`request<T>` and `Promise<Tag>` are type arguments, not JSX: with no page
    calling createTag, the client's own POST is not Tag's create."""
    files = {
        "frontend/src/api.ts": (
            "async function request<T>(method: string, path: string, body?: unknown): Promise<T> {\n"
            "  const res = await fetch(path, { method, body: JSON.stringify(body) });\n"
            "  return res.json() as Promise<T>;\n}\n"
            "export const listTags = (): Promise<Tag> => request<Tag>('GET', '/tag/');\n"
            "export const createTag = (b: Tag) => request<Tag>('POST', '/tag/', b);\n"),
        "frontend/src/pages/TagList.tsx": (
            "import { listTags } from '../api';\nexport default function TagList() {\n"
            "  listTags();\n  return <ul></ul>;\n}\n"),
    }
    assert _creates(tmp_path, files, "Tag") == {"Tag": False}
