"""The acceptance matrix attributes each frontend create to the entity it creates.

Observed on a live run: every list page rendered one shared ``EntityList``
component that called ``api.post(path, payload)`` with the page's ``path`` prop,
and the api module issued the request through ``request('POST', ...)``.

* ``EntityList`` was not seen as a generic page, because its doc comment gave
  ``"/shop/"`` as an example path, so the two entities whose only create went
  through it were reported "no frontend create path" although both were
  creatable in the browser.
* The entities that did pass, passed for the wrong reason: a "run" page that
  named them in dropdowns POSTed to a different resource, and any POST in a file
  that mentions an entity counted as that entity's create.
"""

from besser.BUML.metamodel.structural import Class, DomainModel, PrimitiveDataType, Property
from besser.spec_driven_agent.validation.acceptance import build_acceptance_matrix

_STR = PrimitiveDataType("str")

API = """\
const BASE = 'http://localhost:8000';
async function request(method, path, body) {
  const res = await fetch(`${BASE}${path}`, { method, body: JSON.stringify(body) });
  return res.json();
}
export const api = {
  get: (path) => request('GET', path),
  post: (path, body) => request('POST', path, body ?? {}),
};
"""

ENTITY_LIST = """\
import { useState } from 'react';
import { api } from '../api';
/**
 * Generic CRUD page.
 *  path - collection route with trailing slash, e.g. "/shop/"
 */
export default function EntityList({ title, path }) {
  const [form, setForm] = useState({});
  const submit = async () => { await api.post(path, form); };
  return <form onSubmit={submit}><h1>{title}</h1></form>;
}
"""


def _page(name, path):
    return (
        "import EntityList from '../components/EntityList';\n"
        f"export default function {name}() {{\n"
        f"  return <EntityList title=\"{name}\" path=\"{path}\" />;\n"
        "}\n"
    )


RUN_PAGE = """\
import { api } from '../api';
export default function RunTracker({ shops, products }) {
  const run = async (shopId, productId) =>
    api.post('/stockcheck/execute/', { shop: shopId, product: productId });
  return <div>{shops.length} shops, {products.length} products</div>;
}
"""


def _model(*names):
    classes = set()
    for name in names:
        cls = Class(name=name)
        cls.attributes = {Property(name="label", type=_STR)}
        classes.add(cls)
    return DomainModel(name="Shop", types=classes)


def _workspace(tmp_path, files):
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(tmp_path)


def _app(tmp_path, entity_list=ENTITY_LIST):
    return _workspace(tmp_path, {
        "frontend/src/api.js": API,
        "frontend/src/components/EntityList.jsx": entity_list,
        "frontend/src/pages/ShopList.jsx": _page("ShopList", "/shop/"),
        "frontend/src/pages/ShopURLList.jsx": _page("ShopURLList", "/shopurl/"),
        "frontend/src/pages/RunTracker.jsx": RUN_PAGE,
    })


def test_a_shared_list_component_whose_comment_names_an_entity_still_creates(tmp_path):
    matrix = build_acceptance_matrix(_app(tmp_path), _model("Shop", "ShopURL", "Product", "StockCheck"))
    assert matrix["Shop"]["create"] is True
    assert matrix["ShopURL"]["create"] is True


def test_a_post_to_another_resource_is_not_this_entitys_create(tmp_path):
    matrix = build_acceptance_matrix(_app(tmp_path), _model("Shop", "ShopURL", "Product", "StockCheck"))
    assert matrix["Product"]["page"] is True
    assert matrix["Product"]["create"] is False
    assert matrix["StockCheck"]["create"] is True


def test_a_shared_component_that_names_an_entity_in_code_is_not_generic(tmp_path):
    shop_only = ENTITY_LIST.replace("await api.post(path, form);", "await api.post(path, { ...form, shop: 1 });")
    matrix = build_acceptance_matrix(_app(tmp_path, entity_list=shop_only), _model("Shop", "ShopURL"))
    assert matrix["ShopURL"]["create"] is False


def test_a_resource_path_with_separators_still_names_the_entity(tmp_path):
    root = _workspace(tmp_path, {
        "frontend/src/pages/OrderLines.jsx": (
            "import api from '../api';\n"
            "export const save = (line) => api.post('/order-lines/', line);\n"
        ),
    })
    matrix = build_acceptance_matrix(root, _model("OrderLine"))
    assert matrix["OrderLine"]["create"] is True
