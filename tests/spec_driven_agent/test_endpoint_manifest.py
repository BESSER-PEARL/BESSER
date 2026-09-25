"""Tests for build_endpoint_manifest — the exact backend route list handed to
the Phase-2 agent so the LLM-authored frontend stops guessing URLs (the #1
"builds but 404s" failure).

Correctness is about producing the EXACT served path: right prefix, right
trailing slash, no invented plurals. These pin the parser against the real
generator shape (full path in the decorator, no prefix) plus the defensive
prefix cases.
"""
from besser.spec_driven_agent.agent.prompt_builder import build_endpoint_manifest, build_mutation_manifest


def _mk(tmp_path, files: dict) -> str:
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return str(tmp_path)


def test_full_path_in_decorator_no_prefix(tmp_path):
    # The real BackendGenerator shape: APIRouter() with no prefix, full path
    # (singular, trailing slash) baked into each decorator.
    d = _mk(tmp_path, {
        "backend/main_api.py": (
            "from fastapi import FastAPI\n"
            "from routers import book as book_router\n"
            "app = FastAPI()\n"
            "app.include_router(book_router.router)\n"
            'if __name__ == "__main__":\n'
            "    import uvicorn\n"
            '    uvicorn.run(app, host="0.0.0.0", port=8000)\n'
        ),
        "backend/routers/book.py": (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            '@router.get("/book/")\n'
            "def list_book(): ...\n"
            '@router.post("/book/")\n'
            "def create_book(): ...\n"
            '@router.get("/book/{book_id}/")\n'
            "def get_book(): ...\n"
            '@router.put("/book/{book_id}/")\n'
            "def update_book(): ...\n"
            '@router.delete("/book/{book_id}/")\n'
            "def delete_book(): ...\n"
        ),
    })
    m = build_endpoint_manifest(d)
    assert "http://localhost:8000" in m
    # exact served paths, singular, trailing slash
    assert "/book/" in m
    assert "/book/{book_id}/" in m
    # methods merged per path
    assert "GET, POST" in m
    assert "GET, PUT, DELETE" in m
    # never invents a plural or an /api prefix as an actual route line
    assert "/books/" not in m
    assert "\n  GET                  /api" not in m


def test_apirouter_prefix_is_prepended(tmp_path):
    d = _mk(tmp_path, {
        "app/main.py": "from fastapi import FastAPI\napp = FastAPI()\n",
        "app/routers/item.py": (
            "from fastapi import APIRouter\n"
            'router = APIRouter(prefix="/items")\n'
            '@router.get("/")\n'
            "def list_items(): ...\n"
            '@router.get("/{item_id}")\n'
            "def get_item(): ...\n"
        ),
    })
    m = build_endpoint_manifest(d)
    assert "/items/" in m
    assert "/items/{item_id}" in m


def test_include_router_prefix_via_import_alias(tmp_path):
    d = _mk(tmp_path, {
        "main.py": (
            "from fastapi import FastAPI\n"
            "from routers import user as user_router\n"
            "app = FastAPI()\n"
            'app.include_router(user_router.router, prefix="/api/v1")\n'
        ),
        "routers/user.py": (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            '@router.get("/user/")\n'
            "def list_user(): ...\n"
        ),
    })
    m = build_endpoint_manifest(d)
    assert "/api/v1/user/" in m


def test_no_routes_returns_empty(tmp_path):
    d = _mk(tmp_path, {"backend/database.py": "engine = None\n"})
    assert build_endpoint_manifest(d) == ""


def test_default_port_when_no_uvicorn(tmp_path):
    d = _mk(tmp_path, {
        "routers/x.py": (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            '@router.get("/x/")\n'
            "def x(): ...\n"
        ),
    })
    m = build_endpoint_manifest(d)
    assert "http://localhost:8000" in m


def test_skips_node_modules(tmp_path):
    d = _mk(tmp_path, {
        "routers/real.py": (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            '@router.get("/real/")\n'
            "def r(): ...\n"
        ),
        "frontend/node_modules/pkg/decoy.py": (
            'router.get("/should_not_appear/")\n'
        ),
    })
    m = build_endpoint_manifest(d)
    assert "/real/" in m
    assert "should_not_appear" not in m


def test_no_double_slashes(tmp_path):
    d = _mk(tmp_path, {
        "main.py": (
            "from routers import a as a_router\n"
            'app.include_router(a_router.router, prefix="/api/")\n'
        ),
        "routers/a.py": (
            "from fastapi import APIRouter\n"
            'router = APIRouter(prefix="/a/")\n'
            '@router.get("/")\n'
            "def a(): ...\n"
        ),
    })
    m = build_endpoint_manifest(d)
    # inspect only the route lines (indented "METHOD  /path"), not the http:// header
    route_lines = [ln for ln in m.splitlines() if ln.startswith("  ") and "/" in ln]
    assert route_lines
    assert all("//" not in ln for ln in route_lines)
    assert "/api/a/" in m


def test_relationship_mutation_map_covers_native_reverse_inherited_and_bulk_inputs(tmp_path):
    d = _mk(tmp_path, {
        "backend/main_api.py": (
            "app = FastAPI()\n"
            "from routers import staff as staff_router, product as product_router, item as item_router\n"
            "app.include_router(staff_router.router, prefix='/api')\n"
            "app.include_router(product_router.router, prefix='/api')\n"
            "app.include_router(item_router.router, prefix='/api')\n"
        ),
        "backend/sql_alchemy.py": (
            "class Customer(Base):\n    __tablename__ = 'customer'\n"
            "class Staff(Customer):\n    __tablename__ = 'staff'\n"
            "class Order(Base):\n    __tablename__ = 'order'\n"
            "class Product(Base):\n    __tablename__ = 'product'\n"
            "class OrderItem(Base):\n    __tablename__ = 'item'\n"
            "Customer.orders = relationship('Order')\n"
            "Order.products = relationship('Product', secondary=OrderItem.__table__, viewonly=True)\n"
            "Product.orders = relationship('Order', secondary=OrderItem.__table__, viewonly=True)\n"
            "OrderItem.order = relationship('Order')\n"
            "OrderItem.products = relationship('Product')\n"
        ),
        "backend/pydantic_classes.py": (
            "class CustomerCreate(BaseModel):\n    name: str\n    orders: list[int]\n"
            "class StaffCreate(CustomerCreate):\n    title: str\n"
            "class OrderItemLinkCreate(BaseModel):\n    target: int\n    agreedAmount: float\n"
            "class ProductCreate(BaseModel):\n    capacity: int\n    orders: list['OrderItemLinkCreate']\n"
            "class OrderItemCreate(BaseModel):\n    order: int\n    products: int\n    agreedAmount: float\n"
        ),
        "backend/routers/staff.py": (
            "router = APIRouter(prefix='/staff')\n"
            "@router.post('/')\ndef create_staff(data: StaffCreate):\n    return Staff()\n"
            "@router.post('/bulk/')\ndef bulk_staff(items: list[StaffCreate]):\n    return []\n"
        ),
        "backend/routers/product.py": (
            "router = APIRouter(prefix='/products')\n"
            "@router.put('/{id}/')\ndef edit_product(data: ProductCreate):\n    return Product()\n"
            "@router.delete('/{id}/')\ndef delete_product(id: int):\n    return Product()\n"
        ),
        "backend/routers/item.py": (
            "router = APIRouter()\n"
            "@router.post('/item/')\ndef create_item(data: OrderItemCreate):\n    return OrderItem()\n"
            "@router.delete('/item/{order}/{products}/')\ndef delete_item(order: int, products: int):\n    return OrderItem()\n"
        ),
    })
    manifest = build_endpoint_manifest(d)
    assert "Relationship mutation coverage" in manifest
    assert "Order <-> Product via OrderItem" in manifest
    assert "OrderItem.products->Product" in manifest and "OrderItem.product->" not in manifest
    assert "StaffCreate: name:str, orders:list[int], title:str" in manifest
    assert "OrderItemLinkCreate: agreedAmount:float, target:int" in manifest
    assert "POST /api/staff/bulk/ body=StaffCreate" in manifest
    assert "PUT /api/products/{id}/ body=ProductCreate" in manifest
    assert "DELETE /api/item/{order}/{products}/" in manifest
    assert "@backend/routers/item.py:" in manifest
    customer_group = next(line for line in manifest.splitlines() if line.startswith("- Customer <-> Order:"))
    staff_ids = [line.split()[1] for line in manifest.splitlines() if line.startswith("- W") and " /api/staff/" in line]
    assert staff_ids and all(identifier in customer_group for identifier in staff_ids)
    assert "UNKNOWN mount" not in manifest
    assert "BEFORE any recompute action" in manifest and "source AND destination" in manifest


def test_mutation_inventory_marks_unknowns_and_budget_omissions_without_executing_source(tmp_path):
    d = _mk(tmp_path, {
        "backend/models.py": (
            "raise RuntimeError('must never execute generated code')\n"
            "class Item(Base):\n    __tablename__ = 'item'\n"
            "class Owner(Base):\n    __tablename__ = 'owner'\n"
            "Item.owners = relationship('Owner')\n"
        ),
        "backend/routes.py": (
            "router = APIRouter(prefix=configuration.prefix)\n"
            "@router.api_route('/item/{id}/', methods=['PATCH', 'DELETE'])\n"
            "def write_item(data: dict):\n    return Item()\n"
        ),
        "tests/decoy.py": "@router.post('/never/')\ndef decoy(): pass\n",
    })
    manifest = build_mutation_manifest(d)
    assert "PATCH /item/{id}/" in manifest and "DELETE /item/{id}/" in manifest
    assert "UNKNOWN mount" in manifest and "UNKNOWN input shape" in manifest
    assert "/never/" not in manifest
    clipped = build_mutation_manifest(d, max_chars=700)
    assert len(clipped) <= 700 and "TRUNCATED" in clipped and "UNVERIFIED" in clipped


def test_mutation_observations_use_executed_paths_body_shapes_and_freshness_not_scenario_names(tmp_path):
    d = _mk(tmp_path, {
        "backend/main.py": (
            "app = FastAPI()\n"
            "class Product(Base):\n    __tablename__ = 'product'\n"
            "class Order(Base):\n    __tablename__ = 'order'\n"
            "Product.orders = relationship('Order')\n"
            "class ProductCreate(BaseModel):\n    name: str\n    capacity: int\n    orders: list[int]\n"
            "@app.post('/products/')\ndef create_product(data: ProductCreate):\n    return Product()\n"
            "@app.put('/products/{id}/')\ndef update_product(data: ProductCreate):\n    return Product()\n"
        ),
    })

    def record(name, revision, requests, responses, *, passed=True):
        return {"scenario_id": name, "revision": revision,
                "scenario": {"requests": requests, "backend": "backend"},
                "report": {"boot": "ok", "status": "passed" if passed else "failed", "responses": responses}}

    records = [
        record("update_completely_tested", "now", [
            {"method": "POST", "path": "/products/", "json": {"name": "private-fixture-value", "capacity": 2}},
        ], [{"index": 0, "method": "POST", "path": "/products/", "status": 200, "json": {"id": 7}}]),
        record("old_nonempty_links", "before", [
            {"method": "PUT", "path": "/products/7/", "json": {"orders": [9]}},
        ], [{"index": 0, "method": "PUT", "path": "/products/7/", "status": 200}]),
        record("never_dispatched", "now", [
            {"method": "PUT", "path": "/products/{{0.id}}/", "json": {"orders": [10]}},
        ], [{"index": 0, "method": "PUT", "path": "/products/{{0.id}}/", "failures": ["no preceding value"]}], passed=False),
        record("current_refusal", "now", [
            {"method": "GET", "path": "/products/7/"},
            {"method": "PUT", "path": "/products/{{0.id}}/", "json": {"orders": "{{0.orders}}"}, "expected_status": 409},
        ], [{"index": 0, "method": "GET", "path": "/products/7/", "status": 200, "json": {"id": 7, "orders": []}},
            {"index": 1, "method": "PUT", "path": "/products/7/", "status": 409}]),
        record("old_report_without_revision", None, [
            {"method": "POST", "path": "/products/", "json": {"orders": [4]}},
        ], [{"index": 0, "method": "POST", "path": "/products/", "status": 200}]),
    ]
    manifest = build_mutation_manifest(d, scenario_records=records, current_revision="now")
    post = next(line for line in manifest.splitlines() if "POST /products/: requests=" in line)
    put = next(line for line in manifest.splitlines() if "PUT /products/{id}/: requests=" in line)
    assert "'current': 1" in post and "'unknown-revision': 1" in post
    assert "relation keys NOT supplied in 1/1 current bodies: orders" in post
    assert "'stale': 1" in put and "'unexecuted': 1" in put and "'current': 1" in put
    assert '"orders"=array:empty' in put and "current:HTTP409" in put and "stale:HTTP200" in put
    assert "array:nonempty" not in put  # Old/non-dispatched bodies cannot become current observations.
    assert "S1[0]" in post and "S1[0]" not in put  # A misleading scenario name does not cover PUT.
    assert 'id="never_dispatched": current; executed=0/1; saved assertions=failed/unknown' in manifest
    assert "private-fixture-value" not in manifest
    assert "NOT invariant coverage" in manifest and "not a requirement to test every route" in manifest
    unknown = build_mutation_manifest(d, scenario_records=records[:1])
    assert "unknown-revision" in unknown and "current supplied JSON=NONE observed" in unknown
