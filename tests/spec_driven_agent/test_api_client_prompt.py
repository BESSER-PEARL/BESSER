"""The prompt points frontend code at the generated typed client, when there is one.

The endpoint section told the model to call the raw routes "from the frontend
verbatim" and described the response wrappers the generated client hides, so a
model that saw the client in the inventory still wrote its own fetch layer. With
a client in the workspace the section names it, lists its functions from the
contract it was rendered from, and keeps only the routes the client does not
cover as raw routes (for test_api and calls the agent adds itself).
"""
import json

from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, IntegerType, Method, MethodImplementationType, Multiplicity,
    Property, StringType,
)
from besser.generators.backend import BackendGenerator
from besser.generators.backend.api_contract import build_api_contract
from besser.generators.react import ReactGenerator
from besser.spec_driven_agent.agent.prompt_builder import (
    API_CLIENT_SECTION_CHARS, build_endpoint_manifest, build_inventory,
)


def _model() -> DomainModel:
    customer = Class(name="Customer", attributes={Property(name="name", type=StringType)})
    order = Class(name="Order", attributes={Property(name="total", type=IntegerType)})
    close = Method(name="close", parameters=[], type=None, implementation_type=MethodImplementationType.BAL)
    close.code = "def close() -> nothing {\n    this.total = this.total\n}"
    order.add_method(close)
    placed = BinaryAssociation(name="Placed", ends={
        Property(name="orders", type=order, multiplicity=Multiplicity(0, "*")),
        Property(name="customer", type=customer, multiplicity=Multiplicity(0, 1))})
    return DomainModel(name="Shop", types={customer, order}, associations={placed})


def _workspace(tmp_path, model=None, custom_route=False):
    """A web app's backend and typed client, laid out as the web app generator writes them."""
    model = model or _model()
    BackendGenerator(model, output_dir=str(tmp_path / "web_app" / "backend")).generate()
    ReactGenerator(model, None, output_dir=str(tmp_path / "web_app" / "frontend")).generate_api_client()
    if custom_route:
        (tmp_path / "web_app" / "backend" / "routers" / "report.py").write_text(
            "from fastapi import APIRouter\nrouter = APIRouter()\n"
            '@router.get("/report/summary/")\ndef summary(): ...\n', encoding="utf-8")
    return str(tmp_path), model


def test_the_section_names_the_client_and_its_functions(tmp_path):
    workspace, model = _workspace(tmp_path)

    section = build_endpoint_manifest(workspace, response_model=model, use_client=True)

    assert "web_app/frontend/src/api/generated/client.ts" in section
    assert "verbatim" not in section
    assert "apiRequest(method, path" in section
    assert "- customerApi (Customer, body CustomerCreate): CRUD(customer_id) at /customer/" in section
    assert "; relations.orders: list(customer_id) GET /customer/{customer_id}/orders/" in section
    assert "; methods: close(order_id) POST /order/{order_id}/methods/close/" in section
    # Every served route is in the client: no heading over an empty route list.
    assert "The backend serves no other routes" in section
    assert "not in the client" not in section
    assert "\n\n\n" not in section


def test_routes_the_client_covers_are_not_listed_again_but_custom_ones_are(tmp_path):
    workspace, model = _workspace(tmp_path, custom_route=True)

    section = build_endpoint_manifest(workspace, response_model=model, use_client=True)
    routes = [line for line in section.splitlines() if line.startswith("  ") and " /" in line]

    assert any(line.endswith(" /report/summary/") for line in routes)
    assert not any("/customer/" in line or "/order/" in line for line in routes)
    # The system routes are the client's apiSystem object, read from the contract.
    assert "- apiSystem: root() GET /; health() GET /health; statistics() GET /statistics" in section
    assert not any(line.endswith((" /", " /health", " /statistics")) for line in routes)
    # The response shapes and request bodies test_api needs stay.
    assert "Response shapes (generated routers)" in section
    assert "Request shapes" in section


def test_without_a_client_the_section_is_the_route_manifest(tmp_path):
    model = _model()
    BackendGenerator(model, output_dir=str(tmp_path / "backend")).generate()

    with_flag = build_endpoint_manifest(str(tmp_path), response_model=model, use_client=True)

    assert with_flag == build_endpoint_manifest(str(tmp_path), response_model=model)
    assert "verbatim" in with_flag


def test_the_validators_still_see_every_served_route(tmp_path):
    """Checks parse the manifest built without use_client: the client must not hide routes from them."""
    workspace, model = _workspace(tmp_path)

    manifest = build_endpoint_manifest(workspace)

    assert "/customer/{customer_id}/" in manifest and "/order/{order_id}/methods/close/" in manifest


def test_the_class_listing_is_capped_and_the_rest_stay_raw_routes(tmp_path):
    classes = {Class(name=f"Entity{index:02d}", attributes={Property(name="label", type=StringType)})
               for index in range(60)}
    workspace, model = _workspace(tmp_path, DomainModel(name="Wide", types=classes))

    section = build_endpoint_manifest(workspace, response_model=model, use_client=True)
    listed = [line for line in section.splitlines() if line.startswith("- entity")]

    assert sum(len(line) + 1 for line in listed) <= API_CLIENT_SECTION_CHARS
    assert len(listed) < 60
    assert f"{60 - len(listed)} more classes" in section
    # A class past the cap keeps its raw routes.
    assert "/entity59/{entity59_id}/" in section


def test_the_client_is_read_from_the_contract_not_from_router_source(tmp_path):
    """A router edit that leaves the contract alone does not change the client listing."""
    workspace, model = _workspace(tmp_path)
    contract = json.loads((tmp_path / "web_app" / "backend" / "api_contract.json").read_text(encoding="utf-8"))
    assert contract == json.loads(json.dumps(build_api_contract(model)))
    (tmp_path / "web_app" / "backend" / "routers" / "customer.py").write_text("", encoding="utf-8")

    section = build_endpoint_manifest(workspace, response_model=model, use_client=True)

    assert "- customerApi (Customer, body CustomerCreate): CRUD(customer_id) at /customer/" in section


def test_the_inventory_keeps_functions_ahead_of_type_exports_under_its_cap(tmp_path):
    """A client of many types and per-class objects still names every object it can."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "client.ts").write_text(
        "".join(f"export interface Shape{index} {{\n  id: number;\n}}\n" for index in range(9))
        + "export const apiRequest = () => null;\nexport const orderApi = {};\nexport const taskApi = {};\n",
        encoding="utf-8")

    inventory = build_inventory(str(tmp_path), None, "x")

    line = next(line for line in inventory.splitlines() if "client.ts" in line)
    assert "apiRequest@28, orderApi@29, taskApi@30, Shape0@1" in line and "(+2 more)" in line, line


def test_the_inventory_lists_type_exports_and_component_props(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "types.ts").write_text(
        'export type Status = "OPEN" | "DONE";\n'
        "export interface Order {\n  id: number;\n}\n"
        "export enum Color { Red }\n", encoding="utf-8")
    (source / "Panel.tsx").write_text(
        "interface PanelOptions {\n  title?: string;\n  rows: number;\n}\n"
        "export interface PanelProps {\n  id: string;\n  options?: PanelOptions; // how it looks\n"
        "  onSelect?: (row: any) => void;\n}\n"
        "export const Panel = (props: PanelProps) => null;\n", encoding="utf-8")

    inventory = build_inventory(str(tmp_path), None, "x")

    types_line = next(line for line in inventory.splitlines() if "types.ts" in line)
    assert "Status@1" in types_line and "Order@2" in types_line and "Color@5" in types_line
    panel_line = next(line for line in inventory.splitlines() if "Panel.tsx" in line)
    assert "PanelOptions{title?, rows}" in panel_line
    assert "PanelProps{id, options?, onSelect?}" in panel_line
    assert "Panel@10" in panel_line
