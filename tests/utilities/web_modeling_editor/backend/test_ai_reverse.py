"""AI fallback reader: a repo with no data classes the exact reader understands gets a model proposed by an LLM."""

import asyncio
import json
import os
import textwrap
import time

import httpx
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.routers import spec_driven_router
from besser.utilities.web_modeling_editor.backend.services.reverse_engineering import (
    ai_reverse,
    analyze_repository,
    propose_model_with_ai,
)
from besser.utilities.web_modeling_editor.backend.services.reverse_engineering.ai_reverse import (
    AI_MODEL_SOURCE,
    build_repo_digest,
)
from besser.utilities.web_modeling_editor.backend.services.reverse_engineering.code_reverse import CodeAnalysis
from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner as runner_module

PRISMA = '''
model Customer {
  id     Int     @id @default(autoincrement())
  email  String  @unique
  orders Order[]
}

model Order {
  id         Int      @id
  placedAt   DateTime
  customer   Customer @relation(fields: [customerId], references: [id])
  customerId Int
}
'''

TS_TYPES = '''
export interface Order {
  id: number;
  total: number;
  status: OrderStatus;
}
export type OrderStatus = "OPEN" | "PAID";
'''

SECRET = "sk-live-abcdefghijklmnopqrstuvwxyz123456"

SPEC = {
    "systemName": "Shop",
    "classes": [
        {"className": "Customer", "attributes": [{"name": "email", "type": "String"}]},
        {"className": "Order", "attributes": [
            {"name": "total", "type": "number"},
            {"name": "placedAt", "type": "DateTime"},
            {"name": "status", "type": "OrderStatus"},
            {"name": "customer", "type": "Customer"},  # a reference, not an attribute
        ]},
    ],
    "enumerations": [{"name": "OrderStatus", "literals": ["OPEN", "PAID"]}],
    "relationships": [{"type": "Association", "source": "Customer", "target": "Order",
                       "sourceMultiplicity": "1", "targetMultiplicity": "*", "name": "orders"}],
}


def _repo(root, files):
    for path, content in files.items():
        target = os.path.join(root, *path.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(textwrap.dedent(content))
    return str(root)


def _shop_repo(root):
    return _repo(root, {
        "README.md": "# Shop\nAn online shop with customers and orders.\n",
        "prisma/schema.prisma": PRISMA,
        "src/types/order.ts": TS_TYPES,
        "src/app.ts": "export const start = () => 1;\n",
        "node_modules/lib/types/index.ts": "export interface Noise { x: number }\n",
        ".env": f"OPENAI_API_KEY={SECRET}\n",
        "package.json": '{"dependencies": {"react": "18.3.1"}}',
    })


def _free_tier(monkeypatch):
    monkeypatch.setenv("BESSER_FREE_LLM_BASE_URL", "http://free.test/v1")
    monkeypatch.setenv("BESSER_FREE_LLM_MODEL", "free-model")


def test_digest_reads_schema_and_types_and_skips_dependencies_and_secrets(tmp_path):
    digest = build_repo_digest(_shop_repo(tmp_path))

    assert digest.files == ["prisma/schema.prisma", "src/types/order.ts", "README.md"]
    assert "model Customer" in digest.text and "export interface Order" in digest.text
    assert "node_modules" not in digest.text and "Noise" not in digest.text
    assert ".env" not in digest.text and SECRET not in digest.text
    assert "src/app.ts" in digest.text  # listed in the tree, not quoted


def test_digest_reads_plain_classes_in_an_mvc_model_folder(tmp_path):
    """github.com/ArmenSl/MVC-Java-App: plain Java classes in Model/, no @Entity.

    The live reading quoted only README.txt, so the classes were guessed from prose.
    """
    root = _repo(tmp_path, {
        "Model/Enfant.java": "public class Enfant extends Personne {\n    private int age;\n}\n",
        "Model/Reservation.java": "public class Reservation {\n    private java.util.Date jour;\n}\n",
        "Controller/controleReservation.java": "public class controleReservation {\n}\n",
        "Vue/pageReservation.java": "public class pageReservation extends javax.swing.JFrame {\n}\n",
        "README.txt": "Projet Java cantine\n",
    })
    files = build_repo_digest(root).files
    assert "Model/Enfant.java" in files and "Model/Reservation.java" in files
    assert "Controller/controleReservation.java" not in files
    assert "Vue/pageReservation.java" not in files


def test_digest_redacts_tokens_inside_quoted_files(tmp_path):
    root = _repo(tmp_path, {"db/schema.sql": f"-- token {SECRET}\nCREATE TABLE item (id int);\n"})

    digest = build_repo_digest(root)

    assert digest.files == ["db/schema.sql"] and SECRET not in digest.text


def test_digest_quotes_at_most_the_file_cap(tmp_path):
    files = {f"db/{n:02}.sql": "CREATE TABLE t (id int);\n" for n in range(12)}

    digest = build_repo_digest(_repo(tmp_path, files))

    assert digest.files == [f"db/{n:02}.sql" for n in range(ai_reverse._MAX_DATA_FILES)]


def test_digest_truncates_large_files_and_stays_within_the_total_cap(tmp_path):
    files = {f"db/{n:02}.sql": "CREATE TABLE t (id int);\n" + "-- filler\n" * 2000 for n in range(12)}

    digest = build_repo_digest(_repo(tmp_path, files))

    assert len(digest.text) <= ai_reverse._MAX_DIGEST_CHARS
    assert digest.text.count("[... truncated]") == len(digest.files) > 1


def test_reply_becomes_a_validated_model(monkeypatch, tmp_path):
    reply = "Here is the model:\n```json\n" + json.dumps({
        **SPEC,
        "classes": SPEC["classes"] + [{"className": "bad name!"}, {"className": "1st"}, {"className": "Order"}],
        "relationships": SPEC["relationships"] + [{"type": "Association", "source": "Order", "target": "Ghost"}],
    }) + "\n```"
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt: reply)
    analysis = CodeAnalysis()

    model = propose_model_with_ai(_shop_repo(tmp_path), analysis)

    assert sorted(c.name for c in model.get_classes()) == ["Customer", "Order", "bad_name"]
    order = next(c for c in model.get_classes() if c.name == "Order")
    assert {a.name: a.type.name for a in order.attributes} == {
        "total": "float", "placedAt": "datetime", "status": "OrderStatus"}
    assert [e.name for e in model.get_enumerations()] == ["OrderStatus"]
    assert len(model.associations) == 1
    assert analysis.ai is True and analysis.model_source == AI_MODEL_SOURCE
    assert analysis.entities == ["Customer", "Order", "bad_name"]
    assert analysis.source_files == ["prisma/schema.prisma", "src/types/order.ts", "README.md"]


def test_inheritance_source_is_the_subclass_and_a_two_way_pair_is_dropped(monkeypatch, tmp_path):
    spec = {
        "classes": [{"className": "User"}, {"className": "Admin"}],
        "relationships": [{"type": "Inheritance", "source": "Admin", "target": "User"},
                          {"type": "Inheritance", "source": "User", "target": "Admin"}],
    }
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt: json.dumps(spec))

    model = propose_model_with_ai(_shop_repo(tmp_path), CodeAnalysis())

    assert [(g.specific.name, g.general.name) for g in model.generalizations] == [("Admin", "User")]


def test_class_count_is_capped(monkeypatch, tmp_path):
    spec = {"classes": [{"className": f"C{n}", "attributes": []} for n in range(50)]}
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt: json.dumps(spec))

    model = propose_model_with_ai(_shop_repo(tmp_path), CodeAnalysis())

    assert len(model.get_classes()) == ai_reverse._MAX_CLASSES


def test_unparseable_reply_gives_no_model(monkeypatch, tmp_path):
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt: "I cannot help with that.")
    analysis = CodeAnalysis()

    assert propose_model_with_ai(_shop_repo(tmp_path), analysis) is None
    assert analysis.ai is False


def test_the_exact_reader_is_not_marked_ai(tmp_path):
    root = _repo(tmp_path, {"models.py": "from pydantic import BaseModel\nclass Item(BaseModel):\n    name: str\n"})

    analysis, _model = analyze_repository(root)

    assert analysis.to_dict()["ai"] is False


class _ImportGitHub:
    def __init__(self, files):
        self.files = files

    async def get_branches(self, owner, repo_name, per_page=100):
        return ["main"]

    async def get_branch_head(self, owner, repo_name, branch):
        return "c" * 40

    async def download_repo_tarball(self, owner, repo, ref, dest_dir=None):
        return _repo(dest_dir, self.files)


def _import(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_module, "LLM_RUN_WORKSPACE_ROOT", str(tmp_path))
    fake = _ImportGitHub({"prisma/schema.prisma": PRISMA, "src/types/order.ts": TS_TYPES})
    monkeypatch.setattr(spec_driven_router, "create_github_service", lambda token: fake)
    monkeypatch.setattr(spec_driven_router, "get_user_token", lambda sid: "token")

    async def post():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.post("/besser_api/spec-driven/import-github-run",
                                     json={"owner": "me", "repo": "shop"}, headers={"X-GitHub-Session": "s"})
    return asyncio.run(post())


def test_import_proposes_an_ai_model_when_the_code_has_no_readable_data_classes(monkeypatch, tmp_path):
    _free_tier(monkeypatch)
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt: json.dumps(SPEC))

    r = _import(monkeypatch, tmp_path)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["analysis"]["ai"] is True
    assert body["analysis"]["model_source"] == AI_MODEL_SOURCE
    assert body["analysis"]["source_files"] == ["prisma/schema.prisma", "src/types/order.ts"]
    assert body["analysis"]["entities"] == ["Customer", "Order"]
    assert "an AI model read its code" in body["message"]
    diagram = body["proposed_diagram"]
    assert diagram["diagramType"] == "ClassDiagram" and diagram["model"]["type"] == "ClassDiagram"
    elements = diagram["model"]["elements"].values()
    assert {e["name"] for e in elements if e.get("type") == "Class"} == {"Customer", "Order"}
    assert {e["name"] for e in elements if e.get("type") == "Enumeration"} == {"OrderStatus"}


def test_import_succeeds_without_a_proposal_when_the_ai_fails(monkeypatch, tmp_path):
    _free_tier(monkeypatch)

    def boom(prompt):
        raise RuntimeError("upstream down")
    monkeypatch.setattr(ai_reverse, "_ask_model", boom)

    r = _import(monkeypatch, tmp_path)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["proposed_diagram"] is None and body["run_id"]
    assert body["analysis"]["ai"] is False
    assert "could not read a model" in body["message"]


def test_a_slow_ai_reading_times_out_and_the_import_still_succeeds(monkeypatch, tmp_path):
    _free_tier(monkeypatch)
    monkeypatch.setattr(spec_driven_router, "AI_READ_TIMEOUT_SECONDS", 0.05)

    def slow(prompt):
        time.sleep(0.5)
        return json.dumps(SPEC)
    monkeypatch.setattr(ai_reverse, "_ask_model", slow)

    r = _import(monkeypatch, tmp_path)

    assert r.status_code == 200, r.text
    assert r.json()["proposed_diagram"] is None and r.json()["analysis"]["ai"] is False


def test_import_skips_the_ai_when_the_server_has_no_keyless_model(monkeypatch, tmp_path):
    monkeypatch.delenv("BESSER_FREE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("BESSER_FREE_LLM_MODEL", raising=False)
    calls = []
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt: calls.append(prompt) or "{}")

    r = _import(monkeypatch, tmp_path)

    assert r.status_code == 200, r.text
    assert r.json()["proposed_diagram"] is None and calls == []
    assert "AI reading is not available" in r.json()["message"]
