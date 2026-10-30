"""AI fallback reader: a repo with no data classes the exact reader understands gets a model proposed by an LLM."""

import asyncio
import importlib
import json
import os
import sys
import textwrap
import time

import httpx
import pytest
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.models.spec_driven import ImportGitHubRunRequest
from besser.utilities.web_modeling_editor.backend.routers import spec_driven_router
from besser.utilities.web_modeling_editor.backend.services import reverse_engineering
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


def test_source_files_named_after_keys_or_tokens_are_read_but_secret_files_are_not(tmp_path):
    """A substring match on "key"/"token" dropped entity classes like ``ApiKey.java`` and ``Keyword.java``."""
    root = _repo(tmp_path, {
        "src/main/java/app/model/ApiKey.java": "@Entity\npublic class ApiKey {\n    private String label;\n}\n",
        "src/main/java/app/model/Keyword.java": "@Entity\npublic class Keyword {\n    private String text;\n}\n",
        "src/types/token.ts": "export interface Token {\n  value: string;\n}\n",
        "config/secrets.yml": f"production:\n  api_key: {SECRET}\n",
        "credentials.json": f'{{"token": "{SECRET}"}}',
        "config/master.key": "0123456789abcdef\n",
        "id_rsa": "-----BEGIN OPENSSH PRIVATE KEY-----\n",
    })
    digest = build_repo_digest(root)

    assert {"src/main/java/app/model/ApiKey.java", "src/main/java/app/model/Keyword.java",
            "src/types/token.ts"} <= set(digest.files)
    for skipped in ("secrets.yml", "credentials.json", "master.key", "id_rsa"):
        assert skipped not in digest.text
    assert SECRET not in digest.text


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


def test_digest_reads_typeorm_entities_before_request_dtos(tmp_path):
    """github.com/lujakob/nestjs-realworld-example-app: the live digest quoted only ``dto/`` files."""
    entity = '''
        import { Entity, Column } from 'typeorm';
        @Entity('article')
        export class ArticleEntity {
          @Column()
          slug: string;
        }
        '''
    files = {f"src/article/dto/dto{n}.dto.ts": "export class CreateArticleDto {}" for n in range(8)}
    files["src/article/article.entity.ts"] = entity
    files["src/user/user.entity.ts"] = entity.replace("Article", "User")

    digest = build_repo_digest(_repo(tmp_path, files))

    assert digest.files[:2] == ["src/article/article.entity.ts", "src/user/user.entity.ts"]


def test_digest_labels_request_dtos_as_api_shapes(tmp_path):
    """github.com/lujakob/nestjs-realworld-example-app: the AI drew five ``*Dto`` classes beside the entities."""
    files = {
        "src/user/user.entity.ts": "@Entity('user')\nexport class UserEntity {\n  username: string;\n}\n",
        "src/user/dto/create-user.dto.ts": "export class CreateUserDto {\n  readonly username: string;\n}\n",
        "src/user/dto/index.ts": "export { CreateUserDto } from './create-user.dto';\n",
    }

    digest = build_repo_digest(_repo(tmp_path, files))

    assert digest.files[:2] == ["src/user/user.entity.ts", "src/user/dto/create-user.dto.ts"]
    assert "## File: src/user/dto/create-user.dto.ts (API request/response shapes)" in digest.text
    assert "src/user/dto/index.ts" not in digest.files


def test_prompt_keeps_hidden_credentials_drops_api_shapes_and_reads_id_named_links():
    """Live misses on the RealWorld benchmark: aspnetcore's ``[JsonIgnore] byte[] Hash`` was dropped,
    nestjs's ``FollowsEntity { followerId; followingId }`` became no link, and nestjs/fastapi request
    DTOs and JWT payloads were drawn as classes."""
    prompt = ai_reverse._USER_PROMPT.format(max_classes=30, digest="")

    assert "credentials (password hashes, salts)" in prompt and "byte[]" in prompt
    assert "followerId and followingId both refer to the user" in prompt
    assert "authentication token contents (JWT claims)" in prompt


def test_digest_reads_laravel_migrations_and_the_authenticatable_user(tmp_path):
    """github.com/gothinkster/laravel-realworld-example-app: the live digest quoted three models and no table."""
    files = {f"database/migrations/2017_0{n}_create_t{n}_table.php":
             f"<?php Schema::create('t{n}', function (Blueprint $table) {{ $table->string('name'); }});"
             for n in range(6)}
    files["app/User.php"] = "<?php class User extends Authenticatable implements JWTSubject { }"
    files["app/Article.php"] = "<?php class Article extends Model { }"

    digest = build_repo_digest(_repo(tmp_path, files))

    assert "app/User.php" in digest.files
    assert all(f"database/migrations/2017_0{n}_create_t{n}_table.php" in digest.files for n in range(6))


def test_digest_reads_alembic_migrations(tmp_path):
    """github.com/nsidnev/fastapi-realworld-example-app keeps its only table definitions in Alembic."""
    root = _repo(tmp_path, {"app/db/migrations/versions/fdf8_main_tables.py": API_SCHEMAS[
        "app/db/migrations/versions/fdf8_main_tables.py"]})

    assert build_repo_digest(root).files == ["app/db/migrations/versions/fdf8_main_tables.py"]


def test_digest_redacts_tokens_inside_quoted_files(tmp_path):
    root = _repo(tmp_path, {"db/schema.sql": f"-- token {SECRET}\nCREATE TABLE item (id int);\n"})

    digest = build_repo_digest(root)

    assert digest.files == ["db/schema.sql"] and SECRET not in digest.text


def test_digest_quotes_at_most_the_file_cap(tmp_path):
    files = {f"db/{n:02}.sql": "CREATE TABLE t (id int);\n" for n in range(ai_reverse._MAX_DATA_FILES + 4)}

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
        "classes": SPEC["classes"] + [{"className": "bad name!", "attributes": [{"name": "x", "type": "str"}]},
                                     {"className": "1st"}, {"className": "Order"}],
        "relationships": SPEC["relationships"] + [{"type": "Association", "source": "Order", "target": "Ghost"}],
    }) + "\n```"
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, *_: reply)
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
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, *_: json.dumps(spec))

    model = propose_model_with_ai(_shop_repo(tmp_path), CodeAnalysis())

    assert [(g.specific.name, g.general.name) for g in model.generalizations] == [("Admin", "User")]


def test_one_rejected_relationship_does_not_cost_the_others(monkeypatch, tmp_path):
    """Admin would inherit a second ``items`` end; the metamodel rejected the reply and every link was dropped."""
    spec = {
        "classes": [{"className": "User", "attributes": [{"name": "name", "type": "str"}]},
                    {"className": "Admin", "attributes": [{"name": "level", "type": "int"}]},
                    {"className": "Order", "attributes": [{"name": "total", "type": "float"}]},
                    {"className": "Invoice", "attributes": [{"name": "number", "type": "str"}]}],
        "relationships": [{"type": "Association", "source": "User", "target": "Order", "name": "items"},
                          {"type": "Association", "source": "Admin", "target": "Order", "name": "items"},
                          {"type": "Inheritance", "source": "Admin", "target": "User"},
                          {"type": "Association", "source": "Order", "target": "Invoice", "name": "invoices"}],
    }
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, *_: json.dumps(spec))
    analysis = CodeAnalysis()

    model = propose_model_with_ai(_shop_repo(tmp_path), analysis)

    assert len(model.associations) == 3
    assert model.generalizations == set()
    assert analysis.skipped_relationships == ["Admin -> User"]
    assert analysis.to_dict()["skipped_relationships"] == ["Admin -> User"]


def test_an_inheritance_cycle_through_three_classes_is_cut(monkeypatch, tmp_path):
    """A -> B -> C -> A crashed the metamodel mid-construction, and the reading lost every relationship."""
    spec = {
        "classes": [{"className": name, "attributes": [{"name": "x", "type": "str"}]} for name in "ABC"],
        "relationships": [{"type": "Inheritance", "source": "A", "target": "B"},
                          {"type": "Inheritance", "source": "B", "target": "C"},
                          {"type": "Inheritance", "source": "C", "target": "A"}],
    }
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, *_: json.dumps(spec))

    model = propose_model_with_ai(_shop_repo(tmp_path), CodeAnalysis())

    assert sorted((g.specific.name, g.general.name) for g in model.generalizations) == [("A", "B"), ("B", "C")]


def test_empty_unrelated_classes_are_dropped():
    """github.com/gothinkster/aspnetcore-realworld-example-app, 2nd benchmark run: the model drew each
    many-to-many directly and also kept its join tables as empty boxes with no link."""
    spec = ai_reverse._normalise_spec({"classes": [
        {"className": "Article", "attributes": [{"name": "title", "type": "str"}]},
        {"className": "Tag", "attributes": []},
        {"className": "ArticleTag", "attributes": []},
    ], "relationships": [{"source": "Article", "target": "Tag", "sourceMultiplicity": "*", "targetMultiplicity": "*"}]})

    assert [c["className"] for c in spec["classes"]] == ["Article", "Tag"]


def test_class_count_is_capped(monkeypatch, tmp_path):
    spec = {"classes": [{"className": f"C{n}", "attributes": []} for n in range(50)]}
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, *_: json.dumps(spec))

    model = propose_model_with_ai(_shop_repo(tmp_path), CodeAnalysis())

    assert len(model.get_classes()) == ai_reverse._MAX_CLASSES


def test_unparseable_reply_gives_no_model(monkeypatch, tmp_path):
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, *_: "I cannot help with that.")
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

    async def get_default_branch(self, owner, repo_name):
        return "main"

    async def get_branch_head(self, owner, repo_name, branch):
        return "c" * 40

    async def download_repo_tarball(self, owner, repo, ref, dest_dir=None):
        return _repo(dest_dir, self.files)


def _import(monkeypatch, tmp_path, files=None, **body):
    """Import a repo; the user allowed the AI reading unless ``allow_ai_reading=False`` says otherwise."""
    body = {"allow_ai_reading": True, **body}
    monkeypatch.setattr(runner_module, "LLM_RUN_WORKSPACE_ROOT", str(tmp_path))
    fake = _ImportGitHub(files or {"prisma/schema.prisma": PRISMA, "src/types/order.ts": TS_TYPES})
    monkeypatch.setattr(spec_driven_router, "create_github_service", lambda token: fake)
    monkeypatch.setattr(spec_driven_router, "get_user_token", lambda sid: "token")

    async def post():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.post("/besser_api/spec-driven/import-github-run",
                                     json={"owner": "me", "repo": "shop", **body},
                                     headers={"X-GitHub-Session": "s"})
    return asyncio.run(post())


def test_import_proposes_an_ai_model_when_the_code_has_no_readable_data_classes(monkeypatch, tmp_path):
    _free_tier(monkeypatch)
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, *_: json.dumps(SPEC))

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
    stereotypes = {n["data"]["name"]: n["data"].get("stereotype") for n in diagram["model"]["nodes"]}
    assert {name for name, kind in stereotypes.items() if kind != "Enumeration"} == {"Customer", "Order"}
    assert {name for name, kind in stereotypes.items() if kind == "Enumeration"} == {"OrderStatus"}


def test_import_succeeds_without_a_proposal_when_the_ai_fails(monkeypatch, tmp_path):
    _free_tier(monkeypatch)

    def boom(prompt, *_):
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

    def slow(prompt, *_):
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
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, *_: calls.append(prompt) or "{}")

    r = _import(monkeypatch, tmp_path)

    assert r.status_code == 200, r.text
    assert r.json()["proposed_diagram"] is None and calls == []
    assert "AI reading is not available" in r.json()["message"]


DEMO = "demo-secret-1234"


def _demo_tier(monkeypatch):
    monkeypatch.setenv("BESSER_DEMO_TOKEN", DEMO)
    monkeypatch.setenv("BESSER_SPONSORED_LLM_BASE_URL", "http://sponsored.test/v1")
    monkeypatch.setenv("BESSER_SPONSORED_LLM_TOKEN", "sponsored-token")
    monkeypatch.setenv("BESSER_SPONSORED_LLM_MODEL", "sponsored-model")


def test_a_demo_tab_reads_the_repo_with_the_demo_model(monkeypatch, tmp_path):
    """On experimental the keyless model was out of credits; demo links have a paid one."""
    _free_tier(monkeypatch)
    _demo_tier(monkeypatch)
    tiers = []
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, provider: tiers.append(provider) or json.dumps(SPEC))

    r = _import(monkeypatch, tmp_path, demo_token=DEMO)

    assert r.status_code == 200, r.text
    assert tiers == ["sponsored"]
    assert r.json()["analysis"]["ai"] is True


def test_a_wrong_demo_token_reads_with_the_keyless_model(monkeypatch, tmp_path):
    _free_tier(monkeypatch)
    _demo_tier(monkeypatch)
    tiers = []
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, provider: tiers.append(provider) or json.dumps(SPEC))

    r = _import(monkeypatch, tmp_path, demo_token="not-the-secret")

    assert r.status_code == 200, r.text
    assert tiers == ["free"]


def test_no_demo_secret_on_the_server_never_opens_the_demo_model(monkeypatch, tmp_path):
    _free_tier(monkeypatch)
    _demo_tier(monkeypatch)
    monkeypatch.delenv("BESSER_DEMO_TOKEN")
    tiers = []
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, provider: tiers.append(provider) or json.dumps(SPEC))

    r = _import(monkeypatch, tmp_path, demo_token=DEMO)

    assert r.status_code == 200, r.text
    assert tiers == ["free"]


API_SCHEMAS = {
    "app/models/schemas/users.py": '''
        from pydantic import BaseModel

        class UserInLogin(BaseModel):
            email: str
            password: str
        ''',
    "app/db/migrations/versions/fdf8_main_tables.py": '''
        from alembic import op

        def upgrade():
            op.create_table('users')
        ''',
}


def test_a_repo_with_only_pydantic_models_is_read_by_the_ai_first(monkeypatch, tmp_path):
    """Pydantic models and dataclasses are mostly API shapes: on the FastAPI RealWorld repo the exact
    reader proposed 8 request/response classes and no Article, and the AI never ran."""
    _free_tier(monkeypatch)
    calls = []
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, *_: calls.append(prompt) or json.dumps(SPEC))

    r = _import(monkeypatch, tmp_path, files=API_SCHEMAS)

    assert r.status_code == 200, r.text
    assert len(calls) == 1 and "op.create_table" in calls[0]
    assert r.json()["analysis"]["ai"] is True and r.json()["analysis"]["entities"] == ["Customer", "Order"]


def test_a_failed_ai_reading_falls_back_to_the_pydantic_models(monkeypatch, tmp_path):
    _free_tier(monkeypatch)

    def boom(prompt, *_):
        raise RuntimeError("upstream down")
    monkeypatch.setattr(ai_reverse, "_ask_model", boom)

    r = _import(monkeypatch, tmp_path, files=API_SCHEMAS)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["analysis"]["ai"] is False and body["analysis"]["entities"] == ["UserInLogin"]
    assert body["proposed_diagram"] is not None and "Pydantic models" in body["message"]


def test_orm_models_are_read_exactly_without_asking_the_ai(monkeypatch, tmp_path):
    _free_tier(monkeypatch)
    calls = []
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, *_: calls.append(prompt) or json.dumps(SPEC))
    models = '''
        from sqlalchemy import Column, Integer

        class Room(Base):
            __tablename__ = 'rooms'
            id = Column(Integer, primary_key=True)
        '''

    r = _import(monkeypatch, tmp_path, files={"app/models.py": models})

    assert r.status_code == 200, r.text
    assert calls == [] and r.json()["analysis"]["entities"] == ["Room"]


# ---------------------------------------------------------------------
# The AI reading is the user's choice
# ---------------------------------------------------------------------


def test_without_the_users_consent_the_code_goes_to_no_model(monkeypatch, tmp_path):
    """Review of #693: the import sent up to 40k characters of the repo's code to the free LLM chain,
    whose first provider is a third party, without asking."""
    _free_tier(monkeypatch)
    calls = []
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, *_: calls.append(prompt) or json.dumps(SPEC))

    r = _import(monkeypatch, tmp_path, allow_ai_reading=False)

    assert r.status_code == 200, r.text
    body = r.json()
    assert calls == []
    assert body["ai_reading_offered"] is True
    assert body["proposed_diagram"] is None and body["analysis"]["ai"] is False
    assert "could not read this repo's data classes directly" in body["message"]
    assert "if you allow it" in body["message"]


def test_without_consent_the_exact_reading_is_still_proposed(monkeypatch, tmp_path):
    _free_tier(monkeypatch)
    calls = []
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, *_: calls.append(prompt) or json.dumps(SPEC))

    r = _import(monkeypatch, tmp_path, files=API_SCHEMAS, allow_ai_reading=False)

    assert r.status_code == 200, r.text
    body = r.json()
    assert calls == []
    assert body["ai_reading_offered"] is True
    assert body["analysis"]["entities"] == ["UserInLogin"] and body["proposed_diagram"] is not None
    assert "Pydantic models" in body["message"] and "if you allow it" in body["message"]


def test_no_reading_is_offered_when_the_server_has_no_model(monkeypatch, tmp_path):
    monkeypatch.delenv("BESSER_FREE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("BESSER_FREE_LLM_MODEL", raising=False)

    r = _import(monkeypatch, tmp_path, allow_ai_reading=False)

    assert r.status_code == 200, r.text
    assert r.json()["ai_reading_offered"] is False
    assert "AI reading is not available" in r.json()["message"]


def test_an_allowed_reading_offers_nothing_more(monkeypatch, tmp_path):
    _free_tier(monkeypatch)
    monkeypatch.setattr(ai_reverse, "_ask_model", lambda prompt, *_: json.dumps(SPEC))

    r = _import(monkeypatch, tmp_path)

    assert r.status_code == 200, r.text
    assert r.json()["ai_reading_offered"] is False and r.json()["analysis"]["ai"] is True


def test_the_ai_reader_refuses_a_request_that_did_not_allow_it(tmp_path):
    request = ImportGitHubRunRequest(owner="me", repo="shop")

    with pytest.raises(PermissionError):
        asyncio.run(spec_driven_router._read_repo_with_ai(request, str(tmp_path), CodeAnalysis(), "me", "shop"))


def test_the_ai_reader_does_not_import_the_deprecated_kg_module(monkeypatch):
    """``besser.utilities.kg_to_buml`` is a deprecated alias that warns on import."""
    monkeypatch.delitem(sys.modules, "besser.utilities.kg_to_buml", raising=False)
    monkeypatch.delitem(sys.modules, ai_reverse.__name__)
    # Importing a submodule rebinds the package attribute; the original is put back after.
    monkeypatch.setattr(reverse_engineering, "ai_reverse", ai_reverse)

    importlib.import_module(ai_reverse.__name__)

    assert "besser.utilities.kg_to_buml" not in sys.modules
