"""Opening a GitHub repo BESSER did not create: its stack, and a model read from its code."""

import asyncio
import os
import textwrap

import httpx
from httpx._transports.asgi import ASGITransport

from besser.utilities.web_modeling_editor.backend.backend import app
from besser.utilities.web_modeling_editor.backend.routers import spec_driven_router
from besser.utilities.web_modeling_editor.backend.services.reverse_engineering import analyze_repository
from besser.utilities.web_modeling_editor.backend.services.spec_driven import runner as runner_module
from besser.utilities.web_modeling_editor.backend.services.spec_driven.runner import run_changes

SQLALCHEMY_MODELS = '''
from typing import Optional
from sqlalchemy import Column, Date, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Room(Base):
    __tablename__ = "rooms"
    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False)
    capacity = Column(Integer)

class Booking(Base):
    __tablename__ = "bookings"
    id: Mapped[int] = mapped_column(primary_key=True)
    day: Mapped[Optional[Date]] = mapped_column(Date)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id"))
    room = relationship("Room")
'''

DJANGO_MODELS = '''
from django.db import models

class Category(models.Model):
    title = models.CharField(max_length=40)

class Product(models.Model):
    name = models.CharField(max_length=80)
    price = models.DecimalField(max_digits=6, decimal_places=2)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    tags = models.ManyToManyField("Tag", blank=True)

class Tag(models.Model):
    label = models.CharField(max_length=20)
'''


def _repo(root, files):
    for path, content in files.items():
        target = os.path.join(root, *path.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(textwrap.dedent(content))
    return str(root)


def _attrs(model, name):
    cls = next(c for c in model.get_classes() if c.name == name)
    return {a.name: (a.type.name, a.is_id, a.multiplicity.min) for a in cls.attributes}


def _ends(model):
    return sorted(
        sorted(f"{e.name}->{e.type.name}[{e.multiplicity.min}..{e.multiplicity.max}]" for e in a.ends)
        for a in model.associations
    )


def test_sqlalchemy_models_become_classes_and_foreign_keys_associations(tmp_path):
    root = _repo(tmp_path, {
        "app/main.py": "from fastapi import FastAPI\napp = FastAPI()\n",
        "app/models.py": SQLALCHEMY_MODELS,
        # API schemas restate the ORM; the ORM wins.
        "app/schemas.py": "from pydantic import BaseModel\nclass RoomOut(BaseModel):\n    id: int\n",
        "web/package.json": '{"dependencies": {"react": "18.3.1"}}',
    })

    analysis, model = analyze_repository(root)

    assert analysis.stack == ["FastAPI", "React"]
    assert analysis.model_source == "SQLAlchemy models"
    assert analysis.entities == ["Booking", "Room"]
    assert _attrs(model, "Room") == {"id": ("int", True, 1), "name": ("str", False, 1), "capacity": ("int", False, 0)}
    assert _attrs(model, "Booking") == {"id": ("int", True, 1), "day": ("date", False, 0)}
    assert _ends(model) == [["bookings->Booking[0..9999]", "room->Room[1..1]"]]


def test_django_models_with_foreign_keys_and_many_to_many(tmp_path):
    analysis, model = analyze_repository(_repo(tmp_path, {"manage.py": "", "shop/models.py": DJANGO_MODELS}))

    assert analysis.stack == ["Django"] and analysis.model_source == "Django models"
    assert _attrs(model, "Product") == {"name": ("str", False, 1), "price": ("float", False, 1)}
    assert _ends(model) == [
        ["category->Category[1..1]", "products->Product[0..9999]"],
        ["product_set->Product[0..9999]", "tags->Tag[0..9999]"],
    ]


def test_a_repo_without_data_classes_reports_its_stack_and_no_model(tmp_path):
    analysis, model = analyze_repository(_repo(tmp_path, {"web/package.json": '{"dependencies": {"vue": "3"}}'}))

    assert model is None
    assert analysis.stack == ["Vue"] and analysis.entities == []


def test_code_is_read_never_run(tmp_path):
    marker = tmp_path / "ran.txt"
    _repo(tmp_path, {"models.py": f"open({str(marker)!r}, 'w').write('x')\nfrom pydantic import BaseModel\n"
                                  "class Item(BaseModel):\n    name: str\n"})

    _analysis, model = analyze_repository(str(tmp_path))

    assert not marker.exists()
    assert [c.name for c in model.get_classes()] == ["Item"]


class _ImportGitHub:
    def __init__(self, files):
        self.files = files

    async def get_branches(self, owner, repo_name, per_page=100):
        return ["main"]

    async def get_default_branch(self, owner, repo_name):
        return "main"

    async def get_branch_head(self, owner, repo_name, branch):
        return "b" * 40

    async def download_repo_tarball(self, owner, repo, ref, dest_dir=None):
        return _repo(dest_dir, self.files)


def test_importing_a_non_besser_repo_proposes_a_model_the_editor_can_load(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_module, "LLM_RUN_WORKSPACE_ROOT", str(tmp_path))
    fake = _ImportGitHub({"app/main.py": "from fastapi import FastAPI\napp = FastAPI()\n", "app/models.py": SQLALCHEMY_MODELS})
    monkeypatch.setattr(spec_driven_router, "create_github_service", lambda token: fake)
    monkeypatch.setattr(spec_driven_router, "get_user_token", lambda sid: "token")

    async def post():
        async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.post("/besser_api/spec-driven/import-github-run",
                                     json={"owner": "me", "repo": "rooms"}, headers={"X-GitHub-Session": "s"})
    r = asyncio.run(post())

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_model"] is False and body["run_id"]  # the code is still the next run's base
    assert body["analysis"]["entities"] == ["Booking", "Room"]
    diagram = body["proposed_diagram"]
    assert diagram["diagramType"] == "ClassDiagram" and diagram["model"]["type"] == "ClassDiagram"
    names = {e.get("name") for e in diagram["model"]["elements"].values() if e.get("type") == "Class"}
    assert names == {"Booking", "Room"}


def test_a_finished_run_says_what_it_changed():
    recipe = {
        "history": [{"mode": "create"}, {"mode": "modify"}],
        "tool_calls": [
            {"tool": "modify_file", "input": {"path": "web_app/backend/main_api.py"}},
            {"tool": "write_file", "input": {"path": "web_app/frontend/src/Rooms.tsx"}},
            {"tool": "modify_file", "input": {"path": "web_app/backend/main_api.py"}},
            {"tool": "read_file", "input": {"path": "README.md"}},
            {"tool": "delete_file", "input": {"path": "old.py"}},
        ],
        "model_change": {"changes": ["class Room attribute capacity: int added"]},
    }

    assert run_changes(recipe) == {
        "mode": "modify",
        "changedFiles": ["old.py", "web_app/backend/main_api.py", "web_app/frontend/src/Rooms.tsx"],
        "changedFileCount": 3,
        "modelChanges": ["class Room attribute capacity: int added"],
    }


GENERATED = os.path.join(os.path.dirname(__file__), "fixtures", "besser_generated_sql_alchemy.py.txt")


def test_a_besser_generated_backend_reads_back_with_its_types_enums_and_relationships(tmp_path):
    """BESSER's own generator writes aliased names (Mapped_, Integer_, ForeignKey_, dt_datetime).

    Read live from github.com/ArmenSl/taskflow-besser: every attribute came back
    as str and no foreign key became an association.
    """
    with open(GENERATED, encoding="utf-8") as fh:
        root = _repo(tmp_path, {"backend/sql_alchemy.py": fh.read()})
    _analysis, model = analyze_repository(root)

    item = _attrs(model, "TodoItem")
    assert item["id"] == ("int", True, 1)
    assert item["isArchived"][0] == "bool"
    assert item["dueDate"][:3:2] == ("date", 0)
    assert item["completedAt"][0] == "datetime"
    assert item["status"][0] == "TodoStatus" and item["priority"][0] == "Priority"
    assert "todolist_id" not in item
    assert {e.name for e in model.get_enumerations()} == {"TodoStatus", "Priority"}
    assert {lit.name for lit in next(e for e in model.get_enumerations() if e.name == "Priority").literals} == {
        "LOW", "MEDIUM", "HIGH", "URGENT"}

    ends = _ends(model)
    # Role names are the code's own relationship names.
    assert ["items->TodoItem[0..9999]", "todolist->TodoList[1..1]"] in ends
    assert ["user->User[0..1]", "userComment->Comment[0..9999]"] in ends
    # The ``tags`` secondary table is a many-to-many association.
    assert ["tags->Tag[0..9999]", "todoitem->TodoItem[0..9999]"] in ends


def test_django_models_extending_the_repos_own_abstract_base_are_read(tmp_path):
    """github.com/gothinkster/django-realworld-example-app: every model extends ``TimestampedModel``.

    Read live, the proposal was that abstract base alone, with no Article, Comment or User.
    """
    root = _repo(tmp_path, {
        "manage.py": "",
        "conduit/apps/core/models.py": '''
            from django.db import models

            class TimestampedModel(models.Model):
                created_at = models.DateTimeField(auto_now_add=True)

                class Meta:
                    abstract = True
            ''',
        "conduit/apps/articles/models.py": '''
            from django.db import models
            from conduit.apps.core.models import TimestampedModel

            class Article(TimestampedModel):
                title = models.CharField(max_length=255)
                author = models.ForeignKey('profiles.Profile', on_delete=models.CASCADE, related_name='articles')
                tags = models.ManyToManyField('articles.Tag', related_name='articles')

            class Tag(TimestampedModel):
                tag = models.CharField(max_length=255)
            ''',
        "conduit/apps/profiles/models.py": '''
            from django.db import models
            from conduit.apps.core.models import TimestampedModel

            class Profile(TimestampedModel):
                user = models.OneToOneField('authentication.User', on_delete=models.CASCADE)
                bio = models.TextField(blank=True)
                follows = models.ManyToManyField('self', related_name='followed_by', symmetrical=False)
            ''',
        "conduit/apps/authentication/models.py": '''
            from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
            from django.db import models
            from conduit.apps.core.models import TimestampedModel

            class User(AbstractBaseUser, PermissionsMixin, TimestampedModel):
                email = models.EmailField(unique=True)
            ''',
    })

    analysis, model = analyze_repository(root)

    assert analysis.entities == ["Article", "Profile", "Tag", "User"]
    assert _attrs(model, "Article") == {"title": ("str", False, 1), "created_at": ("datetime", False, 1)}
    # AbstractBaseUser declares the password the RealWorld spec asks for.
    assert _attrs(model, "User")["password"][0] == "str"
    ends = _ends(model)
    assert ["articles->Article[0..9999]", "author->Profile[1..1]"] in ends
    assert ["articles->Article[0..9999]", "tags->Tag[0..9999]"] in ends
    assert ["followed_by->Profile[0..9999]", "follows->Profile[0..9999]"] in ends
    assert ["profile->Profile[0..1]", "user->User[1..1]"] in ends


def test_flask_sqlalchemy_types_mixins_column_helpers_and_self_many_to_many(tmp_path):
    """github.com/gothinkster/flask-realworld-example-app (the cookiecutter-flask layout).

    Read live, every type came back as str (``db.String(80)``), ``reference_col`` foreign
    keys were lost with five of six relations, and the ``SurrogatePK`` mixin became a class.
    """
    root = _repo(tmp_path, {
        "conduit/database.py": '''
            from .extensions import db

            Column = db.Column
            Model = db.Model

            class SurrogatePK(object):
                id = db.Column(db.Integer, primary_key=True)

            def reference_col(tablename, nullable=False, pk_name='id', **kwargs):
                return db.Column(db.ForeignKey('{0}.{1}'.format(tablename, pk_name)), nullable=nullable, **kwargs)
            ''',
        "conduit/user/models.py": '''
            from conduit.database import Column, Model, SurrogatePK, db

            class User(SurrogatePK, Model):
                __tablename__ = 'users'
                username = Column(db.String(80), unique=True, nullable=False)
                created_at = Column(db.DateTime, nullable=False)
            ''',
        "conduit/profile/models.py": '''
            from conduit.database import Model, SurrogatePK, db, reference_col, relationship

            followers_assoc = db.Table("followers_assoc",
                                       db.Column("follower", db.Integer, db.ForeignKey("userprofile.user_id")),
                                       db.Column("followed_by", db.Integer, db.ForeignKey("userprofile.user_id")))

            class UserProfile(Model, SurrogatePK):
                __tablename__ = 'userprofile'
                user_id = reference_col('users', nullable=False)
                user = relationship('User', backref=db.backref('profile', uselist=False))
                follows = relationship('UserProfile', secondary=followers_assoc, backref='followed_by')
            ''',
        "conduit/articles/models.py": '''
            from conduit.database import Column, Model, SurrogatePK, db, reference_col, relationship

            class Comment(Model, SurrogatePK):
                __tablename__ = 'comment'
                body = Column(db.Text)
                author_id = reference_col('userprofile', nullable=False)
                author = relationship('UserProfile', backref=db.backref('comments'))
            ''',
    })

    analysis, model = analyze_repository(root)

    assert analysis.entities == ["Comment", "User", "UserProfile"]
    assert _attrs(model, "User") == {
        "id": ("int", True, 1), "username": ("str", False, 1), "created_at": ("datetime", False, 1)}
    ends = _ends(model)
    assert ["profile->UserProfile[0..9999]", "user->User[1..1]"] in ends
    assert ["author->UserProfile[1..1]", "comments->Comment[0..9999]"] in ends
    assert ["followed_by->UserProfile[0..9999]", "follows->UserProfile[0..9999]"] in ends


def test_pydantic_models_extending_the_repos_own_base_model_are_read(tmp_path):
    """github.com/nsidnev/fastapi-realworld-example-app: domain models extend ``RWModel(BaseModel)``."""
    root = _repo(tmp_path, {
        "app/models/domain/rwmodel.py": "from pydantic import BaseModel\nclass RWModel(BaseModel):\n    pass\n",
        "app/models/domain/users.py": "from app.models.domain.rwmodel import RWModel\n"
                                      "class User(RWModel):\n    username: str\n",
    })

    analysis, _model = analyze_repository(root)

    assert analysis.entities == ["User"]


def test_sqlalchemy_joined_inheritance_is_a_generalization(tmp_path):
    """BESSER's SQLAlchemy generator writes ``Author(Person)`` with a key joining the two tables and a
    ``type_spec`` discriminator; the round-trip benchmark read no generalization and an
    ``Author -> Person`` association instead. A self link's two ends came back as ``authors``."""
    root = _repo(tmp_path, {"backend/sql_alchemy.py": '''
        from sqlalchemy import ForeignKey, String
        from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

        class Base(DeclarativeBase):
            pass

        class Person(Base):
            __tablename__ = "person"
            id: Mapped[int] = mapped_column(primary_key=True)
            name: Mapped[str] = mapped_column(String(100))
            type_spec: Mapped[str] = mapped_column(String(50))
            __mapper_args__ = {"polymorphic_identity": "person", "polymorphic_on": "type_spec"}

        class Author(Person):
            __tablename__ = "author"
            id: Mapped[int] = mapped_column(ForeignKey("person.id"), primary_key=True)
            email: Mapped[str] = mapped_column(String(100))
            mentor_id: Mapped[int] = mapped_column(ForeignKey("author.id"), nullable=True)
            __mapper_args__ = {"polymorphic_identity": "author"}

        Author.mentees: Mapped[list["Author"]] = relationship(
            "Author", back_populates="mentor", foreign_keys=[Author.mentor_id])
        Author.mentor: Mapped["Author"] = relationship(
            "Author", back_populates="mentees", uselist=False, foreign_keys=[Author.mentor_id],
            remote_side=[Author.id])
        '''})

    _analysis, model = analyze_repository(root)

    assert {(g.specific.name, g.general.name) for g in model.generalizations} == {("Author", "Person")}
    assert _attrs(model, "Person") == {"id": ("int", True, 1), "name": ("str", False, 1)}
    assert _attrs(model, "Author") == {"email": ("str", False, 1)}
    assert _ends(model) == [["mentees->Author[0..9999]", "mentor->Author[0..1]"]]


def test_django_multi_table_inheritance_and_text_choices(tmp_path):
    """BESSER's Django generator writes ``class Genre(models.TextChoices)`` and
    ``CharField(choices=Genre.choices)``: the enum and the generalization were both lost."""
    root = _repo(tmp_path, {"library/models.py": '''
        from django.db import models

        class Genre(models.TextChoices):
            FICTION = 'FICTION', 'FICTION'
            SCIENCE = 'SCIENCE', 'SCIENCE'

        class Person(models.Model):
            name = models.CharField(max_length=255)

        class Author(Person):
            email = models.CharField(max_length=255)

        class Book(models.Model):
            class Format(models.IntegerChoices):
                PAPER = 1
                EBOOK = 2

            genre = models.CharField(max_length=255, choices=Genre.choices)
            format = models.IntegerField(choices=Format.choices)
        '''})

    _analysis, model = analyze_repository(root)

    assert {(g.specific.name, g.general.name) for g in model.generalizations} == {("Author", "Person")}
    assert {e.name: {lit.name for lit in e.literals} for e in model.get_enumerations()} == {
        "Genre": {"FICTION", "SCIENCE"}, "Format": {"PAPER", "EBOOK"}}
    assert _attrs(model, "Book") == {"genre": ("Genre", False, 1), "format": ("Format", False, 1)}


def test_a_subclass_does_not_repeat_the_fields_it_inherits(tmp_path):
    """Pydantic subclasses restate parent fields (``UserInDB(User)``); B-UML inherits them."""
    root = _repo(tmp_path, {"app/models.py": '''
        from typing import Optional
        from pydantic import BaseModel

        class User(BaseModel):
            username: str
            email: str

        class UserInDB(User):
            email: Optional[str]
            hashed_password: str
        '''})

    _analysis, model = analyze_repository(root)

    assert {(g.specific.name, g.general.name) for g in model.generalizations} == {("UserInDB", "User")}
    assert set(_attrs(model, "UserInDB")) == {"hashed_password"}


def test_pydantic_links_declared_on_both_sides_and_key_lists_are_one_association_each(tmp_path):
    """BESSER's Pydantic generator writes each end on its class, and a many-to-many as a list of ids."""
    root = _repo(tmp_path, {"app/models.py": '''
        from typing import List, Optional
        from pydantic import BaseModel

        class Library(BaseModel):
            name: str
            books: Optional[List["Book"]] = None  # 1:N Relationship (optional)

        class Book(BaseModel):
            title: str
            authors: List[int]  # N:M Relationship
            library: "Library"  # N:1 Relationship

        class Author(BaseModel):
            email: str
            mentees: Optional[List["Author"]] = None
            mentor: Optional["Author"] = None
            books: Optional[List[int]] = None
        '''})

    _analysis, model = analyze_repository(root)

    assert _ends(model) == [
        ["authors->Author[0..9999]", "books->Book[0..9999]"],
        ["books->Book[0..9999]", "library->Library[1..1]"],
        ["mentees->Author[0..9999]", "mentor->Author[0..1]"],
    ]
    assert set(_attrs(model, "Book")) == {"title"}
    assert set(_attrs(model, "Author")) == {"email"}
