"""A class with an attribute and an association end of one name (a text attribute
``tags`` next to an association whose end is ``tags``).

The SQLAlchemy model declared a column and a relationship under that name, the
Create schema declared the field twice and the router passed the list of ids
into the column: every POST of the class, and of its subclasses, answered 500.
The generators now keep the association end and leave the attribute out.
"""
from besser.BUML.metamodel.structural import (
    BinaryAssociation, Class, DomainModel, Generalization, IntegerType, Multiplicity, Property, StringType,
)
from besser.generators.pydantic_classes import PydanticGenerator
from besser.generators.sql_alchemy import SQLAlchemyGenerator

SCRIPT = """
    author = client.post("/author/", json={"name": "Ann"}).json()["author"]["id"]
    tag = client.post("/tag/", json={"label": "howto"}).json()["tag"]["id"]
    article = client.post("/article/", json={"title": "Maps", "author": author, "tags": [tag]})
    tutorial = client.post("/tutorial/", json={"title": "Charts", "level": 2, "author": author, "tags": [tag]})
    article_id = article.json()["article"]["id"]
    tutorial_id = tutorial.json()["tutorial"]["id"]
    updated = client.put(f"/article/{article_id}/", json={"title": "Maps 2", "author": author, "tags": []})
    print(json.dumps({
        "created": [article.status_code, tutorial.status_code, updated.status_code],
        "article": client.get(f"/article/{article_id}/").json(),
        "tutorial": client.get(f"/tutorial/{tutorial_id}/").json(),
        "detailed": client.get("/tutorial/", params={"detailed": True}).json(),
        "search": [client.get("/article/search/", params={"title": "Maps"}).status_code,
                   client.get("/tutorial/search/", params={"title": "Charts"}).json()],
        "filters": sorted(p["name"] for p in client.get("/openapi.json").json()
                          ["paths"]["/tutorial/search/"]["get"]["parameters"]),
    }))
"""


def _model() -> DomainModel:
    author = Class(name="Author", attributes={Property(name="name", type=StringType)})
    tag = Class(name="Tag", attributes={Property(name="label", type=StringType)})
    article = Class(name="Article", attributes={
        Property(name="title", type=StringType),
        Property(name="tags", type=StringType),
        Property(name="author", type=StringType),
    })
    tutorial = Class(name="Tutorial", attributes={Property(name="level", type=IntegerType)})
    tagging = BinaryAssociation(name="Tagging", ends={
        Property(name="tags", type=tag, multiplicity=Multiplicity(0, "*")),
        Property(name="tagged", type=article, multiplicity=Multiplicity(0, "*")),
    })
    writing = BinaryAssociation(name="Writing", ends={
        Property(name="author", type=author, multiplicity=Multiplicity(1, 1)),
        Property(name="articles", type=article, multiplicity=Multiplicity(0, "*")),
    })
    return DomainModel(name="Blog", types={author, tag, article, tutorial}, associations={tagging, writing},
                       generalizations={Generalization(general=article, specific=tutorial)})


def test_the_class_and_its_subclass_are_created_and_read_back(run_generated_backend):
    result = run_generated_backend(_model(), SCRIPT)
    assert result["created"] == [200, 200, 200]

    article = result["article"]
    assert article["article"]["title"] == "Maps 2" and article["article"]["author_id"] == 1
    assert article["tag_ids"] == []

    tutorial = result["tutorial"]
    assert tutorial["tutorial"]["level"] == 2 and tutorial["tutorial"]["author_id"] == 1
    assert tutorial["tag_ids"] == [1]
    listed = result["detailed"][0]
    assert listed["author"]["name"] == "Ann"
    assert [tag["label"] for tag in listed["tags"]] == ["howto"]

    assert result["search"][0] == 200
    assert [item["title"] for item in result["search"][1]] == ["Charts"]
    assert result["filters"] == ["level", "title"]


def test_each_name_is_emitted_once(tmp_path):
    SQLAlchemyGenerator(model=_model(), output_dir=str(tmp_path)).generate()
    PydanticGenerator(model=_model(), backend=True, output_dir=str(tmp_path)).generate()
    orm = (tmp_path / "sql_alchemy.py").read_text(encoding="utf-8")
    schema = (tmp_path / "pydantic_classes.py").read_text(encoding="utf-8")

    assert "    tags: Mapped_[str]" not in orm and "    author: Mapped_[str]" not in orm
    assert "Article.tags: Mapped_[List_[\"Tag\"]]" in orm and "author_id: Mapped_[int]" in orm
    article_schema = schema[schema.index("class ArticleCreate"):schema.index("class TutorialCreate")]
    assert article_schema.count("    tags:") == 1 and article_schema.count("    author:") == 1
    assert "tags: Optional[List[int]] = None" in article_schema
    assert "author: int" in article_schema
