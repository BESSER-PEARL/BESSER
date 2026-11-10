"""``besser generators``: the registry, as text and as JSON."""

from besser.generators.registry import GENERATORS, get_generator


def test_every_registered_generator_is_listed(besser):
    run = besser("generators")

    assert run.returncode == 0
    for spec in GENERATORS:
        assert f"\n{spec.id}  [{', '.join(spec.models)}]  {spec.summary}\n" in run.stdout + "\n"


def test_options_are_shown_the_way_they_are_typed(besser):
    stdout = besser("generators").stdout

    assert "    --option dbms=sqlite|postgresql|mysql|mssql|mariadb|oracle  (default: sqlite)\n" in stdout
    assert "    --option containerization=true|false  (default: false)\n" in stdout
    assert "    --option http_methods=GET,POST,PUT,PATCH,DELETE  (default: GET,POST,PUT,DELETE)\n" in stdout
    assert "    --option shots=<integer>\n" in stdout


def test_json_lists_each_generator_with_its_models_and_option_schemas(besser):
    result = besser("generators", "--json").json

    assert result["ok"] is True and result["errors"] == [] and result["files"] == []
    assert [entry["id"] for entry in result["generators"]] == [spec.id for spec in GENERATORS]
    by_id = {entry["id"]: entry for entry in result["generators"]}
    assert by_id["react"]["models"] == ["domain", "gui"]
    assert by_id["sqlalchemy"] == {
        "id": "sqlalchemy", "summary": "SQLAlchemy ORM models", "category": "database",
        "models": ["domain"], "options": get_generator("sqlalchemy").options,
    }
    assert by_id["sqlalchemy"]["options"]["dbms"]["enum"][:2] == ["sqlite", "postgresql"]


def test_every_generator_says_what_it_generates():
    assert all(spec.summary for spec in GENERATORS)
