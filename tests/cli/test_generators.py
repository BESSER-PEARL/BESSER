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
    assert "    --option http_methods=GET,POST,PUT,PATCH,DELETE\n" in stdout
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
        "supports_regenerate": True,
    }
    assert by_id["sqlalchemy"]["options"]["dbms"]["enum"][:2] == ["sqlite", "postgresql"]


def test_json_says_which_generators_regenerate_supports(besser):
    supported = {entry["id"]: entry["supports_regenerate"] for entry in besser("generators", "--json").json["generators"]}

    assert supported == {spec.id: spec.load_class().regenerable for spec in GENERATORS}
    assert supported["python"] is True and supported["flutter"] is False and supported["supabase"] is False


def test_the_list_ends_with_the_generators_regenerate_supports(besser):
    last_line = besser("generators").stdout.splitlines()[-1]

    assert last_line.startswith("'besser regenerate' keeps hand edits with: python, java, ")
    assert "sqlalchemy" in last_line and "flutter" not in last_line and "terraform" not in last_line


def test_every_generator_says_what_it_generates():
    assert all(spec.summary for spec in GENERATORS)


def test_quiet_and_verbose_print_nothing_on_standard_error(besser):
    for flag in ("-q", "-v"):
        run = besser("generators", flag)

        assert run.returncode == 0
        assert run.stderr == ""
