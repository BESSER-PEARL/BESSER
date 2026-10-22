"""Every Flutter file lands in ``output_dir`` (#679).

FlutterPubspecGenerator passed ``output_dir`` to the base class as the model,
so its output_dir was None and pubspec.yaml went to ``<cwd>/output``.
"""
from besser.BUML.metamodel.gui import GUIModel, Module, Screen
from besser.BUML.metamodel.structural import Class, DomainModel, Property, StringType
from besser.generators.flutter import FlutterGenerator


def test_all_files_written_to_output_dir(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    out = tmp_path / "out"

    item = Class(name="Item", attributes={Property(name="name", type=StringType)})
    model = DomainModel(name="Shop", types={item})
    home = Screen(name="Home", description="", x_dpi="x", y_dpi="y", screen_size="Small",
                  view_elements=set(), is_main_page=True)
    module = Module(name="main", screens={home})
    gui = GUIModel(name="shop", package="com.example.shop", versionCode="1", versionName="1.0",
                   description="", screenCompatibility=True, modules={module})

    FlutterGenerator(model, gui, home, module=module, output_dir=str(out)).generate()

    assert sorted(p.name for p in out.iterdir()) == ["main.dart", "pubspec.yaml", "sql_helper.dart"]
    assert not any(cwd.rglob("*")), "files were written into the working directory"
