"""Method parameters keep their declared order through a B-UML export (#679).

The builder wrote ``parameters={Parameter(...), ...}``, a set literal: loading
the exported script put the parameters in set order, so a re-export (and the
editor after an import) showed ``configure(alpha, delta, beta, ...)`` for a
method declared ``configure(alpha, beta, gamma, ...)``. Parameters are an
ordered list in the metamodel, and the builder now writes a list.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROUNDTRIP = textwrap.dedent('''
    import sys
    from pathlib import Path
    from besser.BUML.metamodel.structural import Class, DomainModel, Method, Parameter, StringType
    from besser.utilities.buml_code_builder.domain_model_builder import domain_model_to_code
    from besser.utilities.web_modeling_editor.backend.services.converters.buml_to_json.class_diagram_converter import (
        parse_buml_content,
    )

    out = Path(sys.argv[1])
    names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    method = Method(name="configure", parameters=[Parameter(name=n, type=StringType) for n in names])
    model = DomainModel(name="Devices", types={Class(name="Device", methods={method})})
    domain_model_to_code(model, str(out / "first.py"))
    loaded = parse_buml_content((out / "first.py").read_text())
    domain_model_to_code(loaded, str(out / "second.py"))
    loaded_method = next(iter(loaded.get_class_by_name("Device").methods))
    (out / "names.txt").write_text(",".join(p.name for p in loaded_method.parameters))
''')

SEEDS = ("0", "1", "2", "3")
DECLARED = "alpha,beta,gamma,delta,epsilon,zeta"


def _roundtrip(out: Path, seed: str) -> dict[str, bytes]:
    out.mkdir()
    repo = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo)}
    subprocess.run([sys.executable, "-c", ROUNDTRIP, str(out)], env=env, check=True,
                   cwd=repo, capture_output=True)
    return {name: (out / name).read_bytes() for name in ("first.py", "second.py", "names.txt")}


def test_reexport_keeps_declared_parameter_order(tmp_path):
    files = _roundtrip(tmp_path / "out", "0")
    assert files["names.txt"].decode() == DECLARED
    assert files["second.py"] == files["first.py"]


def test_reexport_is_identical_across_processes(tmp_path):
    runs = {seed: _roundtrip(tmp_path / seed, seed) for seed in SEEDS}
    assert len({files["second.py"] for files in runs.values()}) == 1
