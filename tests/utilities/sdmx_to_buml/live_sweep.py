"""Live sweep: import real SDMX dataflows and run every class-diagram generator on each model.

Not collected by pytest (it needs the network and takes minutes). Run it from the repository root:

    python -m tests.utilities.sdmx_to_buml.live_sweep --out sweep_report
    python -m tests.utilities.sdmx_to_buml.live_sweep --only ecb,eurostat --skip-generators

For each dataflow it checks: the import (the editor's safe mode), ``validate()``, the web-editor
round trip, then the Python, Pydantic, SQLAlchemy, SQL, JSON schema, Java, RDF, Django, REST API and
backend generators. Generated Python is imported (or compiled), SQLAlchemy tables are created in
an in-memory SQLite database, the SQL script is executed in SQLite, Turtle is parsed when ``rdflib``
is installed, Java is compiled when ``javac`` is on the path, and ``manage.py check`` runs on the
Django project. The report is a Markdown matrix plus the same results as JSON.
"""

import argparse
import importlib.util
import json
import py_compile
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from besser.utilities.sdmx_to_buml import SDMXClient, structure_to_buml

DATAFLOWS = [
    # (provider key or base URL, dataflow reference, label)
    ("eurostat", "nama_10_gdp", "Eurostat GDP"),
    ("eurostat", "une_rt_m", "Eurostat unemployment"),
    ("eurostat", "demo_pjan", "Eurostat population"),
    ("eurostat", "prc_hicp_manr", "Eurostat HICP"),
    ("eurostat", "env_air_gge", "Eurostat greenhouse gases"),
    ("eurostat-comext", "DS-045409", "Eurostat Comext trade"),
    ("lustat", "DSD_INT_TRAD@DF_E4103", "LUSTAT foreign trade"),
    ("lustat", "DF_A1100", "LUSTAT A1100"),
    ("lustat", "DF_A3100", "LUSTAT A3100"),
    ("lustat", "DF_A1102", "LUSTAT A1102"),
    ("ecb", "EXR", "ECB exchange rates"),
    ("ecb", "ICP", "ECB HICP"),
    ("ecb", "BSI", "ECB balance sheets"),
    ("oecd", "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,1.1", "OECD quarterly national accounts"),
    ("oecd", "OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0", "OECD unemployment"),
    ("oecd", "OECD.CFE.EDS,DSD_FUA_CLIM@DF_TEMPERATURES", "OECD city temperatures"),
    ("imf", "IMF.STA,CPI", "IMF CPI"),
    ("imf", "IMF.STA,BOP", "IMF balance of payments"),
    ("bis", "WS_CBPOL", "BIS policy rates"),
    ("bis", "WS_XRU", "BIS exchange rates"),
    ("ilo", "DF_UNE_DEAP_SEX_AGE_RT", "ILO unemployment rate"),
    ("ilo", "DF_EAP_DWAP_SEX_AGE_RT", "ILO participation rate"),
    ("unicef", "CME", "UNICEF child mortality"),
    ("abs", "CPI", "ABS CPI"),
    ("abs", "LF", "ABS labour force"),
    ("insee", "IPC-2015", "INSEE CPI"),
    ("insee", "CHOMAGE-TRIM-NATIONAL", "INSEE unemployment"),
    ("istat", "IT1,22_289", "Istat population"),
    ("unsd", "IAEG-SDGs,DF_SDG_GLH", "UN SDG indicators"),
    ("bundesbank", "BBEX3", "Bundesbank exchange rates"),
    ("bundesbank", "BBAF3", "Bundesbank financial accounts"),
    ("https://sdmx.oecd.org/public/rest", "OECD.SDD.STES,DSD_STES@DF_CLI", "OECD leading indicators (custom URL)"),
]

GENERATORS = ["python", "pydantic", "sqlalchemy", "sql", "json_schema", "java", "rdf", "django", "rest_api",
              "backend"]
CHECKS = ["import", "validate", "round_trip"] + GENERATORS


def _short(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ").replace("|", "/")
    return text[:220]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _compile_tree(folder: Path) -> int:
    count = 0
    for py in folder.rglob("*.py"):
        py_compile.compile(str(py), doraise=True)
        count += 1
    return count


def run_generator(name: str, model, out: Path) -> str:
    """Run one generator into ``out`` and check what it wrote; returns a short note."""
    # pylint: disable=import-outside-toplevel
    if name == "python":
        from besser.generators.python_classes import PythonGenerator
        PythonGenerator(model, output_dir=str(out)).generate()
        _load_module(out / "classes.py", f"sweep_py_{id(out)}")
        return "imported"
    if name == "pydantic":
        from besser.generators.pydantic_classes import PydanticGenerator
        PydanticGenerator(model, output_dir=str(out)).generate()
        _load_module(out / "pydantic_classes.py", f"sweep_pyd_{id(out)}")
        return "imported"
    if name == "sqlalchemy":
        from sqlalchemy import create_engine

        from besser.generators.sql_alchemy import SQLAlchemyGenerator
        SQLAlchemyGenerator(model, output_dir=str(out)).generate(dbms="sqlite")
        module = _load_module(out / "sql_alchemy.py", f"sweep_sa_{id(out)}")
        engine = create_engine("sqlite://")
        module.Base.metadata.create_all(engine)
        return f"{len(module.Base.metadata.tables)} tables created"
    if name == "sql":
        from besser.generators.sql import SQLGenerator
        SQLGenerator(model, output_dir=str(out), sql_dialect="sqlite").generate()
        script = next(out.glob("*.sql")).read_text(encoding="utf-8")
        con = sqlite3.connect(":memory:")
        con.executescript(script)
        tables = con.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        return f"{tables} tables executed"
    if name == "json_schema":
        from besser.generators.json import JSONSchemaGenerator
        JSONSchemaGenerator(model, output_dir=str(out)).generate()
        for f in out.glob("*.json"):
            json.loads(f.read_text(encoding="utf-8"))
        return "valid JSON"
    if name == "java":
        from besser.generators.java_classes import JavaGenerator
        JavaGenerator(model, output_dir=str(out)).generate()
        sources = [str(p) for p in out.rglob("*.java")]
        javac = shutil.which("javac")
        if javac and sources:
            proc = subprocess.run([javac, "-d", str(out / "classes"), *sources], capture_output=True, text=True,
                                  timeout=300, check=False)
            if proc.returncode:
                raise RuntimeError("javac: " + (proc.stderr or proc.stdout).strip().splitlines()[0])
            return f"{len(sources)} files compiled"
        return f"{len(sources)} files (javac not found)"
    if name == "rdf":
        from besser.generators.rdf import RDFGenerator
        RDFGenerator(model, output_dir=str(out)).generate()
        try:
            import rdflib
        except ImportError:
            return "written (rdflib not installed)"
        graph = rdflib.Graph()
        for f in out.glob("*.ttl"):
            graph.parse(str(f), format="turtle")
        return f"{len(graph)} triples parsed"
    if name == "django":
        from besser.generators.django import DjangoGenerator
        DjangoGenerator(model, project_name="sweep_project", app_name="sweep_app", output_dir=str(out)).generate()
        manage = next(out.rglob("manage.py"))
        proc = subprocess.run([sys.executable, str(manage), "check"], capture_output=True, text=True, timeout=300,
                              cwd=str(manage.parent), check=False)
        if proc.returncode:
            lines = [line for line in (proc.stderr or proc.stdout).strip().splitlines() if line.strip()]
            raise RuntimeError("manage.py check: " + " / ".join(lines[-3:]))
        return "manage.py check passed"
    if name == "rest_api":
        from besser.generators.rest_api import RESTAPIGenerator
        RESTAPIGenerator(model, output_dir=str(out)).generate()
        return f"{_compile_tree(out)} files compiled"
    if name == "backend":
        from besser.generators.backend import BackendGenerator
        BackendGenerator(model, output_dir=str(out)).generate()
        return f"{_compile_tree(out)} files compiled"
    raise ValueError(name)


def sweep(dataflows, out_dir: Path, generators=GENERATORS, timeout: float = 180) -> list[dict]:
    from tests.utilities.sdmx_to_buml.roundtrip import round_trip_differences  # pylint: disable=import-outside-toplevel
    rows = []
    for provider, flow, label in dataflows:
        row = {"provider": provider if not provider.startswith("http") else "custom", "dataflow": flow,
               "label": label, "results": {}, "notes": {}}
        rows.append(row)
        started = time.time()
        try:
            dsd = SDMXClient(provider, timeout=timeout, safe=True).fetch_structure(flow)
            model, mapping = structure_to_buml(dsd)
            kinds = [d.kind for d in mapping.dimensions]
            row["results"]["import"] = "ok"
            row["notes"]["import"] = (f"{time.time() - started:.1f} s, {len(kinds)} dims: {kinds.count('enum')} enum, "
                                      f"{kinds.count('class')} class, {kinds.count('string')} string")
        except Exception as exc:  # pylint: disable=broad-except
            row["results"]["import"] = _short(exc)
            print(f"{label:38} import FAILED {_short(exc)}", flush=True)
            continue
        result = model.validate(raise_exception=False)
        row["results"]["validate"] = "ok" if result["success"] else "; ".join(result["errors"])[:220]
        try:
            diffs = round_trip_differences(model)
            row["results"]["round_trip"] = "ok" if not diffs else "differs: " + "; ".join(diffs)[:200]
        except Exception as exc:  # pylint: disable=broad-except
            row["results"]["round_trip"] = _short(exc)
        for name in generators:
            target = out_dir / row["provider"] / flow.replace(",", "_").replace("@", "_") / name
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)
            try:
                row["notes"][name] = run_generator(name, model, target)
                row["results"][name] = "ok"
            except Exception as exc:  # pylint: disable=broad-except
                row["results"][name] = _short(exc)
                row["notes"][name] = traceback.format_exc(limit=3)[-600:]
        failed = [k for k, v in row["results"].items() if v != "ok"]
        print(f"{label:38} {'all ok' if not failed else 'FAILED: ' + ', '.join(failed)}", flush=True)
    return rows


def markdown(rows, checks) -> str:
    lines = ["| Provider | Dataflow | " + " | ".join(checks) + " |",
             "|---|---|" + "---|" * len(checks)]
    for row in rows:
        cells = []
        for check in checks:
            value = row["results"].get(check)
            cells.append("-" if value is None else ("ok" if value == "ok" else "FAIL"))
        lines.append(f"| {row['provider']} | {row['dataflow']} | " + " | ".join(cells) + " |")
    lines += ["", "## Failures", ""]
    for row in rows:
        for check, value in row["results"].items():
            if value != "ok":
                lines.append(f"- **{row['provider']} {row['dataflow']}**, {check}: {value}")
    lines += ["", "## Imports", ""]
    for row in rows:
        if row["results"].get("import") == "ok":
            lines.append(f"- {row['provider']} {row['dataflow']} ({row['label']}): {row['notes']['import']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", default=None, help="folder for the report and the generated code")
    parser.add_argument("--only", default="", help="comma-separated provider keys to sweep")
    parser.add_argument("--skip-generators", action="store_true")
    parser.add_argument("--generators", default="", help="comma-separated subset of: " + ", ".join(GENERATORS))
    args = parser.parse_args(argv)
    out = Path(args.out or tempfile.mkdtemp(prefix="sdmx_sweep_"))
    out.mkdir(parents=True, exist_ok=True)
    only = {p.strip() for p in args.only.split(",") if p.strip()}
    flows = [f for f in DATAFLOWS if not only or f[0] in only]
    generators = [] if args.skip_generators else (
        [g.strip() for g in args.generators.split(",") if g.strip()] or GENERATORS)
    rows = sweep(flows, out / "generated", generators)
    checks = ["import", "validate", "round_trip"] + generators
    (out / "sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "sweep.md").write_text(markdown(rows, checks), encoding="utf-8")
    print(f"\nReport: {out / 'sweep.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
