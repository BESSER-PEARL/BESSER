# A2 — All-Generators Smoke Test (final-analysis wave)

Owner: A2 (final-analysis wave)
Branch: `claude/refine-local-plan-sS9Zv`
Script: `tests/utilities/web_modeling_editor/backend/smoke_all_generators.py`

## Goal

Run every backend code generator once on a representative v4 fixture and
confirm the output is "reasonable":

* output directory non-empty
* at least one Python file parses cleanly with `ast.parse`
  (where the generator emits Python)
* template/text files have non-zero size

This is the cross-cutting check across the registry — not a per-generator
correctness audit. Pass means the generator is wired up, accepts the v4
domain/agent/quantum/NN/GUI model produced by the JSON-to-BUML processors
in the backend services layer, and produces non-empty output. Failures
here indicate either a registry/wiring break, a converter regression, or
a runtime dep that the smoke harness must explicitly skip.

## Targets and fixtures

The registry source of truth is
`besser/utilities/web_modeling_editor/backend/config/generators.py`. There
are 19 entries today (`pytorch` and `tensorflow` are conditional on the
optional torch / tensorflow imports):

| Category          | Generators                                     |
|-------------------|------------------------------------------------|
| object_oriented   | `python`, `java`, `pydantic`                   |
| web_framework     | `django`, `backend`, `web_app`, `rest_api`     |
| database          | `sqlalchemy`, `sql`                            |
| data_format       | `jsonschema`, `rdf`                            |
| object_model      | `jsonobject`                                   |
| ai_agent          | `agent` (BAFGenerator)                         |
| quantum           | `qiskit`                                       |
| frontend          | `react`, `flutter`                             |
| deployment        | `terraform`                                    |
| neural_network    | `pytorch`, `tensorflow`                        |

Fixtures (from `tests/fixtures/v4/`):

* `class_diagram_basic.json` — Library/Author/Genre, used by all
  class-diagram-driven generators (`python`, `java`, `pydantic`,
  `backend`, `sqlalchemy`, `sql`, `jsonschema`, `rdf`, `rest_api`).
* `webapp_smoke_project.json` — Library + minimal GUI (one Home page),
  used by `web_app`, `react`, `flutter`.
* `agent_diagram_basic.json` — `GreetBot` (one `Greeting` state, one
  `greet` intent), used by `agent`.
* `object_diagram_basic.json` — `myBook : Book` instance with embedded
  `referenceDiagramData`, used by `jsonobject`.
* `nn_diagram_basic.json` — Tiny MLP (`fc1` → `fc2`), used by `pytorch`
  and `tensorflow`.
* Programmatic `QuantumCircuit` (one Hadamard + one Z-basis measurement)
  for `qiskit`, because the v4 quantum diagram is a column-of-symbols
  shape rather than a node/edge graph and the smoke fixture catalogue
  does not yet ship a quantum JSON sample. The processor is exercised
  separately by `tests/generators/qiskit/`.
* No fixture for `terraform` — the deployment metamodel needs a populated
  `DeploymentModel` *and* a `.conf` file the generator reads with
  `open()`; that is out of scope for this smoke.

## How the script works

`smoke_all_generators.py` imports the FastAPI `app` first (to break the
circular-import chain through `services.deployment.github_deploy_api`),
then walks `SUPPORTED_GENERATORS`, dispatches to a per-generator runner
(or the default class-diagram runner), and reports PASS / FAIL / SKIP.
Each runner spins up a `tempfile.TemporaryDirectory`, instantiates the
generator with the right kwargs, calls `.generate()`, and tallies output
size and `ast.parse` success counts.

Run with:

```bash
python tests/utilities/web_modeling_editor/backend/smoke_all_generators.py
```

## Results (run on this branch)

```
SUMMARY: {'PASS': 17, 'FAIL': 0, 'SKIP': 2}
```

| Generator    | Required diagrams           | Status | Output size           | Notes                                      |
|--------------|-----------------------------|--------|-----------------------|--------------------------------------------|
| `python`     | ClassDiagram                | PASS   | 3.0 KiB (1 file)      | `py=1/1` parses                            |
| `java`       | ClassDiagram                | PASS   | 1.2 KiB (2 files)     | one `.java` per class, no `.py`            |
| `pydantic`   | ClassDiagram                | PASS   | 695 B (1 file)        | `py=1/1`                                   |
| `django`     | ClassDiagram                | SKIP   | 0 B                   | `django-admin` CLI not installed in env    |
| `backend`    | ClassDiagram                | PASS   | 27.1 KiB (4 files)    | `py=3/3` parse (FastAPI scaffold)          |
| `web_app`    | ClassDiagram + GUI(+Agent)  | PASS   | 138.3 KiB (36 files)  | `py=3/3` backend, plus React + Docker      |
| `sqlalchemy` | ClassDiagram                | PASS   | 1.6 KiB (1 file)      | `py=1/1`                                   |
| `sql`        | ClassDiagram                | PASS   | 284 B (1 file)        | `tables.sql`                               |
| `jsonschema` | ClassDiagram                | PASS   | 2.3 KiB (1 file)      | `json_schema.json`                         |
| `jsonobject` | ObjectDiagram (+ ref Class) | PASS   | 196 B (1 file)        | `object_model.json`                        |
| `agent`      | AgentDiagram                | PASS   | 4.5 KiB (3 files)     | `py=1/1` (`GreetBot.py` parses)            |
| `qiskit`     | (programmatic) QuantumCircuit | PASS | 880 B (1 file)        | `py=1/1` (`qiskit_circuit.py` parses)      |
| `rdf`        | ClassDiagram                | PASS   | 1.1 KiB (1 file)      | `vocabulary.ttl`                           |
| `rest_api`   | ClassDiagram                | PASS   | 5.7 KiB (3 files)     | `py=2/2` parse                             |
| `react`      | ClassDiagram + GUI          | PASS   | 112.1 KiB (29 files)  | TSX/CSS/HTML scaffold, no `.py`            |
| `flutter`    | ClassDiagram + GUI          | PASS   | 8.3 KiB (2 files)     | Dart + pubspec; pass main_page from GUI    |
| `terraform`  | DeploymentModel             | SKIP   | 0 B                   | no v4 deployment fixture (`.conf` needed)  |
| `pytorch`    | NNDiagram                   | PASS   | 486 B (1 file)        | `py=1/1` (`pytorch_nn_subclassing.py`)     |
| `tensorflow` | NNDiagram                   | PASS   | 450 B (1 file)        | `py=1/1` (`tf_nn_subclassing.py`)          |

## Observations

* **Wiring is clean across the registry** — every PASS generator accepts
  the v4 BUML model produced by the converters under
  `services/converters/json_to_buml/`. No generator failed because of
  the v4 wire-shape cutover.

* **`django` is SKIP, not FAIL** — `DjangoGenerator.generate()` shells
  out to `django-admin startproject`, which requires the `django`
  package's CLI on `PATH`. The runner deliberately checks
  `shutil.which("django-admin")` and reports SKIP when missing so this
  environmental gap doesn't pollute the regression matrix. CI installs
  Django in `requirements.txt`; on this sandbox it is absent.

* **`terraform` is SKIP** — `TerraformGenerator` needs a
  `DeploymentModel` populated with `PublicCluster` instances, each
  pointing at a `.conf` file the generator opens directly. The test
  fixture catalogue does not yet include such a deployment fixture, and
  manufacturing one is out of scope for the smoke wave. Per-cluster
  unit tests live under `tests/generators/terraform/` with their own
  pinned configs.

* **`flutter` invocation** — the top-level `FlutterGenerator` constructor
  is `(model, gui_model, main_page, module=None, output_dir)`. The
  smoke runner walks the parsed `GUIModel.modules[0].screens[0]` to
  pick a main page. This matches how the production `_generate_web_app`
  / Flutter integrations resolve a primary screen.

* **`qiskit` programmatic input** — the v4 quantum diagram is a
  column-of-symbols shape (`model.cols[col][qubit] = "•" / "Swap" / ...`)
  parsed by `process_quantum_diagram`. Building a tiny diagram in that
  shape just to round-trip a one-gate test added more risk than value
  for this smoke; the runner builds a `QuantumCircuit(name, qubits=1)`
  with one `HadamardGate(target_qubit=0)` and one
  `Measurement(target_qubit=0, output_bit=0)` directly. The full
  JSON-to-BUML round trip is covered by
  `tests/utilities/web_modeling_editor/backend/test_v4_round_trip.py`.

## Next actions

* Add a v4 quantum diagram fixture (`quantum_diagram_basic.json`) in a
  later wave so the qiskit smoke can also exercise
  `process_quantum_diagram`.
* If the deployment metamodel gets v4 JSON support, add a
  `deployment_basic.json` fixture and flip `terraform` to PASS.
* CI runners that ship Django will record `django` as PASS automatically;
  no script change is needed.
