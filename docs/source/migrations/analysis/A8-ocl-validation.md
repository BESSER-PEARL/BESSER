# A8 — OCL Validation Path Analysis

**Wave**: BESSER WME final-analysis
**Scope**: Confirm UserDiagram + ClassDiagram OCL constraints still flow correctly through the v4 wire to the BOCL evaluator.
**Verdict**: **WORKS — v4 native end-to-end. One critical and one minor caveat (see §5).**

---

## 1. Endpoint Flow Trace

`POST /besser_api/validate-diagram`

File: `besser/utilities/web_modeling_editor/backend/routers/validation_router.py:49-235`.

```
DiagramInput { title, model: { type, version, nodes, edges, ... } }
   │
   ▼
diagram_type = input_data.model.get("type")
   │
   ├── "ClassDiagram"
   │     buml_model = process_class_diagram({title, model})           # v.r.:74
   │     buml_model.validate(raise_exception=False)                   # metamodel validation
   │
   ├── "ObjectDiagram"
   │     reference_data = model["referenceDiagramData"]
   │     buml_model = process_class_diagram({title, model: reference_data})
   │     domain_model.validate(...)
   │     object_model = process_object_diagram(input_data.model_dump(), buml_model)
   │     object_model.validate(...)
   │
   └── "UserDiagram"                                                  # v.r.:113
         buml_model = user_reference_domain_model                     # static Python module
         buml_model.validate(...)
         object_model = process_object_diagram(input_data.model_dump(),
                                               user_reference_domain_model)
         object_model.validate(...)

   ▼
if diagram_type in ("ClassDiagram","ObjectDiagram","UserDiagram") and no errors:
    ocl_results = check_ocl_constraint(buml_model[, object_model])    # v.r.:204-209
   ▼
ValidationResponse {
    isValid, errors, warnings, message,
    valid_constraints: ["✅ [Account inv positive] '...'", ...],
    invalid_constraints: ["❌ [Account inv positive] '...' - <reason>", ...],
    ocl_message
}
```

### Inside `check_ocl_constraint`
File: `besser/utilities/web_modeling_editor/backend/services/validators/ocl_checker.py`.

1. `_collect_all_constraints(domain_model)` (line 103) yields `(label, kind, c)`
   for every `domain_model.constraints` (invariants) and every `cls.methods[*]
   .pre/.post` (method contracts).
2. For each constraint:
   - **Pre/post or no object model**: `_parse_only(expression)` invokes `BOCLLexer +
     BOCLParser` for syntax-only check (lines 14-29).
   - **Invariant + object model**: `OCLWrapper(domain_model, object_model).evaluate(c)`
     evaluates against object instances of the context class (lines 187-219).
3. Result strings are bullet-pointed unicode-tagged labels of the form
   `[ContextClass inv NAME]` / `[Class::method pre NAME]`.

---

## 2. Constraint Discovery — ClassDiagram (v4)

**Location**: `data.oclConstraints` array on each class node, per the v4 spec
(`docs/source/migrations/uml-v4-shape.md:163-164,235-240`).

```ts
// ClassNodeData
oclConstraints?: { id: string; name: string; expression: string }[];
```

**Reader**: `_process_constraints` in
`besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/class_diagram_processor.py:628-734`.

Walks `node.data.oclConstraints` for every `type: 'class'` node (lines 690-692)
plus the rare free-standing `ClassOCLConstraint` node fallback
(`stereotype == 'oclconstraint'`, lines 122-123, 683-688).

For each row:
1. `_ocl_box_to_full_text(row, owner_class, ...)` (line 561) returns canonical
   `context X (inv|pre|post): ...` text. The v4 spec requires `expression` to
   be canonical full text; legacy body-only text + a `kind`/`targetMethodId`
   tuple is still accepted via `legacy_body_only_to_text`.
2. `process_ocl_constraints(text, domain_model, counter, ...)` in
   `parsers/ocl_parser.py:159-254` splits on `context` boundaries, parses each
   block via `parse_constraint_text` (line 92), and stores the canonical text
   on `constraint.expression` (line 152, 231).
3. Routing:
   - `kind == "invariant"` → appended to `domain_model.constraints`.
   - `kind == "precondition"` → `method.add_pre(constraint)`.
   - `kind == "postcondition"` → `method.add_post(constraint)`.

Pre/post are matched against methods by (class, method) via
`method_by_qualified_name`, which walks the inheritance chain (lines 640-669).

---

## 3. Constraint Discovery — UserDiagram (v4)

**Frontend metamodel**: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/services/userMetaModel/usermetamodel.json` — a v3-shaped JSON `{elements, relationships}` (4230 lines) shipped to the editor as a read-only reference for the user profile schema. **It is never sent over the wire**; the validator never opens this file.

**Backend authority**: `besser/utilities/web_modeling_editor/backend/constants/user_buml_model.py` (577 lines) — a hand-coded Python module that imports `Class`/`Property`/`Constraint`/etc. and constructs `domain_model` (line 549). Four `Constraint` invariants (lines 520-545) are baked in:

- `pi_age_range` — `context Personal_Information inv pi_age_range: self.age >= 0 and self.age <= 120`
- `skill_name_not_empty` — `context Skill inv skill_name_not_empty: self.name <> ''`
- `education_required_fields` — `context Education inv education_required_fields: self.degreeName <> '' and self.providedBy <> ''`
- `disability_description_not_empty` — `context Disability inv disability_description_not_empty: self.description <> ''`

**Mapping v4 → reference model**: the v4 UserDiagram input (`{nodes, edges}` with `type: "UserDiagram"`, `node.type == "UserModelName"`) is fed through `process_object_diagram(input_data.model_dump(), user_reference_domain_model)` (validation_router.py:120). The processor (`object_diagram_processor.py:114-`) reads each `UserModelName` node's `data.classId` and resolves it against:
1. `referenceDiagramData` first (per `_reference_class_name`, line 91), but the UserDiagram path does **not** ship reference data — so it falls through.
2. `data.className` (line 163).
3. `data.name` matched against `user_reference_domain_model.get_class_by_name(...)` (line 167).

The class-name string match against the static reference domain model is what binds a v4 `UserModelName` node to one of the four constrained classes (`Personal_Information`, `Skill`, `Education`, `Disability`).

**Constraint evaluation**: `check_ocl_constraint(buml_model, object_model)` runs each invariant against the constructed `ObjectModel`. Because `object_model is not None`, it reaches the `evaluable=True` branch (ocl_checker.py:171) and `OCLWrapper.evaluate(constraint)` returns `True`/`False`/value.

The spec line `usermetamodel_buml_short.json … backend remains the OCL validation authority` (uml-v4-shape.md:717-719) is honoured: the JSON in `library/lib/services/userMetaModel/` is editor-only.

---

## 4. Element ID / Line Mapping in Errors

The validation response carries:

```python
valid_constraints:   ["✅ [Account inv positive] 'context Account inv positive: self.balance >= 0' — <description>"]
invalid_constraints: ["❌ [Account inv positive] 'context Account inv positive: self.balance >= 0' - Constraint violation: <reason>"]
```

**What the webapp gets**:

- A `[ContextClass inv ConstraintName]` / `[Class::method pre ConstraintName]` label.
- The full canonical OCL expression text.
- A reason string (`Evaluates to False`, the BOCL parser error text, or the
  natural-language `description` field of the OCL row).

**What the webapp does NOT get**:

- The v4 node id (`row.id`) of the failing OCL row — it is **not** propagated
  into the `OCLConstraint` object or into the result string. The webapp can
  only map back by matching `(class_name, constraint_name)` against the
  `data.oclConstraints[*].name` it sent.
- File paths or line numbers — there is no source file at this layer; the
  expression came from a JSON field. BOCL parser errors include byte offsets
  inside the expression string but not character columns within the JSON
  document.

This is sufficient for highlighting the offending constraint row in the editor
UI by `name`-matching, but **not** robust if the user has duplicate
constraint names — A8 spec already emits a duplicate-name warning
(class_diagram_processor.py:730-733).

---

## 5. Critical Issues & Test Corpus

### 5.1 Critical: lossy round-trip when OCL row has no `name` (medium severity)
`process_ocl_constraints` auto-generates a constraint name (`{Class}_inv_{counter}_{block_idx}`) when the BOCL header omits one (ocl_parser.py:219-225) and **rewrites the canonical expression to include it** (lines 238-245). On the way out (BUML → v4 JSON), if the converter re-emits `row.expression` it will carry the synthesised name; `row.name` (the v4 row's own field) may diverge unless explicitly synced. Round-trip identity needs a paired test (none exist for this case in the v4 corpus).

### 5.2 Critical: webapp cannot mark a specific row by node id
The `[ContextClass inv name]` label and the canonical expression text are the only handles into the v4 row. If a user creates two `oclConstraints` rows on the same class with the same `name` (no UI uniqueness guarantee), violations cannot be unambiguously routed to one row. A pragmatic mitigation already exists (duplicate-name warning); a richer mitigation would propagate `row.id` into `OCLConstraint` (e.g. on `description`/metadata) so the response can include it.

### 5.3 Test corpus coverage of OCL with v4

| Coverage | Status |
|---|---|
| v4 ClassDiagram `data.oclConstraints` invariant routed to `domain_model.constraints` | covered — `tests/utilities/web_modeling_editor/backend/services/converters/test_ocl_pre_post.py:228-234` |
| v4 ClassDiagram pre routed to `Method.pre` | covered — same file:237-243 |
| v4 ClassDiagram post routed to `Method.post` | covered — same file:246-252 |
| Pre/post pointing at unknown method emits warning | covered — same file:255-264 |
| Malformed expression emits warning, doesn't abort | covered — same file:267-285 (file extends to ~530 lines) |
| `/validate-diagram` end-to-end on a valid v4 ClassDiagram | covered — `tests/utilities/web_modeling_editor/backend/test_api_integration.py:425-456` |
| `/validate-diagram` returns `valid_constraints` / `invalid_constraints` arrays | implicit — schema enforced by `ValidationResponse` model |
| `/validate-diagram` on UserDiagram triggers OCL with object model | covered — same file:669-715 (uses monkeypatched stubs; does **not** exercise real BOCL eval) |
| `/validate-diagram` on UserDiagram with **real** `user_reference_domain_model` and a failing OCL invariant | **MISSING** — flag |
| End-to-end test that crafts a v4 ClassDiagram with an OCL row that **violates** at object-model time and asserts `invalid_constraints` is populated | **MISSING** — flag (existing tests only cover routing/parsing, not evaluation against an ObjectModel through the wire) |
| Round-trip test: v4 → BUML → v4 preserves `oclConstraints` row IDs and names | **MISSING** — flag |

---

## 6. Hand-crafted failing v4 ClassDiagram

The minimal v4 payload below should produce one OCL **syntax** error (the
expression is parsed but evaluation requires an ObjectDiagram, which we do not
have for ClassDiagram-only validation — so the path that fails is parsing;
to produce a *runtime* violation requires routing through `ObjectDiagram` or
`UserDiagram`):

```json
{
  "title": "FailingOCL",
  "model": {
    "version": "4.0.0",
    "type": "ClassDiagram",
    "nodes": [
      {
        "id": "n-account",
        "type": "class",
        "position": {"x": 0, "y": 0},
        "width": 160, "height": 100,
        "measured": {"width": 160, "height": 100},
        "data": {
          "name": "Account",
          "stereotype": null,
          "attributes": [
            {"id": "a-balance", "name": "balance",
             "attributeType": "int", "visibility": "public"}
          ],
          "methods": [],
          "oclConstraints": [
            {"id": "ocl-bad", "name": "broken",
             "expression": "context Account inv broken: self."}
          ]
        }
      }
    ],
    "edges": []
  }
}
```

**Expected response** (per `validation_router.py:204-217` + `ocl_checker.py:179-186`):

```json
{
  "isValid": true,
  "errors": [],
  "warnings": ["Warning: Invalid OCL syntax in 'context Account inv broken: self.': <BOCL syntax error>"],
  "message": "✅ Diagram is valid",
  "valid_constraints": [],
  "invalid_constraints": [],
  "ocl_message": "Found 0 valid and 0 invalid constraints"
}
```

**Subtle behaviour**: a syntactically broken `oclConstraints` row is **swallowed at parse time** (warning, not error) before the constraint reaches `domain_model.constraints` — so it never appears in `invalid_constraints`. To exercise the `invalid_constraints` path you need either (a) a parseable but evaluable-False invariant + an `ObjectDiagram` that supplies instances, or (b) a parseable-but-fails-at-evaluation expression. The flow for path (a) is the same routing as above, with `check_ocl_constraint(domain_model, object_model)` reaching `parser.evaluate(constraint) → False` (ocl_checker.py:204-215) and emitting `❌ [Account inv balance_pos] '...' - Constraint violation: Evaluates to False`.

**Failing-evaluation example** (requires ObjectDiagram path):

```json
"oclConstraints": [
  {"id": "ocl-1", "name": "balance_pos",
   "expression": "context Account inv balance_pos: self.balance >= 0",
   "description": "Balance must be non-negative"}
]
```

with an object whose `balance = -10`. Response:

```json
{
  "isValid": false,
  ...
  "invalid_constraints": [
    "❌ [Account inv balance_pos] 'context Account inv balance_pos: self.balance >= 0' - Constraint violation: Balance must be non-negative"
  ]
}
```

---

## 7. Summary

- Wire reading: ClassDiagram `data.oclConstraints` is correctly walked and
  parsed into BUML invariants and method pre/post conditions for v4 input.
- UserDiagram OCL: not driven by the editor's bundled
  `usermetamodel.json` — the backend's `user_buml_model.py` is the single
  source of truth, and the v4 UserDiagram nodes resolve to its classes
  by class-name matching through `process_object_diagram`.
- BOCL parse + evaluate paths are reachable from the v4 wire with no
  v3↔v4 conversion shim required.
- Error messages carry constraint name + canonical OCL text but do **not**
  propagate v4 node/row ids; webapp must name-match to highlight the row.
- Test corpus covers parsing + routing thoroughly but does **not** exercise
  the full `/validate-diagram` → `OCLWrapper.evaluate` path with a real
  ObjectModel, nor does it cover the round-trip invariant of `row.id`
  preservation.
