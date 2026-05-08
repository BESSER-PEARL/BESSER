# A4 - Test Coverage Map

**Wave**: BESSER WME final-analysis (A4)
**Date**: 2026-05-08
**Branch**: `claude/refine-local-plan-sS9Zv`
**Submodule HEAD**: tracked by parent at `besser/utilities/web_modeling_editor/frontend`
**Scope**: produce per-diagram and per-feature coverage map for the Apollon -> v4 migration; surface gaps the migration may have introduced.

This is a pure read-only audit. No source files were modified; counts are reproducible from `pytest --collect-only` and a `grep -c "it("` on the vitest files.

---

## 1. Backend test counts (parent repo)

`python -m pytest tests/utilities/web_modeling_editor/ --collect-only -q` reports **412 tests** across **13 files**.

### 1.1 Counts per file

| File | # tests |
|------|--------:|
| `tests/utilities/web_modeling_editor/backend/services/converters/parsers/test_parsers.py` | 112 |
| `tests/utilities/web_modeling_editor/backend/services/converters/test_converter_roundtrip.py` | 73 |
| `tests/utilities/web_modeling_editor/backend/test_api_integration.py` | 69 |
| `tests/utilities/web_modeling_editor/converters/nn/test_nn_diagram_processor.py` | 52 |
| `tests/utilities/web_modeling_editor/backend/services/utils/test_agent_config_recommendation_utils.py` | 25 |
| `tests/utilities/web_modeling_editor/backend/services/converters/test_ocl_pre_post.py` | 18 |
| `tests/utilities/web_modeling_editor/backend/services/utils/test_user_profile_utils.py` | 14 |
| `tests/utilities/web_modeling_editor/backend/services/converters/parsers/test_constraint_descriptions.py` | 13 |
| `tests/utilities/web_modeling_editor/backend/services/converters/test_nn_buml_to_json_safety.py` | 12 |
| `tests/utilities/web_modeling_editor/backend/services/utils/test_agent_config_manual_mapping_utils.py` | 11 |
| `tests/utilities/web_modeling_editor/backend/test_v4_round_trip.py` | 5 |
| `tests/utilities/web_modeling_editor/backend/test_spreadsheet_import.py` | 4 |
| `tests/utilities/web_modeling_editor/converters/nn/test_nn_templates.py` | 4 |
| **Total** | **412** |

### 1.2 Counts per area

| Area | # tests | Notes |
|------|--------:|-------|
| **converters / parsers** (attribute / method / multiplicity / constraint / sanitize) | 125 | `test_parsers.py` (112) + `test_constraint_descriptions.py` (13) |
| **converters / round-trip** (json<->buml across diagrams) | 73 | `test_converter_roundtrip.py` (Class 19, NN 40, Object 5, StateMachine 9). **No Agent or User class.** |
| **API integration** (FastAPI routers) | 69 | `test_api_integration.py` |
| **converters / NN-specific** (lib processor + safety + templates) | 68 | `test_nn_diagram_processor.py` (52) + `test_nn_buml_to_json_safety.py` (12) + `test_nn_templates.py` (4) |
| **services / utils** (agent recommendation + manual mapping + user profile) | 50 | three files combined |
| **OCL pre/post** | 18 | `test_ocl_pre_post.py` |
| **v4 round-trip fixtures** | 5 | `test_v4_round_trip.py` (one per diagram) |
| **spreadsheet import** | 4 | `test_spreadsheet_import.py` |

### 1.3 API integration: per-class breakdown (69 tests)

| Class | # tests | Endpoint(s) |
|-------|--------:|-------------|
| `TestValidateDiagram` | 15 | `/validate-diagram` |
| `TestGenerateOutput` | 13 | `/generate-output` |
| `TestRecommendationEndpoints` | 9 | `/recommend-agent-config-llm`, `/recommend-agent-config-mapping`, `/agent-config-manual-mapping` |
| `TestEdgeCases` | 6 | mixed (`/generate-output`, `/validate-diagram`) |
| `TestExportBuml` | 5 | `/export-buml` |
| `TestMiddlewareAndRequestValidation` | 5 | mixed |
| `TestFeedbackEndpoint` | 3 | `/feedback` |
| `TestGetJsonModel` | 3 | `/get-json-model` |
| `TestProjectGeneration` | 3 | `/generate-output-from-project` |
| `TestRecommendationEndpointsAuth` | 3 | recommend endpoints auth gating |
| `TestHealthEndpoints` | 2 | `/health`, `/besser_api/` |
| `TestStandaloneChatbotDeploy` | 1 | `/github/deploy-webapp` |
| `TestDeprecatedEndpoints` | 1 | `/check-ocl` |

### 1.4 Per-diagram count (backend tests that exercise each of the 6 BESSER diagrams)

A test "exercises" a diagram when it loads or constructs that diagram type and asserts on its conversion / generation / validation. Counts below are conservative (named test classes / fixture-driven tests).

| Diagram | Round-trip class | NN-only | API integration (input shape) | v4 fixtures | OCL | **Total exercised** |
|---------|-----------------:|--------:|------------------------------:|------------:|----:|--------------------:|
| **ClassDiagram** | 19 | - | most generate/validate/export-buml tests use `class_diagram_input` | 1 | 18 | **~50+** |
| **StateMachineDiagram** | 9 | - | a handful in `TestValidateDiagram` (state-machine fixtures) | 1 | - | ~12 |
| **ObjectDiagram** | 5 | - | (none observed via fixture name) | 1 | - | 6 |
| **AgentDiagram** | 0 (no `TestAgentDiagramRoundtrip` class) | - | recommendation + chatbot-deploy tests indirectly use agent payloads | 1 | - | ~14 |
| **NNDiagram** | 40 | 68 (processor + safety + templates) | (none in `test_api_integration.py`) | 1 | - | **109** (highest) |
| **UserDiagram** | 0 (no class) | - | (none) | 0 (no `user_diagram_basic.json` fixture) | - | **0** |

NN coverage is by far the deepest in the backend (109 tests); UserDiagram has **zero backend tests**.

---

## 2. Frontend test counts (submodule, read-only)

### 2.1 Vitest files inventory

| Location | # files | # `it()` cases |
|----------|--------:|---------------:|
| `packages/library/tests/round-trip/` | 6 | 28 |
| `packages/library/tests/unit/` | 17 | 594 |
| `packages/webapp/src/**/__tests__/` | 10 | 172 |
| **Total vitest** | **33** | **794** |

`packages/webapp/tests/e2e/` adds 8 Playwright spec files (not vitest); not counted in the 794.
`packages/server/` and `packages/editor/` have no test files.
`node_modules/@reduxjs/toolkit/...` test files are vendored deps and are excluded.

### 2.2 Per-diagram round-trip + node-render counts

Nothing in the library test tree explicitly does a "node-render" test (no `@testing-library/react` invocations under `packages/library/tests/unit/` for the `lib/nodes/<diagram>/` components). All per-diagram coverage in the library lives in the round-trip suite.

| Diagram | Round-trip `it()` cases | Round-trip describe blocks | Node-render tests | Total distinct cases |
|---------|------------------------:|---------------------------:|------------------:|---------------------:|
| **ClassDiagram** | 4 | 1 | 0 | 4 |
| **StateMachineDiagram** | 3 | 1 | 0 | 3 |
| **ObjectDiagram** | 4 | 1 | 0 | 4 |
| **AgentDiagram** | 6 | 3 (main + transition-shape parameterized + v4->v3 export) | 0 | 6 |
| **NNDiagram** | 7 | 1 | 0 | 7 |
| **UserDiagram** | 4 | 2 (main + attribute-operator synthesis) | 0 | 4 |

**All six diagrams have >= 3 distinct test cases** in the library round-trip suite (smallest: StateMachineDiagram at 3). However: **none of the six has a dedicated node-render test in the library** — the `lib/nodes/<diagram>/*.tsx` components are exercised only via store / round-trip integration, not as React components. This is a category gap, not a per-diagram gap.

### 2.3 Library unit tests by file

| File | # `it()` cases | What it tests |
|------|--------------:|----------------|
| `versionConverter.test.ts` | 127 | v3 -> v4 handle / type / message converters (the migrator core) |
| `edgeUtils.test.ts` | 125 | edge geometry / labels |
| `exportUtils.test.ts` | 64 | model -> { json, png, svg, pdf } export |
| `pathParsing.test.ts` | 50 | SVG path parsing |
| `copyPasteUtils.test.ts` | 30 | clipboard ops |
| `bpmnConstraints.test.ts` | 27 | BPMN validation rules |
| `connection.test.ts` | 23 | edge connection rules |
| `nodeUtils.test.ts` | 22 | node geometry / containment |
| `storeUtils.test.ts` | 21 | zustand store helpers |
| `helpers.test.ts` | 20 | misc utilities |
| `diagramTypeUtils.test.ts` | 17 | diagram type detection |
| `alignmentUtils.test.ts` | 16 | guide / snap utils |
| `svgTextLayout.test.ts` | 16 | text layout |
| `quadrantUtils.test.ts` | 13 | quadrant / position utils |
| `MultilineText.test.tsx` | 8 | multiline text component |
| `textUtils.test.ts` | 8 | text helpers |
| `supportsMultilineName.test.ts` | 7 | multiline-name predicate |

### 2.4 Webapp unit tests

| File | # `it()` cases |
|------|--------------:|
| `features/editors/uml/__tests__/multiplicity.test.ts` | 28 |
| `shared/utils/__tests__/projectExportUtils.test.ts` | 23 |
| `features/assistant/services/__tests__/modifiers.test.ts` | 22 |
| `shared/services/storage/__tests__/ProjectStorageRepository.test.ts` | 22 |
| `shared/types/__tests__/project.test.ts` | 19 |
| `app/shell/__tests__/WorkspaceSidebar.test.tsx` | 17 |
| `features/editors/diagram-tabs/__tests__/DiagramTabs.test.tsx` | 17 |
| `features/project/__tests__/ProjectSettingsPanel.test.tsx` | 14 |
| `features/editors/__tests__/HiddenPerspectivesBanner.test.tsx` | 5 |
| `features/editors/gui/__tests__/diagram-helpers.test.ts` | 5 |

---

## 3. Coverage matrix (6 diagrams × {nodes, edges, inspectors, migrator, round-trip})

`Y` = at least one test directly exercises this layer; `N` = no test was found; `~` = exercised only indirectly (e.g. via round-trip).

| Diagram | Nodes (lib) | Edges (lib) | Inspectors (lib) | Migrator v3->v4 | Round-trip JSON | Round-trip BUML (backend) | API integration |
|---------|:-----------:|:-----------:|:----------------:|:---------------:|:---------------:|:-------------------------:|:---------------:|
| ClassDiagram         | ~ (only via round-trip) | ~ | ~ | Y (`versionConverter.test.ts` covers handle/type/edge maps generically; classDiagram round-trip covers SA-2.1 OCL/LinkRel edges) | Y (4) | Y (19) | Y (most generate/validate/export tests) |
| StateMachineDiagram  | ~ | ~ | ~ | Y (covered by `versionConverter.test.ts` generic mappers; round-trip enforces shape) | Y (3) | Y (9) | Y (a few in `TestValidateDiagram`) |
| ObjectDiagram        | ~ | ~ | ~ | Y (round-trip; SA-2.1 `associationId`) | Y (4) | Y (5) | ~ (only via round-trip; no dedicated `TestObjectDiagram*` API test) |
| AgentDiagram         | ~ | ~ (3 describe blocks incl. transition shapes) | ~ | Y (round-trip; AgentStateTransition shapes 1-5 parameterized) | Y (6) | **N** (no `TestAgentDiagramRoundtrip` class in `test_converter_roundtrip.py`) | Y (recommendation + chatbot-deploy paths) |
| NNDiagram            | ~ | ~ | Y (`SA-2.2 #29` filter logic spans `NNComponentEditPanel` and is implicitly covered by 40 backend round-trip tests; still no React-level inspector test) | Y (`nnDiagram.test.ts` 7 cases + `versionConverter` generic) | Y (7) | Y (40 + 12 safety + 4 templates) | **N** (no NNDiagram fixture in `test_api_integration.py`) |
| UserDiagram          | ~ | ~ | ~ | Y (`userDiagram.test.ts` 4 + SA-2.2 #38 attributeOperator synthesis) | Y (4) | **N** (no `user_diagram_basic.json` v4 fixture, no `TestUserDiagramRoundtrip` class) | **N** (no API tests) |

### 3.1 Inspector panel coverage gap

`packages/library/lib/components/inspectors/` ships **22 panel `.tsx` files** across the 6 diagrams:

| Diagram | Inspector panel files | Inspector tests |
|---------|---------------------:|----------------:|
| AgentDiagram         | 9 (`AgentDiagramEdgeEditPanel`, `AgentDiagramInitEdgeEditPanel`, `AgentIntentBodyEditPanel`, `AgentIntentDescriptionEditPanel`, `AgentIntentEditPanel`, `AgentIntentObjectComponentEditPanel`, `AgentRagElementEditPanel`, `AgentStateBodyEditPanel`, `AgentStateEditPanel`, plus `RagDbFields`) | 0 |
| StateMachineDiagram  | 7 (`StateActionNodeEditPanel`, `StateBodyEditPanel`, `StateCodeBlockEditPanel`, `StateEditPanel`, `StateLabelEditPanel`, `StateMachineDiagramEdgeEditPanel`, `StateObjectNodeEditPanel`) | 0 |
| NNDiagram            | 3 (`NNComponentEditPanel`, `NNContainerEditPanel`, `NNReferenceEditPanel`) | 0 |
| ClassDiagram         | 2 (`ClassEditPanel`, `ClassEdgeEditPanel`) | 0 |
| ObjectDiagram        | 2 (`ObjectEditPanel`, `ObjectLinkEditPanel`) | 0 |
| UserDiagram          | 2 (`UserModelNameEditPanel`, `UserModelAttributeEditPanel`) | 0 |
| **Total** | **25** | **0** |

**Zero React-level tests cover any of the 25 inspector panels.** The branch logic inside `NNComponentEditPanel` (`SA-2.2 #29` per-layer optional-field gating) and `ClassEdgeEditPanel` (`SA-2.1` BESSER-specific edge picker) is only exercised transitively by round-trip JSON tests, which do not render React.

### 3.2 SA-1..SA-2.2 lib files vs corresponding tests

| SA milestone | Lib surface | Test |
|--------------|-------------|------|
| SA-1 (inspector registry) | `lib/components/inspectors/registry.ts`, `lib/components/inspectors/index.ts`, `lib/store/propertiesPanelStore.ts` | **No direct test for `registry.ts`**; behaviour is exercised transitively when a node type is rendered with a panel. The `propertiesPanelStore` is not in the unit test list. |
| SA-2 / SA-2.1 ClassDiagram | `lib/components/inspectors/classDiagram/{ClassEditPanel,ClassEdgeEditPanel}.tsx`; `nodes/classDiagram/`; `edges/edgeTypes/ClassDiagramEdge.tsx` | Round-trip only (4 tests, of which 1 is the SA-2.1 `ClassOCLLink + ClassLinkRel` regression). No React-level test for `ClassEdgeEditPanel`. |
| SA-2.1 ObjectDiagram | `lib/components/svgs/nodes/objectDiagram/ObjectNameSVG.tsx` (icon-view toggle) | Round-trip only; no SVG-render test. |
| SA-2.2 NN | `lib/components/inspectors/nnDiagram/NNComponentEditPanel.tsx` (audit fixes #29-#33), `NNContainerEditPanel`, `NNReferenceEditPanel` | Round-trip + 12 BUML-side `nn_buml_to_json_safety` checks. **No inspector React test for the optional-field gating.** |
| SA-2.2 UserDiagram (#37 / #38) | `lib/nodes/userDiagram/UserModelAttribute.tsx`, `UserModelAttributeEditPanel.tsx` (comparator-from-name) | `userDiagram.test.ts` lines 135-214 cover #38 attribute-operator synthesis. **#37 (Integer-gated comparator dropdown) is JSX state in the panel and has no React test.** |

---

## 4. Coverage gaps - prioritized

### CRITICAL (production wire path with no test, or recently changed surface with no test)

1. **No backend round-trip class for `AgentDiagram` and `UserDiagram`.**
   `test_converter_roundtrip.py` defines `TestClassDiagramRoundtrip`, `TestStateMachineRoundtrip`, `TestObjectDiagramRoundtrip`, `TestNNDiagramRoundtrip` only. `process_agent_diagram` and the user-diagram converter are reached **only** by:
   - `test_v4_round_trip.py::test_v4_agent_diagram_round_trip` (1 fixture-shape assertion), and
   - frontend round-trip simulations (which don't actually run Python).
   No backend test asserts that an agent or user payload survives `json -> buml -> json` byte-stably. **Production path; covered fixture-only.**

2. **`test_api_integration.py` has no NNDiagram or UserDiagram input.**
   Of the 6 diagrams, only ClassDiagram, StateMachine, Agent (indirectly via recommend/chatbot endpoints), and ObjectDiagram (indirectly) have integration coverage. Posting an NN payload or User payload to `/generate-output`, `/validate-diagram`, `/export-buml`, or `/get-json-model` is **untested at the FastAPI boundary**, despite the NN converter being the most-tested unit-level component.

3. **Inspector panels: 25 files, 0 React tests.** The SA-2.2 audit fixes (#29 NN per-layer gating, #33 list-shape placeholder, #37 Integer-gated comparator, ClassDiagram SA-2.1 edge picker, AgentRAG `RagDbFields`) are pure React state logic, and the migrator-era refactor moved this logic out of Apollon's old MobX bodies. None of it is unit-tested at the React layer. A regression in any panel ships silently (no failing test), and only manifests when a user opens the inspector.

### HIGH (less production-critical but still missing)

4. **Conversion router endpoints with no integration test:**
   `/csv-to-domain-model`, `/get-json-model-from-image`, `/get-json-model-from-kg`, `/transform-agent-model-json`, `/get-project-json-model`, `/export-project-as-buml`. The set of `client.post` URLs in `test_api_integration.py` is `{generate-output, generate-output-from-project, validate-diagram, export-buml, get-json-model, check-ocl, recommend-agent-config-llm, recommend-agent-config-mapping, agent-config-manual-mapping, feedback, github/deploy-webapp}` plus `GET /health`, `GET /besser_api/`. The seven listed conversion endpoints are unreached by `test_api_integration.py`. (A7 covers `/get-json-model-from-image` separately; CSV is covered by `test_spreadsheet_import.py` at the service layer but not via HTTP.)

5. **No test for the `SA-1` inspector registry (`lib/components/inspectors/registry.ts`).** Adding a new diagram or panel without an entry crashes silently in the panel renderer. A small unit test that asserts every node type from `lib/nodes/<diagram>/index.ts` has a matching registry entry would catch the whole class of mistakes.

6. **No test for `lib/services/diagramBridge.ts` or `lib/sync/yjsSyncClass.ts`.** Both are migration-era new files (`SA-1`-era plumbing) and both are reached at runtime by every editor session. Neither shows up in `tests/unit/`.

7. **Library node-render coverage is structurally absent.** `packages/library/tests/unit/` has zero React-Testing-Library tests for any of the ~95 `lib/nodes/<diagram>/*.tsx` components. The Apollon -> v4 migration replaced MobX-rendered nodes with React-Flow-rendered ones; round-trip tests catch JSON shape but not what a user actually sees on screen.

### MEDIUM

8. **`lib/utils/` files without a unit test:** `classifierMemberDisplay`, `deepPartial`, `interactiveUtils`, `labelUtils`, `layoutUtils`, `multiplicity` (covered by webapp instead, but not by lib), `popoverUtils`, `requiredInterfaceUtils`, `typeNormalization`, `v2Typings`, `v3Typings`. Most are typing-only or trivial, but `interactiveUtils`, `layoutUtils`, `requiredInterfaceUtils`, and `typeNormalization` have non-trivial branch logic.

9. **No round-trip fixture for UserDiagram in `tests/fixtures/v4/`.** The other five diagrams have one each (`agent_diagram_basic.json`, `class_diagram_basic.json`, `nn_diagram_basic.json`, `object_diagram_basic.json`, `state_machine_basic.json`); UserDiagram is missing. `test_v4_round_trip.py` only covers 5 of 6 diagrams as a result.

10. **`store` slices** (`alignmentGuidesStore`, `assessmentSelectionStore`, `metadataStore`, `popoverStore`, `propertiesPanelStore`, `settingsStore`, `diagramStore`) are not directly unit-tested — `storeUtils.test.ts` exercises helpers, not the slices themselves.

---

## 5. Quick numbers

- **Backend tests**: 412 across 13 files.
- **Frontend vitest tests**: 794 `it()` cases across 33 files (28 round-trip + 594 library unit + 172 webapp unit). Plus 8 Playwright e2e files.
- **Diagrams with < 3 frontend round-trip cases**: **none** (smallest is StateMachineDiagram at 3).
- **Diagrams with 0 backend round-trip class**: **2** (AgentDiagram, UserDiagram).
- **Diagrams with 0 API integration test**: **2** (NNDiagram, UserDiagram).
- **Diagrams with 0 inspector React test**: **6** (all of them).
- **v4 fixtures present**: 5 of 6 (UserDiagram missing).

---

## 6. Recommendations (out of scope for A4 - read-only - but listed for the wave)

1. Add `TestAgentDiagramRoundtrip` and `TestUserDiagramRoundtrip` classes to `test_converter_roundtrip.py`, following the `TestClassDiagramRoundtrip` pattern.
2. Add a `user_diagram_basic.json` fixture to `tests/fixtures/v4/` and a `test_v4_user_diagram_round_trip` to `test_v4_round_trip.py`.
3. Extend `test_api_integration.py::TestGenerateOutput` and `TestValidateDiagram` with NN and User payloads.
4. Add a registry-completeness unit test in `packages/library/tests/unit/inspectorRegistry.test.ts` that iterates `lib/nodes/<diagram>/index.ts` and asserts each node type has an entry.
5. Add at least one React-Testing-Library test per inspector panel; prioritise `NNComponentEditPanel` (SA-2.2 #29 / #33), `UserModelAttributeEditPanel` (SA-2.2 #37 / #38), and `ClassEdgeEditPanel` (SA-2.1 BESSER-specific edges) where the migration changed the most state-handling logic.
6. Add HTTP-level integration tests for the seven conversion-router endpoints currently unreached.
