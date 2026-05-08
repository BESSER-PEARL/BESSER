# Webapp Cutover Checklist (SA-7 input)

Reference: library at 503660a (SA-2.1); SA-2.2 may add fields to the lib's surface but no breaking changes expected.

This checklist enumerates every webapp call site that imports from `@besser/wme`
when the alias still points at `packages/editor/`. Phase 7 flips that alias to
`packages/library/`. The library exports today (per `packages/library/lib/index.tsx`):

- `./typings` — `UMLDiagramType`, `UMLModel` (v4 shape: `nodes`/`edges`),
  `ApollonNode`, `ApollonEdge`, `ApollonOptions`, `ApollonMode`, `ApollonView`,
  `Locale`, `Assessment`, `ExportOptions`, `SVG`, `Subscribers`, `Unsubscriber`,
  `InteractiveElements`, `SvgExportMode`, plus type-only `Styles`,
  `DiagramNodeType`, `DiagramEdgeType`.
- `./apollon-editor` — class `ApollonEditor` with constructor + `model`
  getter/setter, `subscribeToModelChange(cb): number`, `unsubscribe(id)` (no
  `unsubscribeFromModelChange`), `subscribeToDiagramNameChange`,
  `subscribeToAssessmentSelection`, `exportAsSVG`, `getNodes/getEdges`,
  `destroy()`, `view` getter/setter, `getSelectedElements`,
  `addOrUpdateAssessment`, `updateDiagramTitle`, `toggleInteractiveElementsMode`,
  `getInteractiveForSerialization`, `getDiagramMetadata`,
  `sendBroadcastMessage`, `receiveBroadcastedMessage`,
  static `exportModelAsSvg`, static `generateInitialSyncMessage`. **No
  `nextRender` getter**, **no `unsubscribeFromModelChange` method**.
- `./utils/helpers`, `./utils/versionConverter`, `./utils` (barrel) — exposes
  `parseMultiplicity`, `toERCardinality`, `erCardinalityToUML`, `normalizeType`,
  `TYPE_ALIASES`, `Visibility`, `VISIBILITY_SYMBOLS`, `SYMBOL_TO_VISIBILITY`,
  `convertV3ToV4`, `convertV2ToV4`, `migrate{Class|Object|StateMachine|Agent|User|NN}DiagramV3ToV4`,
  `convertV4ToV3{Class|StateMachine|Agent|User|NN}`, `isV2Format`, `isV3Format`,
  `isV4Format`, `importDiagram`, plus the SVG/text/layout/edge utils.
- `./services/diagramBridge` — `diagramBridge` singleton +
  `IClassDiagramData = { nodes: any[], edges: any[] }` (was
  `{ elements, relationships }` in v3).
- `./services/settingsService` — `settingsService`, `SettingsService`,
  `IApplicationSettings`, `DEFAULT_SETTINGS`, `ClassNotation`.
- `./logger` — `log`, `setLogLevel`, `setLogger`, `LogLevel`.
- `userMetaModel` — JSON re-exported (SA-4 already shipped this).

> The lib exports `parseMultiplicity` / `toERCardinality` / `erCardinalityToUML`
> from `./utils` (barrel). It does **not** currently re-export
> `formatDisplayName`. None of the 42 call sites imports `formatDisplayName`,
> so this is informational only.

---

## Counts

- **Total webapp files importing `@besser/wme`**: 42 (43 import statements —
  `features/export/ExportDialog.tsx` has two separate import lines for
  `ApollonEditor` and `UMLModel`).
- **Total LoC across those 42 files**: 13,547.
- **Unique symbols imported**: 14 — `ApollonEditor`, `UMLModel`,
  `UMLDiagramType`, `ApollonMode`, `Locale`, `Styles`, `SVG`, `diagramBridge`,
  `settingsService`, `ClassNotation`, `parseMultiplicity`, `toERCardinality`,
  `erCardinalityToUML`, plus the `userMetaModel` JSON (currently imported via
  a relative path from `editor/src/...`, will move to `@besser/wme` per SA-4).
- **Dynamic imports** (`await import('@besser/wme')`): 2 sites
  (`workspaceSlice.ts` x2 internally + `useImportDiagram.ts`).
- **`vi.mock('@besser/wme', ...)`** test mocks: 2 sites
  (`DiagramTabs.test.tsx`, `ProjectSettingsPanel.test.tsx`).

### Distribution by category

| Bucket | Count | Symbols |
|---|---|---|
| **A. ApollonEditor class** | 13 sites | `ApollonEditor` |
| **B. Type-only imports** | 31 sites | `UMLDiagramType`, `UMLModel`, `ApollonMode`, `Locale`, `Styles`, `SVG`, `ClassNotation` |
| **C. Subscription API** | 1 site (call) | `subscribeToModelChange`, **`unsubscribeFromModelChange` (gone)** |
| **D. Model getter/setter** | 1 site (call) | `editor.model = ...` in `ApollonEditorComponent` |
| **E. Helper functions** | 1 site | `parseMultiplicity`, `toERCardinality`, `erCardinalityToUML` |
| **F. Services** | 4 sites | `diagramBridge`, `settingsService` |
| **G. Misc / non-import-site ripples** | 9+ sites | v3 `model.elements` / `model.relationships` walks |

(Several files belong to multiple buckets — e.g. `ApollonEditorComponent.tsx`
hits A + B + C + D + F.)

---

## Per-file action table

> Paths are relative to `packages/webapp/`. SA-7 actions:
> **No-op** (still exported with same shape) /
> **Rename** (moved or renamed) /
> **Shape adjust** (symbol present, shape changed — e.g. v3 → v4) /
> **Polyfill** (needs a thin compat shim) /
> **Replace** (call site needs a different approach) /
> **Block** (no replacement; call site has to be rewritten or ripped out).

| File | Line | Symbols imported | Category | SA-7 action | Notes |
|---|---|---|---|---|---|
| `src/main/app/application.tsx` | 7 | `ApollonEditor` | A | No-op | Type-only re-export to `ApollonEditorProvider`; class export survives. |
| `src/main/app/hooks/useProject.ts` | 1 | `UMLDiagramType`, `UMLModel` | B | Shape adjust | `UMLModel` is now `{nodes, edges}` v4. Any consumer reading `.elements/.relationships` needs rework — grep this file for that. |
| `src/main/app/shell/WorkspaceShell.tsx` | 3 | `UMLDiagramType` | B | No-op | Enum-like `UMLDiagramType` survives unchanged in `lib/types/DiagramType.ts`. |
| `src/main/app/shell/WorkspaceSidebar.tsx` | 2 | `UMLDiagramType` | B | No-op | Same as above. |
| `src/main/app/shell/__tests__/WorkspaceSidebar.test.tsx` | 4 | `UMLDiagramType` | B | No-op | Test consumer; behaviour identical. |
| `src/main/app/shell/menus/MobileNavigation.tsx` | 2 | `UMLDiagramType` | B | No-op | |
| `src/main/app/shell/topbar-types.ts` | 1 | `UMLDiagramType` | B | No-op | |
| `src/main/app/shell/workspace-navigation.tsx` | 2 | `UMLDiagramType` | B | No-op | |
| `src/main/app/store/workspaceSlice.ts` | 2 | `ApollonMode`, `Locale`, `Styles`, `UMLDiagramType`, `UMLModel` | B + F | **Shape adjust** | Calls `diagramBridge.setClassDiagramData(classDiagram.model)` (lines 135, 143). New bridge expects `{nodes, edges}` instead of v3 `{elements, relationships}`. The Redux state stores the *active model*; if the upstream model is still v3-shape (until project storage is migrated), this call will fail to populate the bridge. Two dynamic `await import('@besser/wme')` calls — both still resolve. Also imports `userMetaModel` via deep relative path `../../../../../editor/src/main/packages/user-modeling/usermetamodel_buml_short.json` (line 22) — switch to `import { userMetaModel } from '@besser/wme'`. |
| `src/main/features/agent-config/AgentConfigurationPanel.tsx` | 3 | `UMLDiagramType`, `UMLModel` | B | Shape adjust | Likely walks `model.elements` to enumerate agent intents — requires a follow-up grep within this 2,318-LoC file. |
| `src/main/features/deploy/utils/agentPersonalizationPayload.ts` | 1 | `UMLDiagramType` | B | No-op | |
| `src/main/features/deploy/utils/restoreBaseAgentModels.ts` | 1 | `UMLDiagramType` | B | No-op | |
| `src/main/features/editors/__tests__/HiddenPerspectivesBanner.test.tsx` | 4 | `UMLDiagramType`, `UMLModel` (type) | B | No-op or fixture refresh | If the test instantiates a v3-shape `UMLModel` literal, the literal will fail TS compile against the new type. |
| `src/main/features/editors/diagram-tabs/DiagramTabs.tsx` | 3 | `diagramBridge` | F | **Shape adjust** | Reads `diagram.model.elements` / `.relationships` (lines 84, 85, 188), passes a model into `diagramBridge.setClassDiagramData(refModel)` (line 154). Awaits `apollonEditor.nextRender` twice (lines 204, 206) — **`nextRender` is gone in the new editor**. Either polyfill `nextRender` on the new ApollonEditor or replace those awaits with a different ready signal (e.g. a `ready: Promise<void>` exposed by the constructor). |
| `src/main/features/editors/diagram-tabs/__tests__/DiagramTabs.test.tsx` | 4 | `UMLDiagramType` | B + test | No-op | Uses `vi.mock('@besser/wme')` returning a mocked `diagramBridge`, `setClassDiagramData`. Mock surface doesn't change. |
| `src/main/features/editors/diagram-tabs/scaffoldObjectsFromClasses.ts` | 1 | `UMLDiagramType`, `UMLModel` | B | **Shape adjust** | This file is a ~400 LoC v3 walker — reads `classModel.elements`, `objectModel.elements`, `objectModel.relationships`, `classModel.relationships`. Top-of-list rewrite candidate. |
| `src/main/features/editors/uml/ApollonEditorComponent.tsx` | 1 | `ApollonEditor`, `UMLModel`, `diagramBridge` | A + B + C + D + F | **Replace + Rename** | (1) `editor.unsubscribeFromModelChange(modelSubscriptionRef.current)` (line 66) — method **renamed** to `unsubscribe(subscriberId)`. (2) `await nextEditor.nextRender` (line 117) — **getter gone**; replace with whatever ready signal SA-2.2 exposes (or strip the await; the constructor renders synchronously into the React root and React Flow init is signalled by `setReactFlowInstance`). (3) `nextEditor.model = currentDiagram.model` (line 125) — model setter present but expects v4 shape; whole flow blocked until the stored model is v4. (4) `subscribeToModelChange((model: UMLModel) => …)` survives shape-wise (returns `number`). (5) `diagramBridge.setStateMachineDiagrams` / `setQuantumCircuitDiagrams` survive verbatim. |
| `src/main/features/editors/uml/__tests__/multiplicity.test.ts` | 2 | `parseMultiplicity`, `toERCardinality`, `erCardinalityToUML` | E | No-op | Pure helpers, identical behaviour. |
| `src/main/features/editors/uml/apollon-editor-context.ts` | 1 | `ApollonEditor` | A | No-op | Class export survives. |
| `src/main/features/export/ExportDialog.tsx` | 2, 7 | `ApollonEditor`, `UMLModel` | A + B | No-op (type) / Shape adjust (model usage) | Two separate import lines — coalesce while you're there. |
| `src/main/features/export/useExportBuml.ts` | 2 | `ApollonEditor`, `UMLModel` | A + B | No-op | Calls `editor.model` getter then POSTs to backend. Backend will need to accept v4 wire shape (out-of-scope for SA-7 but flag-worthy). |
| `src/main/features/export/useExportJson.ts` | 2 | `ApollonEditor` | A | No-op | `editor.model` getter still returns a `UMLModel`; downstream JSON shape changes. |
| `src/main/features/export/useExportPng.ts` | 2 | `ApollonEditor`, `SVG` | A + B | No-op | `editor.exportAsSVG()` survives, returns `{svg, clip}`. |
| `src/main/features/export/useExportSvg.ts` | 2 | `ApollonEditor`, `SVG` | A + B | No-op | Same as PNG. |
| `src/main/features/generation/hooks/useDeployLocally.ts` | 2 | `ApollonEditor` | A | No-op | Reads `editor.model`. |
| `src/main/features/generation/hooks/useGenerateCode.ts` | 2 | `ApollonEditor` | A | Shape adjust | Reads `editor.model` and ships it to backend. Whole "model JSON" payload changes from v3 to v4 — backend must accept v4 (out-of-scope for SA-7 import rewrite, but a coordinated change). |
| `src/main/features/generation/useGeneratorExecution.ts` | 18 | `ApollonEditor`, `UMLDiagramType` | A + B | No-op | |
| `src/main/features/import/useBumlToDiagram.ts` | 2 | `UMLDiagramType`, `UMLModel` | B | Shape adjust | Receives `UMLModel` from backend after BUML import — backend must emit v4 shape (or the webapp must call `convertV3ToV4` from the lib). |
| `src/main/features/project/ProjectSettingsPanel.tsx` | 2 | `settingsService`, `ClassNotation` | F + B | No-op | Both survive verbatim. |
| `src/main/features/project/TemplateLibraryDialog.tsx` | 2 | `UMLDiagramType` | B | No-op | |
| `src/main/features/project/__tests__/ProjectSettingsPanel.test.tsx` | 6 | `UMLDiagramType` | B + test | No-op | `vi.mock('@besser/wme')` mock surface unchanged. |
| `src/main/features/project/create-diagram-from-template-modal/software-pattern/software-pattern-types.ts` | 2 | `UMLModel` | B | **Shape adjust** | Type only; but the consumer uses it for v3-shape `template.umlModel` JSON — see `template-factory.ts`. |
| `src/main/features/project/create-diagram-from-template-modal/template-factory.ts` | 7 | `UMLDiagramType` | B + G | **Shape adjust** | Imports many v3-shape JSON template files (`Library_Complete.json`, `dpp.json`, `nexacrm.json`, `team_player_ocl.json`, agent templates, …) — every one starts with `"version":"3.0.0"` and has v3 `elements`/`relationships`. SA-7 must either re-serialize them all to v4 OR pipe them through `convertV3ToV4()` at template-load time. |
| `src/main/features/project/create-diagram-from-template-modal/template-types.ts` | 1 | `UMLDiagramType`, `UMLModel` | B | Shape adjust | Same template typing thread. |
| `src/main/shared/services/agent-variants/agent-variants-service.ts` | 1 | `UMLDiagramType`, `UMLModel` (type) | B | Shape adjust | Persists `UMLModel` snapshots into localStorage (`besser_userProfiles`, `besser_agentBaseModels`). Existing snapshots will be v3 — SA-7 (or a webapp-side migrator) must migrate on load. |
| `src/main/shared/services/project-import/projectImport.ts` | 12 | `UMLDiagramType` | B | No-op | |
| `src/main/shared/services/storage/local-storage-repository.ts` | 17 | `UMLModel` | B | Shape adjust | Stores/loads `UMLModel` from `besser_projects` / `besser_latest`. v3 payloads in users' localStorage need a one-shot migration on launch. |
| `src/main/shared/services/storage/local-storage-types.ts` | 1 | `UMLDiagramType`, `UMLModel` | B | Shape adjust | Type definitions; will compile-fail wherever v3 fields are accessed. |
| `src/main/shared/services/validation/validateDiagram.ts` | 4 | `ApollonEditor` | A | No-op | Reads `editor.model`. |
| `src/main/shared/types/__tests__/project.test.ts` | 2 | `UMLDiagramType`, `UMLModel` (type) | B | Shape adjust | Test fixtures construct v3-shape `UMLModel` literals. |
| `src/main/shared/types/project.ts` | 1 | `UMLDiagramType`, `UMLModel` | B | **Replace** | (1) `createEmptyDiagram` (line 245) creates a v3 model literal: `{ version: '3.0.0', elements: {}, relationships: {}, … }`. Rewrite to `{ version: '4.0.0', nodes: [], edges: [], assessments: {} }`. (2) `isUMLModel()` (line 537) checks `typeof candidate.elements === 'object'` — rewrite to check `Array.isArray(candidate.nodes) && Array.isArray(candidate.edges)`. (3) `diagramHasContent()` (line 612) checks `model.elements`/`model.relationships` lengths — rewrite to `model.nodes.length || model.edges.length`. (4) Bump `PROJECT_SCHEMA_VERSION = 5` and add a `migrateProjectToV5(project)` step that walks every diagram model and runs `convertV3ToV4()` from the lib. |
| `src/main/shared/utils/__tests__/projectExportUtils.test.ts` | 2 | `UMLDiagramType` | B + test | Shape adjust | Test constructs v3 `model.elements = {…}` and `model.relationships = {…}` literals (lines 31, 79, 339, 340). Refactor to v4 fixtures or use `convertV3ToV4`. |

---

## Top 5 highest-effort files

### 1. `src/main/features/editors/uml/ApollonEditorComponent.tsx`

The single biggest API-surface collision. SA-7 must (a) replace
`editor.unsubscribeFromModelChange(id)` with the new lib's
`editor.unsubscribe(id)` (one-line rename); (b) remove or replace
`await nextEditor.nextRender` — the new `ApollonEditor` exposes no
`nextRender` getter, so SA-7 has to either polyfill one in the lib (tracking
React Flow init via the existing `onReactFlowInit` hook in the constructor)
or restructure the setup effect to drive off a different signal (a
`ready: Promise<void>` arg passed to the constructor, or a callback option
similar to React Flow's `onInit`); (c) confirm the `editor.model = …` setter
does the right thing for v4 shape — it does, but the `currentDiagram.model`
flowing in must already be v4, which gates this on the project-storage
migration described in `shared/types/project.ts`. The `subscribeToModelChange`
call (line 129) is unchanged. The `diagramBridge.setStateMachineDiagrams` /
`setQuantumCircuitDiagrams` calls in the supporting effect (lines 84-85)
are unchanged.

### 2. `src/main/shared/types/project.ts`

The localStorage / cross-feature type backbone. SA-7 must rewrite
`createEmptyDiagram` (UML branch) to emit a v4 literal
(`{version: '4.0.0', nodes: [], edges: [], assessments: {}, ...}` instead of
`{version: '3.0.0', elements: {}, relationships: {}, interactive: {…}, …}`),
rewrite `isUMLModel` to validate v4 fields (`nodes` and `edges` arrays in
place of `elements`/`relationships` records), and rewrite `diagramHasContent`
to count nodes/edges. Bump `PROJECT_SCHEMA_VERSION` from 4 to 5 and add a
`migrateProjectToV5(project: BesserProject): BesserProject` that walks each
`project.diagrams[type][i].model` and pipes UML models through
`convertV3ToV4` from the lib. Wire the migrator into
`local-storage-repository.ts` on load. This is the linchpin of the migration:
without it, every other touched file will compile-pass but break at runtime
on the first localStorage hydrate.

### 3. `src/main/features/editors/diagram-tabs/scaffoldObjectsFromClasses.ts`

Pure v3 walker — ~400 LoC of `Object.values(classModel.elements ?? {})`,
`classModel.relationships`, `objectModel.elements`, `objectModel.relationships`.
Every reference has to be rewritten to walk `nodes` (filter by type) and
`edges`. Helper-function logic (slot scaffolding from default values, inherited
attribute resolution, association → object-link generation) is unchanged in
intent but every property access shifts. This is the deepest rewrite in the
webapp, but the only `@besser/wme` import is the type pair `UMLDiagramType,
UMLModel`. SA-7 should consider extracting a shared "v4 model walker" helper
either here or in the lib so this file plus `gui/diagram-helpers.ts` (a
non-import-site ripple) can share traversal code.

### 4. `src/main/app/store/workspaceSlice.ts`

The Redux slice + bridge fan-out. The two
`diagramBridge.setClassDiagramData(classDiagram.model)` calls (lines 135, 143)
plug a webapp-stored model directly into the bridge. The new bridge expects
`{nodes, edges}`; it has v3 fallback at read time (`getClassDiagramData`
handles `parsed.elements`) but no v3 fallback at write time, so the model
must be v4 before reaching it. The deep relative
`import userMetaModel from '../../../../../editor/src/main/packages/user-modeling/usermetamodel_buml_short.json'`
on line 22 should switch to `import { userMetaModel } from '@besser/wme'`
(SA-4 already wired the export). Two `await import('@besser/wme')` dynamic
imports survive untouched.

### 5. `src/main/features/project/create-diagram-from-template-modal/template-factory.ts` (+ template JSON files)

The template factory imports a dozen v3-shape JSON files
(`Library_Complete.json`, `Library_OCL.json`, `team_player_ocl.json`,
`dpp.json`, `nexacrm.json`, `ai_sandbox.json`, plus agent and state-machine
templates). All start with `"version":"3.0.0"` and have v3
`elements`/`relationships` records. Two options for SA-7:
(a) regenerate every JSON template against the v4 schema (one-shot, deterministic,
keeps load time fast, but big diff and easy to miss a feature in the
re-export); (b) leave the JSON as-is and pipe each loaded template through
`convertV3ToV4()` at the template-application boundary (smaller diff, but
adds a runtime conversion step per template apply and trusts the converter).
Recommend (b) for SA-7 — defer (a) to a follow-up cleanup. Either way, the
import line `import { UMLDiagramType }` is itself a no-op.

---

## Webapp-side helper changes

These are the non-import-site changes SA-7 has to make. They follow from the
shape change and the editor API delta, not from the alias flip itself.

- **`shared/types/project.ts` — bump `PROJECT_SCHEMA_VERSION` from 4 to 5**
  and add `migrateProjectToV5(project)` that runs `convertV3ToV4` (from
  `@besser/wme`) over every `diagram.model` whose `type` is a UML diagram
  type. Wire it into `LocalStorageRepository.loadProject` /
  `ProjectStorageRepository`. **Critical: missing this means every
  user's existing project breaks on first launch after the cutover.**
- **`shared/types/project.ts:createEmptyDiagram`** — rewrite UML branch from
  v3 literal to v4 literal (`nodes: []`, `edges: []`, no
  `size`/`elements`/`relationships`/`interactive` keys; use `version: '4.0.0'`).
- **`shared/types/project.ts:isUMLModel`** — replace `elements`/`relationships`
  guards with `Array.isArray(candidate.nodes) && Array.isArray(candidate.edges)`.
- **`shared/types/project.ts:diagramHasContent`** — replace
  `Object.keys(model.elements).length` checks with
  `model.nodes.length || model.edges.length`.
- **`features/editors/gui/diagram-helpers.ts`** — non-import-site, but reaches
  into `classDiagram.elements` / `classDiagram.relationships` ~14 times.
  Mirror the v4 walker rewrite from `scaffoldObjectsFromClasses.ts`.
- **`features/editors/diagram-tabs/scaffoldObjectsFromClasses.ts`** — full
  rewrite of `Object.values(model.elements)` walks to `model.nodes.filter(...)`,
  and `model.relationships` walks to `model.edges.filter(...)`.
- **`features/editors/diagram-tabs/DiagramTabs.tsx`** — remove the two
  `await apollonEditor.nextRender` calls (lines 204, 206); rewrite the
  v3 element/relationship counters (lines 84, 85, 188) to count nodes/edges.
- **`features/editors/uml/ApollonEditorComponent.tsx`** — replace
  `editor.unsubscribeFromModelChange(id)` with `editor.unsubscribe(id)`;
  remove `await nextEditor.nextRender` or replace with the new ready signal.
- **`shared/utils/projectExportUtils` test** + **`shared/types/project` test** —
  refresh fixtures to v4 literals (or factor through `convertV3ToV4`).
- **`features/agent-config/AgentConfigurationPanel.tsx`** (2,318 LoC) needs
  a follow-up grep for direct `.elements` / `.relationships` access; it
  almost certainly walks the agent diagram model to build intent / state lists.
- **`features/import/useImportDiagram.ts`** — already uses
  `await import('@besser/wme')` for `diagramBridge`; if the imported model
  arriving via BUML is v3, route it through `convertV3ToV4` before bridge
  population.
- **`features/editors/diagram-tabs/__tests__/DiagramTabs.test.tsx`** and
  **`features/project/__tests__/ProjectSettingsPanel.test.tsx`** — both use
  `vi.mock('@besser/wme', ...)`. Because the mocked surface only exposes
  `diagramBridge.setClassDiagramData` (or `settingsService` methods), the
  mock contract doesn't change shape, just the alias resolution.
- **`workspaceSlice.ts:22`** — the deep relative import of
  `usermetamodel_buml_short.json` should switch to
  `import { userMetaModel } from '@besser/wme'` (SA-4 already exposed it
  from the lib's `index.tsx`).
- **Template JSON files** under `src/main/templates/pattern/{structural,
  agent, state-machine, …}/` — either re-serialize to v4 or wrap consumption
  in `convertV3ToV4`. Recommended: convert at apply time (smaller diff).
- **`editorRevision` audit** (per `frontend/CLAUDE.md`'s warning) — the
  setup effect in `ApollonEditorComponent` reacts to
  `editorRevision` bumps to recreate the editor. After the cutover, the
  `nextRender` removal changes the timing of when the editor is
  "ready"; double-check that no other code assumes `editor.nextRender` is
  awaited before `setEditor!(editor)` is called. The audit isn't an
  alias-flip change, but SA-7 is the natural moment to reverify.

---

## SA-7 prioritized work order

1. **Land the v3→v4 project storage migration** in
   `shared/types/project.ts` first: rewrite `createEmptyDiagram` (UML
   branch), `isUMLModel`, `diagramHasContent`; bump
   `PROJECT_SCHEMA_VERSION` to 5; add `migrateProjectToV5` and wire it into
   `LocalStorageRepository`. **This unblocks every other call site that
   reads a stored model.**
2. **Flip the alias** in `tsconfig.json` and `vite.config.ts` from
   `packages/editor/src/main/index.ts` to
   `packages/library/lib/index.tsx`. After this step, the type-only sites
   (Bucket B, ~31 of the 42) compile-pass with zero edit if the v4 type is
   shape-compatible at the `UMLModel` level — but anything walking
   `.elements`/`.relationships` immediately breaks at runtime.
3. **Rewrite the model walkers**:
   `features/editors/diagram-tabs/scaffoldObjectsFromClasses.ts`,
   `features/editors/gui/diagram-helpers.ts`,
   `features/editors/diagram-tabs/DiagramTabs.tsx` (counters),
   `features/agent-config/AgentConfigurationPanel.tsx`. Consider extracting
   a shared `v4ModelWalker` helper to `shared/utils/`.
4. **Fix the editor-call deltas** in
   `features/editors/uml/ApollonEditorComponent.tsx`: rename
   `unsubscribeFromModelChange` → `unsubscribe`; replace `nextRender` with
   the SA-2.2 ready signal (or polyfill `nextRender` in the lib if SA-2.2
   doesn't ship one — preferred: polyfill to keep webapp diff small).
5. **Fix the second `nextRender` site** in `DiagramTabs.tsx` (lines 204, 206).
6. **Normalize the userMetaModel import** in `workspaceSlice.ts` (relative
   path → `@besser/wme`).
7. **Pipe templates through `convertV3ToV4`** in `template-factory.ts` (or
   re-serialize the JSON — recommend conversion).
8. **Update test fixtures**: `shared/types/__tests__/project.test.ts`,
   `shared/utils/__tests__/projectExportUtils.test.ts`, the two
   `vi.mock('@besser/wme')` sites, and any v3 literal in
   `HiddenPerspectivesBanner.test.tsx`.
9. **Coalesce the duplicate import** in `features/export/ExportDialog.tsx`
   (line 2 + line 7 → single line).
10. **Run the full webapp test suite, then `npm run build`**, then walk
    a fresh-launch flow with a pre-existing v4-incompatible localStorage
    project to confirm the migrator runs cleanly.
