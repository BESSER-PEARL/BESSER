# Public API Surface Diff: `@besser/wme` (editor) → `@tumaet/apollon` (library)

**Status**: SA-API-DIFF audit, read-only.
**Audience**: SA-7 (Phase 7 cutover sub-agent).
**Scope**: contrast every barrel-exported symbol from the old fork's
`packages/editor/src/main/index.ts` against the new lib's
`packages/library/lib/index.tsx`, then map the diff to concrete webapp
call sites SA-7 has to touch.

Reference SHAs:

- editor (old): `503660a` working tree (no edits to `packages/editor` since
  branching).
- library (new): `503660a` (SA-2.1 close of ClassDiagram + ObjectDiagram
  parity gaps). SA-2.2 may have un-pushed working-tree edits; this audit
  reads the committed blob.

## Top-line: SA-7 work estimate

The cutover is **dominated by `UMLModel`-shape changes**, not by the
class-method surface. The webapp's only `@besser/wme` consumers are 14
distinct symbols (43 import sites). All of them have a counterpart in the
new lib **except `diagramBridge`, `settingsService`, and `ClassNotation`,
which exist in `packages/library/lib/services/` but are not yet
re-exported from the barrel** — three lines in `index.tsx` close that
gap.

The hard work is the **274 `model.elements` / `model.relationships`
accesses** spread across 19 webapp files (heaviest in
`features/assistant/services/modifiers/*`,
`features/editors/gui/diagram-helpers.ts`,
`features/assistant/services/UMLModelingService.ts`). Every one is a
SA-7 fix point: v3 `{elements: Record<id, El>, relationships: Record<id,
Rel>}` becomes v4 `{nodes: ApollonNode[], edges: ApollonEdge[]}`.

Five `ApollonEditor` methods the webapp actually uses are still
**missing** from the new lib at 503660a and must be added by SA-7 (per
the plan):

- `editor.ready` (replacement for `nextRender`) — used at 6 call sites
  (`UMLModelingService.ts` ×4, `DiagramTabs.tsx` ×2,
  `ApollonEditorComponent.tsx` ×1).
- `editor.unsubscribeFromModelChange(id)` alias to `unsubscribe(id)` —
  used at `ApollonEditorComponent.tsx:66`. (Could also be a rename in the
  webapp.)

No showstoppers.  Estimated SA-7 effort: ~1–1.5 days for the
elements/relationships rewrite, ~2 hours for the small API additions and
re-exports.

---

## A. Public surface inventory

### Editor (old) — 113 exports

Source: `packages/editor/src/main/index.ts` re-exports from `./typings`,
`./apollon-editor`, `./compat/helpers`, `./services/diagram-bridge`,
`./services/settings/settings-service`,
`./packages/common/uml-association/multiplicity`, plus `Patch` and
`UMLModelCompat` types.

| Symbol | Category | Source module |
|---|---|---|
| `addOrUpdateAssessment` | function | `compat/helpers` |
| `addOrUpdateElement` | function | `compat/helpers` |
| `addOrUpdateRelationship` | function | `compat/helpers` |
| `AgentIntent` | interface | `typings` |
| `AgentModelElement` | interface | `typings` |
| `AgentRagElement` | interface | `typings` |
| `AgentState` | interface | `typings` |
| `AgentStateTransition` | type | `typings` |
| `ApollonEditor` | class | `apollon-editor` |
| `ApollonMode` | enum | `typings` (re-export from `editor-types`) |
| `ApollonOptions` | type | `typings` |
| `Assessment` | type | `typings` |
| `BPMNEndEvent` | type | `typings` |
| `BPMNFlow` | type | `typings` |
| `BPMNGateway` | type | `typings` |
| `BPMNIntermediateEvent` | type | `typings` |
| `BPMNStartEvent` | type | `typings` |
| `BPMNTask` | type | `typings` |
| `ClassNotation` | type | `services/settings/settings-service` |
| `DEFAULT_SETTINGS` | const | `services/settings/settings-service` |
| `DiagramBridgeService` | class | `services/diagram-bridge` |
| `DiagramReference` | type | `typings` |
| `diagramBridge` | const | `services/diagram-bridge` |
| `erCardinalityToUML` | function | `packages/common/uml-association/multiplicity` |
| `ExportOptions` | type | `typings` |
| `FeedbackCorrectionStatus` | type | `typings` |
| `findAssessment` | function | `compat/helpers` |
| `findElement` | function | `compat/helpers` |
| `findRelationship` | function | `compat/helpers` |
| `IApplicationSettings` | interface | `services/settings/settings-service` |
| `IAssociationInfo` | interface | `services/diagram-bridge` |
| `IAttributeInfo` | interface | `services/diagram-bridge` |
| `IClassDiagramData` | interface | `services/diagram-bridge` |
| `IClassInfo` | interface | `services/diagram-bridge` |
| `IDiagramBridgeService` | interface | `services/diagram-bridge` |
| `IDiagramReference` | interface | `services/diagram-bridge` |
| `ISettingsService` | interface | `services/settings/settings-service` |
| `IUMLObjectAttribute` | interface | `typings` |
| `IUMLObjectLink` | interface | `typings` |
| `IUMLObjectName` | interface | `typings` |
| `isInteractiveElement` | function | `compat/helpers` |
| `isInteractiveRelationship` | function | `compat/helpers` |
| `Locale` | enum | `typings` (re-export) |
| `MethodImplementationType` | type | `typings` |
| `parseMultiplicity` | function | `packages/common/uml-association/multiplicity` |
| `Patch` | type | `services/patcher` (type-only re-export) |
| `Selection` | type | `typings` |
| `setInteractiveElement` | function | `compat/helpers` |
| `setInteractiveRelationship` | function | `compat/helpers` |
| `SettingsService` | class | `services/settings/settings-service` |
| `settingsService` | const | `services/settings/settings-service` |
| `Styles` | type | `typings` (re-export from `theme/styles`) |
| `SVG` | type | `typings` |
| `toERCardinality` | function | `packages/common/uml-association/multiplicity` |
| `UMLAssociation` | type | `typings` |
| `UMLClassifier` | type | `typings` |
| `UMLClassifierMember` | type | `typings` |
| `UMLCommunicationLink` | type | `typings` |
| `UMLComponentComponent` | type | `typings` |
| `UMLComponentSubsystem` | type | `typings` |
| `UMLDeploymentComponent` | type | `typings` |
| `UMLDeploymentNode` | type | `typings` |
| `UMLDiagramType` | enum | `typings` (re-export) |
| `UMLElement` | type | `typings` |
| `UMLElementType` | enum | `typings` (re-export) |
| `UMLModel` | type | `typings` (v3 shape) |
| `UMLModelCompat` | type | `compat` (type-only re-export) |
| `UMLModelElement` | type | `typings` |
| `UMLModelElementType` | type | `typings` |
| `UMLPetriNetPlace` | type | `typings` |
| `UMLReachabilityGraphMarking` | type | `typings` |
| `UMLRelationship` | type | `typings` |
| `UMLRelationshipType` | enum | `typings` (re-export) |
| `UMLReply` | interface | `typings` |
| `UMLState` | interface | `typings` |
| `UMLStateTransition` | type | `typings` |
| `Visibility` | type | `typings` |

(Several BPMN marker / event sub-enums are also pulled in transitively
via `typings` imports but are not re-exported as named values; they
appear only inside type aliases.)

### Library (new) — ~120 exports

Source: `packages/library/lib/index.tsx` re-exports from `./typings`,
`./apollon-editor`, `./utils/helpers`, `./utils/versionConverter`,
`./utils`, plus `log`, `setLogLevel`, `setLogger`, `LogLevel` from
`./logger`, plus the inline `userMetaModel` JSON.

| Symbol | Category | Source module |
|---|---|---|
| `addOrUpdateAssessment` | method on `ApollonEditor` (not standalone) | `apollon-editor.tsx` |
| `adjustSourceCoordinates` | function | `utils/edgeUtils` |
| `adjustTargetCoordinates` | function | `utils/edgeUtils` |
| `AlignmentInfo` | type | `utils/alignmentUtils` |
| `ApollonEdge` | type | `typings` |
| `ApollonEditor` | class | `apollon-editor` |
| `ApollonMode` | enum | `typings` |
| `ApollonNode` | type | `typings` |
| `ApollonOptions` | type | `typings` |
| `ApollonView` | enum | `typings` |
| `Assessment` | type | `typings` |
| `AssessmentViewData` | type | `utils/helpers` |
| `bpmnConstraints.canDropIntoParent` | function | `utils/bpmnConstraints` |
| `calculateAlignmentGuides` | function | `utils/alignmentUtils` |
| `calculateDynamicEdgeLabels` | function | `utils/edgeUtils` |
| `calculateInnerMidpoints` | function | `utils/edgeUtils` |
| `calculateMinHeight` | function | `utils/layoutUtils` |
| `calculateMinWidth` | function | `utils/layoutUtils` |
| `calculateOffsets` | function | `utils/labelUtils` |
| `calculateOverlayPath` | function | `utils/edgeUtils` |
| `calculateStraightPath` | function | `utils/edgeUtils` |
| `calculateTextPlacement` | function | `utils/edgeUtils` |
| `convertV2ToV4` | function | `utils/versionConverter` |
| `convertV3EdgeTypeToV4` | function | `utils/versionConverter` |
| `convertV3HandleToV4` | function | `utils/versionConverter` |
| `convertV3MessagesToV4` | function | `utils/versionConverter` |
| `convertV3NodeTypeToV4` | function | `utils/versionConverter` |
| `convertV3ToV4` | function | `utils/versionConverter` |
| `convertV4ToV3Agent` | function | `utils/versionConverter` |
| `convertV4ToV3Class` | function | `utils/versionConverter` |
| `convertV4ToV3NN` | function | `utils/versionConverter` |
| `convertV4ToV3StateMachine` | function | `utils/versionConverter` |
| `convertV4ToV3User` | function | `utils/versionConverter` |
| `deepEqual` | function | `utils/storeUtils` |
| `DeepPartial` | type | `utils/deepPartial` |
| `DiagramEdgeType` | type | `typings` |
| `DiagramNodeType` | type | `typings` |
| `EdgeMarkerStyles` | interface | `utils/edgeUtils` |
| `ExportOptions` | type | `typings` |
| `FeedbackCorrectionStatus` | type | `typings` |
| `filterRenderedElements` | function | `utils/exportUtils` |
| `findClosestHandle` | function | `utils/edgeUtils` |
| `generateUUID` | function | `utils/index` |
| `getAssessmentNameForArtemis` | function | `utils/helpers` |
| `getConnectionLineType` | function | `utils/edgeUtils` |
| `getCustomColorsFromData` | function | `utils/layoutUtils` |
| `getCustomColorsFromDataForEdge` | function | `utils/layoutUtils` |
| `getDefaultEdgeType` | function | `utils/edgeUtils` |
| `getDiagramBounds` | function | `utils/exportUtils` |
| `getEdgeAssessmentDataById` | function | `utils/helpers` |
| `getEdgeMarkerStyles` | function | `utils/edgeUtils` |
| `getEllipseHandlePosition` | function | `utils/edgeUtils` |
| `getMarkerSegmentPath` | function | `utils/edgeUtils` |
| `getNodeAssessmentDataByNodeElementId` | function | `utils/helpers` |
| `getNodeBounds` | function | `utils/alignmentUtils` |
| `getPopoverOrigin` | function | `utils/popoverUtils` |
| `getPositionOnCanvas` | function | `utils/nodeUtils` |
| `getQuadrant` | function | `utils/quadrantUtils` |
| `getRenderedDiagramBounds` | function | `utils/exportUtils` |
| `getSVG` | function | `utils/exportUtils` |
| `importDiagram` | function | `utils/versionConverter` |
| `InteractiveElements` | type | `typings` |
| `isParentNodeType` | function | `utils/nodeUtils` |
| `isV2Format` | function | `utils/versionConverter` |
| `isV3Format` | function | `utils/versionConverter` |
| `isV4Format` | function | `utils/versionConverter` |
| `layoutTextInDiamond` | function | `utils/svgTextLayout` |
| `layoutTextInEllipse` | function | `utils/svgTextLayout` |
| `Locale` | enum | `typings` |
| `log` | const | `logger` |
| `LogLevel` | type | `logger` |
| `mapFromReactFlowEdgeToApollonEdge` | function | `utils/diagramTypeUtils` |
| `mapFromReactFlowNodeToApollonNode` | function | `utils/diagramTypeUtils` |
| `maxLinesForHeight` | function | `utils/svgTextLayout` |
| `measureTextWidth` | function | `utils/textUtils` |
| `migrateAgentDiagramV3ToV4` | function | `utils/versionConverter` |
| `migrateClassDiagramV3ToV4` | function | `utils/versionConverter` |
| `migrateNNDiagramV3ToV4` | function | `utils/versionConverter` |
| `migrateObjectDiagramV3ToV4` | function | `utils/versionConverter` |
| `migrateStateMachineDiagramV3ToV4` | function | `utils/versionConverter` |
| `migrateUserDiagramV3ToV4` | function | `utils/versionConverter` |
| `parseDiagramType` | function | `utils/diagramTypeUtils` |
| `parseSvgPath` | function | `utils/edgeUtils` |
| `removeDuplicatePoints` | function | `utils/edgeUtils` |
| `rendersNameLabel` | function | `utils/nodeUtils` |
| `resizeAllParents` | function | `utils/nodeUtils` |
| `resolveRequiredInterfaceEdgeType` | function | `utils/requiredInterfaceUtils` |
| `setLogger` | function | `logger` |
| `setLogLevel` | function | `logger` |
| `ShapeLayout` | type | `utils/svgTextLayout` |
| `simplifyPoints` | function | `utils/edgeUtils` |
| `simplifySvgPath` | function | `utils/edgeUtils` |
| `snapNodeToGuides` | function | `utils/alignmentUtils` |
| `sortNodesTopologically` | function | `utils/nodeUtils` |
| `Styles` | type | `typings` |
| `Subscribers` | type | `typings` |
| `supportsMultilineName` | function | `utils/nodeUtils` |
| `SVG` | type | `typings` |
| `SvgExportMode` | type | `typings` |
| `SvgFontSpec` | type | `utils/svgTextLayout` |
| `UMLDiagramType` | enum | `typings` |
| `UMLModel` | type | `typings` (v4 shape) |
| `UMLModelElementType` | type | `typings` |
| `Unsubscriber` | type | `typings` |
| `userMetaModel` | const | `services/userMetaModel/usermetamodel.json` |
| `WhiteSpaceMode` | type | `utils/svgTextLayout` |
| `wrapTextInRect` | function | `utils/svgTextLayout` |
| `WrappedText` | type | `utils/svgTextLayout` |

(Plus the full `__testing` export from `exportUtils` and a few internal
types from `utils/exportUtils` / `utils/edgeUtils`; the shape stays as
emitted by the lib.)

---

## B. Diff

### OLD-ONLY (gone or replaced)

| Symbol | Category | Webapp uses? | Suggested SA-7 action |
|---|---|---|---|
| `AgentIntent` | interface | no | Drop. |
| `AgentModelElement` | interface | no | Drop. |
| `AgentRagElement` | interface | no | Drop. |
| `AgentState` | interface | no | Drop. |
| `AgentStateTransition` | type | no | Drop. |
| `ApollonMode`/`Locale`/`Styles`/`UMLDiagramType` | enum/type | yes | Same name, present in lib — re-export verified. **No action.** |
| `addOrUpdateAssessment` (standalone fn) | function | no | Drop (new lib has it as `ApollonEditor` method). |
| `addOrUpdateElement` | function | no | Drop. v4 mutates `model.nodes[]` directly. |
| `addOrUpdateRelationship` | function | no | Drop. v4 mutates `model.edges[]` directly. |
| `BPMNEndEvent`/`BPMNFlow`/`BPMNGateway`/`BPMNIntermediateEvent`/`BPMNStartEvent`/`BPMNTask` | type | no | Drop (BPMN diagram is dropped in the migration plan). |
| `ClassNotation` | type | **yes** (`ProjectSettingsPanel.tsx:2`) | Add `export type { ClassNotation }` to lib `index.tsx` from `services/settingsService`. |
| `DEFAULT_SETTINGS` | const | no | Optional re-export; lib has the same const internally. |
| `DiagramBridgeService` | class | no (only `diagramBridge` instance is) | Optional: re-export class for symmetry. |
| `DiagramReference` | type | no | Drop. |
| `diagramBridge` | const | **yes** (`DiagramTabs.tsx:3`, `ApollonEditorComponent.tsx:1`) | Add `export { diagramBridge } from "./services/diagramBridge"` to lib `index.tsx`. |
| `erCardinalityToUML` | function | yes (test only: `multiplicity.test.ts:2`) | Port from old `packages/common/uml-association/multiplicity.ts` — not present in new lib. SA-7 must add the function (and `parseMultiplicity` / `toERCardinality`) under e.g. `lib/utils/multiplicity.ts`. |
| `findAssessment`/`findElement`/`findRelationship` | function | no | Drop. |
| `IApplicationSettings`/`ISettingsService`/`SettingsService` | interface/class | no | Optional re-export for type clarity. |
| `IAssociationInfo`/`IAttributeInfo`/`IClassDiagramData`/`IClassInfo`/`IDiagramBridgeService`/`IDiagramReference` | interface | no (only consumed via `diagramBridge` instance) | Optional re-export; useful for callers that type-annotate diagramBridge results. |
| `IUMLObjectAttribute`/`IUMLObjectLink`/`IUMLObjectName` | interface | no | Drop (v4 `ApollonNode.data` carries `classId` / `attributeId` etc. in untyped slots). |
| `MethodImplementationType` | type | no | Drop. |
| `parseMultiplicity` | function | yes (test only: `multiplicity.test.ts:2`) | Port — same as `erCardinalityToUML`. |
| `Patch` | type | no | Drop (patcher subsystem retired per plan). |
| `Selection` | type (old `{elements, relationships}` map) | no (only via `editor.select`, which the webapp does not call on Apollon) | Drop. The new lib's `editor.select(ids: string[])` (planned addition) takes a string array. |
| `isInteractiveElement`/`isInteractiveRelationship`/`setInteractiveElement`/`setInteractiveRelationship` | function | no | Drop. v4 has `editor.toggleInteractiveElementsMode()` / `editor.getInteractiveForSerialization()`. |
| `settingsService` | const | **yes** (`ProjectSettingsPanel.tsx:2`) | Add `export { settingsService } from "./services/settingsService"` to lib `index.tsx`. |
| `toERCardinality` | function | yes (test only: `multiplicity.test.ts:2`) | Port — same as `erCardinalityToUML`. |
| `UMLAssociation`/`UMLClassifier`/`UMLClassifierMember`/`UMLCommunicationLink`/`UMLComponentComponent`/`UMLComponentSubsystem`/`UMLDeploymentComponent`/`UMLDeploymentNode`/`UMLElement`/`UMLPetriNetPlace`/`UMLReachabilityGraphMarking`/`UMLRelationship`/`UMLReply`/`UMLState`/`UMLStateTransition` | type/interface | no | Drop. v3-shaped element types — replaced by `ApollonNode.data` being `{[k]: unknown}`. Webapp narrows with its own type guards. |
| `UMLElementType`/`UMLRelationshipType` | enum | no (webapp uses string literals) | Drop. v4 uses `DiagramNodeType` / `DiagramEdgeType` (string-literal unions). |
| `UMLModelCompat` | type | no | Drop. |
| `UMLModelElement` | type | no | Drop. |
| `Visibility` | type | no | Drop. |

### NEW-ONLY (added)

| Symbol | Category | Notes |
|---|---|---|
| `ApollonEdge` | type | v4 React-Flow edge shape with `{source, target, sourceHandle, targetHandle, data: {points, ...}}`. Replaces `UMLRelationship`. |
| `ApollonNode` | type | v4 React-Flow node shape with `{position, width, height, measured, data, parentId}`. Replaces `UMLElement`. |
| `ApollonView` | enum | `Modelling` / `Exporting` / `Highlight`. New view-mode concept. |
| `DiagramEdgeType` / `DiagramNodeType` | type | String-literal unions; new lib's discriminator for nodes/edges. |
| `InteractiveElements` | type | `{elements: Record<string, boolean>; relationships: Record<string, boolean>}`. Same shape as old `Selection`, different name and purpose. |
| `Subscribers` / `Unsubscriber` | type | New subscription-bookkeeping types. |
| `SvgExportMode` | type | `"web"` / `"compat"` — new `ExportOptions.svgMode` field. |
| `userMetaModel` | const | Shipped JSON metamodel for user-modelling reference. |
| `log` / `setLogLevel` / `setLogger` / `LogLevel` | function/type | Logging hooks that ship with the lib; pure additions. |
| ~50 utility functions (`measureTextWidth`, `wrapTextInRect`, `getDiagramBounds`, `convertV3ToV4`, `migrate*V3ToV4`, etc.) | function | Internal tooling now exported. SA-7 doesn't have to use them, but they're available. |
| `AssessmentViewData`, `EdgeMarkerStyles`, `WrappedText`, `ShapeLayout`, `AlignmentInfo`, `DeepPartial`, … | type | Helper types for the above. |

### Shape-changed (PRESENT in both, different shape)

| Symbol | Old shape | New shape | Webapp impact |
|---|---|---|---|
| `UMLModel` | `{version: "3.x.y", type, size, elements: Record<id, UMLElement>, interactive: Selection, relationships: Record<id, UMLRelationship>, assessments: Record<id, Assessment>, referenceDiagramData?}` | `{version: "4.x.y", id, title, type, nodes: ApollonNode[], edges: ApollonEdge[], assessments: Record<id, Assessment>, interactive?: InteractiveElements}` | **Massive** — 274 access sites in 19 webapp files. See section D. Notable: `size` and `referenceDiagramData` removed; `id` and `title` added; `elements`/`relationships` become `nodes`/`edges`. |
| `Assessment` | `{modelElementId, elementType: UMLElementType \| UMLRelationshipType, score, feedback?, dropInfo?, label?, labelColor?, correctionStatus?}` | `{modelElementId, elementType: string, score, feedback?, dropInfo?: unknown, label?, labelColor?, correctionStatus?}` | Low — webapp doesn't construct assessments; only flows through `editor.model.assessments`. The `elementType` widening (`UMLElementType \| UMLRelationshipType` → `string`) is implicit; no fix needed. |
| `ApollonOptions` | `{type?, mode?, readonly?, enablePopups?, model?, theme?, locale?, copyPasteToClipboard?, colorEnabled?, scale?}` | adds `view?, availableViews?, debug?, collaborationEnabled?, scrollLock?` on top of old fields | Pure addition; existing call sites still type-check. |
| `ExportOptions` | `{margin?, keepOriginalSize?, include?, exclude?}` | adds `svgMode?: "web" \| "compat"` | Pure addition; no fix needed. Webapp may opt into `svgMode: "compat"` for PowerPoint export. |
| `Locale` | TS enum (`Locale.en` / `Locale.de`) | TS enum (same, identical members) | No change. |
| `ApollonMode` | TS enum (`Modelling`, `Exporting`, `Assessment`) | TS enum (same members) | No change. |
| `UMLDiagramType` | TS enum | TS enum (re-exported from `lib/types/DiagramType`) | Members may differ for diagrams dropped in v4 (BPMN, etc.). Webapp uses the names; SA-7 should grep for missing members. |
| `Styles` | DeepPartial-friendly theme shape | DeepPartial-friendly theme shape | Same role; field names match (`packages/library/lib/styles/theme`). Visual diff only inside the lib. |
| `SVG` | `{svg: string; clip: {x, y, width, height}}` | identical | No change. |
| `ApollonEditor` | see section C | see section C | Several methods missing from new lib; see below. |

---

## C. `ApollonEditor` method-level diff

Sources:

- editor (old): `packages/editor/src/main/apollon-editor.ts` (557 lines)
- library (new): `packages/library/lib/apollon-editor.tsx` (466 lines)

| Old method | New method | Status | Notes |
|---|---|---|---|
| `constructor(container, options)` | `constructor(element, options?)` | PRESENT | Same. New constructor's `options` is optional; old fork's was required. Non-breaking. |
| `get model: UMLModel` | `get model: UMLModel` | PRESENT, **shape-changed** | Returns v4 `{nodes, edges, …}` instead of v3 `{elements, relationships, …}`. Every call-site that destructures the result needs the v4 shape (see section D). |
| `set model(model: UMLModelCompat)` | `set model(model: UMLModel)` | PRESENT, **shape-changed** | Setter takes v4 shape. Old fork's setter accepted v2/v3 via `UMLModelCompat`; new lib expects pure v4. Webapp must convert (use `convertV3ToV4` from the new lib). |
| `set type(diagramType)` | `set diagramType(type: UMLDiagramType)` | **RENAMED** | `type` → `diagramType`. Webapp does not currently set this property (no hits in `grep`), so no fix required. |
| `set locale(locale)` | (none) | **DROPPED** | Locale is set at construction only in the new lib. Webapp does not set it post-init (no hits). |
| static `exportModelAsSvg(model, options?, theme?)` | static `exportModelAsSvg(model, options?, theme?)` | PRESENT | Same signature; new version respects `options.svgMode`. |
| `destroy()` | `destroy()` | PRESENT | Same. Webapp calls in `ApollonEditorComponent.tsx:44` and `GraphicalUIEditor.tsx:51`. |
| `select(selection: Selection)` | (planned: `select(ids: string[])`) | **MISSING + signature change** | Old took `{elements, relationships}` Selection. New lib doesn't expose it yet. Per plan, new will take `string[]`. **Webapp does not call `editor.select(...)`** on Apollon — only on grapesjs (`GraphicalUIEditor.tsx`), which is a different `editor`. SA-7 should still add this for API parity but no webapp fix is required. |
| `subscribeToSelectionChange(cb)` | (planned addition) | **MISSING** | Webapp does not subscribe to selection changes. SA-7 may add per plan, but not blocking. |
| `unsubscribeFromSelectionChange(id)` | (none) | **DROPPED** | No webapp use. |
| `subscribeToAssessmentChange(cb)` | (none) | **DROPPED** | No webapp use; replaced by `subscribeToAssessmentSelection`. |
| `unsubscribeFromAssessmentChange(id)` | (none) | **DROPPED** | No webapp use. |
| `subscribeToModelChange(cb)` | `subscribeToModelChange(cb)` | PRESENT | Same callback shape (now sees v4 `UMLModel`). Webapp uses at `ApollonEditorComponent.tsx:129`. |
| `unsubscribeFromModelChange(id)` | (planned alias to `unsubscribe(id)`) | **MISSING (alias)** | Webapp uses at `ApollonEditorComponent.tsx:66`. SA-7 must either (a) add alias on lib, or (b) rename the call site to `editor.unsubscribe(id)`. Plan calls for the alias. |
| `subscribeToModelDiscreteChange(cb)` | (none) | **DROPPED** | No webapp use. |
| `unsubscribeFromDiscreteModelChange(id)` | (none) | **DROPPED** | No webapp use. |
| `subscribeToModelChangePatches(cb)` | (none) | **DROPPED** (per plan) | No webapp use. Patch system retired. |
| `subscribeToAllModelChangePatches(cb)` | (none) | **DROPPED** (per plan) | No webapp use. |
| `subscribeToModelContinuousChangePatches(cb)` | (none) | **DROPPED** (per plan) | No webapp use. |
| `unsubscribeFromModelChangePatches(id)` | (none) | **DROPPED** (per plan) | No webapp use. |
| `importPatch(patch: Patch)` | (none) | **DROPPED** (per plan) | No webapp use. |
| `subscribeToApollonErrors(cb)` | (planned addition) | **MISSING** | No webapp use today. Plan calls for adding. Not blocking. |
| `unsubscribeToApollonErrors(id)` | (planned addition) | **MISSING** | No webapp use today. |
| `remoteSelect(name, color, select, deselect?)` | (none) | **DROPPED** (per plan) | No webapp use. |
| `pruneRemoteSelectors(allowed)` | (none) | **DROPPED** (per plan) | No webapp use. |
| `exportAsSVG(options?)` | `exportAsSVG(options?)` | PRESENT | Same signature. Webapp uses at `useExportSvg.ts:10`, `useExportPng.ts:10`. |
| `getScaleFactor()` | (none) | **DROPPED** | No webapp use. |
| `get nextRender: Promise<void>` | (planned: `get ready: Promise<void>`) | **MISSING + RENAME** | Used by webapp at 6 sites: `DiagramTabs.tsx:204,206`; `ApollonEditorComponent.tsx:117`; `UMLModelingService.ts:353,355,418,420`. SA-7 fix: either add `ready` getter on lib + rename call sites, or add `nextRender` alias. Plan calls for the rename. |
| (none) | `getNodes(): Node[]` | **NEW** | React-Flow leak; no webapp use today. |
| (none) | `getEdges(): Edge[]` | **NEW** | React-Flow leak; no webapp use today. |
| (none) | `subscribeToDiagramNameChange(cb)` | **NEW** | No webapp use today. |
| (none) | `subscribeToAssessmentSelection(cb)` | **NEW** | No webapp use today. |
| (none) | `unsubscribe(id)` | **NEW** | Generic unsubscriber. Used internally by all subscription handles. |
| (none) | `sendBroadcastMessage(sendFn)` / `receiveBroadcastedMessage(base64Data)` | **NEW** | Yjs collab plumbing; webapp does not touch (collab disabled per plan). |
| (none) | `updateDiagramTitle(name)` | **NEW** | No webapp use today. |
| (none) | `toggleInteractiveElementsMode(forceEnabled?)` | **NEW** | No webapp use today. |
| (none) | `getInteractiveForSerialization()` | **NEW** | No webapp use today. |
| (none) | `getDiagramMetadata()` | **NEW** | No webapp use today. |
| (none) | `getSelectedElements(): string[]` | **NEW** | No webapp use today. |
| (none) | `get/set view: ApollonView` | **NEW** | No webapp use today; plan suggests using for the Highlight view. |
| (none) | `addOrUpdateAssessment(assessment)` | **NEW** (method) | No webapp use today. |
| (none) | static `generateInitialSyncMessage()` | **NEW** | Yjs-only. |
| (none, planned) | `undo()` / `redo()` | **MISSING** (per plan, not yet added) | No webapp use today. |

**Summary**: of the 27 old `ApollonEditor` members, 12 are dropped (per
plan; none used by webapp), 6 are present unchanged, 2 are renamed
(`type` → `diagramType`; `nextRender` → `ready`), 7 are present with
shape-changed `UMLModel`, and the new lib adds 14 net-new methods. The
**only blockers for SA-7** are `nextRender` / `ready` and
`unsubscribeFromModelChange` / `unsubscribe` — every other diff is
either internal to the lib or unused by webapp.

---

## D. `UMLModel`-shape touches in webapp

274 access sites (`grep -rn "\\.elements\\b\\|\\.relationships\\b" packages/webapp/src | grep -i model`)
in 19 files. SA-7 must rewrite each to read `model.nodes` and
`model.edges`. Listed by file with pattern.

1. `packages/webapp/src/main/shared/types/project.ts:617-618` — emptiness
   check `model.elements && Object.keys(model.elements).length > 0`.
   Replace with `Array.isArray(model.nodes) && model.nodes.length > 0`.
2. `packages/webapp/src/main/shared/services/validation/validateDiagram.ts:50-51`
   — same emptiness check. Same fix.
3. `packages/webapp/src/main/shared/utils/__tests__/projectExportUtils.test.ts:31,79,339,340`
   — test fixtures construct v3 models. Replace fixtures with v4
   `{nodes: [], edges: []}`.
4. `packages/webapp/src/main/app/shell/hooks/useDeployment.ts:41` —
   emptiness check on `modelWithElements.elements`. Replace.
5. `packages/webapp/src/main/features/editors/diagram-tabs/DiagramTabs.tsx:84,85,188`
   — element/relationship counts; reference-diagram class count.
   Replace `Object.keys(diagram.model.elements ?? {}).length` →
   `(diagram.model.nodes ?? []).length`. For class count, replace
   `Object.values(refModel.elements ?? {}).filter(el => el.type === "Class")`
   → `(refModel.nodes ?? []).filter(n => n.type === "Class")`.
6. `packages/webapp/src/main/features/editors/diagram-tabs/scaffoldObjectsFromClasses.ts:109,115,273,274,298,312,374`
   — heaviest hit. Walks classes, attributes, relationships to scaffold
   object instances. Each `Object.values(classModel.elements ?? {})` →
   `(classModel.nodes ?? [])`; nested `attributes`/`methods` lookups
   change from id-references-into-elements to inline arrays in
   `node.data.attributes` / `node.data.methods` (per
   `migrations/uml-v4-shape.md`).
7. `packages/webapp/src/main/features/editors/gui/diagram-helpers.ts:43,56,76,110,171,251,265,305,348,420`
   — checks like `if (!isUMLModel(classDiagram) || !classDiagram.elements)` →
   replace with `nodes` array check. Same for `.relationships` →
   `.edges`. The `'relationships' in classDiagram` `in`-checks must
   become `'edges' in classDiagram`.
8. `packages/webapp/src/main/features/generation/useGeneratorExecution.ts:74,75,111,113,121`
   — `elementCount` / `relationshipCount` plus an `Object.values(model.elements)` cast.
   Replace with `nodes.length` / `edges.length` and
   `(model.nodes ?? [])` accessor.
9. `packages/webapp/src/main/features/assistant/hooks/useAssistantLogic.ts:718,719`
   — element/relationship counts. Replace.
10. `packages/webapp/src/main/features/assistant/services/UMLModelingService.ts:398,399,404,408,434,438,570,575,595,638,639,671,672,689,690,693,694`
    — heavy: full v3 model assembly via spreads
    `{...currentModel.elements, ...systemData.elements}`. Each spread
    becomes array concat / merge-by-id over `nodes` / `edges`. The
    `Object.entries(systemData.elements).map(...)` patterns become
    `systemData.nodes.map(...)`.
11. `packages/webapp/src/main/features/assistant/services/AssistantClient.ts:171-173`
    — guard then `Object.values(model.elements)` and
    `Object.values(model.relationships)`. Replace with
    `model.nodes` / `model.edges`.
12. `packages/webapp/src/main/features/assistant/services/modifiers/base.ts:170,176,189,201,209,215,218-222`
    — base modifier walks and deletions. `model.elements[id] = …` →
    push-or-replace into `model.nodes`. `delete model.elements[id]` →
    `model.nodes = model.nodes.filter(n => n.id !== id)`. Same for
    `relationships` → `edges`.
13. `packages/webapp/src/main/features/assistant/services/modifiers/ClassDiagramModifier.ts:110,114,123,133,134,138,183,194,208,209`
    — class-specific modifier; same translation pattern.
14. `packages/webapp/src/main/features/assistant/services/modifiers/StateMachineModifier.ts:52,62,73,91,104,117,129,149,151,159,160,176,199,200,211,219`
    — heavy: state-machine state/transition build-up; same translation.
15. `packages/webapp/src/main/features/assistant/services/modifiers/AgentDiagramModifier.ts`
    (29 hits per `grep -c`) — same.
16. `packages/webapp/src/main/features/assistant/services/modifiers/ObjectDiagramModifier.ts`
    — same.
17. `packages/webapp/src/main/features/assistant/services/modifiers/GUIDiagramModifier.ts:58,83,84,110`
    — same.
18. `packages/webapp/src/main/features/assistant/services/modifiers/QuantumCircuitModifier.ts:44,69,70`
    — same.
19. `packages/webapp/src/main/features/assistant/services/converters/ClassDiagramConverter.ts`
    — converter producing v3 shape; rewrite to emit v4.
20. `packages/webapp/src/main/features/import/useImportDiagramPicture.ts:30`
    — `(activeClassDiagram as any)?.model?.elements`. Replace with
    `model?.nodes`.

Note: `migrations/uml-v4-shape.md` is the authoritative per-diagram
field map; SA-7 should consult it for each modifier rewrite (e.g.
`ClassElement.attributes` was `string[]` of attribute-element IDs in v3;
in v4 it's an inline array of `{id, name, type, ...}` on `node.data`).

---

## E. SA-7 to-do (prioritised)

**P0 — required for the cutover to compile**:

1. Add three barrel re-exports to `packages/library/lib/index.tsx`:
   ```ts
   export { diagramBridge, DiagramBridgeService } from "./services/diagramBridge"
   export type {
     IDiagramBridgeService, IClassDiagramData, IClassInfo,
     IAttributeInfo, IAssociationInfo, IDiagramReference,
   } from "./services/diagramBridge"
   export { settingsService, SettingsService, DEFAULT_SETTINGS } from "./services/settingsService"
   export type { ClassNotation, IApplicationSettings, ISettingsService } from "./services/settingsService"
   ```
2. Add a `ready: Promise<void>` getter on `ApollonEditor` (resolves when
   the React tree's first render commits). Either rename `nextRender`
   call sites to `ready` (6 sites in 3 webapp files) or alias `ready`
   as `nextRender` for one release; the plan calls for the rename. The
   simpler path is: ship `ready` on the lib and patch the 6 webapp call
   sites in the same PR.
3. Add `unsubscribeFromModelChange(id)` as an alias method on
   `ApollonEditor` that delegates to `unsubscribe(id)`. Or rename the
   single webapp call site (`ApollonEditorComponent.tsx:66`).
4. Port `parseMultiplicity`, `toERCardinality`, `erCardinalityToUML`
   from `packages/editor/src/main/packages/common/uml-association/multiplicity.ts`
   into a new `packages/library/lib/utils/multiplicity.ts` and
   re-export from `index.tsx`. The webapp has a test
   (`features/editors/uml/__tests__/multiplicity.test.ts`) that imports
   them directly.
5. Either re-export `UMLDiagramType` enum members the webapp relies on,
   or audit `grep -rn "UMLDiagramType\\." packages/webapp/src` for any
   member name removed in v4 (BPMN diagrams, syntax-tree, flowchart,
   reachability-graph, communication, deployment may be gone). Adjust
   webapp call sites accordingly.

**P1 — required for the webapp to behave correctly post-cutover**:

6. Rewrite all 274 `model.elements` / `model.relationships` access sites
   to v4 `model.nodes` / `model.edges`. Heaviest files:
   `features/assistant/services/modifiers/*.ts` (≈ 110 sites),
   `features/editors/gui/diagram-helpers.ts` (10 sites),
   `features/assistant/services/UMLModelingService.ts` (17 sites),
   `features/editors/diagram-tabs/scaffoldObjectsFromClasses.ts`
   (7 sites). Use `migrations/uml-v4-shape.md` per diagram type.
7. Replace `IUMLObjectName.classId` / `IUMLObjectAttribute.attributeId` /
   `IUMLObjectLink.associationId` references with the equivalent fields
   on v4 `node.data` / `edge.data` (per the v4 shape spec —
   `ObjectDiagram` section).
8. Replace any `interactive: {elements, relationships}` reads/writes
   with the new lib's `editor.getInteractiveForSerialization()` /
   `editor.toggleInteractiveElementsMode()` API, or accept the
   `InteractiveElements` shape as-is (same field names).
9. Audit fixture/test files for v3 model literals and migrate them. At
   least: `shared/utils/__tests__/projectExportUtils.test.ts`,
   `features/editors/__tests__/HiddenPerspectivesBanner.test.tsx`,
   `shared/types/__tests__/project.test.ts`,
   `features/editors/diagram-tabs/__tests__/DiagramTabs.test.tsx`.

**P2 — opportunistic clean-up**:

10. Drop unused import of v3-only types — `Selection`, `UMLElement`,
    `UMLRelationship`, `UMLClassifier`, `UMLClassifierMember`, etc. The
    webapp does not currently import these, so this is a hygiene check.
11. Re-export `ApollonNode`, `ApollonEdge`, `DiagramNodeType`,
    `DiagramEdgeType`, `InteractiveElements` from `@besser/wme` (the
    library already does — confirm webapp has access).
12. Decide on an alias strategy for `nextRender` and
    `unsubscribeFromModelChange`. If SA-7 adds aliases on the lib side,
    the webapp diff shrinks to ~zero on `ApollonEditor`-method changes
    and stays only in `UMLModel`-shape changes — easier to review.
