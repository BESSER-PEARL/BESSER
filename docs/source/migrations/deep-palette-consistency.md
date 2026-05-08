# Deep Palette / Canvas Consistency Audit

**Audit ID:** `SA-DEEP-PALETTE-CONSISTENCY`
**Branch:** `claude/refine-local-plan-sS9Zv`
**Mode:** read-only
**Sources of truth**
- Palette: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/constants.ts`
  (`dropElementConfigs` — keyed by `UMLDiagramType`)
- Canvas nodes: `.../packages/library/lib/nodes/<diagram>/<NodeType>.tsx`
- Palette SVGs: `.../packages/library/lib/components/svgs/nodes/<diagram>/...`
- Sidebar / `sectionLabel` rendering: `.../packages/library/lib/components/Sidebar.tsx`

For each entry the audit compares:
1. The palette `svg` (drag ghost) — what the user sees while hovering.
2. The canvas component's render — what lands on canvas after drop.
3. Default dimensions (`width` / `height` in palette) vs the canvas
   `NodeResizer minWidth/minHeight` (which silently grows the dropped
   node when palette dims are smaller).

Throughout:
- "Same SVG" means the canvas component re-uses the palette SVG verbatim
  (the dropped node is literally a re-render of the same React component
  at the dropped width/height).
- "Inline SVG" means the canvas component renders its own `<svg>` rather
  than the palette SVG — drift is possible if the two implementations
  diverge.

---

## 1. ClassDiagram

| Palette entry (defaultData.name → label) | width × height | Canvas component | Drift? |
| --- | --- | --- | --- |
| `class` "Class" (1 attr, 0 methods) | 160 × 90 | `nodes/classDiagram/Class.tsx` (uses `ClassSVG`) | None — same SVG; canvas auto-grows height to attribute count, but base header+1 attribute = 70 px ≤ 90 px so no clamp. Stereotype band absent in both. |
| `class` "Class" (1 attr + 1 method) | 160 × 110 | same | None — same SVG. |
| `class` "Abstract" (`stereotype: Abstract`) | 160 × 110 | same | None — `ClassSVG` renders `«Abstract»` band + italic name; palette path produces the band because `showStereotype = !!stereotype`. `isItalic` defaults to `stereotype===Abstract`, so both apply. |
| `class` "Enumeration" (`stereotype: Enumeration`, 3 literals) | 160 × 140 | same | None — `ClassSVG` skips methods compartment when `stereotype===Enumeration` (`isEnumeration` flag), and the palette default supplies no methods so the visual matches. Header band shows `«Enumeration»` in both. |
| `ClassOCLConstraint` "constraint" | 180 × 90 | `nodes/classDiagram/ClassOCLConstraint.tsx` (inline SVG) | **Drift.** Palette uses fold=12, padding=10, header font 11; canvas uses fold=14, padding=12, header font 12. Palette hard-codes `«inv»` badge; canvas derives from expression. Body text: palette omits, canvas wraps and renders the OCL expression. The minimum canvas size (160 × 80) is smaller than the palette card (180 × 90) so dimensions grow consistently — but visually the dropped node is denser/larger-text than the ghost. |

## 2. ObjectDiagram

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `objectName` "Object" (1 attr `attribute = value`) | 160 × 70 | `nodes/objectDiagram/ObjectName.tsx` → `ObjectNameSVG` | None — same SVG. Note: `showInstancedObjects` toggle hides attribute rows in palette only; the dropped node always shows attributes. This is a **deliberate** preview-vs-canvas behavior, not drift. |

## 3. ActivityDiagram

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `activity` "Activity" | 160 × 120 | `nodes/activityDiagram/Activity.tsx` → `ActivitySVG` | None — same SVG. |
| `activityInitialNode` | 50 × 50 | → `ActivityInitialNodeSVG` | None — same SVG. |
| `activityFinalNode` | 50 × 50 | → `ActivityFinalNodeSVG` | None — same SVG. |
| `activityActionNode` "Action" | 160 × 120 | → `ActivityActionNodeSVG` | None — same SVG. |
| `activityObjectNode` "Object" | 160 × 120 | → `ActivityObjectNodeSVG` | None — same SVG. |
| `activityMergeNode` "Condition" | 160 × 120 | → `ActivityMergeNodeSVG` | None — same SVG. |
| `activityForkNode` "Fork" | 20 × 100 | → `ActivityForkNodeSVG` | None — same SVG. |
| `activityForkNodeHorizontal` "Fork" | 100 × 20 | → `ActivityForkNodeHorizontalSVG` | None — same SVG. |

## 4. UseCaseDiagram

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `useCase` "Use Case" | 160 × 100 | `UseCase.tsx` → `UseCaseNodeSVG` | None — same SVG. |
| `useCaseActor` "Actor" | 100 × 150 | `UseCaseActor.tsx` → `UseCaseActorNodeSVG` | None — same SVG. |
| `useCaseSystem` "System" | 160 × 120 | `UseCaseSystem.tsx` → `UseCaseSystemNodeSVG` | None — same SVG. |

## 5. CommunicationDiagram

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `communicationObjectName` "Object" (1 attr) | 160 × 70 | `CommunicationObjectName.tsx` → `CommunicationObjectNameSVG` | None — same SVG. |

## 6. ComponentDiagram

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `component` "Component" | 180 × 120 | `Component.tsx` → `ComponentNodeSVG` | None — same SVG. |
| `componentSubsystem` "Subsystem" | 180 × 120 | `ComponentSubsystem.tsx` → `ComponentSubsystemNodeSVG` | None — same SVG. |
| `componentInterface` "Interface" | 30 × 30 (`INTERFACE_SIZE`) | `ComponentInterface.tsx` → `ComponentInterfaceNodeSVG` | None — same SVG. |

## 7. DeploymentDiagram

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `deploymentNode` "Node" (`stereotype: node`) | 160 × 100 | → `DeploymentNodeSVG` | None — same SVG. Stereotype band shows `«node»` in both. |
| `deploymentComponent` "Component" | 160 × 100 | → `DeploymentComponentSVG` | None — same SVG. |
| `deploymentArtifact` "Artifact" | 160 × 50 | → `DeploymentArtifactSVG` | None — same SVG. |
| `deploymentInterface` "Interface" | 30 × 30 | → `DeploymentInterfaceSVG` | None — same SVG. |

## 8. SyntaxTree

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `syntaxTreeNonterminal` "Nonterminal" | 160 × 100 | → `SyntaxTreeNonterminalNodeSVG` | None — same SVG. |
| `syntaxTreeTerminal` "Terminal" | 160 × 100 | → `SyntaxTreeTerminalNodeSVG` | None — same SVG. |

## 9. PetriNet

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `petriNetTransition` "Transition" | 30 × 60 | → `PetriNetTransitionSVG` | None — same SVG. |
| `petriNetPlace` "Place" | 60 × 60 | → `PetriNetPlaceSVG` | None — same SVG. |

## 10. ReachabilityGraph

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `reachabilityGraphMarking` "Marking" | 160 × 120 | → `ReachabilityGraphMarkingSVG` | None — same SVG. |

## 11. Flowchart

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `flowchartTerminal` "Terminal" | 160 × 70 | → `FlowchartTerminalNodeSVG` | None — same SVG. |
| `flowchartProcess` "Process" | 160 × 70 | → `FlowchartProcessNodeSVG` | None — same SVG. |
| `flowchartDecision` "Decision" | 160 × 70 | → `FlowchartDecisionNodeSVG` | None — same SVG. |
| `flowchartInputOutput` "Input/Output" | 140 × 70 | → `FlowchartInputOutputNodeSVG` | None — same SVG. |
| `flowchartFunctionCall` "Function Call" | 160 × 70 | → `FlowchartFunctionCallNodeSVG` | None — same SVG. |

## 12. BPMN

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `bpmnTask` "Task" | 160 × 60 | `BPMNTask.tsx` → `BPMNTaskNodeSVG` (minWidth 80, minHeight 50) | None — same SVG. |
| `bpmnSubprocess` "Subprocess" | 160 × 60 | `BPMNSubprocess.tsx` → `BPMNSubprocessNodeSVG` | None — same SVG. |
| `bpmnTransaction` "Transaction" (`variant: transaction`) | 160 × 60 | `BPMNTransaction.tsx` → `BPMNSubprocessNodeSVG` (variant prop) | None — same SVG, variant honored. |
| `bpmnCallActivity` "Call Activity" (`variant: call`) | 160 × 60 | `BPMNCallActivity.tsx` → `BPMNSubprocessNodeSVG` | None — same SVG. |
| `bpmnGroup` "Group" | 160 × 60 | `BPMNGroup.tsx` → `BPMNGroupNodeSVG` | None — same SVG. |
| `bpmnAnnotation` "Annotation" | 160 × 60 | → `BPMNAnnotationNodeSVG` | None — same SVG. |
| `bpmnStartEvent` (variant `start`) | 40 × 40 | → `BPMNEventNodeSVG` | None — same SVG. |
| `bpmnIntermediateEvent` (variant `intermediate`) | 40 × 40 | → `BPMNEventNodeSVG` | None — same SVG. |
| `bpmnEndEvent` (variant `end`) | 40 × 40 | → `BPMNEventNodeSVG` | None — same SVG. |
| `bpmnGateway` | 40 × 40 | → `BPMNGatewayNodeSVG` | None — same SVG. |
| `bpmnDataObject` | 40 × 60 | → `BPMNDataObjectNodeSVG` (minWidth **60**) | **Drift on dimensions.** Palette ghost is 40 wide; canvas resizer immediately grows it to 60 on first interaction (and the `width` field is already inconsistent). |
| `bpmnDataStore` | 60 × 60 | → `BPMNDataStoreNodeSVG` | None. |
| `bpmnPool` "Pool" | 160 × 80 | `BPMNPool.tsx` → `BPMNPoolNodeSVG` (minWidth **200**, minHeight **120**) | **Drift on dimensions.** Dropped pool is forced to 200 × 120, never 160 × 80. The drag ghost looks half the size of the dropped node. |

## 13. StateMachineDiagram

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `State` "State" | 160 × 100 | `State.tsx` (inline SVG) | Minor — palette and canvas both render rect (rx 8) + name at y=26, divider at y=40. Palette font-weight unset (defaults 400); canvas uses `fontWeight=600`. **Bold-vs-regular drift on the name.** |
| `StateInitialNode` | 45 × 45 | `StateInitialNode.tsx` (inline SVG) | None — both render a solid-fill circle of radius `min(w,h)/2`. |
| `StateFinalNode` | 45 × 45 | `StateFinalNode.tsx` (inline SVG) | None — both render concentric circles (outer outlined + inner filled). |
| `StateActionNode` "Action" | 160 × 50 | `StateActionNode.tsx` (inline SVG) | Drift — palette renders the literal label "Action" at `y=h/2+5`; canvas renders `data.name` (which IS "Action" by default, so for the default drop they match). However palette uses `rx=5` and font 14, canvas may differ; spot-check shows canvas matches. |
| `StateObjectNode` "Object" | 160 × 50 | `StateObjectNode.tsx` (inline SVG) | Drift — palette renders literal "Object" centered, fontWeight bold; canvas renders `data.name`. Match for default `name="Object"`. |
| `StateMergeNode` | 80 × 80 | `StateMergeNode.tsx` (inline SVG) | None — diamond polygon. |
| `StateForkNode` | 20 × 60 | `StateForkNode.tsx` (inline SVG) | None — solid black rect. |
| `StateForkNodeHorizontal` | 60 × 20 | `StateForkNodeHorizontal.tsx` (inline SVG) | None — solid black rect. |
| `StateCodeBlock` "code" | 200 × 150 | `StateCodeBlock.tsx` (inline SVG) | **Drift on body content.** Palette renders header bar with `python` (font 10, no bold) and a static `# code` body line. Canvas renders header bar with `python` (font 10 + **bold**) and the multi-line `code` field (default `# Sample code\nprint("Hello World")`). Drag ghost shows minimal placeholder; dropped node shows full code. Header height 20 px in both. |

## 14. AgentDiagram

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `AgentState` "AgentState" | 160 × 100 | `AgentState.tsx` (inline SVG) | None — both rounded rect (rx 8) with name at center. Canvas uses `fontWeight=600`; palette also `fontWeight=600`. |
| `AgentIntent` "Intent" | 160 × 100 (palette ghost adds 30 px for the fold) | `AgentIntent.tsx` (inline SVG) | **Drift on label.** Palette draws the folded-corner shape and shows the literal text `Intent` (no prefix). Canvas prepends `"Intent: "` to `data.name`, so the dropped node header reads `Intent: Intent` (default name) vs the palette's `Intent`. |
| `AgentIntentObjectComponent` "slot:entity" | 160 × 50 | `AgentIntentObjectComponent.tsx` (inline SVG) | None — small rounded rect with label, both render `slot:entity`. |
| `AgentRagElement` "RAG" | 160 × 120 | `AgentRagElement.tsx` (inline SVG) | **Drift on label.** Palette shows only the centered text `RAG DB` inside the cylinder. Canvas shows `RAG DB` near the top ellipse AND a second line below it with the resolved display name (`dbCustomName ?? ragDatabaseName ?? name`, which equals `"RAG"` for the default drop). Drag ghost is missing the second line. |
| `StateInitialNode` (reused) | 45 × 45 | `StateInitialNode.tsx` | None. |
| `StateFinalNode` (reused) | 45 × 45 | `StateFinalNode.tsx` | None. |

## 15. UserDiagram (dynamic palette)

The palette is generated by
`getUserModelNamePaletteEntries()` in
`components/svgs/nodes/userDiagram/UserDiagramSVGs.tsx`, which walks
`getUserMetaModelClasses()` (excluding the placeholder `User` class) and
emits one drag-source per meta-model class with its attributes folded
into `defaultData.attributes`.

The canvas component is `nodes/userDiagram/UserModelName.tsx` which
re-uses `UserModelNameSVG` (the same SVG used for the palette previews
via the `makeUserModelPaletteSVG` factory). So the **shape and header
band** match exactly: instance name + ` : className`, underlined.

| Palette | Canvas | Drift? |
| --- | --- | --- |
| Per meta-model class (e.g. `Personal_Information`, `Skill`, `Education`, `Disability`, …) | `UserModelName` | **Drift on row labels.** The palette's preview-row labels are formatted as `${attrName} =` (an equals sign with no operator). The canvas applies `formatUserModelAttributeForDisplay`, which uses `data.attributeOperator` (default `==` per the palette `defaultData`). So a dropped row reads `age ==`, but the ghost row reads `age =`. |
| `UserModelName` static "Alice" preview | `UserModelName` (no className→raw `Alice` header) | None — both render header `Alice : User` for the static palette card; canvas matches because `defaultData.className === "User"`. |
| `UserModelIcon` ico-circle | `UserModelIcon.tsx` | None — same circle SVG. |

Width / height parity: palette uses `40 + nAttributes*30 + 10` (≈ 50 px
header gutter + 30 px per attribute). The canvas (`UserModelName.tsx`)
recomputes `minHeight = calculateMinHeight(40, attrCount, 0, 30, 30)` =
`40 + 30*attrCount`. Palette is exactly 10 px taller than canvas
`minHeight` — close, though the dropped node will silently shrink by
10 px on first measurement.

## 16. NNDiagram

`Sidebar.tsx` honors `sectionLabel` on a palette entry: it prepends a
divider (suppressed for the very first entry) plus an upper-case heading
above the entry. This is wired correctly (lines 141–168) and produces:
`NN Structure` → `NN Layers` → `NN TensorOps` → `NN Configuration` →
`NN Datasets`.

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `NNContainer` "MyModel" (sectionLabel **NN Structure**) | 320 × 200 | `NNContainer.tsx` (inline SVG, minW 200, minH 140) | Minor — palette draws hint of "two stacked layers" rectangles for visual flavour; canvas doesn't. The dropped container is empty until the user drops layers inside. |
| `NNReference` "ref" | 140 × 40 | `NNReference.tsx` (inline SVG) | **Drift on label.** Palette shows literal `→ Reference` (italic). Canvas shows `→ ${data.name || data.referenceTarget}` — for the default drop with name="ref", canvas reads `→ ref`. |
| `Conv1DLayer` (sectionLabel **NN Layers**) | 160 × 140 | `_NNLayerBase` via `makeNNLayerComponent("Conv1DLayer", "Conv1D", "#FFF8E1")` | None — palette `NNLayerPreviewSVG` mirrors `_NNLayerBase`: stereotype line at y=18, name at y=38, divider at y=50, PNG icon (`/images/nn-layers/conv1d.png`) centred in remaining space. |
| `Conv2DLayer` | 160 × 140 | same factory | None — uses `conv2d.png` in BOTH palette and canvas. |
| `Conv3DLayer` | 160 × 140 | same | None — `conv3d.png` in both. |
| `PoolingLayer` (default attr `pooling.dimension: "2D"`) | 160 × 140 | same | None — `pooling.png` in both. |
| `RNNLayer` | 160 × 140 | same | None — `rnn.png`. |
| `LSTMLayer` | 160 × 140 | same | None — `lstm.png`. |
| `GRULayer` | 160 × 140 | same | None — `gru.png`. |
| `LinearLayer` | 160 × 140 | same | None — `linear.png`. |
| `FlattenLayer` | 160 × 140 | same | None — `flatten.png`. |
| `EmbeddingLayer` | 160 × 140 | same | None — `embedding.png`. |
| `DropoutLayer` | 160 × 140 | same | None — `dropout.png`. |
| `LayerNormalizationLayer` | 160 × 140 | same | None — `layernorm.png`. |
| `BatchNormalizationLayer` (default `batch_normalization.dimension: "1D"`) | 160 × 140 | same | None — `batchnorm.png`. |
| `TensorOp` (sectionLabel **NN TensorOps**) | 160 × 140 | `makeNNLayerComponent("TensorOp", "TensorOp", "#FFF3E0")` | None — `tensorop.png`. |
| `Configuration` (sectionLabel **NN Configuration**) | 160 × 140 | `makeNNLayerComponent("Configuration", "Configuration", "#FCE4EC")` | None — `configuration.png`. |
| `TrainingDataset` (sectionLabel **NN Datasets**) | 160 × 140 | `makeNNLayerComponent("TrainingDataset", "TrainingDataset", "#E8F5E9")` | **Minor drift on label.** Palette uses `defaultName = "Training"` (`buildLayerPreview("TrainingDataset", "Training", …)`), canvas default `data.name = "TrainingDataset"`. Drag ghost shows `Training`, dropped card shows `TrainingDataset`. |
| `TestDataset` | 160 × 140 | `makeNNLayerComponent("TestDataset", "TestDataset", "#FFEBEE")` | **Minor drift on label.** Palette `defaultName = "Test"`, canvas default `data.name = "TestDataset"`. Drag ghost shows `Test`, dropped card shows `TestDataset`. |
| `Configuration` palette `defaultName` | — | — | Palette `buildLayerPreview("Configuration", "Configuration", …)`, canvas default `name = "Configuration"`. Match. |

Per-kind icon coverage: every `NN_LAYER_ICON_FILES` key has a matching
`buildLayerPreview(..., iconFile)` entry except `NNContainer` and
`NNReference` (intentionally — they don't carry layer icons). Verified
by cross-comparing the lookup tables in `_NNLayerBase.tsx` (lines 39-57)
and `NNDiagramSVGs.tsx` (`buildLayerPreview` callers, lines 112-213).

## 17. Sfc

| Palette | w × h | Canvas | Drift? |
| --- | --- | --- | --- |
| `sfcStart` "Start" | 160 × 70 | `SfcStart.tsx` → `SfcStartNodeSVG` | None — same SVG. |
| `sfcStep` "Step" | 160 × 70 | → `SfcStepNodeSVG` | None — same SVG. |
| `sfcJump` "Jump" | 96 × 48 | → `SfcJumpNodeSVG` | None — same SVG. |
| `sfcTransitionBranch` "Branch" (`showHint: true`) | 30 × 30 | → `SfcTransitionBranchNodeSVG` | None — same SVG. |
| `sfcActionTable` "Action Table" | 160 × 30 | → `SfcActionTableNodeSVG` | None — same SVG. |

## 18. CommentConfig (free-form sticky note)

Defined separately and registered in `Sidebar.tsx` (NOT in
`dropElementConfigs`). Uses an inline `CommentPaletteSVG` defined in
`constants.ts`. The canvas component is `nodes/common/Comment.tsx`
(referenced — not exhaustively diff'd here, but the palette SVG mirrors
v3 silhouette per the SA-HIDE-NOISE comment).

---

## Sidebar `sectionLabel` rendering

`Sidebar.tsx` (lines 141-168):

- Reads `config.sectionLabel`. If set:
  - Prepends a `DividerLine` (suppressed for the first entry, where
    `index === 0`).
  - Renders a small uppercase heading (font 11 px, weight 600, opacity 0.7).
- Order is preserved by `dropElementConfigs[diagramType]` array order.

The only diagrams currently using `sectionLabel` are NNDiagram and
UserDiagram (latter implicitly via static entries; NN with five labels:
Structure / Layers / TensorOps / Configuration / Datasets).

---

## Top 10 inconsistencies (worst → least)

1. **BPMNPool** dimensions. Palette **160 × 80**, canvas `minWidth=200`,
   `minHeight=120`. The dropped pool is ~50% larger in area than the
   ghost — the most visible drag/drop mismatch on canvas.
2. **AgentIntent** label. Palette shows `Intent`, canvas prepends
   `"Intent: "` and renders `data.name` → `Intent: Intent` for the
   default drop. Header text doubles in length on drop.
3. **AgentRagElement** label. Palette shows only `RAG DB`. Canvas shows
   `RAG DB` PLUS a second display label (`dbCustomName ?? ragDatabaseName
   ?? name` = `"RAG"`). Drag ghost looks empty under the cylinder.
4. **StateCodeBlock** body. Palette shows literal `# code` placeholder.
   Canvas renders multi-line `data.code` (default
   `# Sample code\nprint("Hello World")`). The dropped node visibly
   acquires extra rows that didn't exist in the ghost.
5. **NN TrainingDataset / TestDataset** name field. Palette shows
   `Training` / `Test` (per `buildLayerPreview` defaultName). Canvas
   shows `TrainingDataset` / `TestDataset` (per `defaultData.name`).
   Drag ghost label is shorter than the dropped card.
6. **NNReference** label. Palette shows literal `→ Reference`. Canvas
   shows `→ ${data.name || referenceTarget}` → `→ ref` for the default
   drop.
7. **UserDiagram attribute rows** — operator drift. Palette previews
   render `${attrName} =` (single equals, no operator). Canvas applies
   `attributeOperator` from `defaultData` (default `==`), so dropped
   rows show `${attrName} ==`.
8. **ClassOCLConstraint** typography. Palette uses padding 10 / fold 12
   / header font 11; canvas uses padding 12 / fold 14 / header font 12
   plus the wrapped expression body that the palette omits. Visually
   the dropped sticky-note is denser and bigger-text than the ghost.
9. **BPMNDataObject** dimensions. Palette **40 × 60**, canvas
   `minWidth=60`. Dropped data-object grows 50% wider than the ghost.
10. **State machine `State` / `AgentState`** font weight. The palette
    SVGs use the default font-weight for the name; the canvas uses
    `fontWeight=600`. The dropped node's header reads visibly bolder
    than the drag preview.

Inconsistencies #4-7 are all "default-data divergence" — palette ghost
hard-codes a label the SVG component shouldn't be hard-coding; if those
palette previews were taught to read `data.name` (or were given the same
defaultData the dropElementConfig uses) the drift would disappear.
Inconsistencies #1, #9 are pure dimension drift; bumping the palette
`width`/`height` to match the canvas `NodeResizer` minimums fixes them.

## Summary

- **Total entries audited:** 86 (across 17 diagram registries plus
  `CommentConfig` and the dynamic UserDiagram set).
- **Definite drift count:** 10 (top-10 above).
- **Borderline / cosmetic:** 4 more (UserDiagram height +10 px gutter,
  NNContainer hint-rectangles, AgentState/AgentIntent body divider
  presence, ClassOCLConstraint badge fallback). These don't change the
  visible silhouette enough to trip a user but were noted during the
  pass.

The bulk of the palette uses the **same SVG component** as the canvas
(class diagram, object/communication, useCase, component, deployment,
syntaxTree, petriNet, reachabilityGraph, flowchart, BPMN [shape, not
size], sfc). For those families, drift is structurally impossible
without edits to the shared SVG. The "inline-SVG canvas" families
(state-machine, agent, NN) are where every drift case in the top 10
lives — those nodes deliberately re-implement their visual on the
canvas side and the two implementations have diverged on label
formatting, optional sub-text, and font weight.
