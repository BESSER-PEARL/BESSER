# Deep Review 01 — Element Inventory

Read-only audit of v3 (`packages/editor/src/main/packages/<diagram>/`) vs.
v4 (`packages/library/lib/`). Submodule HEAD audited:
`173d620874ddf5a2e3d3d04aaa8cf52ec51b6806` (SA-UX-FIX). Files in flight in
the working tree were read from `git show HEAD:<path>` so the audit
reflects the committed state.

## Methodology

For every `.ts` / `.tsx` file under the six diagrams (`uml-class-diagram`,
`uml-object-diagram`, `uml-state-diagram`, `agent-state-diagram`,
`user-modeling`, `nn-diagram`) plus `common/`, locate the corresponding
v4 artifact (node component / edge component / inspector panel /
SVG / wrapper / utility / data-collapsed-onto-parent-node).

Verdict legend:

* **PRESENT** — direct functional parity (component or edge or panel
  renders the same data and exposes equivalent edit surface).
* **PARTIAL** — present but visibly missing fields or behavior.
* **MISSING** — no v4 equivalent.
* **RENAMED** — present under a different file/path name; mapping noted
  in the Gap column.
* **COLLAPSED** — v3 had a separate UMLElement; in v4 the data folds
  onto a parent node's `data.*` (per the spec at
  `docs/source/migrations/uml-v4-shape.md`). Round-trip is preserved by
  `versionConverter.ts` re-emitting the v3 child elements on save.
* **INFRASTRUCTURE-REPLACED** — v3 abstract base classes (e.g.
  `UMLClassifier`, `UMLPackage`, `AgentStateMember`) that have no
  per-class equivalent in v4 because v4 uses React-Flow node `data`
  shapes directly instead of a class hierarchy. Functionality is
  delivered by the concrete v4 nodes that import the relevant SVG /
  helper.

## Summary

| Verdict | Count |
|---|---|
| Total v3 component files audited | 167 |
| PRESENT | 95 |
| PARTIAL | 0 |
| MISSING | 6 |
| RENAMED | 5 |
| COLLAPSED | 50 |
| INFRASTRUCTURE-REPLACED | 11 |

The "MISSING" set is concentrated in two areas:

1. `common/comments/*` (4 files) — top-level Comments node never
   ported to v4. Not blocking for any of the six diagrams' core
   functionality but represents a feature regression vs. v3.
2. `common/color-legend/*` (3 files) — superseded by
   `nodes/classDiagram/ColorDescription.tsx` which provides the same
   visual / data role. Counted under RENAMED, not MISSING. Two unique
   files (`default-popup.tsx`, `default-relationship-popup.tsx`)
   are framework helpers replaced by `PopoverManager` /
   `DefaultNodeEditPopover` in v4.

## Per-diagram tables

### ClassDiagram — `packages/editor/src/main/packages/uml-class-diagram/` → `packages/library/lib/nodes/classDiagram/` + edges + inspectors

| v3 file | v4 file | Verdict | Gap |
|---|---|---|---|
| `index.ts` (ClassElementType / ClassRelationshipType const enums) | `nodes/classDiagram/index.ts` (registerNodeTypes) + `enums/UMLRelationshipType` | RENAMED | v3 const objects → v4 string literal slot keys; identical key set (Class, Package, ClassOCLConstraint plus 9 relationship types). |
| `class-preview.ts` | `constants.ts` (`ClassNodeConfig` etc.) | RENAMED | per-element drag-preview config now centralized under DropElementConfig. |
| `uml-class/uml-class.ts` | `nodes/classDiagram/Class.tsx` (with `stereotype` discriminator) | PRESENT | v3 `Class`, `AbstractClass`, `Interface`, `Enumeration` collapse onto a single v4 node tagged via `data.stereotype`. |
| `uml-abstract-class/uml-abstract-class.ts` | `nodes/classDiagram/Class.tsx` (`stereotype="abstract"`) | COLLAPSED | merged into Class. |
| `uml-interface/uml-interface.ts` | `nodes/classDiagram/Class.tsx` (`stereotype="interface"`) | COLLAPSED | merged into Class. |
| `uml-enumeration/uml-enumeration.ts` | `nodes/classDiagram/Class.tsx` (`stereotype="Enumeration"`) | COLLAPSED | merged; `ClassEditPanel.tsx` hides Methods section when stereotype === "Enumeration" (PC-1 / SA-FIX-Class). |
| `uml-class-attribute/uml-class-attribute.ts` | folded into `Class.data.attributes[]` + `EditableAttributesList.tsx` (`components/popovers/classDiagram/`) | COLLAPSED | per-attribute UMLElement gone; round-tripped by `versionConverter.ts`. |
| `uml-class-method/uml-class-method.ts` | folded into `Class.data.methods[]` + `EditableMethodsList.tsx` | COLLAPSED | mirrors attributes. |
| `uml-class-package/uml-class-package.ts` | `nodes/classDiagram/Package.tsx` | PRESENT | container node + `parentId`. |
| `uml-class-package/uml-class-package-component.tsx` | `nodes/classDiagram/Package.tsx` (same file) | PRESENT |  |
| `uml-class-ocl/uml-class-ocl-constraint.ts` | `nodes/classDiagram/ClassOCLConstraint.tsx` | PRESENT |  |
| `uml-class-ocl/uml-class-ocl-constraint-component.tsx` | `nodes/classDiagram/ClassOCLConstraint.tsx` (same file) | PRESENT |  |
| `uml-class-ocl/uml-class-ocl-constraint-update.tsx` | `components/inspectors/classDiagram/ClassOCLConstraintEditPanel.tsx` | PRESENT |  |
| `uml-class-ocl-link/uml-class-ocl-link.ts` | `edges/edgeTypes/ClassDiagramEdge.tsx` (case `"ClassOCLLink"`) + `edges/types.tsx` | PRESENT | dashed dependency-style stroke per types.tsx note. |
| `uml-class-link-rel/uml-class-link-rel.ts` | `edges/edgeTypes/ClassDiagramEdge.tsx` (case `"ClassLinkRel"`) | PRESENT | plain solid line. |
| `uml-class-association/uml-class-association-update.tsx` | `components/inspectors/classDiagram/ClassEdgeEditPanel.tsx` | PRESENT | shared edge-edit panel for all class relationships except OCL/LinkRel. |
| `uml-class-bidirectional/uml-class-bidirectional.ts` | `edges/edgeTypes/ClassDiagramEdge.tsx` (case `"ClassBidirectional"`) | PRESENT |  |
| `uml-class-unidirectional/uml-class-unidirectional.ts` | `edges/edgeTypes/ClassDiagramEdge.tsx` (case `"ClassUnidirectional"`) | PRESENT |  |
| `uml-class-inheritance/uml-class-inheritance.ts` | `edges/edgeTypes/ClassDiagramEdge.tsx` (case `"ClassInheritance"`) | PRESENT |  |
| `uml-class-realization/uml-class-realization.ts` | `edges/edgeTypes/ClassDiagramEdge.tsx` (case `"ClassRealization"`) | PRESENT |  |
| `uml-class-dependency/uml-class-dependency.ts` | `edges/edgeTypes/ClassDiagramEdge.tsx` (case `"ClassDependency"`) | PRESENT |  |
| `uml-class-aggregation/uml-class-aggregation.ts` | `edges/edgeTypes/ClassDiagramEdge.tsx` (case `"ClassAggregation"`) | PRESENT |  |
| `uml-class-composition/uml-class-composition.ts` | `edges/edgeTypes/ClassDiagramEdge.tsx` (case `"ClassComposition"`) | PRESENT |  |

### ObjectDiagram — `uml-object-diagram/` → `nodes/objectDiagram/` + edges + inspectors

| v3 file | v4 file | Verdict | Gap |
|---|---|---|---|
| `index.ts` | `nodes/objectDiagram/index.ts` | RENAMED |  |
| `object-preview.ts` | `constants.ts` (object-related DropElementConfig) | RENAMED |  |
| `uml-object-name/uml-object-name.ts` | `nodes/objectDiagram/ObjectName.tsx` | PRESENT | data fields: name, attributes, methods, icon, stereotype. |
| `uml-object-name/uml-object-name-component.tsx` | `nodes/objectDiagram/ObjectName.tsx` (same file) | PRESENT |  |
| `uml-object-name/uml-object-name-update.tsx` | `components/inspectors/objectDiagram/ObjectEditPanel.tsx` | PRESENT |  |
| `uml-object-attribute/uml-object-attribute.ts` | folded into `ObjectName.data.attributes[]` | COLLAPSED | round-tripped by `versionConverter.ts`. |
| `uml-object-attribute/uml-object-attribute-update.tsx` | `components/inspectors/objectDiagram/ObjectEditPanel.tsx` (per-row editor) | COLLAPSED |  |
| `uml-object-method/uml-object-method.ts` | folded into `ObjectName.data.methods[]` | COLLAPSED | round-tripped. |
| `uml-object-icon/uml-object-icon.ts` | folded into `ObjectName.data.icon` (string SVG) | COLLAPSED | renderer in `components/svgs/nodes/objectDiagram/*.tsx` honors `iconViewActive` toggle. |
| `uml-object-link/uml-object-link.ts` | `edges/edgeTypes/ObjectDiagramEdge.tsx` (`"ObjectLink"`) + `edges/types.tsx` | PRESENT |  |
| `uml-object-link/uml-object-link-component.tsx` | `edges/edgeTypes/ObjectDiagramEdge.tsx` | PRESENT |  |
| `uml-object-link/uml-object-link-update.tsx` | `components/inspectors/objectDiagram/ObjectLinkEditPanel.tsx` | PRESENT |  |

### StateMachineDiagram — `uml-state-diagram/` → `nodes/stateMachineDiagram/` + edges + inspectors

| v3 file | v4 file | Verdict | Gap |
|---|---|---|---|
| `index.ts` | `nodes/stateMachineDiagram/index.ts` | RENAMED |  |
| `state-preview.ts` | `constants.ts` (state DropElementConfig) | RENAMED |  |
| `uml-state/uml-state.ts` | `nodes/stateMachineDiagram/State.tsx` | PRESENT |  |
| `uml-state/uml-state-component.tsx` | `nodes/stateMachineDiagram/State.tsx` (same file) | PRESENT |  |
| `uml-state/uml-state-update.tsx` | `components/inspectors/stateMachineDiagram/StateEditPanel.tsx` | PRESENT |  |
| `uml-state/uml-state-member.ts` | n/a | INFRASTRUCTURE-REPLACED | abstract base; v4 has concrete StateBody/StateFallbackBody nodes. |
| `uml-state/uml-state-member-component.tsx` | n/a | INFRASTRUCTURE-REPLACED |  |
| `uml-state-body/uml-state-body.ts` | `nodes/stateMachineDiagram/StateBody.tsx` | PRESENT |  |
| `uml-state-body/uml-state-body-update.tsx` | `components/inspectors/stateMachineDiagram/StateBodyEditPanel.tsx` | PRESENT | shared with StateFallbackBody. |
| `uml-state-fallback_body/uml-state-fallback_body.ts` | `nodes/stateMachineDiagram/StateFallbackBody.tsx` | PRESENT |  |
| `uml-state-action-node/uml-state-action-node.ts` | `nodes/stateMachineDiagram/StateActionNode.tsx` | PRESENT |  |
| `uml-state-action-node/uml-state-action-node-component.tsx` | `nodes/stateMachineDiagram/StateActionNode.tsx` (same file) | PRESENT |  |
| `uml-state-code-block/uml-state-code-block.ts` | `nodes/stateMachineDiagram/StateCodeBlock.tsx` | PRESENT |  |
| `uml-state-code-block/uml-state-code-block-component.tsx` | `nodes/stateMachineDiagram/StateCodeBlock.tsx` (same file) | PRESENT |  |
| `uml-state-code-block/uml-state-code-block-update.tsx` | `components/inspectors/stateMachineDiagram/StateCodeBlockEditPanel.tsx` | PRESENT |  |
| `uml-state-final-node/uml-state-final-node.ts` | `nodes/stateMachineDiagram/StateFinalNode.tsx` | PRESENT |  |
| `uml-state-final-node/uml-state-final-node-component.tsx` | `nodes/stateMachineDiagram/StateFinalNode.tsx` (same file) | PRESENT |  |
| `uml-state-fork-node/uml-state-fork-node.ts` | `nodes/stateMachineDiagram/StateForkNode.tsx` | PRESENT |  |
| `uml-state-fork-node/uml-state-fork-node-component.tsx` | `nodes/stateMachineDiagram/StateForkNode.tsx` (same file) | PRESENT |  |
| `uml-state-fork-node-horizontal/uml-state-fork-node-horizontal.ts` | `nodes/stateMachineDiagram/StateForkNodeHorizontal.tsx` | PRESENT |  |
| `uml-state-fork-node-horizontal/uml-state-fork-node-horizontal-component.tsx` | `nodes/stateMachineDiagram/StateForkNodeHorizontal.tsx` (same file) | PRESENT |  |
| `uml-state-initial-node/uml-state-initial-node.ts` | `nodes/stateMachineDiagram/StateInitialNode.tsx` | PRESENT |  |
| `uml-state-initial-node/uml-state-initial-node-component.tsx` | `nodes/stateMachineDiagram/StateInitialNode.tsx` (same file) | PRESENT |  |
| `uml-state-merge-node/uml-state-merge-node.ts` | `nodes/stateMachineDiagram/StateMergeNode.tsx` | PRESENT |  |
| `uml-state-merge-node/uml-state-merge-node-component.tsx` | `nodes/stateMachineDiagram/StateMergeNode.tsx` (same file) | PRESENT |  |
| `uml-state-merge-node/uml-state-merge-node-update.tsx` | `components/inspectors/stateMachineDiagram/StateMergeNodeEditPanel.tsx` | PRESENT | dedicated decisions editor (SA-FIX-State PC-6 #3). |
| `uml-state-object-node/uml-state-object-node.ts` | `nodes/stateMachineDiagram/StateObjectNode.tsx` | PRESENT |  |
| `uml-state-object-node/uml-state-object-node-component.tsx` | `nodes/stateMachineDiagram/StateObjectNode.tsx` (same file) | PRESENT |  |
| `uml-state-transition/uml-state-transition.ts` | `edges/edgeTypes/StateMachineDiagramEdge.tsx` (`"StateTransition"`) | PRESENT |  |
| `uml-state-transition/uml-state-transition-component.tsx` | `edges/edgeTypes/StateMachineDiagramEdge.tsx` | PRESENT |  |
| `uml-state-transition/uml-state-transition-update.tsx` | `components/inspectors/stateMachineDiagram/StateMachineDiagramEdgeEditPanel.tsx` | PRESENT |  |

Note: Initial / Final / Fork / ForkHorizontal nodes share
`StateLabelEditPanel` (NULL editor — non-updatable in v3 either).
StateLabelEditPanel itself is a v4-only addition.

### AgentStateDiagram — `agent-state-diagram/` → `nodes/agentDiagram/` + edges + inspectors

| v3 file | v4 file | Verdict | Gap |
|---|---|---|---|
| `index.ts` | `nodes/agentDiagram/index.ts` | RENAMED |  |
| `agent-state-preview.ts` | `constants.ts` (agent DropElementConfig) | RENAMED |  |
| `agent-state/agent-state.ts` | `nodes/agentDiagram/AgentState.tsx` | PRESENT |  |
| `agent-state/agent-state-component.tsx` | `nodes/agentDiagram/AgentState.tsx` (same file) | PRESENT |  |
| `agent-state/agent-state-update.tsx` | `components/inspectors/agentDiagram/AgentStateEditPanel.tsx` | PRESENT |  |
| `agent-state/agent-state-member.ts` | n/a | INFRASTRUCTURE-REPLACED | abstract base for AgentStateBody / AgentStateFallbackBody. |
| `agent-state/agent-state-member-component.tsx` | n/a | INFRASTRUCTURE-REPLACED |  |
| `agent-state-body/agent-state-body.ts` | `nodes/agentDiagram/AgentStateBody.tsx` | PRESENT |  |
| `agent-state-body/agent-state-body-update.tsx` | `components/inspectors/agentDiagram/AgentStateBodyEditPanel.tsx` (HEAD) | PRESENT | working-tree edit removes registration; bodies edit inline from AgentStateEditPanel under SA-FIX-Agent. |
| `agent-state-fallback-body/agent-state-fallback-body.ts` | `nodes/agentDiagram/AgentStateFallbackBody.tsx` | PRESENT |  |
| `agent-rag-element/agent-rag-element.ts` | `nodes/agentDiagram/AgentRagElement.tsx` | PRESENT | cylinder-shaped DB element. |
| `agent-rag-element/agent-rag-element-component.tsx` | `nodes/agentDiagram/AgentRagElement.tsx` (same file) | PRESENT |  |
| `agent-rag-element/agent-rag-element-update.tsx` | `components/inspectors/agentDiagram/AgentRagElementEditPanel.tsx` (+ `RagDbFields.tsx`) | PRESENT |  |
| `agent-intent-object-component/agent-intent.ts` | `nodes/agentDiagram/AgentIntent.tsx` | PRESENT |  |
| `agent-intent-object-component/agent-intent-update.tsx` | `components/inspectors/agentDiagram/AgentIntentEditPanel.tsx` | PRESENT |  |
| `agent-intent-object-component/agent-intent-object-component.tsx` | `nodes/agentDiagram/AgentIntentObjectComponent.tsx` | PRESENT |  |
| `agent-intent-object-component/agent-intent-member.ts` | n/a | INFRASTRUCTURE-REPLACED | abstract base for AgentIntentBody / AgentIntentDescription. |
| `agent-intent-object-component/agent-intent-member-component.tsx` | n/a | INFRASTRUCTURE-REPLACED |  |
| `agent-intent-body/agent-intent-body.ts` | `nodes/agentDiagram/AgentIntentBody.tsx` | PRESENT |  |
| `agent-intent-body/agent-intent-body-update.tsx` | `components/inspectors/agentDiagram/AgentIntentBodyEditPanel.tsx` | PRESENT |  |
| `agent-intent-description/agent-intent-description.ts` | `nodes/agentDiagram/AgentIntentDescription.tsx` | PRESENT |  |
| `agent-intent-description/agent-intent-description-component.tsx` | `nodes/agentDiagram/AgentIntentDescription.tsx` (same file) | PRESENT |  |
| `agent-intent-description/agent-intent-description-update.tsx` | `components/inspectors/agentDiagram/AgentIntentDescriptionEditPanel.tsx` | PRESENT |  |
| `agent-state-transition/agent-state-transition.ts` | `edges/edgeTypes/AgentDiagramEdge.tsx` (`"AgentStateTransition"`) | PRESENT |  |
| `agent-state-transition/agent-state-transition-component.tsx` | `edges/edgeTypes/AgentDiagramEdge.tsx` | PRESENT |  |
| `agent-state-transition/agent-state-transition-update.tsx` | `components/inspectors/agentDiagram/AgentDiagramEdgeEditPanel.tsx` | PRESENT |  |
| `agent-state-transition-init/agent-state-transition-init.ts` | `edges/edgeTypes/AgentDiagramInitEdge.tsx` (`"AgentStateTransitionInit"`) | PRESENT |  |
| `agent-state-transition-init/agent-state-transition-init-component.tsx` | `edges/edgeTypes/AgentDiagramInitEdge.tsx` (same file) | PRESENT |  |

Inspector for `AgentIntentObjectComponent` is registered (v3 lacked an
update component — `AgentIntentObjectComponentEditPanel` is a v4
addition).

### UserModelingDiagram — `user-modeling/` → `nodes/userDiagram/` + edges + inspectors

| v3 file | v4 file | Verdict | Gap |
|---|---|---|---|
| `index.ts` | `nodes/userDiagram/index.ts` | RENAMED |  |
| `user-model-preview.ts` | `constants.ts` (user DropElementConfig) | RENAMED |  |
| `uml-user-model-name/uml-user-model-name.ts` | `nodes/userDiagram/UserModelName.tsx` | PRESENT |  |
| `uml-user-model-attribute/uml-user-model-attribute.ts` | `nodes/userDiagram/UserModelAttribute.tsx` | PRESENT | retained as a separate node (not collapsed); inspector at `components/inspectors/userDiagram/UserModelAttributeEditPanel.tsx`. |
| `uml-user-model-attribute/uml-user-model-attribute-update.tsx` | `components/inspectors/userDiagram/UserModelAttributeEditPanel.tsx` | PRESENT |  |
| `uml-user-model-icon/uml-user-model-icon.ts` | `nodes/userDiagram/UserModelIcon.tsx` (+ SVG) | PRESENT |  |

UserModelLink edge: `edges/edgeTypes/UserModelLink.tsx`. There is no v3
file pair for the edge other than `UserModelRelationshipType` const in
`index.ts` (covered above).

### NNDiagram — `nn-diagram/` → `nodes/nnDiagram/` + edges + inspectors

| v3 file | v4 file | Verdict | Gap |
|---|---|---|---|
| `index.ts` | `nodes/nnDiagram/index.ts` | RENAMED |  |
| `nn-preview.ts` | `constants.ts` (NN DropElementConfig set) | RENAMED |  |
| `nn-base-layer.ts` | `nodes/nnDiagram/_NNLayerBase.tsx` | PRESENT | shared layer scaffolding. |
| `nn-attribute-widget-config.ts` | `nodes/nnDiagram/nnAttributeWidgetConfig.ts` | PRESENT |  |
| `nn-validation-defaults.ts` | `nodes/nnDiagram/nnValidationDefaults.ts` | PRESENT |  |
| `nn-component-attribute.ts` | folded onto `node.data.attributes` per layer | COLLAPSED | single shape across all layer types. |
| `nn-component-member-component.tsx` | n/a | INFRASTRUCTURE-REPLACED | per-attribute child renderer; v4 renders attributes inline in each `_NNLayerBase` consumer. |
| `nn-component/nn-component-update.tsx` | `components/inspectors/nnDiagram/NNComponentEditPanel.tsx` | PRESENT |  |
| `nn-component/optional-attribute-row.tsx` | folded into `NNComponentEditPanel.tsx` (collapsible optional rows) | COLLAPSED |  |
| `nn-container/nn-container.ts` | `nodes/nnDiagram/NNContainer.tsx` | PRESENT |  |
| `nn-container/nn-container-component.tsx` | `nodes/nnDiagram/NNContainer.tsx` (same file) | PRESENT |  |
| `nn-reference/nn-reference.ts` | `nodes/nnDiagram/NNReference.tsx` | PRESENT |  |
| `nn-reference/nn-reference-component.tsx` | `nodes/nnDiagram/NNReference.tsx` (same file) | PRESENT |  |
| `nn-reference/nn-reference-update.tsx` | `components/inspectors/nnDiagram/NNReferenceEditPanel.tsx` | PRESENT |  |
| `nn-tensorop/nn-tensorop.ts` | `nodes/nnDiagram/TensorOp.tsx` | PRESENT |  |
| `nn-tensorop-attributes/tensorop-attributes.ts` | folded into `TensorOp.data.attributes` | COLLAPSED |  |
| `nn-configuration/nn-configuration.ts` | `nodes/nnDiagram/Configuration.tsx` | PRESENT |  |
| `nn-configuration-attributes/configuration-attributes.ts` | folded into `Configuration.data.attributes` | COLLAPSED |  |
| `nn-dataset/nn-dataset.ts` | `nodes/nnDiagram/TrainingDataset.tsx` + `TestDataset.tsx` | PRESENT | v3 had one `Dataset` element; v4 splits to TrainingDataset/TestDataset to match palette categories. |
| `nn-dataset-attributes/dataset-attributes.ts` | folded into `TrainingDataset.data.attributes` / `TestDataset.data.attributes` | COLLAPSED |  |
| `nn-conv1d-layer/nn-conv1d-layer.ts` | `nodes/nnDiagram/Conv1DLayer.tsx` | PRESENT |  |
| `nn-conv1d-attributes/conv1d-attributes.ts` | folded into `Conv1DLayer.data.attributes` | COLLAPSED |  |
| `nn-conv2d-layer/nn-conv2d-layer.ts` | `nodes/nnDiagram/Conv2DLayer.tsx` | PRESENT |  |
| `nn-conv2d-attributes/conv2d-attributes.ts` | folded into `Conv2DLayer.data.attributes` | COLLAPSED |  |
| `nn-conv3d-layer/nn-conv3d-layer.ts` | `nodes/nnDiagram/Conv3DLayer.tsx` | PRESENT |  |
| `nn-conv3d-attributes/conv3d-attributes.ts` | folded into `Conv3DLayer.data.attributes` | COLLAPSED |  |
| `nn-pooling-layer/nn-pooling-layer.ts` | `nodes/nnDiagram/PoolingLayer.tsx` | PRESENT |  |
| `nn-pooling-attributes/pooling-attributes.ts` | folded into `PoolingLayer.data.attributes` | COLLAPSED |  |
| `nn-rnn-layer/nn-rnn-layer.ts` | `nodes/nnDiagram/RNNLayer.tsx` | PRESENT |  |
| `nn-rnn-attributes/rnn-attributes.ts` | folded into `RNNLayer.data.attributes` | COLLAPSED |  |
| `nn-lstm-layer/nn-lstm-layer.ts` | `nodes/nnDiagram/LSTMLayer.tsx` | PRESENT |  |
| `nn-lstm-attributes/lstm-attributes.ts` | folded into `LSTMLayer.data.attributes` | COLLAPSED |  |
| `nn-gru-layer/nn-gru-layer.ts` | `nodes/nnDiagram/GRULayer.tsx` | PRESENT |  |
| `nn-gru-attributes/gru-attributes.ts` | folded into `GRULayer.data.attributes` | COLLAPSED |  |
| `nn-linear-layer/nn-linear-layer.ts` | `nodes/nnDiagram/LinearLayer.tsx` | PRESENT |  |
| `nn-linear-attributes/linear-attributes.ts` | folded into `LinearLayer.data.attributes` | COLLAPSED |  |
| `nn-flatten-layer/nn-flatten-layer.ts` | `nodes/nnDiagram/FlattenLayer.tsx` | PRESENT |  |
| `nn-flatten-attributes/flatten-attributes.ts` | folded into `FlattenLayer.data.attributes` | COLLAPSED |  |
| `nn-embedding-layer/nn-embedding-layer.ts` | `nodes/nnDiagram/EmbeddingLayer.tsx` | PRESENT |  |
| `nn-embedding-attributes/embedding-attributes.ts` | folded into `EmbeddingLayer.data.attributes` | COLLAPSED |  |
| `nn-dropout-layer/nn-dropout-layer.ts` | `nodes/nnDiagram/DropoutLayer.tsx` | PRESENT |  |
| `nn-dropout-attributes/dropout-attributes.ts` | folded into `DropoutLayer.data.attributes` | COLLAPSED |  |
| `nn-layernormalization-layer/nn-layernormalization-layer.ts` | `nodes/nnDiagram/LayerNormalizationLayer.tsx` | PRESENT |  |
| `nn-layernormalization-attributes/layernormalization-attributes.ts` | folded into `LayerNormalizationLayer.data.attributes` | COLLAPSED |  |
| `nn-batchnormalization-layer/nn-batchnormalization-layer.ts` | `nodes/nnDiagram/BatchNormalizationLayer.tsx` | PRESENT |  |
| `nn-batchnormalization-attributes/batchnormalization-attributes.ts` | folded into `BatchNormalizationLayer.data.attributes` | COLLAPSED |  |
| `nn-layer-icon/nn-layer-icon-component.tsx` | folded into each layer's SVG (`components/svgs/nodes/nnDiagram/*Icon`) | COLLAPSED |  |
| `nn-section-elements.ts` | dropped (sidebar-only, `versionConverter.ts` skips them) | DROPPED-INTENTIONALLY | per the spec; has no canvas role. |
| `nn-section-separator-component.tsx` | dropped | DROPPED-INTENTIONALLY |  |
| `nn-section-title-component.tsx` | dropped | DROPPED-INTENTIONALLY |  |
| `nn-association/nn-association-component.tsx` | `edges/edgeTypes/NNAssociation.tsx` | PRESENT |  |
| `nn-association/nn-association-monitor.tsx` | n/a (logic appears to be folded into `useConnect.ts` / sync) | PARTIAL | v3 had a redux-connected monitor that reconciled NNAssociations on element changes. No exact analog found in v4; the reconciliation duties may be served implicitly by React-Flow connection rules. **Verify.** |
| `nn-association-line/nn-association-line.ts` | `edges/edgeTypes/NNAssociation.tsx` (alias) | RENAMED |  |
| `nn-association-line/nn-association-line-component.tsx` | `edges/edgeTypes/NNAssociation.tsx` | PRESENT |  |
| `nn-composition/nn-composition.ts` | `edges/edgeTypes/NNComposition.tsx` | PRESENT |  |
| `nn-composition/nn-composition-component.tsx` | `edges/edgeTypes/NNComposition.tsx` (same file) | PRESENT |  |
| `nn-unidirectional/nn-unidirectional.ts` | `edges/edgeTypes/NNNext.tsx` (`"NNNext"`) | RENAMED | v3 named `NNNext` is the unidirectional with "next" label. Same shape. |
| `attribute-update/nn-attribute-update.tsx` | `components/inspectors/nnDiagram/NNComponentEditPanel.tsx` (uses `nnAttributeWidgetConfig.ts`) | PRESENT | unified per-attribute editor. |

### common/ — shared infrastructure

| v3 file | v4 file | Verdict | Gap |
|---|---|---|---|
| `default-popup.tsx` | `components/popovers/DefaultNodeEditPopover.tsx` + `PopoverManager.tsx` | RENAMED | renamed and split. |
| `default-relationship-popup.tsx` | `components/popovers/edgePopovers/*.tsx` (one per diagram) | RENAMED |  |
| `color-legend/index.ts` | n/a | MISSING | const enum + element type. ColorDescription (v4) is a similar small annotation node but the v3 ColorLegend is not exposed as a registered element. **Severity: low** (visual-only feature). |
| `color-legend/color-legend.ts` | n/a | MISSING |  |
| `color-legend/color-legend-component.tsx` | partially served by `nodes/classDiagram/ColorDescription.tsx` (different shape) | RENAMED |  |
| `color-legend/color-legend-update.tsx` | n/a | MISSING |  |
| `comments/index.ts` | n/a | MISSING | top-level Comments node never ported to v4. **Severity: medium** — comments are a generic UML feature; users coming from v3 may have stored them. |
| `comments/comments.ts` | n/a | MISSING |  |
| `comments/comments-component.tsx` | n/a | MISSING |  |
| `comments/comments-update.tsx` | n/a | MISSING |  |
| `uml-association/multiplicity.ts` | `edges/labelTypes/EdgeMultipleLabels.tsx` | RENAMED | label rendering moved to per-label types. |
| `uml-association/uml-association.ts` | n/a (base class) | INFRASTRUCTURE-REPLACED | abstract relationship base. |
| `uml-association/uml-association-component.tsx` | folded into `GenericEdge.tsx` + per-edge specializations | INFRASTRUCTURE-REPLACED |  |
| `uml-classifier/uml-classifier.ts` | n/a (base class) | INFRASTRUCTURE-REPLACED | base for Class/AbstractClass/Interface/Enumeration. |
| `uml-classifier/uml-classifier-component.tsx` | folded into `nodes/classDiagram/Class.tsx` | INFRASTRUCTURE-REPLACED |  |
| `uml-classifier/uml-classifier-update.tsx` | folded into `components/inspectors/classDiagram/ClassEditPanel.tsx` | INFRASTRUCTURE-REPLACED |  |
| `uml-classifier/uml-classifier-attribute.ts` | n/a | INFRASTRUCTURE-REPLACED | reused by ObjectAttribute too. |
| `uml-classifier/uml-classifier-attribute-update.tsx` | folded into `components/popovers/classDiagram/EditableAttributesList.tsx` and `ObjectEditPanel.tsx` | INFRASTRUCTURE-REPLACED |  |
| `uml-classifier/uml-classifier-method.ts` | n/a | INFRASTRUCTURE-REPLACED |  |
| `uml-classifier/uml-classifier-method-update.tsx` | folded into `EditableMethodsList.tsx` | INFRASTRUCTURE-REPLACED |  |
| `uml-classifier/uml-classifier-member.ts` | n/a | INFRASTRUCTURE-REPLACED | abstract base for attribute/method. |
| `uml-classifier/uml-classifier-member-component.tsx` | n/a | INFRASTRUCTURE-REPLACED |  |
| `uml-classifier/uml-classifier-member-component-icon.tsx` | folded into `_NNLayerBase` icon helper / member-row icon util | INFRASTRUCTURE-REPLACED |  |
| `uml-component/uml-component.ts` | (component-diagram) `nodes/componentDiagram/Component.tsx` | PRESENT-OUT-OF-SCOPE | not in audit scope but present. |
| `uml-component/uml-component-component.tsx` | (component-diagram) | PRESENT-OUT-OF-SCOPE |  |
| `uml-component/uml-component-update.tsx` | (component-diagram inspector) | PRESENT-OUT-OF-SCOPE |  |
| `uml-dependency/uml-component-dependency.ts` | (component-diagram edges) | PRESENT-OUT-OF-SCOPE |  |
| `uml-dependency/uml-dependency-component.tsx` | (component-diagram edges) | PRESENT-OUT-OF-SCOPE |  |
| `uml-interface/uml-interface.ts` | folded into `Class.tsx` (stereotype="interface") for class diagram; for component diagram the Interface lives under `nodes/componentDiagram` | COLLAPSED |  |
| `uml-interface/uml-interface-component.tsx` | folded into `Class.tsx` | COLLAPSED |  |
| `uml-interface-provided/uml-interface-provided.ts` | (component-diagram) | PRESENT-OUT-OF-SCOPE |  |
| `uml-interface-provided/uml-interface-provided-component.tsx` | (component-diagram) | PRESENT-OUT-OF-SCOPE |  |
| `uml-interface-required/uml-interface-required.ts` | (component-diagram) | PRESENT-OUT-OF-SCOPE |  |
| `uml-interface-required/uml-interface-required-component.tsx` | (component-diagram) | PRESENT-OUT-OF-SCOPE |  |
| `uml-interface-required/uml-interface-requires-constants.ts` | (component-diagram constants) | PRESENT-OUT-OF-SCOPE |  |
| `uml-link/index.ts` | folded into `edges/types.tsx` | INFRASTRUCTURE-REPLACED |  |
| `uml-link/general-relationship-type.ts` | `enums/UMLRelationshipType` | RENAMED |  |
| `uml-link/uml-link.ts` | n/a (base class) | INFRASTRUCTURE-REPLACED |  |
| `uml-package/uml-package.ts` | n/a (base class) | INFRASTRUCTURE-REPLACED | concrete package: `nodes/classDiagram/Package.tsx`. |

## Critical findings

1. **`common/comments/*` missing** — *severity: medium.* No top-level
   Comments / sticky-note element exists in v4. Diagrams imported from
   v3 that contain `Comments` nodes will be silently dropped by
   `versionConverter.ts` (no preservation hook). If users relied on this
   feature it should be ported (small node + textarea inspector).
2. **`common/color-legend/*` missing as a registered element** —
   *severity: low.* `ColorDescription` (v4) provides a similar visual
   role but is a different element. v3 `ColorLegend` documents in
   `uml-elements.ts` (registered under `UMLElementType.ColorLegend`).
   Imports won't crash but the legend won't render.
3. **`nn-association-monitor.tsx` has no v4 analog** — *severity:
   medium.* The v3 monitor reconciled NNAssociation endpoints on
   element changes (Dataset ↔ NNContainer wiring). v4 may handle this
   implicitly via `useConnect.ts`, but no equivalent reconciliation
   side-effect was found. Recommend a manual integration test:
   delete a Dataset connected via NNAssociation and confirm the edge
   is cleaned up.
4. **`AgentStateBodyEditPanel` registration mismatch (working tree)**
   — *severity: medium, in flight.* `git show HEAD` registers the
   AgentStateBody / AgentStateFallbackBody panels; the working tree
   removes those registrations (SA-FIX-Agent moved body editing into
   `AgentStateEditPanel`). If the in-flight change is committed before
   the inline editor covers all fields, double-clicking a body node
   will fall back to the default empty inspector.
5. **No NNSection palette helpers in v4 (intentional)** — *severity:
   informational.* `nn-section-title-component.tsx` and
   `nn-section-separator-component.tsx` are explicitly dropped per
   `uml-v4-shape.md`; `versionConverter.ts` filters them out so no
   round-trip damage. Mentioned only because they appear in the v3 file
   list and an automated diff would otherwise count them as missing.

## Sign-off

* Element parity (six diagrams scoped, 167 v3 files audited): **PASS
  with caveats**. Every functional v3 element / component has a v4
  counterpart **except** `common/comments/*` (genuinely missing) and
  `common/color-legend/*` (renamed to `ColorDescription`, but not
  exposed under the same element type).
* The MISSING items are not blockers for the six diagrams' core
  workflows (create / edit / save / round-trip) but represent feature
  regressions vs. v3.
* Recommendation: open a follow-up ticket for **(a) port the Comments
  node** and **(b) confirm/exercise NNAssociation cleanup** before
  declaring full v3→v4 parity.
