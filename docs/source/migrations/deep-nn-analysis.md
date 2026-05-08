# Deep Analysis — NNDiagram v3 vs v4 (SA-DEEP-NN)

Read-only, evidence-based audit of the BESSER NNDiagram migration. Compares the
legacy v3 implementation under
`besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/nn-diagram/`
against the SA-5 / SA-2.2 / SA-FIX-NN / SA-FIX-NN-DROPS / SA-UX-FIX-2 v4
implementation under
`besser/utilities/web_modeling_editor/frontend/packages/library/lib/`. Each row
cites a load-bearing line range. Status: **PASS** = present and matches the
brief, **GAP** = regressed or differs, **N/A** = not applicable to v4.

Scope of inputs:

- v3 (~50 files): `nn-base-layer.ts`, `nn-attribute-widget-config.ts`,
  `nn-validation-defaults.ts`, every per-layer dir, `nn-component-attribute.ts`,
  `nn-association`, `nn-association-line`, `nn-composition`, `nn-container`,
  `nn-reference`, `nn-section-title-component.tsx`,
  `nn-section-separator-component.tsx`, `nn-layer-icon`, etc.
- v4: `lib/nodes/nnDiagram/` (19 .tsx + `_NNLayerBase.tsx` +
  `nnAttributeWidgetConfig.ts` + `nnValidationDefaults.ts`),
  `lib/components/svgs/nodes/nnDiagram/NNDiagramSVGs.tsx`,
  `lib/components/inspectors/nnDiagram/{NNComponentEditPanel,
  NNContainerEditPanel, NNReferenceEditPanel}.tsx`,
  `lib/edges/edgeTypes/{NNNext,NNComposition,NNAssociation}.tsx`,
  `lib/utils/versionConverter.ts` (NN section), `lib/constants.ts` NN palette
  block, `lib/utils/nodeUtils.ts::isParentNodeType`,
  `lib/utils/bpmnConstraints.ts::canDropIntoParent`.
- Backend: `besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/nn_diagram_processor.py`.

---

## A. Palette (Sidebar)

Source: `lib/constants.ts:1028-1184` (`UMLDiagramType.NNDiagram`).
Section labels mirror v3 `nn-preview.ts:333,336,339,342` (Structure / Layers /
TensorOps / Configuration / Datasets) plus the omitted v3 `Comment` section
(handled cross-cutting elsewhere). Section dividers in v4 are emitted by
`Sidebar.tsx` whenever the palette entry's `sectionLabel` field is set
(`constants.ts:370-376`, `:1041,1059,1159,1167,1175`).

| Palette entry | v3 element | v4 default `data` shape | Status |
|---|---|---|---|
| NNContainer (Structure §) | `NNContainer` | `{ name: "MyModel" }` | **PASS** (`constants.ts:1036-1042`) |
| NNReference (Structure §) | `NNReference` | `{ name: "ref" }` | **PASS** (`:1043-1049`) |
| Conv1DLayer (Layers §) | `Conv1DLayer` | `{ name: "Conv1D", attributes: {} }` | **PASS** (`:1053-1060`) |
| Conv2DLayer | `Conv2DLayer` | `{ name: "Conv2D", attributes: {} }` | **PASS** (`:1061-1067`) |
| Conv3DLayer | `Conv3DLayer` | `{ name: "Conv3D", attributes: {} }` | **PASS** (`:1068-1074`) |
| PoolingLayer | `PoolingLayer` | `{ name: "Pooling", attributes: { "pooling.dimension": "2D" } }` (qualified slug) | **PASS** (`:1075-1085`) |
| RNNLayer | `RNNLayer` | `{ name: "RNN", attributes: {} }` | **PASS** (`:1086-1092`) |
| LSTMLayer | `LSTMLayer` | `{ name: "LSTM", attributes: {} }` | **PASS** (`:1093-1099`) |
| GRULayer | `GRULayer` | `{ name: "GRU", attributes: {} }` | **PASS** (`:1100-1106`) |
| LinearLayer | `LinearLayer` | `{ name: "Linear", attributes: {} }` | **PASS** (`:1107-1113`) |
| FlattenLayer | `FlattenLayer` | `{ name: "Flatten", attributes: {} }` | **PASS** (`:1114-1120`) |
| EmbeddingLayer | `EmbeddingLayer` | `{ name: "Embedding", attributes: {} }` | **PASS** (`:1121-1127`) |
| DropoutLayer | `DropoutLayer` | `{ name: "Dropout", attributes: {} }` | **PASS** (`:1128-1134`) |
| LayerNormalizationLayer | `LayerNormalizationLayer` | `{ name: "LayerNorm", attributes: {} }` | **PASS** (`:1135-1141`) |
| BatchNormalizationLayer | `BatchNormalizationLayer` | `{ name: "BatchNorm", attributes: { "batch_normalization.dimension": "1D" } }` (qualified slug) | **PASS** (`:1142-1152`) |
| TensorOp (TensorOps §) | `TensorOp` | `{ name: "TensorOp", attributes: {} }` | **PASS** (`:1153-1160`) |
| Configuration (Configuration §) | `Configuration` | `{ name: "Configuration", attributes: {} }` | **PASS** (`:1161-1168`) |
| TrainingDataset (Datasets §) | `TrainingDataset` | `{ name: "TrainingDataset", attributes: {} }` | **PASS** (`:1169-1176`) |
| TestDataset | `TestDataset` | `{ name: "TestDataset", attributes: {} }` | **PASS** (`:1177-1183`) |

Default `attributes` is a **dict** (`{slug: value}`) on every layer entry, NOT
an array — diverges from ClassDiagram's array form. This dict shape is what
crashed the universal id-rewriter in `DraggableGhost.tsx` until commit
`a31c4ec`. Section separators (NN Structure / NN Layers / NN TensorOps / NN
Configuration / NN Datasets) are present and ordered identically to v3.

---

## B. Canvas rendering per layer kind

Source: `lib/nodes/nnDiagram/_NNLayerBase.tsx:39-57` (icon table) and
`:111-249` (renderer). Each of the 17 `*.tsx` layer files is a one-line
delegator to `makeNNLayerComponent(nodeType, kindLabel)`.

`NN_LAYER_ICON_FILES` table at `_NNLayerBase.tsx:39-57` maps every layer kind
to its PNG asset under `packages/webapp/assets/images/nn-layers/` — verified
all 17 PNGs exist on disk:
`batchnorm.png, configuration.png, conv1d.png, conv2d.png, conv3d.png,
dropout.png, embedding.png, flatten.png, gru.png, layernorm.png, linear.png,
lstm.png, pooling.png, rnn.png, tensorop.png, test_data.png, train_data.png`.

| Aspect | Brief | v4 implementation | Status |
|---|---|---|---|
| PNG icon present per layer | 17 layer kinds, 17 PNGs | `_NNLayerBase.tsx:230-239` `<image href="/images/nn-layers/{kind}.png">` | **PASS** |
| Stereotype-style card (SA-2.2 #34) | Replaced v3 110×110 layer card | `_NNLayerBase.tsx:201-209` `«kindLabel»` header | **PASS** |
| Card height accommodates icon | SA-UX-FIX-2 (B4) bumped 60→140 | `constants.ts:1050-1052` comment + height: 140 on every layer entry | **PASS** |
| NNContainer custom renderer (no icon) | Container has no PNG | `NNContainer.tsx:24-98` plain rounded rect + header line | **PASS** |
| NNReference custom renderer (no icon) | Reference has no PNG | `NNReference.tsx:19-80` dashed border, italic `→ name` | **PASS** |

The v3 110×110 layer card visual was retired per SA-2.2 #34 in favour of the
shared stereotype card; the PNG icon was reinstated under SA-UX-FIX-2 (B4)
when user testing reported the icon loss.

---

## C. Inspector panel

Source: `lib/components/inspectors/nnDiagram/NNComponentEditPanel.tsx:137-399`.
Single generic component drives all 17 layer-style kinds via
`getLayerSchema(layerKind)` (`nnAttributeWidgetConfig.ts:766-770`).
`NNContainer` / `NNReference` use dedicated panels
(`inspectors/nnDiagram/index.ts:46-47`).

| Behaviour | Brief | v4 location | Status |
|---|---|---|---|
| Single panel reads schema by kind | `getLayerSchema(layerKind)` | `NNComponentEditPanel.tsx:151` | **PASS** |
| Per-kind ordered field schema | `LAYER_ATTRIBUTE_SCHEMA` | `nnAttributeWidgetConfig.ts:475-495` 17 entries | **PASS** |
| Conv (1D/2D/3D) shared schema (12 fields) | name + kernel_dim + out_channels + 9 optionals | `:197-233` `CONV_FIELDS` | **PASS** |
| Pooling schema (13 fields) | adds `pooling_type`, `dimension`, `output_dim` | `:235-282` `POOLING_FIELDS` | **PASS** |
| Recurrent schema (10) | name + hidden_size + 8 optionals | `:284-313` `RECURRENT_FIELDS` | **PASS** |
| Linear (6) / Flatten (6) / Embedding (6) / Dropout (4) | per-layer | `:315-362` | **PASS** |
| LayerNorm (5) / BatchNorm (6) | with `dimension` collision-aware on BN | `:364-398` | **PASS** |
| TensorOp (8) | `tns_type` + 6 conditional optionals | `:400-420` | **PASS** |
| Configuration (8) | 6 mandatory + 2 optional | `:422-441` | **PASS** |
| Dataset (6) | name + path_data + 4 optionals | `:443-468` | **PASS** |
| TensorOp conditional filter by `tns_type` | reshape/concatenate/multiply/matmultiply/transpose/permute | `NNComponentEditPanel.tsx:65-83`, `:282-293` | **PASS** |
| Pooling conditional filter by `pooling_type` | global_*/adaptive_*/standard | `:86-119`, `:294-305` (mirrors v3 `nn-component-update.tsx:649-669`) | **PASS** |
| Dataset conditional filter by `input_format` | non-images hides shape/normalize | `:121-131`, `:306-318` | **PASS** |
| Per-row "enable optional" checkbox | toggles attribute presence on `data.attributes` | `:368-393`, `:421-450` | **PASS** |
| Mandatory auto-fill on first mount (#30) | mirrors v3 `componentDidMount` | `:238-269` `useEffect` | **PASS** |
| List-shape placeholder + warning (#33) | `getListExpectation` + `LIST_STRICT_REGEX` | `:431-442`, `:537-558` | **PASS** |
| Pooling placeholder re-resolves on `dimension` change | reads node attr | `:321-326`, passed into row | **PASS** |
| `predecessor` widget (sibling layers in same container) | dropdown of sibling names | `:158-170`, `:483-507` | **PASS** |
| `layers_of_tensors` widget (TensorOp) | comma-separated text | `:509-528` | **PASS** |
| Dataset `task_type` (binary/multi_class/regression) | dropdown | `nnAttributeWidgetConfig.ts:447-452` | **PASS** |
| Dataset `input_format` (csv/images) | dropdown | `:453-459` | **PASS** |
| Pooling `pooling_type` includes `global_*` | for v3 fixture round-trip (SA-2.2 #32) | `:66-73` `POOLING_TYPE_OPTIONS` | **PASS** |
| Activation function options | relu/leaky_relu/sigmoid/softmax/tanh | `:41-47` | **PASS** |

Spot-check: every field present in v3 `WIDGET_CONFIG_MAP`
(`nn-attribute-widget-config.ts:32-136`) has a matching slug in
`V3_ATTRIBUTE_TYPE_TO_SLUG` (`:502-662`) and a slot in `LAYER_ATTRIBUTE_SCHEMA`.
No widget kind is dropped.

---

## D. Slug collision: `dimension`

`pooling.dimension` (Pooling) vs `batch_normalization.dimension` (BatchNorm)
are explicitly disambiguated on **both sides**.

| Side | Source | Mechanism |
|---|---|---|
| Frontend collision set | `nnAttributeWidgetConfig.ts:83` `COLLIDING_SLUGS = new Set(["dimension"])` | shared by inspector reads (`NNComponentEditPanel.tsx:197-211, 225-231`) and migrator |
| Frontend prefix lookup | `nnAttributeWidgetConfig.ts:96-134` `kindToSlugPrefix` | maps `PoolingLayer→pooling`, `BatchNormalizationLayer→batch_normalization` |
| Frontend write | `qualifySlug(layerKind, slug)` `:90-93` | inspector writes `pooling.dimension`, default-data uses qualified form (`constants.ts:1082, :1149`) |
| Frontend read tolerance | `getAttribute()` `:137-147` reads either form | both qualified & unqualified accepted |
| Migrator emits qualified | `versionConverter.ts:1374-1383` `collapseV3LayerAttributes` | v3 `DimensionAttributePooling`/`DimensionAttributeBatchNormalization` → qualified slug |
| Backend collision set | `nn_diagram_processor.py:118` `_COLLIDING_SLUGS = frozenset({'dimension'})` | mirror of frontend |
| Backend prefix table | `:119-122` `_LAYER_KIND_PREFIX` | `PoolingLayer→pooling`, `BatchNormalizationLayer→batch_normalization` |
| Backend resolver | `:161-167` in `get_element_attribute` | tries qualified first, falls back to unqualified |

**Status: PASS — bidirectional disambiguation verified.**

---

## E. Edges (NNNext / NNComposition / NNAssociation)

Sources: `lib/edges/edgeTypes/NNNext.tsx:35-194`, `NNComposition.tsx:12`,
`NNAssociation.tsx:12`. Marker styles
`lib/utils/edgeUtils.ts:357-380` (`getEdgeMarkerStyles`).

| Edge | Renderer | Marker | Brief | Status |
|---|---|---|---|---|
| NNNext (sequential, layer→layer) | `NNNext.tsx` full renderer with `EdgeInlineMarkers` | `markerEnd: url(#black-arrow)` (`edgeUtils.ts:361-367`) | filled-arrow head, unidirectional | **PASS** |
| NNComposition (NNContainer↔layer ownership) | aliases `NNNext` (`NNComposition.tsx:12`) | `markerStart: url(#black-rhombus)` (`edgeUtils.ts:368-374`) | diamond on **source** (container) side | **PASS** |
| NNAssociation (Dataset↔NNContainer) | aliases `NNNext` (`NNAssociation.tsx:12`) | no marker, plain stroke (`edgeUtils.ts:375-380`) | undirected line | **PASS** |
| v3→v4 edge type passthrough | `NNNext`, `NNComposition`, `NNAssociation` mapped 1:1 | `versionConverter.ts:454-456` | preserved verbatim | **PASS** |

Marker positions correct: composition diamond is `markerStart`
(source/container side), next is `markerEnd` (filled triangle on target),
association has neither.

---

## F. Drop validation

Sources: `lib/utils/nodeUtils.ts:230-259` (`isParentNodeType`),
`lib/utils/bpmnConstraints.ts:60-122` (`canDropIntoParent`).

| Check | Brief | v4 location | Status |
|---|---|---|---|
| `NNContainer` recognised as parent | `isParentNodeType("NNContainer")` returns true | `nodeUtils.ts:254` | **PASS** |
| `State` recognised as parent | for state body legacy nodes | `:255` | **PASS** |
| `AgentState` recognised as parent | bodies inlined, but flagged so default doesn't apply | `:256` (returns true) + `bpmnConstraints.ts:119-121` (returns false on `canDropIntoParent`) | **PASS** |
| `AgentIntent` recognised as parent | for intent body/description/object | `:257` | **PASS** |
| `NNContainer` accepts the 14 layer kinds + NNReference | drop allowed | `bpmnConstraints.ts:60-76` `NN_LAYER_KINDS_IN_CONTAINER` set: Conv1D/2D/3D, Pooling, RNN/LSTM/GRU, Linear, Flatten, Embedding, Dropout, LayerNorm, BatchNorm, TensorOp, NNReference (15 entries) | **PASS** |
| Top-level only nodes excluded | Configuration / TrainingDataset / TestDataset NOT in container set | `bpmnConstraints.ts:60-76` (intentionally absent per `:50-58` comment) | **PASS** |
| `State` parents 3 body shapes | `StateBody`/`StateFallbackBody`/`StateCodeBlock` | `bpmnConstraints.ts:97-103` | **PASS** |
| `AgentIntent` parents 3 body kinds | `AgentIntentBody`/`Description`/`ObjectComponent` | `:108-114` | **PASS** |
| `AgentState` returns false for canDropIntoParent | bodies are inlined onto data, not nested | `:119-121` | **PASS** |

---

## G. Auto-name uniqueness

Source: `lib/nodes/nnDiagram/_NNLayerBase.tsx:73-95`
(`nextUniqueNNLayerName`) + `:131-155` (effect that runs on first mount).
Mirrors v3 `nn-component-update.tsx:561-585` counter loop.

| Behaviour | Brief | Status |
|---|---|---|
| Two Conv2D drops produce `Conv2D` and `Conv2D2` | suffix counter starts at 2 | **PASS** (`:91-94`) |
| Sibling check filtered by `nodeType` | Conv2D and Conv1D do not collide | **PASS** (`:83-85`) |
| `selfId` excluded from sibling set | re-renders don't loop | **PASS** (`:84`) |
| Effect runs only once per node mount | `useEffect(..., [id])` | **PASS** (`:131-155`) |
| Modifiable-diagram guard | dedupe disabled in read-only diagrams | **PASS** (`:132`) |

---

## H. User-reported "we cant drop the class" / NN drop crash

Root cause: `DraggableGhost.tsx:90-105` (pre-fix) called
`defaultData.attributes.map(...)` unconditionally during the universal id
rewrite. NN palette entries store `attributes` as `{slug: value}` dict (see §A
above), not an array, so `.map is not a function` aborted the drop handler.

Fix: commit `a31c4ec` adds `Array.isArray(...)` guard around both
`defaultData.methods` and `defaultData.attributes` rewrites
(`packages/library/lib/components/DraggableGhost.tsx`, the original `+9 -3`
diff). Confirmed via `git show a31c4ec -- packages/library/lib/components/DraggableGhost.tsx`:

```diff
- if (defaultData.attributes) {
+ if (Array.isArray(defaultData.attributes)) {
```

**Drop now succeeds** for every NN palette entry. NN attribute dicts skip
the array rewrite entirely (the slug→value keys are stable and don't need
new IDs).

---

## I. Backend layer creators (spot-check)

Source: `nn_diagram_processor.py`. Verified:

| Layer | Function | Mandatory attrs read | Optional attrs read |
|---|---|---|---|
| Conv2D | `_create_conv_layer(node, Conv2D, …)` `:709-793` | `NameAttribute`, `KernelDimAttribute`, `OutChannelsAttribute` (`:712,716,728`) | `StrideDimAttribute`, `InChannelsAttribute`, `PaddingAmountAttribute`, `PaddingTypeAttribute`, `ActvFuncAttribute`, `NameModuleInputAttribute`, `InputReusedAttribute`, `PermuteInAttribute`, `PermuteOutAttribute` (`:739-791`) |
| Pooling | `create_pooling_layer(node)` `:796-882` | `NameAttribute`, `PoolingTypeAttribute`, `DimensionAttribute` (collision-aware `:811-814`) | `KernelDimAttribute`, `StrideDimAttribute`, `PaddingAmountAttribute`, `PaddingTypeAttribute`, `OutputDimAttribute`, `ActvFuncAttribute`, `NameModuleInputAttribute`, `InputReusedAttribute`, `PermuteInAttribute`, `PermuteOutAttribute` |
| LSTM | `_create_rnn_like_layer(node, LSTMLayer)` `:884-943` | `NameAttribute`, `HiddenSizeAttribute` (`:887,891`) | `ReturnTypeAttribute`, `InputSizeAttribute`, `BidirectionalAttribute`, `DropoutAttribute`, `BatchFirstAttribute`, `ActvFuncAttribute`, `NameModuleInputAttribute`, `InputReusedAttribute` (`:901-941`) |

Mapping table `_ATTR_KEY_TO_NAME`
(`nn_diagram_processor.py:74-110`) covers every slug in the frontend's
`V3_ATTRIBUTE_TYPE_TO_SLUG`. Both sides agree.

17 layer kinds correspond to 17 backend creator entry points (3 conv +
pooling + 3 recurrent + linear + flatten + embedding + dropout + layernorm +
batchnorm + tensorop + configuration + dataset (training/test share)).

---

## Verdict

The NNDiagram migration is **functionally complete**. All 18 node types
(17 layer-style + container + reference, the 18th being NNContainer with
NNReference making 19 total kinds in the registry — counting differs because
TrainingDataset and TestDataset are separate kinds that share the dataset
schema) register, render with their PNG icons, dispatch to the generic
inspector with per-kind schema, and round-trip the dimension-slug collision
through to the backend. The three NN edges have correct markers. Drop
validation accepts the 14 in-container layer kinds plus NNReference inside
NNContainer; State / AgentState / AgentIntent parent-child rules are also
covered. The DraggableGhost crash that blocked all NN drops is fixed in
`a31c4ec`. Auto-name uniqueness reproduces the v3 suffix-counter behaviour.

No blocking gaps observed in this audit. Minor observations:

1. `_NNLayerBase.tsx` falls back to "no icon" when the asset is missing
   (`:169` `showIcon = !!iconFile && iconSize >= 24`). NNContainer / NNReference
   have no entry in `NN_LAYER_ICON_FILES`, so they render iconless by design —
   no fallback `default.png` is shipped. Document comment at `:36-38` says
   "fall back to `default.png` if the asset is present" but the asset
   intentionally isn't shipped. Minor doc/code drift, not functional.
2. The dimension-collision contract is enforced only for `dimension`;
   `_COLLIDING_SLUGS` is a single-element set. If new colliding slugs are
   introduced later they must be added in two places (frontend
   `nnAttributeWidgetConfig.ts:83` and backend `nn_diagram_processor.py:118`).
3. The inspector's mandatory auto-fill effect (`NNComponentEditPanel.tsx:238-269`)
   keys on `[elementId]` and runs once. If the layer kind ever changes
   in-place (which the public API does not currently allow), pre-existing
   attributes wouldn't be re-validated against the new schema. Theoretical, not
   reachable from current UX.
