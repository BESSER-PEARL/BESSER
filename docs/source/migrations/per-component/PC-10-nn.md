# PC-10 — NNDiagram (container + 18 layer kinds + edges + icons) parity audit

Read-only audit of the NNDiagram migration. Compares the v3 implementation
under `packages/editor/src/main/packages/nn-diagram/` against the SA-5 /
SA-2.2 v4 implementation under `packages/library/lib/nodes/nnDiagram/`,
`packages/library/lib/components/inspectors/nnDiagram/`, and the three
`packages/library/lib/edges/edgeTypes/NN*.tsx` files.

- **v3 source-of-truth (old)**:
  - `besser/utilities/web_modeling_editor/frontend/packages/editor/src/main/packages/nn-diagram/` (entire package)
  - `…/nn-attribute-widget-config.ts` — per-attribute widget metadata
  - `…/nn-validation-defaults.ts` — `LIST_STRICT_REGEX`, `getListExpectation`, attribute defaults
  - `…/nn-component/nn-component-update.tsx` — inspector behaviour: `createMandatoryAttributes`, `getTensorOpOptionalAttributes`, `getPoolingOptionalAttributes`, `getDatasetOptionalAttributes`
  - `…/nn-layer-icon/nn-layer-icon-component.tsx` — v3 icon card renderer (PNG, not SVG)
- **v3 layer-icon assets**: `besser/utilities/web_modeling_editor/frontend/packages/webapp/assets/images/nn-layers/{conv1d,conv2d,conv3d,pooling,linear,flatten,embedding,dropout,rnn,lstm,gru,layernorm,batchnorm,tensorop,configuration,train_data,test_data}.png` — 17 PNGs, all still present in the webapp asset tree.
- **v4 target (new)**:
  - Nodes: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/nnDiagram/` (18 `.tsx` + `_NNLayerBase.tsx` + `nnAttributeWidgetConfig.ts` + `nnValidationDefaults.ts` + `index.ts`)
  - Inspectors: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/inspectors/nnDiagram/{NNComponentEditPanel,NNContainerEditPanel,NNReferenceEditPanel}.tsx`
  - Edges: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/edges/edgeTypes/{NNNext,NNComposition,NNAssociation}.tsx`
  - Palette previews: `besser/utilities/web_modeling_editor/frontend/packages/library/lib/components/svgs/nodes/nnDiagram/NNDiagramSVGs.tsx`

Status legend: **PASS** = present and matches the brief. **GAP** = missing,
regressed, or differs from the brief. **N/A in v3** = brief asks v4 to
expose something v3 did not.

---

## 1. Node-type registrations (18 kinds)

Verified via `lib/nodes/nnDiagram/index.ts:40-60` (`registerNodeTypes`)
and the inspector binding loop at
`lib/components/inspectors/nnDiagram/index.ts:22-47`.

| Kind | v3 element | v4 component | Inspector | Status |
|---|---|---|---|---|
| Conv1DLayer | `nn-conv1d-layer/` | `Conv1DLayer.tsx` (delegates `_NNLayerBase`) | `NNComponentEditPanel` | **PASS** |
| Conv2DLayer | `nn-conv2d-layer/` | `Conv2DLayer.tsx` | `NNComponentEditPanel` | **PASS** |
| Conv3DLayer | `nn-conv3d-layer/` | `Conv3DLayer.tsx` | `NNComponentEditPanel` | **PASS** |
| PoolingLayer | `nn-pooling-layer/` | `PoolingLayer.tsx` | `NNComponentEditPanel` | **PASS** |
| RNNLayer | `nn-rnn-layer/` | `RNNLayer.tsx` | `NNComponentEditPanel` | **PASS** |
| LSTMLayer | `nn-lstm-layer/` | `LSTMLayer.tsx` | `NNComponentEditPanel` | **PASS** |
| GRULayer | `nn-gru-layer/` | `GRULayer.tsx` | `NNComponentEditPanel` | **PASS** |
| LinearLayer | `nn-linear-layer/` | `LinearLayer.tsx` | `NNComponentEditPanel` | **PASS** |
| FlattenLayer | `nn-flatten-layer/` | `FlattenLayer.tsx` | `NNComponentEditPanel` | **PASS** |
| EmbeddingLayer | `nn-embedding-layer/` | `EmbeddingLayer.tsx` | `NNComponentEditPanel` | **PASS** |
| DropoutLayer | `nn-dropout-layer/` | `DropoutLayer.tsx` | `NNComponentEditPanel` | **PASS** |
| LayerNormalizationLayer | `nn-layernormalization-layer/` | `LayerNormalizationLayer.tsx` | `NNComponentEditPanel` | **PASS** |
| BatchNormalizationLayer | `nn-batchnormalization-layer/` | `BatchNormalizationLayer.tsx` | `NNComponentEditPanel` | **PASS** |
| TensorOp | `nn-tensorop/` | `TensorOp.tsx` | `NNComponentEditPanel` | **PASS** |
| Configuration | `nn-configuration/` | `Configuration.tsx` | `NNComponentEditPanel` | **PASS** |
| TrainingDataset | `nn-dataset/` (Training variant) | `TrainingDataset.tsx` | `NNComponentEditPanel` | **PASS** |
| TestDataset | `nn-dataset/` (Test variant) | `TestDataset.tsx` | `NNComponentEditPanel` | **PASS** |
| NNContainer | `nn-container/` | `NNContainer.tsx` | `NNContainerEditPanel` | **PASS** |
| NNReference | `nn-reference/` | `NNReference.tsx` | `NNReferenceEditPanel` | **PASS** |

All 18 kinds are present and dispatch to a render component. The
inspector loop binds the generic `NNComponentEditPanel` to all 17
layer-style kinds; `NNContainer` / `NNReference` get dedicated panels.

---

## 2. Per-layer attribute schema

Source: `lib/nodes/nnDiagram/nnAttributeWidgetConfig.ts` exports
`LAYER_ATTRIBUTE_SCHEMA` (line 477) with one ordered field array per
kind. v3 source-of-truth: per-attribute element classes under each
`nn-*-attributes/` directory plus the `WIDGET_CONFIG_MAP` at
`nn-attribute-widget-config.ts:32-136`.

| Layer | v3 attributes (count) | v4 fields (count) | Status |
|---|---|---|---|
| Conv1D / Conv2D / Conv3D | 12 each (name, kernel_dim, out_channels, stride_dim, in_channels, padding_amount, padding_type, actv_func, name_module_input, input_reused, permute_in, permute_out) | `CONV_FIELDS` 12 | **PASS** |
| Pooling | 13 (name, pooling_type, dimension, kernel_dim, stride_dim, padding_amount, padding_type, output_dim, actv_func, name_module_input, input_reused, permute_in, permute_out) | `POOLING_FIELDS` 13 | **PASS** |
| RNN / LSTM / GRU | 10 (name, hidden_size, return_type, input_size, bidirectional, dropout, batch_first, actv_func, name_module_input, input_reused) | `RECURRENT_FIELDS` 10 | **PASS** |
| Linear | 6 (name, out_features, in_features, actv_func, name_module_input, input_reused) | `LINEAR_FIELDS` 6 | **PASS** |
| Flatten | 6 (name, start_dim, end_dim, actv_func, name_module_input, input_reused) | `FLATTEN_FIELDS` 6 | **PASS** |
| Embedding | 6 (name, num_embeddings, embedding_dim, actv_func, name_module_input, input_reused) | `EMBEDDING_FIELDS` 6 | **PASS** |
| Dropout | 4 (name, rate, name_module_input, input_reused) | `DROPOUT_FIELDS` 4 | **PASS** |
| LayerNormalization | 5 (name, normalized_shape, actv_func, name_module_input, input_reused) | `LAYER_NORM_FIELDS` 5 | **PASS** |
| BatchNormalization | 6 (name, num_features, dimension, actv_func, name_module_input, input_reused) | `BATCH_NORM_FIELDS` 6 | **PASS** |
| TensorOp | 8 (name, tns_type, concatenate_dim, layers_of_tensors, reshape_dim, transpose_dim, permute_dim, input_reused) | `TENSOR_OP_FIELDS` 8 | **PASS** |
| Configuration | 8 (batch_size, epochs, learning_rate, optimizer, loss_function, metrics, weight_decay, momentum) | `CONFIGURATION_FIELDS` 8 | **PASS** |
| TrainingDataset / TestDataset | 6 (name, path_data, task_type, input_format, shape, normalize) | `DATASET_FIELDS` 6 | **PASS** |

All 18 kind schemas are present and the slug lists match the v3
`WIDGET_CONFIG_MAP` keys (cross-checked against
`V3_ATTRIBUTE_TYPE_TO_SLUG` at lines 503-662).

---

## 3. SA-2.2 audit-recommendation deltas

| # | Brief | v4 implementation | Status | Notes |
|---|---|---|---|---|
| #29 | Conditional optional-attribute filtering by discriminator (TensorOp `tns_type`, Pooling `pooling_type`, Dataset `input_format`) | `filterTensorOpOptionals` / `filterPoolingOptionals` / `filterDatasetOptionals` at `NNComponentEditPanel.tsx:65-131`. Wired into the render path lines 282-318. Hide-sets ported verbatim from v3 `nn-component-update.tsx:614-669`. | **PASS** | Pooling adaptive vs global vs standard hide-sets are bit-for-bit identical to v3. |
| #30 | Mandatory-attribute auto-creation on node drop | `useEffect` in `NNComponentEditPanel.tsx:238-269` — fills missing mandatory keys with `f.defaultValue` / `NN_ATTRIBUTE_DEFAULTS[slug]` / `data.name` for the `name` slug. | **PASS** | Equivalent to v3 `componentDidMount` → `createMandatoryAttributes` at lines 588-605. **GAP (minor)**: v3 deduplicates auto-generated `name` values across the diagram (counter loop, lines 561-580); v4 just copies `data.name`, so two freshly dropped layers can share a name until the user edits one. |
| #31 | Per-row optional checkbox | `NNAttributeRow` exposes a `Checkbox` when `onEnabledChange` is supplied (`NNComponentEditPanel.tsx:444-450, 380-393`). Unchecking calls `removeAttribute(slug)`; checking writes the schema default. | **PASS** | Mirrors v3 `OptionalAttributeRow` toggle. |
| #32 | `pooling_type` must include `global_average` and `global_max` | `POOLING_TYPE_OPTIONS` at `nnAttributeWidgetConfig.ts:66-73` lists `["average", "max", "adaptive_average", "adaptive_max", "global_average", "global_max"]`. | **PASS** | Includes both `global_*` values; matches the v3 hide-list logic at `nn-component-update.tsx:660`. |
| #33 | List-validation regex on shaped fields (`kernel_dim`, `stride_dim`, `output_dim`, `normalized_shape`, `transpose_dim`) | `LIST_STRICT_REGEX` at `nnValidationDefaults.ts:40` plus `getListExpectation()` lines 58-106. `NNAttributeRow` checks the regex on lines 440-441 and surfaces an `error` + `helperText` placeholder. | **PASS** | Pooling placeholder re-resolves when the user changes `dimension` (panel resolves `poolingDimension` at line 321 and passes it to `getListExpectation`). Regex source `^\[\s*-?\d+(\s*,\s*-?\d+)*\s*\]$` matches v3 verbatim. |
| `qualifySlug` for `dimension` collision | Pooling vs BatchNormalization both define a `dimension` slug | `COLLIDING_SLUGS = new Set(["dimension"])` (line 83); `qualifySlug(layerKind, slug)` (line 90) emits `pooling.dimension` / `batch_normalization.dimension`; readers tolerate both forms via `readAttribute()` in the panel and `getAttribute()` in the config module. | **PASS** | Backend disambiguation already shipped in `4afc909`. The v4 panel writes the qualified key on update (line 198) and accepts either key on read (line 226). |

---

## 4. Edges

| Edge | v3 | v4 | Status |
|---|---|---|---|
| NNNext (sequential flow) | `nn-unidirectional/` (filled-arrow head, layer→layer) | `edges/edgeTypes/NNNext.tsx` — full step-path renderer with `EdgeInlineMarkers`, midpoint dragging, label, reconnection | **PASS** |
| NNComposition (container ⟶ layer, diamond on container side) | `nn-composition/` | `NNComposition.tsx` aliases `NNNext` and registers a separate edge type. Marker (`url(#black-rhombus)` on markerStart) is resolved by `getEdgeMarkerStyles('NNComposition')`. | **PASS (visual via marker registry)** |
| NNAssociation (Dataset ⟷ NNContainer, plain line) | `nn-association/` + `nn-association-line/` | `NNAssociation.tsx` aliases `NNNext`; markers explicitly omitted by `getEdgeMarkerStyles('NNAssociation')`. | **PASS** |

The three v4 files share a single renderer (`NNNext`); aliasing is
intentional per the file headers and matches the spec.

---

## 5. CRITICAL — Layer logos are GONE

**FINDING: confirmed.** SA-2.2 #34 retired the v3 110×110 fixed-icon
layer card in favour of stereotype-style cards (`«Conv2D»` header +
`name`). The current v4 layer renderer at
`lib/nodes/nnDiagram/_NNLayerBase.tsx:54-119` draws **only**:

1. A rounded rectangle (`<rect rx={6}>`).
2. Stereotype text `«${kindLabel}»` at `y={18}`.
3. Bold `data.name` at `y={38}`.
4. A horizontal divider line if the body is tall enough.

There is **no `<image href=…>` element**, no PNG reference, no SVG icon
import. `makeNNLayerComponent(nodeType, kindLabel, defaultFill)` (lines
123-148) is the factory used by all 17 layer files; none of the
per-layer files override the renderer to inject an icon. The palette
previews at
`lib/components/svgs/nodes/nnDiagram/NNDiagramSVGs.tsx:55-77` are also
plain text-only `<rect>` + `<text>` placeholders (`buildLayerPreview`
function lines 13-53) — no `<image>`, no PNG.

**The v3 PNG assets still exist** at
`packages/webapp/assets/images/nn-layers/*.png` (17 files for the 17
layer-style kinds plus `default.png` fallback referenced in the v3
mapping). The v3 renderer at
`packages/editor/src/main/packages/nn-diagram/nn-layer-icon/nn-layer-icon-component.tsx:55-62`
served them via `<image href="/images/nn-layers/${file}.png" width=80
height=80 />` against `ICON_BASE_PATH = '/images/nn-layers/'`.

**Important nomenclature note for the brief**: the brief refers to "the
v3 layer-icon SVGs at `nn-layer-icon/`". The v3 implementation is
**PNG-based, not SVG-based** — `nn-layer-icon-component.tsx` is the
single TSX file that maps a layer type to a PNG filename, and the
assets are PNGs. There are **no v3 SVG files** to port; the
restoration path is to bring back the PNG `<image>` element inside
`_NNLayerBase.tsx` (or to convert the PNGs to SVG as a follow-up). The
PNGs are still in the webapp asset tree, so wiring them back is a
~5-line edit to `_NNLayerBase.tsx` plus a layer-kind→filename lookup
table (already present in the v3 source).

**Status: GAP (regression introduced intentionally by SA-2.2 #34;
user-reported)**.

---

## 6. NNContainer / NNReference

| Feature | v3 | v4 | Status |
|---|---|---|---|
| NNContainer renders as a parent node with header bar | `nn-container/nn-container-component.tsx` | `NNContainer.tsx:54-89` — large rounded rect, header height from `LAYOUT.DEFAULT_HEADER_HEIGHT`, name + divider | **PASS** |
| Layer nesting via `parentId` | v3 used owner pointer | v4 uses React-Flow `parentId = container.id` (per `nodes/nnDiagram/index.ts:14-17` contract) | **PASS** |
| `entryLayerId` (entry-side pointer) | Field carried on container in v3 | Editable in `NNContainerEditPanel.tsx:81-99` via dropdown of children resolved by `parentId` | **PASS** |
| Description field | `StylePane showDescription` | Multiline `MuiTextField` at `NNContainerEditPanel.tsx:71-80` | **PASS** |
| NNReference label + dashed border | `nn-reference/nn-reference-component.tsx` | `NNReference.tsx:33-78` — dashed rect (`strokeDasharray="4 2"`), italic `→ ${name}` label | **PASS** |
| NNReference target dropdown + free-text override | v3 used owner-side picker | `NNReferenceEditPanel.tsx:74-101` — dropdown for same-parent layers + free-text override for cross-container references | **PASS** |

---

## 7. Generic `NNComponentEditPanel` schema mapping (all 18 kinds)

Verified via `getLayerSchema(layerKind)` import at
`NNComponentEditPanel.tsx:18` and the kind-keyed dispatch loop at
`lib/components/inspectors/nnDiagram/index.ts:22-44`.

| Layer kind | `getLayerSchema()` returns | Mandatory subset (excluding `name`) | Discriminator | Status |
|---|---|---|---|---|
| Conv1D / 2D / 3D | `CONV_FIELDS` (12) | `kernel_dim`, `out_channels` | — | **PASS** |
| PoolingLayer | `POOLING_FIELDS` (13) | `pooling_type`, `dimension` | `pooling_type` → `filterPoolingOptionals` | **PASS** |
| RNN / LSTM / GRU | `RECURRENT_FIELDS` (10) | `hidden_size` | — | **PASS** |
| LinearLayer | `LINEAR_FIELDS` (6) | `out_features` | — | **PASS** |
| FlattenLayer | `FLATTEN_FIELDS` (6) | — | — | **PASS** |
| EmbeddingLayer | `EMBEDDING_FIELDS` (6) | `num_embeddings`, `embedding_dim` | — | **PASS** |
| DropoutLayer | `DROPOUT_FIELDS` (4) | `rate` | — | **PASS** |
| LayerNormalizationLayer | `LAYER_NORM_FIELDS` (5) | `normalized_shape` | — | **PASS** |
| BatchNormalizationLayer | `BATCH_NORM_FIELDS` (6) | `num_features`, `dimension` | — | **PASS** |
| TensorOp | `TENSOR_OP_FIELDS` (8) | `tns_type` | `tns_type` → `filterTensorOpOptionals` | **PASS** |
| Configuration | `CONFIGURATION_FIELDS` (8) | `batch_size`, `epochs`, `learning_rate`, `optimizer`, `loss_function`, `metrics` | — | **PASS** |
| TrainingDataset / TestDataset | `DATASET_FIELDS` (6) | `path_data` | `input_format` → `filterDatasetOptionals` | **PASS** |
| NNContainer | n/a — uses `NNContainerEditPanel` | — | — | **PASS** |
| NNReference | n/a — uses `NNReferenceEditPanel` | — | — | **PASS** |

---

## 8. Top gaps (ranked)

1. **Layer-icon regression (#34)** — `_NNLayerBase.tsx` does not render
   any image; the v3 80×80 PNG card is gone. Assets still ship in
   `packages/webapp/assets/images/nn-layers/`. User has flagged this
   as a regression. **Restoration path**: re-introduce
   `<image href="/images/nn-layers/${LAYER_ICONS[nodeType]}" …/>` plus
   a `LAYER_ICONS` mapping inside `_NNLayerBase.tsx` (or a sibling
   helper). All 17 PNG filenames are in
   `nn-layer-icon/nn-layer-icon-component.tsx:6-24`. *(Note: the brief
   describes v3 icons as SVGs — they are actually PNGs; mention this
   when scoping the restoration ticket.)*
2. **Auto-name uniqueness on drop** — v3
   `createMandatoryAttributes()` (`nn-component-update.tsx:561-585`)
   suffixes a counter (`Conv2D2`, `Conv2D3`, …) when the auto-generated
   name collides with an existing layer. The v4
   `useEffect` (`NNComponentEditPanel.tsx:238-269`) just copies
   `data.name` and does not deduplicate, so two newly-dropped layers
   can share a name until the user edits one. Minor UX regression.
3. **Palette previews are placeholder text** — `NNDiagramSVGs.tsx`
   builds every layer thumbnail as a coloured rectangle with the kind
   label only. Once #34 is restored on the canvas, the sidebar drag
   source should also render the icon so the palette and the canvas
   match. (Currently inconsistent even with the icons gone — the
   palette shows `Conv2D`, the canvas shows `«Conv2D»`.)

---

## 9. Files inspected

- `packages/library/lib/nodes/nnDiagram/_NNLayerBase.tsx`
- `packages/library/lib/nodes/nnDiagram/{Conv1D,Conv2D,Conv3D,Pooling,RNN,LSTM,GRU,Linear,Flatten,Embedding,Dropout,LayerNormalization,BatchNormalization,TensorOp,Configuration,TrainingDataset,TestDataset}Layer.tsx` (Conv1DLayer read in full; remainder confirmed via `index.ts` registrations)
- `packages/library/lib/nodes/nnDiagram/{NNContainer,NNReference}.tsx`
- `packages/library/lib/nodes/nnDiagram/{nnAttributeWidgetConfig,nnValidationDefaults,index}.ts`
- `packages/library/lib/components/inspectors/nnDiagram/{NNComponentEditPanel,NNContainerEditPanel,NNReferenceEditPanel,index}.tsx?`
- `packages/library/lib/components/svgs/nodes/nnDiagram/NNDiagramSVGs.tsx`
- `packages/library/lib/edges/edgeTypes/{NNNext,NNComposition,NNAssociation}.tsx`
- `packages/editor/src/main/packages/nn-diagram/{nn-attribute-widget-config,nn-validation-defaults}.ts`
- `packages/editor/src/main/packages/nn-diagram/nn-component/nn-component-update.tsx`
- `packages/editor/src/main/packages/nn-diagram/nn-layer-icon/nn-layer-icon-component.tsx`
- `packages/webapp/assets/images/nn-layers/*.png` (asset inventory)
