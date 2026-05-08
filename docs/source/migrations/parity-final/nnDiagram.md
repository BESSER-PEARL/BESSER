# SA-PARITY-FINAL-6 — NNDiagram

**Verdict: PASS WITH ONE BLOCKING BACKEND GAP.** Frontend v4 ports the
v3 NNDiagram element catalogue, attribute schema, inspector behaviour,
and edge types with full structural fidelity. Round-trip migrator
disambiguates the `dimension` slug collision via qualified keys
(`pooling.dimension` / `batch_normalization.dimension`). However, the
**Python backend at `besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/nn_diagram_processor.py`
does NOT implement the matching qualified-slug lookup**, so a v4
Pooling or BatchNormalization layer authored in the new frontend will
have its mandatory `dimension` read as `None` server-side. This
contradicts the SA-5 inspector contract and breaks code generation
for those two layer kinds.

Everything else in the v4 NN port is at parity with v3.

---

## 1. Element types — PASS

All 18 `NNElementType` layer types from
`packages/editor/src/main/packages/nn-diagram/index.ts` (v3) are
registered as v4 nodes in
`besser/utilities/web_modeling_editor/frontend/packages/library/lib/nodes/nnDiagram/index.ts`:

`Conv1DLayer`, `Conv2DLayer`, `Conv3DLayer`, `PoolingLayer`,
`RNNLayer`, `LSTMLayer`, `GRULayer`, `LinearLayer`, `FlattenLayer`,
`EmbeddingLayer`, `DropoutLayer`, `LayerNormalizationLayer`,
`BatchNormalizationLayer`, `TensorOp`, `Configuration`,
`TrainingDataset`, `TestDataset`, plus `NNContainer` (parent) and
`NNReference`.

**Note:** the round number in the brief is "18 layer types" — the
literal layer-only count is 14 (Conv×3, Pool, RNN/LSTM/GRU, Linear,
Flatten, Embedding, Dropout, LayerNorm, BatchNorm) + `TensorOp` (1)
= 15 layer-style nodes. Adding `Configuration`, `TrainingDataset`,
`TestDataset` lands at 18 non-container/reference nodes, matching
the v3 catalogue.

**v3 sidebar helpers** (`NNSectionTitle`, `NNSectionSeparator`) are
intentionally not ported as nodes — they were sidebar-organisation
glue, not canvas elements. SA-2.2 spec confirms this is intentional.

## 2. Edge types — PASS

`NNNext`, `NNComposition`, `NNAssociation` all exist at
`packages/library/lib/edges/edgeTypes/`. `NNComposition` and
`NNAssociation` are aliased to `NNNext` (visual fidelity is OK
because v3 used the same step-path style; the SA-5 round-trip
covers all three by `data` shape, not by render).

## 3. Per-element attribute fields — PASS

`packages/library/lib/nodes/nnDiagram/nnAttributeWidgetConfig.ts`
defines `LAYER_ATTRIBUTE_SCHEMA` for every layer kind and
`V3_ATTRIBUTE_TYPE_TO_SLUG` for the migrator. Cross-checked
against v3 source-of-truth
`packages/editor/src/main/packages/nn-diagram/index.ts` and
`nn-attribute-widget-config.ts`:

| Layer kind | v3 attribute keys | v4 schema slugs | Status |
|---|---|---|---|
| Conv1D / Conv2D / Conv3D | name, kernel_dim, out_channels, stride_dim, in_channels, padding_amount, padding_type, actv_func, name_module_input, input_reused, permute_in, permute_out | identical 12 slugs | PASS |
| Pooling | name, pooling_type, dimension, kernel_dim, stride_dim, padding_amount, padding_type, output_dim, actv_func, name_module_input, input_reused, permute_in, permute_out | identical 13 slugs (`dimension` → qualified) | PASS |
| RNN / LSTM / GRU | name, hidden_size, return_type, input_size, bidirectional, dropout, batch_first, actv_func, name_module_input, input_reused | identical 10 slugs | PASS |
| Linear | name, out_features, in_features, actv_func, name_module_input, input_reused | identical 6 slugs | PASS |
| Flatten | name, start_dim, end_dim, actv_func, name_module_input, input_reused | identical 6 slugs | PASS |
| Embedding | name, num_embeddings, embedding_dim, actv_func, name_module_input, input_reused | identical 6 slugs | PASS |
| Dropout | name, rate, name_module_input, input_reused | identical 4 slugs | PASS |
| LayerNormalization | name, normalized_shape, actv_func, name_module_input, input_reused | identical 5 slugs | PASS |
| BatchNormalization | name, num_features, dimension, actv_func, name_module_input, input_reused | identical 6 slugs (`dimension` → qualified) | PASS |
| TensorOp | name, tns_type, concatenate_dim, layers_of_tensors, reshape_dim, transpose_dim, permute_dim, input_reused | identical 8 slugs | PASS |
| Configuration | batch_size, epochs, learning_rate, optimizer, loss_function, metrics, weight_decay, momentum | identical 8 slugs | PASS |
| TrainingDataset / TestDataset | name, path_data, task_type, input_format, shape, normalize | identical 6 slugs | PASS |

**Reconciliation with audit prompt wording.** The brief lists some
PyTorch-style names (`groups`, `bias`, `dilation`, `padding`,
`padding_idx`, `eps`, `elementwise_affine`, `momentum` (per-layer),
`affine`, `track_running_stats`, `lr`, `loss`, `epochs`, `batch_size`,
`metrics`, `train_size`, `test_size`) that **were never present in
v3**. The v3 metamodel uses `kernel_dim` / `stride_dim` /
`padding_amount` / `padding_type` / `permute_in` / `permute_out` for
Conv layers and does not expose `groups` / `bias` / `dilation`. The
v4 schema correctly mirrors v3 — these are not gaps.

**Pooling option list — PASS.** v4
`POOLING_TYPE_OPTIONS = ['average', 'max', 'adaptive_average',
'adaptive_max', 'global_average', 'global_max']` matches the v3
`getPoolingOptionalAttributes` filter universe.

**TensorOp option list — PASS.** v4 `TNS_TYPE_OPTIONS = ['reshape',
'concatenate', 'multiply', 'matmultiply', 'transpose', 'permute']`
is verbatim from v3 `nn-attribute-widget-config.ts`.

## 4. Inspector form parity — PASS

`packages/library/lib/components/inspectors/nnDiagram/NNComponentEditPanel.tsx`
implements every behaviour the SA-5 brief calls for:

- **#29 conditional optional filtering** — `filterTensorOpOptionals`
  / `filterPoolingOptionals` / `filterDatasetOptionals` are
  byte-for-byte ports of the v3 helpers at
  `nn-component-update.tsx:614-669`.
- **#30 mandatory auto-fill** — `useEffect([elementId])` populates
  missing mandatory keys with schema `defaultValue` /
  `NN_ATTRIBUTE_DEFAULTS` / the layer's own `data.name`. Mirrors v3
  `componentDidMount` → `createMandatoryAttributes`.
- **#31 optional-row checkbox** — per-row `<Checkbox>` toggles
  enabled state, calls `removeAttribute` (with both qualified and
  unqualified key removal) on uncheck. Mirrors v3
  `OptionalAttributeRow`.
- **#33 list validation** — `LIST_STRICT_REGEX` warning + per-list
  `getListExpectation` placeholder; Pooling re-resolves the
  expectation when the user changes `dimension`.
- **Type-correct widgets** — `text` / `dropdown` / `predecessor` /
  `layers_of_tensors` widgets implemented in `NNAttributeRow`.
  Mandatory rows omit the checkbox by passing
  `onEnabledChange={undefined}`.
- **NNContainer panel** — dedicated `NNContainerEditPanel` for the
  container's name + entry-layer pointer.
- **NNReference panel** — dedicated `NNReferenceEditPanel`.

All 17 layer kinds + Container + Reference are wired into the
inspector registry at `inspectors/nnDiagram/index.ts`.

## 5. Slug collision handling — FAIL (backend leg only)

**Frontend leg — PASS.** `qualifySlug(layerKind, slug)` exists in
`nnAttributeWidgetConfig.ts` and is consumed by:

- `NNComponentEditPanel.updateAttribute` / `removeAttribute` /
  `readAttribute` (storage keyed on `pooling.dimension` /
  `batch_normalization.dimension` for both layer kinds).
- `versionConverter.collapseV3LayerAttributes` (migrator emits the
  qualified key on output).

**Backend leg — FAIL (BLOCKER).**
`besser/utilities/web_modeling_editor/backend/services/converters/json_to_buml/nn_diagram_processor.py:114-152`
implements `get_element_attribute(node, 'DimensionAttribute')` by
looking up the bare `dimension` key on `node.data.attributes`. There
is no qualified-slug fallback (`pooling.dimension` /
`batch_normalization.dimension`). Mirror gap on the buml→json side:
`buml_to_json/nn_diagram_converter.py:110,213` writes back the bare
`dimension` slug.

Concrete consequence: a v4-shaped diagram authored through the new
inspector stores `attributes['pooling.dimension'] = '2D'`. When that
diagram is sent to `/generate-output`, the backend reads
`attrs.get('dimension')` → `None` → raises
`"PoolingLayer '<name>' missing mandatory 'dimension' attribute"`.
Same failure mode for BatchNormalization.

This is a real, reproducible blocker for any user-authored Pooling
or BatchNormalization layer flowing through the v4 inspector.

**Suggested fix (for a follow-up PR, not this audit):**
in `_ATTR_KEY_TO_NAME` lookup at the v4 branch of
`get_element_attribute`, after the bare-slug `attrs.get(name)`,
fall back to `attrs.get(f"{kind_prefix}.{name}")` for the
collision-prone slug `dimension` (mirror of frontend's
`COLLIDING_SLUGS`). Same change on the buml→json emit side.

## 6. Visual shape — PASS

`docs/source/migrations/uml-v4-shape.md` § "Visual deviations from
v3" (lines 949–994) documents the deliberate retirement of the v3
110×110 fixed-icon layer card in favour of the
`_NNLayerBase.tsx` stereotype card. The doc covers the rationale
(uniformity / authoring / theme fidelity) and the migration
contract (icon view could be opt-in later without round-trip
impact). No further action needed.

## 7. Round-trip — PASS

`packages/library/tests/round-trip/nnDiagram.test.ts` (205 lines)
covers 7 cases:

1. v3 fixture → v4 with structural fidelity.
2. **Conv2D 5+ attribute collapse stress test** (the case the brief
   specifically calls out).
3. `dimension` slug collision disambiguation (open question #2).
4. Configuration's 6 mandatory training fields collapse.
5. Dataset attributes including the path field.
6. v4 → v3 → v4 structural equality.
7. Conv2D attribute edit preserved through v4 → v3 → v4.

All cases exercise the migrator's qualified-slug rules.

---

## Critical gaps

1. **Backend qualified-slug lookup missing for Pooling /
   BatchNormalization `dimension`.** Frontend writes
   `pooling.dimension` / `batch_normalization.dimension`; backend
   reads the bare `dimension` key in both `nn_diagram_processor.py`
   and `nn_diagram_converter.py`. This breaks code generation for
   v4-authored Pooling / BatchNormalization layers.

## Non-issues (clarifications)

- The audit prompt enumerated PyTorch-style attribute names not
  present in v3 (e.g. `groups`, `bias`, `dilation`, `padding_idx`,
  `eps`, `elementwise_affine`, `affine`, `track_running_stats`,
  `lr` (vs `learning_rate`), `loss` (vs `loss_function`),
  `train_size`, `test_size`). v4 correctly mirrors v3 by NOT having
  them. Not a gap.
- Sidebar-only `NNSectionTitle` / `NNSectionSeparator` nodes are
  intentionally not ported. Not a gap.
