# BESSER UML v3 → v4 Wire Shape Specification

**Status**: SA-1 Wave 1 hand-off contract.
**Audience**: every Wave-2 sub-agent (frontend nodes/edges, backend converters).
**Source of truth**: this document. No SA-to-SA chatter.

This file describes, per BESSER diagram type, the **v3** shape that the legacy
`packages/editor` Apollon fork currently emits and the **v4** shape that the
new React-Flow library (`packages/library`, `@tumaet/apollon` 4.x) accepts.

The mapping is exhaustive enough that:

- frontend `nodes/<diagram>/*.tsx` and `edges/edgeTypes/*.tsx` know the exact
  `data` field schema for each node/edge `type`,
- the Python `json_to_buml/<diagram>_diagram_processor.py` and
  `buml_to_json/<diagram>_diagram_converter.py` parse and emit the correct
  shape without consulting frontend code,
- the TS `migrate-uml-v3-to-v4.ts` migrator and the Python normalizer are
  byte-equivalent on every legacy fixture.

When in doubt, prefer **explicit fields on `node.data`** over inferring from
labels. Inheritance from v3:

- `id`, `name`, `type`, `bounds`, `owner`, `highlight`, `fillColor`,
  `strokeColor`, `textColor`, `description`, `icon`, `uri`, `assessmentNote`
  are common UMLModelElement fields. In v4, they all move into `node.data`
  (except `id`, which stays on the `Node`, and `bounds`, which is replaced by
  `position` + `width` + `height` + `measured`).
- v3 `owner: string | null` becomes v4 `parentId?: string` on the React-Flow
  Node. **v4 has no separate "owner" string in `data`** — React Flow's
  `parentId` is the only parent reference. Children whose `parentId` is set
  are positioned relative to the parent.
- v3 `bounds: { x, y, width, height }` ⇒ v4 `position: { x, y }` and
  `width` / `height` / `measured` on the React-Flow node.
- v3 `path: IPath` on relationships moves to v4 `edge.data.points: IPoint[]`.
- v3 `source: { element, direction }` and `target: { element, direction }`
  on relationships become v4 `edge.source` / `edge.target` (element id) plus
  `edge.sourceHandle` / `edge.targetHandle` (encoding direction). Roles and
  multiplicities for class/agent/etc. associations move into `edge.data` —
  see per-diagram sections.

---

## Project envelope

### v3 (today)

`BesserProject.diagrams[T][i].model` is per-diagram a v3 `UMLModel`:

```ts
type V3UMLModel = {
  version: `3.${number}.${number}`;
  type: UMLDiagramType;
  size: { width: number; height: number };
  elements: { [id: string]: UMLElement };
  relationships: { [id: string]: UMLRelationship };
  interactive: { elements: Record<string, boolean>; relationships: Record<string, boolean> };
  assessments: { [id: string]: Assessment };
  referenceDiagramData?: any;
};
```

`PROJECT_SCHEMA_VERSION = 4` wraps these into `BesserProject` (see
`packages/webapp/src/main/shared/types/project.ts`).

### v4 (target)

```ts
type V4UMLModel = {
  version: `4.${number}.${number}`;
  id: string;
  title: string;
  type: UMLDiagramType;
  size?: { width: number; height: number };  // optional, recomputable from nodes
  nodes: ApollonNode[];
  edges: ApollonEdge[];
  interactive?: { elements: Record<string, boolean>; relationships: Record<string, boolean> };
  assessments: { [id: string]: Assessment };
};

type ApollonNode = {
  id: string;
  type: DiagramNodeType;       // string union, see per-diagram sections
  position: { x: number; y: number };
  width: number;
  height: number;
  measured: { width: number; height: number };
  data: Record<string, unknown>;
  parentId?: string;           // replaces v3 owner
};

type ApollonEdge = {
  id: string;
  source: string;              // node id
  target: string;              // node id
  type: DiagramEdgeType;
  sourceHandle: string;        // direction encoded as handle id
  targetHandle: string;
  data: { points: IPoint[]; [key: string]: unknown };
};
```

`PROJECT_SCHEMA_VERSION` bumps to `5` to mark the project envelope contains v4
diagram models. The migrator runs on read in
`packages/webapp/src/main/shared/services/storage/local-storage-repository.ts`.

### Project mapping rules

- For every diagram in `project.diagrams[T][i]`, check `model.version`:
  - starts with `'3.'` → run the v3→v4 migrator on that single model in
    place, leaving the surrounding `ProjectDiagram` envelope alone.
  - starts with `'4.'` → leave alone.
- `currentDiagramType`, `currentDiagramIndices`, `references`, and
  `settings` are unchanged.
- `GUINoCodeDiagram` and `QuantumCircuitDiagram` keep their pre-existing
  formats (GrapesJS / custom canvas). They are **not** v3 UML models and the
  migrator must skip them based on diagram type, not on the absence of
  `elements`.

---

## ClassDiagram

### v3 element subtypes (mined from `packages/editor/src/main/packages/uml-class-diagram/`)

- `Package` — container for classes.
- `Class` / `AbstractClass` / `Interface` / `Enumeration` (`stereotype` field
  on the type or the `type` value itself disambiguates).
- `ClassAttribute`, `ClassMethod` — children of a class via `owner`. The
  parent class lists their ids in `attributes: string[]` / `methods: string[]`.
- `ClassOCLConstraint` — free-standing OCL constraint node.

### v3 relationship subtypes

- `ClassBidirectional`, `ClassUnidirectional`, `ClassAggregation`,
  `ClassComposition`, `ClassInheritance`, `ClassRealization`,
  `ClassDependency`, `ClassOCLLink`, `ClassLinkRel`.

Associations carry `source.role`, `source.multiplicity`, `target.role`,
`target.multiplicity` (see `UMLAssociation` in `typings.ts`). Inheritance
points from child (source) to parent (target).

### v4 node types

```ts
// node.type values
'package' | 'class'

// node.data for Package
{
  name: string;
  fillColor?: string; strokeColor?: string; textColor?: string;
  description?: string; assessmentNote?: string;
}

// node.data for class | interface | abstract | enumeration
type ClassNodeData = {
  name: string;
  stereotype?: 'abstract' | 'interface' | 'enumeration' | string | null;
  // member rows; v3 attributes/methods were separate UMLElements with owner=parent
  attributes: ClassifierMember[];
  methods: ClassifierMember[];
  // OCL constraints attached directly to this class (was a separate element pointing via owner in v3)
  oclConstraints?: { id: string; name: string; expression: string }[];
  fillColor?: string; strokeColor?: string; textColor?: string;
  description?: string; icon?: string; uri?: string; assessmentNote?: string;
};

type ClassifierMember = {
  id: string;
  name: string;
  attributeType: string;            // canonical Python-style: 'str','int','float','bool','date','datetime','time','any', or custom
  visibility: 'public' | 'private' | 'protected' | 'package';
  code?: string;
  implementationType?: 'none' | 'code' | 'bal' | 'state_machine' | 'quantum_circuit';
  stateMachineId?: string;
  quantumCircuitId?: string;
  isOptional?: boolean;
  isDerived?: boolean;
  isId?: boolean;
  isExternalId?: boolean;
  defaultValue?: unknown;
};
```

`stereotype` is encoded as a single string. Map v3 element `type`:

| v3 `type`      | v4 `node.type` | v4 `data.stereotype` |
|----------------|----------------|----------------------|
| `Class`        | `class`        | `null`               |
| `AbstractClass`| `class`        | `'abstract'`         |
| `Interface`    | `class`        | `'interface'`        |
| `Enumeration`  | `class`        | `'enumeration'`      |
| `Package`      | `package`      | n/a                  |

### v4 edge types

`'ClassBidirectional' | 'ClassUnidirectional' | 'ClassAggregation' |
'ClassComposition' | 'ClassInheritance' | 'ClassRealization' |
'ClassDependency' | 'ClassOCLLink' | 'ClassLinkRel'`.

```ts
// edge.data for class associations (Bidirectional, Unidirectional, Aggregation, Composition, Dependency)
{
  name?: string;                    // association name
  sourceRole?: string;
  sourceMultiplicity?: string;
  targetRole?: string;
  targetMultiplicity?: string;
  isManuallyLayouted?: boolean;
  points: IPoint[];
}

// edge.data for ClassInheritance, ClassRealization, ClassOCLLink, ClassLinkRel
{
  name?: string;
  isManuallyLayouted?: boolean;
  points: IPoint[];
}
```

### Mapping rules (ClassDiagram)

- Each v3 `Class`/`AbstractClass`/`Interface`/`Enumeration` element →
  one v4 node with `type: 'class'`, `data.attributes` rebuilt from the
  v3 `attributes: string[]` lookups against `model.elements[attrId]`, and
  similarly for `methods`. The child elements `ClassAttribute` /
  `ClassMethod` are **not** emitted as v4 nodes — they collapse into rows
  on the parent.
- Position: `node.position = { x: v3.bounds.x, y: v3.bounds.y }`,
  `node.width = v3.bounds.width`, `node.height = v3.bounds.height`,
  `node.measured = { width: v3.bounds.width, height: v3.bounds.height }`.
- `Package` v3 elements stay as nodes with `type: 'package'`; child
  classes that had `owner: <packageId>` get `parentId: <packageId>`.
- `ClassOCLConstraint` v3 elements: collapse onto their owner class as a
  row in `data.oclConstraints` if `owner` points to a class; otherwise
  emit a free-standing node with `type: 'class'` + `data.stereotype:
  'oclConstraint'` (rare, treat as fallback). Recommend: always collapse;
  drop free-standing OCL constraint nodes since the editor never produced
  them top-level in practice.
- Associations: `edge.source = v3.source.element`, `edge.target =
  v3.target.element`. Direction mapping uses
  `sourceHandle = v3.source.direction` and `targetHandle =
  v3.target.direction` (the strings `'Up'|'Down'|'Left'|'Right'` are
  preserved verbatim — React Flow accepts arbitrary handle ids).
- Roles & multiplicities lift directly into `edge.data.sourceRole` etc.
- `path` becomes `edge.data.points`. If `isManuallyLayouted` is set, copy
  it through; otherwise omit.
- Member name parsing: if a v3 `ClassAttribute.name` is `"+ counter:
  int"` and `attributeType` is undefined, fall back to
  `parseLegacyNameFormat` (see `utils/classifierMemberDisplay.ts`) — the
  migrator must accept both shapes. After migration, write the
  canonical separate fields.

---

## ObjectDiagram

### v3 element subtypes (`packages/editor/src/main/packages/uml-object-diagram/`)

- `ObjectName` (top-level container; carries optional `classId` linking to
  a class in a sibling ClassDiagram).
- `ObjectAttribute` (children, with optional `attributeId`).
- `ObjectMethod` (children).
- `ObjectIcon`.

### v3 relationship subtypes

- `ObjectLink` (with optional `associationId`).

### v4 node types

```ts
'objectName'

type ObjectNodeData = {
  name: string;                              // e.g. "myInstance: Customer"
  classId?: string;                          // link to ClassDiagram class
  attributes: ObjectAttribute[];
  methods: ObjectAttribute[];
  fillColor?: string; strokeColor?: string; textColor?: string;
  description?: string; assessmentNote?: string;
};

type ObjectAttribute = {
  id: string;
  name: string;            // "attribute = value" or just "attribute"
  attributeId?: string;    // link to ClassDiagram attribute
  attributeType?: string;
  defaultValue?: unknown;
};
```

### v4 edge types

`'ObjectLink'`

```ts
{
  name?: string;
  associationId?: string;
  sourceRole?: string;
  sourceMultiplicity?: string;
  targetRole?: string;
  targetMultiplicity?: string;
  points: IPoint[];
}
```

### Mapping rules (ObjectDiagram)

- v3 `ObjectName` → v4 `objectName` node, with `attributes`/`methods`
  collapsed in the same way as ClassDiagram members.
- v3 `ObjectIcon` is a standalone presentation element; collapse into the
  owning object's node as `data.icon: string`. Do not emit a separate v4
  node.
- `ObjectLink.associationId` survives end-to-end on `edge.data.associationId`.

---

## StateMachineDiagram

### v3 element subtypes (`packages/editor/src/main/packages/uml-state-diagram/`)

- `State` (container; has `bodies: string[]`, `fallbackBodies: string[]`,
  `stereotype: string | null`, `italic`, `underline`, `deviderPosition`,
  `hasBody`, `hasFallbackBody`).
- `StateBody`, `StateFallbackBody` (children of State).
- `StateCodeBlock` (free-floating code panel attached to a state).
- `StateActionNode`, `StateObjectNode`, `StateInitialNode`,
  `StateFinalNode`, `StateMergeNode`, `StateForkNode`,
  `StateForkNodeHorizontal`.

### v3 relationship subtypes

- `StateTransition` with `params: { [id]: string }` and `guard?: string`.

### v4 node types

```ts
'State'                       // container
'StateActionNode'
'StateObjectNode'
'StateInitialNode'
'StateFinalNode'
'StateMergeNode'
'StateForkNode'
'StateForkNodeHorizontal'
'StateCodeBlock'
```

(v4 collapses `StateBody` and `StateFallbackBody` into the parent state's
`data.bodies` / `data.fallbackBodies` arrays — they are never separate
React-Flow nodes in v4.)

```ts
type StateNodeData = {
  name: string;
  stereotype?: string | null;
  italic?: boolean;
  underline?: boolean;
  bodies: { id: string; name: string }[];
  fallbackBodies: { id: string; name: string }[];
  // Display-only (recomputed at render time): deviderPosition, hasBody, hasFallbackBody.
  description?: string;
  fillColor?: string; strokeColor?: string; textColor?: string;
  assessmentNote?: string;
};

type StateActionNodeData = {
  name: string;
  description?: string;
  fillColor?: string; strokeColor?: string; textColor?: string;
};

type StateObjectNodeData = {
  name: string;
  classId?: string;        // optional link to ClassDiagram if used
  description?: string;
};

// StateInitialNode, StateFinalNode, StateMergeNode, StateForkNode, StateForkNodeHorizontal
type StateMarkerNodeData = {
  name: string;            // usually empty
};

type StateCodeBlockData = {
  name: string;
  code: string;            // Python BAL code
  language?: 'python' | 'bal';
};
```

### v4 edge types

`'StateTransition'`

```ts
{
  name?: string;            // edge label
  guard?: string;           // optional guard expression in [brackets]
  params: { [key: string]: string };  // ordered dictionary of parameters; v3 stored as { '0', '1', ... }
  points: IPoint[];
}
```

### Mapping rules (StateMachineDiagram)

- A v3 `State` and its child `StateBody`/`StateFallbackBody` elements
  collapse: every `StateBody` whose `owner === stateId` becomes an entry
  in `data.bodies` (preserving id + name). Same for fallback bodies. The
  child elements **do not** emit v4 nodes.
- All other state element types each map 1:1 to a v4 node with the same
  type name (no rename).
- A `StateCodeBlock` is its own node in v4 (was not parented in v3).
- `StateTransition.params` is normalized to a dict in v4 just like v3 —
  pass through. If a legacy diagram stored params as `string` or
  `string[]`, the migrator coerces to dict (`{ '0': v }` or
  `{ '0': v[0], '1': v[1], ... }`).
- Inheritance/initial relationships: `StateInitialNode → State` is
  represented by a regular `StateTransition` whose source is the initial
  node — there is no separate edge type.

---

## AgentDiagram

### v3 element subtypes (`packages/editor/src/main/packages/agent-state-diagram/`)

The agent diagram **inherits all StateMachine element types** plus its own:

- Inherited: `State`, `StateBody`, `StateFallbackBody`, `StateActionNode`,
  `StateFinalNode`, `StateForkNode`, `StateForkNodeHorizontal`,
  `StateInitialNode`, `StateMergeNode`, `StateObjectNode`,
  `StateCodeBlock`.
- Agent-specific:
  - `AgentState` — extends `State` with `replyType: string`.
  - `AgentStateBody`, `AgentStateFallbackBody`.
  - `AgentIntent` — has `bodies: string[]` and `intent_description: string`.
  - `AgentIntentBody`, `AgentIntentDescription`,
    `AgentIntentObjectComponent`.
  - `AgentRagElement` — extends UMLElement with optional
    `ragDatabaseName`, `dbSelectionType`, `dbCustomName`, `dbQueryMode`,
    `dbOperation`, `dbSqlQuery`.

### v3 relationship subtypes

- `AgentStateTransition` — see *Legacy AgentStateTransition shapes* below.
- `AgentStateTransitionInit` — initial-state marker edge.

### v4 node types (agent diagram)

```ts
// State + StateMachine inherited types are reused as-is.
'AgentState'
'AgentIntent'
'AgentRagElement'
```

(`AgentStateBody` / `AgentStateFallbackBody` / `AgentIntentBody` /
`AgentIntentDescription` / `AgentIntentObjectComponent` collapse into the
parent's `data` exactly like StateBody — v3-only sub-elements that never
emit v4 nodes.)

```ts
type AgentStateNodeData = StateNodeData & {
  replyType: string;        // 'text' | 'image' | 'json' | 'llm' | ...
};

type AgentIntentNodeData = {
  name: string;
  intent_description: string;
  bodies: { id: string; name: string }[];      // intent body rows (training utterances)
  description?: string;
  fillColor?: string; strokeColor?: string; textColor?: string;
};

type AgentRagElementNodeData = {
  name: string;
  ragDatabaseName?: string;
  dbSelectionType?: string;       // 'predefined' | 'custom'
  dbCustomName?: string;
  dbQueryMode?: string;           // 'sql' | 'natural_language'
  dbOperation?: string;
  dbSqlQuery?: string;
};
```

### v4 edge types (agent diagram)

`'AgentStateTransition' | 'AgentStateTransitionInit'`

`AgentStateTransitionInit` carries no extra payload beyond `points`.

`AgentStateTransition` is the most complex edge in the migration. The v4
`edge.data` shape is the **canonical** form:

```ts
type AgentStateTransitionData = {
  name?: string;
  params: { [key: string]: string };
  transitionType: 'predefined' | 'custom';
  predefined?: {
    predefinedType: string;             // e.g. 'when_intent_matched', 'when_no_intent_matched', 'auto', 'when_variable_operation_matched', 'when_file_received', 'custom_transition'
    intentName?: string;                // for when_intent_matched
    fileType?: string;                  // for when_file_received
    conditionValue?:
      | string
      | { variable: string; operator: string; targetValue: string };
  };
  custom?: {
    event:
      | 'None'
      | 'DummyEvent'
      | 'WildcardEvent'
      | 'ReceiveMessageEvent'
      | 'ReceiveTextEvent'
      | 'ReceiveJSONEvent'
      | 'ReceiveFileEvent';
    condition: string[];
  };
  points: IPoint[];
};
```

**`predefined` is filled when `transitionType === 'predefined'` and `custom`
is filled when `transitionType === 'custom'`.** The migrator always emits
the canonical shape and strips the legacy flat fields.

### Legacy AgentStateTransition shapes (must round-trip)

The v3 deserializer at
`packages/editor/src/main/packages/agent-state-diagram/agent-state-transition/agent-state-transition.ts`
accepts at least 5 historical shapes. The migrator must collapse them all
to the canonical v4 shape above. Reference fixtures:

#### 1. Canonical predefined (current writer output)

```json
{
  "transitionType": "predefined",
  "predefined": { "predefinedType": "when_intent_matched", "intentName": "greet" },
  "custom": { "condition": [] }
}
```

→ v4: keep `predefined` as-is; drop `custom`.

#### 2. Canonical custom (current writer output)

```json
{
  "transitionType": "custom",
  "predefined": { "predefinedType": "" },
  "custom": { "event": "ReceiveTextEvent", "condition": ["len(msg) > 0"] }
}
```

→ v4: keep `custom` as-is; drop `predefined`.

#### 3. Legacy flat predefined (pre-2024)

```json
{
  "predefinedType": "when_variable_operation_matched",
  "variable": "score",
  "operator": ">=",
  "targetValue": "10"
}
```

→ v4:

```json
{
  "transitionType": "predefined",
  "predefined": {
    "predefinedType": "when_variable_operation_matched",
    "conditionValue": { "variable": "score", "operator": ">=", "targetValue": "10" }
  }
}
```

#### 4. Legacy flat custom (`condition` was a string)

```json
{
  "condition": "custom_transition",
  "customEvent": "WildcardEvent",
  "customConditions": ["x == 1"]
}
```

→ v4:

```json
{
  "transitionType": "custom",
  "custom": { "event": "WildcardEvent", "condition": ["x == 1"] }
}
```

#### 5. Legacy nested `conditionValue.events`/`conditions`

```json
{
  "transitionType": "custom",
  "conditionValue": {
    "events": ["ReceiveMessageEvent"],
    "conditions": ["msg == 'hi'"]
  }
}
```

→ v4:

```json
{
  "transitionType": "custom",
  "custom": { "event": "ReceiveMessageEvent", "condition": ["msg == 'hi'"] }
}
```

#### 6. Legacy `predefinedType: 'when_file_received'` (file selector)

```json
{
  "predefinedType": "when_file_received",
  "fileType": "image/png"
}
```

→ v4:

```json
{
  "transitionType": "predefined",
  "predefined": {
    "predefinedType": "when_file_received",
    "fileType": "image/png"
  }
}
```

The Python normalizer (`json_to_buml/agent_diagram_processor.py`) and the TS
migrator must implement identical fallthrough order:

1. If `transitionType === 'custom'` **or** legacy `condition === 'custom_transition'`
   **or** `custom.event` non-empty/`custom.condition` non-empty: emit
   `transitionType: 'custom'` with `custom` filled.
2. Else: emit `transitionType: 'predefined'` with `predefined` filled.
3. Inside `predefined`, the type comes from
   `predefined.predefinedType ?? predefinedType ?? (legacy condition string) ?? 'when_intent_matched'`.
4. `intentName` / `variable`/`operator`/`targetValue` / `fileType` extracted
   per the per-type rules above.

### Mapping rules (AgentDiagram)

- `AgentState` and `AgentIntent` collapse their body children just like
  `State` collapses `StateBody`. `AgentRagElement` has no children to
  collapse.
- `replyType` defaults to `'text'` when missing.
- `AgentIntent.intent_description` defaults to `''`.
- `AgentRagElement` may appear inside an `AgentState` as a child via
  `owner` — preserve that as `parentId`.

---

## UserDiagram

### v3 element subtypes (`packages/editor/src/main/packages/user-modeling/`)

- `UserModelName` (top-level user node; like ObjectName).
- `UserModelAttribute`.
- `UserModelIcon`.

### v3 relationship subtypes

- `UserModelLink`.

### v4 node types

```ts
'UserModelName'

type UserModelNameData = {
  name: string;                  // user identifier (e.g. "Alice: Customer")
  attributes: UserModelAttribute[];
  description?: string;
  fillColor?: string; strokeColor?: string; textColor?: string;
  icon?: string;
};

type UserModelAttribute = {
  id: string;
  name: string;
  attributeType?: string;
  defaultValue?: unknown;
};
```

### v4 edge types

`'UserModelLink'`

```ts
{
  name?: string;
  points: IPoint[];
}
```

### Mapping rules (UserDiagram)

- `UserModelAttribute` and `UserModelIcon` v3 children collapse into the
  parent `UserModelName` data — same shape as ObjectDiagram.
- The reference metamodel JSON (`usermetamodel_buml_short.json`) is bundled
  in the new lib at `services/userMetaModel/usermetamodel.json` — backend
  remains the OCL validation authority.

---

## NNDiagram

This is the largest single transformation: **v3 represents each layer
attribute as its own UMLElement** (e.g. `NameAttributeConv2D`,
`KernelDimAttributeConv2D`, …), all owned by the layer; **v4 collapses
every layer's attributes into `node.data.attributes: Record<string,
unknown>`**.

### v3 element subtypes (`packages/editor/src/main/packages/nn-diagram/`)

Layer types:

- `Conv1DLayer`, `Conv2DLayer`, `Conv3DLayer`
- `PoolingLayer`
- `RNNLayer`, `LSTMLayer`, `GRULayer`
- `LinearLayer`
- `FlattenLayer`
- `EmbeddingLayer`
- `DropoutLayer`
- `LayerNormalizationLayer`
- `BatchNormalizationLayer`
- `TensorOp`
- `Configuration`
- `TrainingDataset`, `TestDataset`

Attribute element types: see
`packages/editor/src/main/packages/nn-diagram/index.ts` for the full
enumeration; per layer kind there are 3–13 attribute element types whose
names start with the attribute slug and end in the layer slug, e.g.
`NameAttributeConv2D`, `KernelDimAttributeConv2D`,
`InputReusedAttributeConv2D`.

Section helper element types (`NNSectionTitle`, `NNSectionSeparator`) and
container types (`NNContainer`, `NNReference`).

### v3 relationship subtypes

- `NNNext` — sequential layer flow (unidirectional with "next" label).
- `NNComposition` — diamond on container side.
- `NNAssociation` — dataset ↔ container.

### v4 node types

```ts
'Conv1DLayer' | 'Conv2DLayer' | 'Conv3DLayer'
'PoolingLayer'
'RNNLayer' | 'LSTMLayer' | 'GRULayer'
'LinearLayer'
'FlattenLayer' | 'EmbeddingLayer' | 'DropoutLayer'
'LayerNormalizationLayer' | 'BatchNormalizationLayer'
'TensorOp'
'Configuration'
'TrainingDataset' | 'TestDataset'
'NNContainer' | 'NNReference'
```

```ts
type NNLayerNodeData = {
  name: string;                                 // layer instance name
  attributes: Record<string, unknown>;          // per-layer attribute schema (see widget config)
  description?: string;
  fillColor?: string; strokeColor?: string; textColor?: string;
  assessmentNote?: string;
};

type NNContainerNodeData = {
  name: string;                                 // model name
};

type NNReferenceNodeData = {
  name: string;
  referenceTarget?: string;                     // id of referenced container, if any
};
```

The keys used in `attributes` follow the **attribute slug** of the v3 element
type (without the layer suffix), normalized to `snake_case`:

| v3 element type                       | v4 attribute key      |
|---------------------------------------|-----------------------|
| `NameAttributeConv2D`                 | `name`*               |
| `KernelDimAttributeConv2D`            | `kernel_dim`          |
| `OutChannelsAttributeConv2D`          | `out_channels`        |
| `StrideDimAttributeConv2D`            | `stride_dim`          |
| `InChannelsAttributeConv2D`           | `in_channels`         |
| `PaddingAmountAttributeConv2D`        | `padding_amount`      |
| `PaddingTypeAttributeConv2D`          | `padding_type`        |
| `ActvFuncAttributeConv2D`             | `actv_func`           |
| `NameModuleInputAttributeConv2D`      | `name_module_input`   |
| `InputReusedAttributeConv2D`          | `input_reused`        |
| `PermuteInAttributeConv2D`            | `permute_in`          |
| `PermuteOutAttributeConv2D`           | `permute_out`         |
| `HiddenSizeAttribute*`                | `hidden_size`         |
| `ReturnTypeAttribute*`                | `return_type`         |
| `InputSizeAttribute*`                 | `input_size`          |
| `BidirectionalAttribute*`             | `bidirectional`       |
| `DropoutAttribute*` (within RNN/LSTM) | `dropout`             |
| `BatchFirstAttribute*`                | `batch_first`         |
| `RateAttributeDropout`                | `rate`                |
| `OutFeaturesAttributeLinear`          | `out_features`        |
| `InFeaturesAttributeLinear`           | `in_features`         |
| `StartDimAttributeFlatten`            | `start_dim`           |
| `EndDimAttributeFlatten`              | `end_dim`             |
| `NumEmbeddingsAttributeEmbedding`     | `num_embeddings`      |
| `EmbeddingDimAttributeEmbedding`      | `embedding_dim`       |
| `NormalizedShapeAttributeLayerNormalization` | `normalized_shape` |
| `NumFeaturesAttributeBatchNormalization`     | `num_features`     |
| `DimensionAttributePooling`           | `dimension`           |
| `DimensionAttributeBatchNormalization`| `dimension`           |
| `PoolingTypeAttributePooling`         | `pooling_type`        |
| `OutputDimAttributePooling`           | `output_dim`          |
| `TnsTypeAttributeTensorOp`            | `tns_type`            |
| `ConcatenateDimAttributeTensorOp`     | `concatenate_dim`     |
| `LayersOfTensorsAttributeTensorOp`    | `layers_of_tensors`   |
| `ReshapeDimAttributeTensorOp`         | `reshape_dim`         |
| `TransposeDimAttributeTensorOp`       | `transpose_dim`       |
| `PermuteDimAttributeTensorOp`         | `permute_dim`         |
| `BatchSizeAttributeConfiguration`     | `batch_size`          |
| `EpochsAttributeConfiguration`        | `epochs`              |
| `LearningRateAttributeConfiguration`  | `learning_rate`       |
| `OptimizerAttributeConfiguration`     | `optimizer`           |
| `LossFunctionAttributeConfiguration`  | `loss_function`       |
| `MetricsAttributeConfiguration`       | `metrics`             |
| `WeightDecayAttributeConfiguration`   | `weight_decay`        |
| `MomentumAttributeConfiguration`      | `momentum`            |
| `PathDataAttributeDataset`            | `path_data`           |
| `TaskTypeAttributeDataset`            | `task_type`           |
| `InputFormatAttributeDataset`         | `input_format`        |
| `ShapeAttributeDataset`               | `shape`               |
| `NormalizeAttributeDataset`           | `normalize`           |
| `NameAttributeDataset`                | `name`*               |

\* The `Name*` v3 attribute element holds the **same** value as the layer
element's `name` field. The migrator should prefer the more recently set
value if they diverge; otherwise just copy the layer's `name` and drop the
attribute element.

The full attribute schema and validation defaults live at
`packages/editor/src/main/packages/nn-diagram/nn-attribute-widget-config.ts`
and `nn-validation-defaults.ts`. SA-5 (NNDiagram) ports these verbatim
into `packages/library/lib/nodes/nnDiagram/nnAttributeWidgetConfig.ts`
and `nnValidationDefaults.ts` as data-only modules.

### v4 edge types

`'NNNext' | 'NNComposition' | 'NNAssociation'`

```ts
type NNEdgeData = {
  name?: string;
  points: IPoint[];
};
```

### Mapping rules (NNDiagram)

- For each v3 layer element, walk `model.elements` for siblings whose
  `owner === layerId` and whose `type` is one of the per-layer attribute
  element types. Drop them from the v4 node list, accumulate them into the
  parent's `data.attributes` keyed by the snake_case slug above. Their
  `value` field becomes the value (string).
- For boolean attributes (`'true'`/`'false'`), normalize to JS `boolean`
  in v4 — the widget config's `BOOLEAN_OPTIONS` is the source of truth for
  which keys are boolean.
- For numeric attributes (e.g. `out_channels`, `kernel_dim`,
  `learning_rate`), keep as **strings** in v4 to match the widget which
  edits them as text — Python codegen parses with `int(...)` / `float(...)`
  exactly as today.
- Layers nested under an `NNContainer` get `parentId: <containerId>`.
- `NNReference` retains its `referenceTarget` as plain data; no edge
  rewrite required.
- v3 `NNSectionTitle` and `NNSectionSeparator` are sidebar-only helpers —
  the migrator drops them entirely.

---

## Conversion direction guarantees

Two directions, both must be implemented:

- **Frontend (TS)**: `migrate-uml-v3-to-v4.ts` reads a v3 model and emits a
  v4 model. Pure, no IO. Same fallthrough rules as the Python normalizer.
- **Backend (Python)**: `json_to_buml/<diagram>_diagram_processor.py`
  parses v4 directly. `buml_to_json/<diagram>_diagram_converter.py` emits
  v4 directly. **No v3 emission path** anywhere on the backend after Wave
  3 (the old fork is deleted).

Round-trip tests must hold for **every diagram type** for at least:

1. v4 fixture → backend BUML → backend re-emit v4 → structural diff = 0.
2. v3 fixture → TS migrator → v4 → backend BUML → frozen golden diff = 0.

---

## Open questions surfaced during spec authoring

These are flagged for user resolution before Wave-2 fan-out so SA-2..6
don't have to invent assumptions:

1. **`UserDiagram` references** — does a `UserModelName` carry a `classId`
   linkage like `ObjectName` does? Source `uml-user-model-name.ts` would
   resolve this; if not yet defined, recommend adding `classId?: string`
   for parity with `ObjectName`.
2. **NN attribute keys for ambiguous slugs** — e.g. `DimensionAttribute*`
   appears on both Pooling and BatchNormalization. The mapping table above
   uses `dimension` for both; SA-5 should confirm by inspection that no
   layer carries two `dimension` attributes that differ in meaning.
3. **`ClassOCLConstraint` collapse policy** — recommended above to
   collapse onto the owner class. Confirm this matches how the backend
   processor (`class_diagram_processor.py`) currently consumes them; if
   the backend reads them as standalone elements with cross-references,
   the v4 spec needs a top-level `oclConstraint` node type instead.
4. **`StateObjectNode` cross-diagram reference** — does it carry a
   `classId` linking to a sibling ClassDiagram (similar to `ObjectName`)?
   The above schema includes the field as optional; SA-3 should confirm.
5. **`AgentRagElement.dbCustomName` vs `ragDatabaseName`** — both fields
   exist in the v3 typings; the migrator preserves both verbatim. SA-4
   should confirm the runtime semantics so the BAF generator picks the
   correct one.

If any of the above is answered "different from spec", patch the spec
**before** the Wave 2 sub-agents read it. The hand-off contract is the
spec, not the implementation.

---

## Visual deviations from v3

### NN layer cards (SA-2.2 #34)

**v3:** `NNBaseLayer` (and every concrete layer kind it spawned —
Conv1D / Conv2D / Conv3D / Pooling / RNN / LSTM / GRU / Linear /
Flatten / Embedding / Dropout / LayerNormalization /
BatchNormalization / TensorOp / Configuration / TrainingDataset /
TestDataset) rendered as a fixed **110×110 px** card with a custom
SVG glyph centred in the body and `hasAttributes = false` so no row
section was drawn. The icon glyphs lived under
`packages/editor/src/main/packages/nn-diagram/nn-layer-icon/` and
were imported per layer kind.

**v4:** `_NNLayerBase.tsx` renders a generic `«KindLabel»` stereotype
card with the layer's `name` underneath, a horizontal separator under
the header, and a default fill colour per kind. The card auto-resizes
(min 120×50) and is consistent with SA-3's State / SA-4's Agent
visuals, so the inspector experience is uniform across every diagram
type.

**Decision: retire the v3 fixed-icon visual.** The SA-PARITY-2 audit
flagged the change (round 2 #34) as MEDIUM. Per the SA-2.2 brief,
visual restoration is in scope only if the v3 SVG icons port cleanly
within ~2 hours; the icon directory contains 18 hand-tuned SVG
symbols that would need re-export to React component + theme-aware
fill threading. SA-2.2 explicitly retires the icon look in favour of
the stereotype-card style for these reasons:

- **Uniformity** — every other diagram type in the v4 lib uses the
  stereotype-card pattern; introducing a per-diagram icon view would
  reintroduce SA-2's pre-spec hot path of bespoke per-kind renderers.
- **Authoring affordance** — the v3 110×110 card hid the layer's
  `name` from the canvas (only the kind label was visible). The v4
  card surfaces both, which is the dominant authoring affordance the
  v3 inspector users asked for.
- **Theme fidelity** — v3 icons baked the stroke colour into the SVG
  source. The stereotype-card style respects `data.fillColor` /
  `data.strokeColor` / `data.textColor` end-to-end through
  `getCustomColorsFromData`, so theme + per-node overrides work the
  same way they do for class / agent / state cards.

If a future contributor wants to restore the icon-view as an opt-in
toggle (à la SA-2.1's `showIconView` for ObjectName), the SA-2.2
inspector schema and v3 ↔ v4 round-trip are unaffected — the icons
would be a render-time-only enhancement on top of `_NNLayerBase.tsx`.

### Header underline on UserModelName (SA-2.2 #35)

`HeaderSection.tsx` now applies `textDecoration="underline"` on both
the parent `<text>` element and the inner name `<tspan>`.
Chromium-based browsers historically dropped underline on tspans when
the parent had a `dy` offset (which happens for the SA-2 / SA-4 cards
with a stereotype line above the name); explicit duplication on the
tspan keeps both the ObjectName and UserModelName headers correctly
underlined regardless of browser quirks.
