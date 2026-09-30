# BESSER UML v4 Wire Shape Specification (v3 Legacy Notes)

**Status**: Canonical spec for the v4 React Flow editor.
**Audience**: every contributor touching `packages/library` (frontend nodes/edges)
or `besser/utilities/web_modeling_editor/backend/services/converters/` (Python).
**Source of truth**: this document.

This file describes, per BESSER diagram type, the **v4** shape that the
React-Flow library at `packages/library` emits and accepts as the canonical
on-disk / on-wire format. For backward compatibility it also documents the
legacy **v3** shape that the retired SVG/Redux editor used to emit; the
v3→v4 migrator at `packages/library/lib/utils/versionConverter.ts`
upgrades old fixtures to v4 on load.

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
  nodes: BesserNode[];
  edges: BesserEdge[];
  interactive?: { elements: Record<string, boolean>; relationships: Record<string, boolean> };
  assessments: { [id: string]: Assessment };
};

type BesserNode = {
  id: string;
  type: DiagramNodeType;       // string union, see per-diagram sections
  position: { x: number; y: number };
  width: number;
  height: number;
  measured: { width: number; height: number };
  data: Record<string, unknown>;
  parentId?: string;           // replaces v3 owner
};

type BesserEdge = {
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

### v3 element subtypes (ClassDiagram)

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
  name: string;                     // bare identifier; method rows do NOT fuse the signature in here
  attributeType: string;            // canonical Python-style: 'str','int','float','bool','date','datetime','time','any', or custom
  visibility: 'public' | 'private' | 'protected' | 'package';
  code?: string;
  implementationType?: 'none' | 'code' | 'bal' | 'state_machine' | 'quantum_circuit' | 'neural_network';
  stateMachineId?: string;
  quantumCircuitId?: string;
  neuralNetworkId?: string;         // method rows with implementationType 'neural_network': id of the
                                    // project NNDiagram that implements the method (a network name is
                                    // also accepted, for models imported from BUML)
  isOptional?: boolean;
  isDerived?: boolean;
  isId?: boolean;
  isExternalId?: boolean;
  defaultValue?: unknown;
  // Method rows only — structured signature (the inspector mirrors
  // returnType onto attributeType; 'any' means "no explicit return type":
  // the backend reads it as an untyped method (Method.type = None), and
  // buml_to_json writes 'any' back for one, so JSON -> B-UML -> JSON is stable).
  parameters?: { id: string; name: string; parameterType?: string; defaultValue?: unknown }[];
  returnType?: string;
};
```

`stereotype` is encoded as a single string (capitalized canonical forms —
the frontend compares case-sensitively; readers stay case-insensitive for
legacy lowercase data). Map v3 element `type`:

| v3 `type`      | v4 `node.type` | v4 `data.stereotype` |
|----------------|----------------|----------------------|
| `Class`        | `class`        | `null`               |
| `AbstractClass`| `class`        | `'Abstract'`         |
| `Interface`    | `class`        | `'Interface'`        |
| `Enumeration`  | `class`        | `'Enumeration'`      |
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
  sourceNavigable?: boolean;        // per-end navigability (associations, aggregations, compositions)
  targetNavigable?: boolean;
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
- `ClassOCLConstraint`: the canonical v4 shape is a free-standing node
  with `type: 'ClassOCLConstraint'` (sticky-note rendering;
  `data.expression` carries the full `context …` OCL text,
  `data.description` the optional natural-language note), tethered to its
  anchoring class by a visual `ClassOCLLink` edge. This is the shape the
  editor authors (`ClassOCLConstraintEditPanel`) and the shape the
  backend converter always emits — including for class invariants and
  method pre/post conditions, mirroring the develop baseline where
  constraints are always visible boxes. The v3→v4 migrator may still
  collapse owner-linked constraints onto `data.oclConstraints` rows; the
  backend processor accepts both shapes on ingest (plus the legacy
  `type: 'class'` + `data.stereotype: 'oclConstraint'` fallback) and the
  context class is re-derived from the OCL text itself, with the
  `ClassOCLLink` edge resolving owners for legacy body-only rows.
  Legacy body-only constraints (v3 `constraint` holding just the body plus
  `kind: 'invariant' | 'precondition' | 'postcondition'`) keep their v3
  metadata as `data.constraintName` (the name used for the synthesised
  `context …` header) and `data.targetMethodId` (pre/post only: the target
  method row id) — on the node's `data` or on the `data.oclConstraints` row.
  Without them a pre/post body-only row cannot be resolved (`unknown_method`).
  The backend processor reads both fields from either place: an invariant
  keeps `constraintName` as its constraint name, and a pre/post condition
  attaches to the method whose row `id` equals `targetMethodId` (its header
  is synthesised from that method's signature). BOCL names only
  invariants, so pre/post conditions get generated names
  (`<method>_pre_<n>_<m>`), as they did before v4.
- Associations: `edge.source = v3.source.element`, `edge.target =
  v3.target.element`. Direction mapping uses
  `sourceHandle = v3.source.direction` and `targetHandle =
  v3.target.direction` (the strings `'Up'|'Down'|'Left'|'Right'` are
  preserved verbatim — React Flow accepts arbitrary handle ids).
- Roles & multiplicities lift directly into `edge.data.sourceRole` etc.
- Navigability: v3 `source.navigable` / `target.navigable` lift into
  `edge.data.sourceNavigable` / `edge.data.targetNavigable`. A plain
  association is always `ClassBidirectional`; `ClassUnidirectional` is
  **legacy** and is read as `ClassBidirectional` with
  `sourceNavigable: false, targetNavigable: true` (by the TS migrators and
  by the backend processor). When the flags are absent the legacy default
  applies (`ClassUnidirectional` → source non-navigable, everything else →
  both navigable); a present but non-boolean value falls back to that
  default with a warning. Rules enforced on ingest (corrected with a
  warning, never a hard failure): at least one end is navigable, and the
  part (source) end of a composition is navigable — the composite (whole,
  diamond) end is the target. BUML → JSON always emits `ClassBidirectional`
  (or `ClassComposition`) with both flags, keeping the drawn orientation
  (the processor records the source role in the layout side-channel).
- Association-end names are deduplicated per owning class (the class at the
  opposite end): a colliding role gets a `_1`, `_2`, … suffix instead of
  failing the conversion.
- `path` becomes `edge.data.points`. If `isManuallyLayouted` is set, copy
  it through; otherwise omit.
- Member name parsing: if a v3 `ClassAttribute.name` is `"+ counter:
  int"` and `attributeType` is undefined, fall back to
  `parseLegacyNameFormat` (see `utils/classifierMemberDisplay.ts`) — the
  migrator must accept both shapes. After migration, write the
  canonical separate fields.

---

## ObjectDiagram

### v3 element subtypes (ObjectDiagram)

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

### v3 element subtypes (StateMachineDiagram)

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

### v3 element subtypes (AgentDiagram)

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
// Canvas node types. State + StateMachine inherited types are reused as-is.
'AgentState'   // standard and reasoning states (data.stateType)
'comment'      // tethered to states by 'CommentLink' edges
```

`AgentIntent`, `AgentRagElement`, `AgentLLM`, `AgentTool`, `AgentSkill`,
`AgentWorkspace` (and `AgentGUI`) are **no longer v4 node types**: they are
off-canvas *components* stored in the model's top-level `components` map (see
*Components* below). Nodes of those types are accepted as **legacy input
only** (React Flow models saved before the Components page) and are migrated
into `components` on load by the library (`normalizeAgentComponents`), the
webapp and the backend processor. `AgentReasoningState` is likewise a legacy
input type, read as `AgentState` + `stateType: 'reasoning'`.

(`AgentStateBody` / `AgentStateFallbackBody` collapse into the state's
`data.bodies` / `data.fallbackBodies` rows exactly like StateBody — v3-only
sub-elements that never emit v4 nodes.)

```ts
type AgentStateNodeData = {
  name: string;
  initial?: boolean;                 // exactly one state carries true (replaces StateInitialNode + init edge)
  stateType?: 'standard' | 'reasoning';
  bodies: AgentActionRow[];          // ordered: executed in this order
  fallbackBodies: AgentActionRow[];
  fallbackBodyEnabled?: boolean;     // absent = true
  // reasoning states only
  llm_name?: string; max_steps?: number; enable_task_planning?: boolean;
  stream_steps?: boolean; system_prompt?: string; fallback_message?: string;
  fillColor?: string; strokeColor?: string; textColor?: string;
};

// One action of a state body. Field names are the old editor's AgentStateMember
// serialization verbatim; only the fields of the row's action type are required.
type AgentActionRow = {
  id: string;
  name: string;                      // text of a text reply / source of a code row / display label otherwise
  replyType: 'text' | 'llm' | 'llm_chat' | 'rag' | 'db_reply' | 'code' | 'web_crawl_llm'
    | 'ws_markdown' | 'ws_html' | 'ws_speech' | 'ws_options' | 'ws_location'
    | 'ws_file' | 'ws_image' | 'ws_dataframe' | 'ws_plotly' | 'gui_reply';
  actionType?: string;               // metamodel class name, e.g. 'LLMReplyAction' (writer emits both; reader prefers actionType)
  code?: string;                     // code rows: the source (reader falls back to name)
  llm_name?: string;                 // registered LLM by name; '' = agent default
  system_message?: string;           // llm / llm_chat system prompt
  ragDatabaseName?: string; prompt?: string;                      // rag
  dbSelectionType?: string; dbCustomName?: string; dbQueryMode?: string;
  dbOperation?: string; dbSqlQuery?: string;                      // db_reply
  // prompt customisation & session data flow
  inputPromptMode?: 'last_user_message' | 'custom';               // llm, rag, db_reply
  customInputPrompt?: string; customInputPromptUseSessionVars?: boolean;
  systemPromptUseSessionVars?: boolean;                           // llm, llm_chat
  promptUseSessionVars?: boolean;                                 // rag
  useSessionVars?: boolean;                                       // text, ws_markdown, ws_html, ws_speech
  storeInSession?: string;           // session variable receiving the generated answer
  sendReply?: boolean;               // default true; false = silent (store only)
  // web_crawl_llm
  initial_url?: string; max_depth?: number; max_pages?: number; crawl_format?: string;
  base_url_prefix?: string; run_crawl?: boolean; no_crawl_error_message?: string;
  system_message_prefix?: string; systemMessagePrefixUseSessionVars?: boolean;
  // websocket replies
  ws_message?: string; ws_audio_speed?: number | null; ws_options?: string; // options: one per line
  ws_latitude?: number; ws_longitude?: number;
  guiId?: string;                    // gui_reply: gui_id of an AgentGUI component
};
```

`replyType` ↔ `actionType`: text↔TextReplyAction, llm↔LLMReplyAction,
llm_chat↔LLMChatAction, rag↔RAGReplyAction, db_reply↔DBAction,
code↔CustomCodeAction, web_crawl_llm↔WebCrawlLLMAction,
ws_markdown/ws_html/ws_speech/ws_options/ws_location/ws_file/ws_image/ws_dataframe/ws_plotly
↔ WebSocketReply{Markdown,HTML,Speech,Options,Location,File,Image,Dataframe,Plotly}Action,
gui_reply↔GUIReplyAction. Legacy rows: an `llm` / `llm_chat` row without
`system_message` had its prompt on `name` (ignored when it is the
placeholder `"AI response 🪄"`).

### Components (agent diagram)

```ts
type AgentDiagramModel = UMLModel & {
  components?: { [id: string]: AgentComponent };   // sibling of nodes / edges
  config?: { default_llm_name?: string; [key: string]: unknown };
};

// Flat entry, no geometry (bounds / position). Identical to the v3
// smart-generator `model.components` entry, so v3 and v4 share it verbatim.
type AgentComponent = { id: string; type: AgentComponentType; name: string; owner?: string | null } & TypeFields;
```

| `type` | fields (defaults) |
|---|---|
| `AgentLLM` | `provider` (`'openai'`), `parameters` (`{}`), `num_previous_messages` (1), `global_context` (`''`) |
| `AgentIntent` | `intent_description` (`''`), `bodies: string[]` — ordered ids of its `AgentIntentBody` entries |
| `AgentIntentBody` | `name` = one training sentence, `owner` = intent id |
| `AgentRagElement` | `llm_name` (`''` = default LLM), `llm_prompt`, `k` (4), `num_previous_messages` (0), `embedding_provider` (`'openai'` \| `'ollama'`), `embedding_base_url`, `embedding_model` (Ollama defaults `http://localhost:11434` / `nomic-embed-text`), `use_hybrid_rag` (false), `bm25_weight` (0.6, 0 < w < 1) |
| `AgentTool` | `description`, `code` |
| `AgentSkill` | `description`, `content` |
| `AgentWorkspace` | `path`, `description`, `writable` (true), `max_read_bytes` (200000) |
| `AgentGUI` | `gui_id`, `persist` (true), `width` (`''`), `is_form` (false), `guiModel` (GrapesJS GUI JSON, or `null` = not designed yet) |

Canvas → component references are **by name** (`intentName`, `llm_name`,
`ragDatabaseName`) or by **`gui_id`** (`guiId`, `formGuiId`, `guiEventGuiId`).

Legacy locations, merged into `components` on read (later wins; the
canonical `model.components` always wins):
1. v4 `nodes` of a component type — `node.data` is flattened onto the entry;
   an intent's `data.training_phrases` (or `data.bodies`) rows `{id, name}`
   become `AgentIntentBody` entries owned by the intent. Edges touching a
   migrated node are dropped.
2. v3 `model.elements` of a component type (backend processor only).
3. Diagram-level (`ProjectDiagram.agentComponents`) and model-level
   `model.agentComponents` maps (an early components-panel build).

`buml_to_json` emits components **only** in `model.components`, never as nodes.

### v4 edge types (agent diagram)

`'AgentStateTransition' | 'CommentLink'`

(`AgentStateTransitionInit` is legacy input only: the initial state is the
`data.initial` flag.)

`AgentStateTransition` is the most complex edge in the migration. The v4
`edge.data` shape is the **canonical** form:

```ts
type AgentStateTransitionData = {
  name?: string;
  params?: { [key: string]: string };
  transitionType: 'predefined' | 'custom';
  predefined?: {
    predefinedType: string;             // 'when_intent_matched' | 'when_no_intent_matched' | 'auto' | 'when_variable_operation_matched' | 'when_file_received' | 'when_form_submitted'
    intentName?: string;                // for when_intent_matched
    fileType?: string;                  // for when_file_received: one or more comma-separated MIME types or extensions ('pdf, csv, image/png')
    formGuiId?: string;                 // for when_form_submitted: gui_id of a form AgentGUI ('' = any form)
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
      | 'ReceiveFileEvent'
      | 'GUIEvent';
    condition: string[];
    guiEventGuiId?: string;             // for GUIEvent: GUIEvent.message_id (gui_id of the AgentGUI)
  };
  points: IPoint[];
};
```

**`predefined` is filled when `transitionType === 'predefined'` and `custom`
is filled when `transitionType === 'custom'`.** The migrator always emits
the canonical shape and strips the legacy flat fields (legacy flat
`formGuiId` / `guiEventGuiId` are lifted into the blocks).

### Legacy AgentStateTransition shapes (must round-trip)

The legacy v3 deserializer accepted at least 5 historical shapes. The
migrator must collapse them all to the canonical v4 shape above. Reference
fixtures:

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

- `AgentState` collapses its body children into `data.bodies` /
  `data.fallbackBodies` rows just like `State` collapses `StateBody`.
- v3 component elements (`AgentIntent` + `AgentIntentBody`, `AgentLLM`,
  `AgentRagElement`, `AgentTool`, `AgentSkill`, `AgentWorkspace`,
  `AgentGUI`) and v3 `model.components` / `agentComponents` go to the v4
  `components` map verbatim (bounds stripped) — never to `nodes`.
- `StateInitialNode` + `AgentStateTransitionInit` fold into `data.initial`
  on the target state.
- `replyType` defaults to `'text'` when missing (or is derived from
  `actionType`).
- `AgentIntent.intent_description` defaults to `''`.

---

## UserDiagram

### v3 element subtypes (UserDiagram)

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

### v3 element subtypes (NNDiagram)

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

Attribute element types: per layer kind there are 3–13 attribute element
types in the legacy v3 shape, whose
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
  input_var?: string;                           // NN.input_var (forward-pass input variable)
  return_vars?: string[];                       // NN.return_vars, one entry per returned variable
                                                // (readers also accept the comma-separated string)
};

type NNReferenceNodeData = {
  name: string;
  referenceTarget?: string;                     // id of the referenced NNContainer node (the editor
                                                // writes the id; the backend emits the container
                                                // name — readers resolve id first, then name)
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

\* The `Name*` v3 attribute element holds the layer's **instance** name
(`conv1d_layer`, `l1`, …); the v3 layer element's own `name` is only the
palette label (`Conv1D Layer`). v4 keeps the instance name on both
`data.name` (what the canvas shows) and `data.attributes.name` (what
`json_to_buml/nn_diagram_processor.py` reads — it rejects a layer or dataset
without it). The migrator prefers the attribute value and falls back to the
layer name; the inspector keeps the two in sync. Default names are
lowercase identifiers (`conv1d_layer`, `pooling_layer`, `tensorop`,
`dataset`, …).

Extended attribute keys (smart-generator; all optional, stored as strings /
booleans like the table above):

| Layer kinds | v4 attribute keys |
|-------------|-------------------|
| every layer (inherited from `Layer`) | `is_layer_call`, `input_var`, `output_var` |
| Conv1D/2D/3D | `dilation`, `groups`, `bias` |
| Linear | `bias` |
| RNN / LSTM / GRU | `bias`, `hx_source`, `hidden_state_var`, `hidden_unused`, `hidden_subscript_source`, `hidden_subscript_target` |
| LSTM only | `cell_state_var`, `cell_unused` |
| Embedding | `padding_idx`, `permute_in`, `permute_out` |
| Dropout | `dimension` (plain key, `1D`/`2D`/`3D`), `permute_in`, `permute_out` |
| LayerNormalization | `eps`, `affine` |
| BatchNormalization | `eps`, `momentum`, `affine`, `track_running_stats`, `permute_in`, `permute_out` |
| TensorOp | `input_var`, `output_var`, `output_vars` (split), `permute_in`, `permute_out`, `reduce_dim`, `reduce_keepdims`, `shape_dim`, `actual_vars`, `subscript_indices` (JSON list of `{type: "index"\|"slice", …}`), `repeat_dim`, `interpolate_size`, `interpolate_scale`, `interpolate_mode`, `pad_amount` (`[[l, r], …]`), `pad_mode`, `pad_value`, `dropout_rate`, `dropout_training_aware`, `split_dim`, `split_sizes` |

`tns_type` takes the metamodel's `ALLOWED_TENSOR_OP_TYPES` (reshape,
concatenate, transpose, permute, multiply, matmultiply, split,
binop_add/subtract/multiply/divide/floor_divide, mean, max, squeeze,
unsqueeze, shape_dim, normalize, repeat, zeros_like, interpolate, pad,
dropout, subscript, identity). `layers_of_tensors` is a list literal whose
items are quoted module names (or `'INPUT'`, the network input) or bare
numeric literals for the binary ops, e.g. `['conv_1', 1.5]`. Only the
Pooling and BatchNormalization `dimension` keys are qualified
(`pooling.dimension` / `batch_normalization.dimension`).

The full attribute schema and validation defaults live at
`packages/library/lib/nodes/nnDiagram/nnAttributeWidgetConfig.ts` and
`nnValidationDefaults.ts` as data-only modules.

### v4 edge types

`'NNNext' | 'NNComposition' | 'NNAssociation'`

```ts
type NNEdgeData = {
  name?: string;
  points: IPoint[];
};
```

### Mapping rules (NNDiagram)

- For each v3 layer element, collect its attribute elements — the ids in the
  layer's `attributes` list (what the old backend read) plus any element with
  `owner === layerId` — whose `type` is one of the per-layer attribute
  element types, or, failing that, whose `attributeName` names a field of the
  layer's schema (the old backend matched on `attributeName`). Drop them from
  the v4 node list, accumulate them into the parent's `data.attributes` keyed
  by the snake_case slug above. Their `value` field becomes the value (string).
- Dropdown values outside the current whitelist (e.g. padding `zeros`,
  optimizer `rmsprop`, loss `cross_entropy`) are preserved verbatim — never
  coerced to a default — so the backend reports the real value as a
  validation error, exactly as it did for the v3 file.
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

## BPMNDiagram

Model `type` on the wire is **`"BPMNDiagram"`** (`UMLDiagramType.BPMN`);
readers also accept the legacy `"BPMN"`. The project-envelope bucket key is
`project.diagrams.BPMN` (not the model type). Backend constants:
`BPMN_DIAGRAM_TYPE`, `BPMN_DIAGRAM_TYPES`, `BPMN_PROJECT_DIAGRAM_KEY`,
`BPMN_FLOW_EDGE_TYPES` in `backend/constants/constants.py`.

### v4 node types (BPMNDiagram)

| v3 `type` | v4 `node.type` | `node.data` fields (besides `name`, colours, `highlight`) |
|-----------|----------------|------------------------------------------------------------|
| `BPMNTask` | `bpmnTask` | `taskType` (`default`/`user`/`service`/`send`/`receive`/`manual`/`business-rule`/`script`), `marker` (`none`/`loop`/`parallel multi instance`/`sequential multi instance`) |
| `BPMNSubprocess` / `BPMNTransaction` / `BPMNCallActivity` | `bpmnSubprocess` / `bpmnTransaction` / `bpmnCallActivity` | `marker`; `isExpanded?` (subprocess/transaction) |
| `BPMNStartEvent` / `BPMNIntermediateEvent` / `BPMNEndEvent` | `bpmnStartEvent` / `bpmnIntermediateEvent` / `bpmnEndEvent` | `eventType` (see `bpmn_event_mapping.py`) |
| `BPMNGateway` | `bpmnGateway` | `gatewayType` (`exclusive`/`parallel`/`inclusive`/`event-based`/`complex`) |
| `BPMNDataObject` / `BPMNDataStore` | `bpmnDataObject` / `bpmnDataStore` | — |
| `BPMNAnnotation` | `bpmnAnnotation` | `name` carries the annotation text |
| `BPMNGroup` | `bpmnGroup` | — |
| `BPMNPool` | `bpmnPool` | — |
| `BPMNSwimlane` | `bpmnSwimlane` | — (always `parentId` = its pool) |

Containment is `parentId` (v3 `owner`): a lane's parent is a pool; a flow
node's parent is a lane, a pool, or a subprocess/transaction. React Flow moves
a node with a container only through `parentId`, and the backend assigns flow
nodes to a pool's process / lane only through `parentId` (an unparented node
lands in a synthetic top-level process). A top-level flow node, data object,
annotation or group whose bounds lie inside a pool / lane / expanded
subprocess is therefore adopted on load (`normalizeV4Model` and
`migrateBpmnDiagramV3ToV4` → `adoptBpmnContainment`, `utils/bpmnContainment.ts`):
parent = the smallest enclosing container (a straddling node in a laned pool
goes to the lane under its centre), position made parent-relative, lane header
strip kept clear. Already-parented nodes are never touched; parents precede
children in `nodes`.

### v4 edge types (BPMNDiagram)

One edge `type` per flow kind (v3 used a single `BPMNFlow` + `flowType`):
`BPMNSequenceFlow` (`sequence`), `BPMNMessageFlow` (`message`),
`BPMNAssociationFlow` (`association`), `BPMNDataAssociationFlow`
(`data association`).

```ts
// edge.data
{
  name?: string;               // backend reads name, falls back to label
  label?: string;              // BPMNDiagramEdge.tsx renders data.label; the backend emits both
  isDefault?: boolean;         // BPMNSequenceFlow only
  isManuallyLayouted?: boolean;
  points: IPoint[];
}
```

### Mapping rules (BPMNDiagram)

- v3 → v4: `BPMNFlow` + `flowType` picks the edge type. The old editor emitted
  `'sequence' | 'message' | 'association' | 'data association'` (with a
  space); the migrator also accepts `dataAssociation` / `data_association` /
  `data-association`, and a missing `flowType` means `sequence`. v3
  `isDefault: true` becomes `edge.data.isDefault: true`.

- `isDefault` is legal only on a sequence flow whose source is an activity or an
  exclusive / inclusive / complex gateway, and a source has at most one default
  flow. The editor clears the flag when a sibling is made default, when the
  gateway type changes to parallel / event-based, and when a flip or endpoint
  reconnect gives the flow an ineligible source; the backend downgrades an
  illegal flag to `false` with a warning.
- Children of a `bpmnSwimlane` keep `position.x >= 30` (the lane header strip,
  `LANE_HEADER_WIDTH`); lanes sit at `x = 40` (`POOL_HEADER_WIDTH`) inside their
  pool, stacked vertically.
- The backend stashes `id`, `parentId`, geometry and colours in the metamodel's
  opaque `layout` so JSON → B-UML → JSON keeps ids and positions; B-UML `.py`
  import goes through the safe AST-allowlist loader (`safe_load_buml`).

---

## Conversion direction guarantees

Two directions, both must be implemented:

- **Frontend (TS)**: `migrate-uml-v3-to-v4.ts` reads a v3 model and emits a
  v4 model. Pure, no IO. Same fallthrough rules as the Python normalizer.
- **Backend (Python)**: `json_to_buml/<diagram>_diagram_processor.py`
  parses v4 directly. `buml_to_json/<diagram>_diagram_converter.py` emits
  v4 directly. **No v3 emission path** anywhere on the backend after Wave
  3 (the old fork is deleted).
- **Backend rejects v3 on ingest.** The backend has no v3 *read* path
  either, so a v3 UML model would convert to an empty B-UML model. Every
  endpoint that takes diagram or project JSON therefore refuses one with
  HTTP 400 (`LegacyDiagramFormatError`, a `ConversionError`; the check is
  `services/validators/legacy_format.py`, run by the `DiagramInput` /
  `SimulationSessionInput` model validators — so also for every diagram
  nested in a `ProjectInput` — and explicitly on the raw-dict bodies of
  `/github/deploy-webapp` and the `userProfileModel` of the
  personalization endpoints). A model is legacy when it has a UML `type`
  and a `version` starting with `3.`, or `elements` / `relationships` and
  no `nodes`, including an object diagram's `referenceDiagramData`.
  `GUINoCodeDiagram` (GrapesJS) and `QuantumCircuitDiagram` models keep
  their own formats and are never treated as legacy. The message tells the
  user to open the diagram in the current editor, whose migrator upgrades it
  on load, or to re-export it from there. Saving a v3 project to GitHub is
  storage and still succeeds; `/github/project/save` skips the B-UML export
  of its legacy diagrams.

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

### NN layer cards (SA-2.2 #34, restored in SA-UX-FIX-2)

v3 layer icons restored in SA-UX-FIX-2 — `_NNLayerBase.tsx` now
renders the per-kind PNG (`/images/nn-layers/<kind>.png`) above the
stereotype/name header, mirroring v3 visuals while keeping the v4
stereotype-card structure. Default layer drop height bumped from 60
→ 140 px to make room.

### Header underline on UserModelName (SA-2.2 #35)

`HeaderSection.tsx` now applies `textDecoration="underline"` on both
the parent `<text>` element and the inner name `<tspan>`.
Chromium-based browsers historically dropped underline on tspans when
the parent had a `dy` offset (which happens for the SA-2 / SA-4 cards
with a stereotype line above the name); explicit duplication on the
tspan keeps both the ObjectName and UserModelName headers correctly
underlined regardless of browser quirks.
