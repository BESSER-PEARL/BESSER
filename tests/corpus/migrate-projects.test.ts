/**
 * A9 corpus test: simulate `migrateProjectToV5` on real v3 BesserProject
 * snapshots and confirm clean upgrade.
 *
 * Loads each fixture under ./v3-projects, runs the v4 → v5 migrator, and
 * asserts:
 *   - returned project has `schemaVersion: 5`
 *   - every UMLModel inside is v4 shape (`nodes: []`, `edges: []`)
 *   - GUI / Quantum diagrams pass through unchanged (deep equality)
 *   - no diagram-content loss: counts of v4 nodes vs v3 elements (minus
 *     types that legitimately collapse onto an owner — e.g. `ClassAttribute`,
 *     `ClassMethod`, `ClassOCLConstraint`, `ObjectAttribute`,
 *     `ObjectMethod`, `ObjectIcon`, `UserModelAttribute`, `UserModelIcon`)
 *     match.
 *
 * Re-implements the webapp's `migrateProjectToV5` locally (copied verbatim
 * from `packages/webapp/src/main/shared/types/project.ts` at commit
 * 7a3b82a) so this test can run from the library's vitest config without
 * having to build the webapp.
 */
import { describe, it, expect } from "vitest"
import {
  migrateClassDiagramV3ToV4,
  migrateObjectDiagramV3ToV4,
  migrateStateMachineDiagramV3ToV4,
  migrateAgentDiagramV3ToV4,
  migrateUserDiagramV3ToV4,
  migrateNNDiagramV3ToV4,
  type UMLModel,
} from "@besser/wme"
import classOnlyProject from "./v3-projects/class-only-project.json"
import stateMachineProject from "./v3-projects/state-machine-project.json"
import agentProject from "./v3-projects/agent-project.json"
import fullStackProject from "./v3-projects/full-stack-project.json"

// ---------------------------------------------------------------------------
// Local copies of the webapp migrator types + functions (verbatim from
// `packages/webapp/src/main/shared/types/project.ts` at 7a3b82a).
// ---------------------------------------------------------------------------

type SupportedDiagramType =
  | "ClassDiagram"
  | "ObjectDiagram"
  | "StateMachineDiagram"
  | "AgentDiagram"
  | "UserDiagram"
  | "GUINoCodeDiagram"
  | "QuantumCircuitDiagram"
  | "NNDiagram"

const ALL_DIAGRAM_TYPES: SupportedDiagramType[] = [
  "ClassDiagram",
  "ObjectDiagram",
  "StateMachineDiagram",
  "AgentDiagram",
  "UserDiagram",
  "GUINoCodeDiagram",
  "QuantumCircuitDiagram",
  "NNDiagram",
]

interface ProjectDiagram {
  id: string
  title: string
  model?: any
  lastUpdate: string
  description?: string
  config?: Record<string, unknown>
  references?: Partial<Record<SupportedDiagramType, string>>
}

interface BesserProject {
  id: string
  type: "Project"
  schemaVersion: number
  name: string
  description: string
  owner: string
  createdAt: string
  currentDiagramType: SupportedDiagramType
  currentDiagramIndices: Record<SupportedDiagramType, number>
  diagrams: Record<SupportedDiagramType, ProjectDiagram[]>
  settings: any
}

const isV3UMLModel = (model: unknown): boolean => {
  if (!model || typeof model !== "object") return false
  const candidate = model as Record<string, unknown>
  return "elements" in candidate && "relationships" in candidate
}

// Webapp dispatcher (mirrors `migrateUMLModelV3ToV4`).
function migrateUMLModelV3ToV4(
  model: any,
  diagramType?: SupportedDiagramType,
): UMLModel {
  const type = diagramType ?? (model && typeof model === "object" ? model.type : undefined)
  switch (type) {
    case "ClassDiagram":
      return migrateClassDiagramV3ToV4(model)
    case "ObjectDiagram":
      return migrateObjectDiagramV3ToV4(model)
    case "StateMachineDiagram":
      return migrateStateMachineDiagramV3ToV4(model)
    case "AgentDiagram":
      return migrateAgentDiagramV3ToV4(model)
    case "UserDiagram":
      return migrateUserDiagramV3ToV4(model)
    case "NNDiagram":
      return migrateNNDiagramV3ToV4(model)
    default:
      throw new Error(`[migrateUMLModelV3ToV4] Unsupported diagram type: ${String(type)}`)
  }
}

// Webapp v4 → v5 migrator (verbatim).
function migrateProjectToV5(project: BesserProject): BesserProject {
  if (project.schemaVersion >= 5) return project
  for (const type of ALL_DIAGRAM_TYPES) {
    if (type === "GUINoCodeDiagram" || type === "QuantumCircuitDiagram") continue
    const diagrams = project.diagrams[type] ?? []
    for (const d of diagrams) {
      if (d.model && isV3UMLModel(d.model)) {
        try {
          d.model = migrateUMLModelV3ToV4(d.model, type)
        } catch (err) {
          // eslint-disable-next-line no-console
          console.error("[migrateProjectToV5] Failed to migrate diagram", d.id, type, err)
        }
      }
    }
  }
  project.schemaVersion = 5
  return project
}

// ---------------------------------------------------------------------------
// Counting helpers
// ---------------------------------------------------------------------------

// v3 child element types that legitimately collapse into their owner
// (and therefore should NOT appear in v4 `nodes`).
const COLLAPSED_V3_TYPES = new Set<string>([
  "ClassAttribute",
  "ClassMethod",
  "ObjectAttribute",
  "ObjectMethod",
  "ObjectIcon",
  "UserModelAttribute",
  "UserModelIcon",
  "NNSectionTitle",
  "NNSectionSeparator",
])

/** v3 elements that survive as top-level nodes after migration. */
function v3SurvivingElementCount(model: any): number {
  if (!model || !model.elements) return 0
  let count = 0
  for (const id of Object.keys(model.elements)) {
    const el = model.elements[id]
    if (!el || typeof el !== "object") continue
    const type = el.type as string
    if (COLLAPSED_V3_TYPES.has(type)) continue
    // ClassOCLConstraint collapses onto its owner if owner exists.
    if (type === "ClassOCLConstraint") {
      const owner = el.owner ? model.elements[el.owner] : undefined
      if (owner && ["Class", "AbstractClass", "Interface", "Enumeration"].includes(owner.type)) {
        continue
      }
    }
    count++
  }
  return count
}

function v3RelationshipCount(model: any): number {
  if (!model || !model.relationships) return 0
  return Object.keys(model.relationships).length
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

interface Snapshot {
  name: string
  data: any
}

const snapshots: Snapshot[] = [
  { name: "class-only-project", data: classOnlyProject },
  { name: "state-machine-project", data: stateMachineProject },
  { name: "agent-project", data: agentProject },
  { name: "full-stack-project", data: fullStackProject },
]

describe("A9 — migrateProjectToV5 on v3 BesserProject corpus", () => {
  // Aggregate counters across the whole corpus, asserted at the end.
  const totals = { v3Nodes: 0, v4Nodes: 0, v3Edges: 0, v4Edges: 0 }

  for (const snapshot of snapshots) {
    it(`${snapshot.name}: clean v4 → v5 upgrade with no content loss`, () => {
      // Deep clone so we can compare GUI/Quantum models pre/post.
      const before = JSON.parse(JSON.stringify(snapshot.data)) as BesserProject
      const project = JSON.parse(JSON.stringify(snapshot.data)) as BesserProject

      // Pre-condition: input is v4.
      expect(before.schemaVersion).toBe(4)

      const migrated = migrateProjectToV5(project)

      // 1. schemaVersion is bumped.
      expect(migrated.schemaVersion).toBe(5)

      // 2. Every UMLModel inside is v4 shape.
      const umlTypes: SupportedDiagramType[] = [
        "ClassDiagram",
        "ObjectDiagram",
        "StateMachineDiagram",
        "AgentDiagram",
        "UserDiagram",
        "NNDiagram",
      ]
      for (const t of umlTypes) {
        for (const d of migrated.diagrams[t] ?? []) {
          if (!d.model) continue
          expect(d.model, `${snapshot.name} ${t} ${d.id}: model present`).toBeTruthy()
          expect(Array.isArray((d.model as any).nodes), `${snapshot.name} ${t} ${d.id}: nodes is array`).toBe(true)
          expect(Array.isArray((d.model as any).edges), `${snapshot.name} ${t} ${d.id}: edges is array`).toBe(true)
          expect(typeof (d.model as any).version === "string" && (d.model as any).version.startsWith("4."),
            `${snapshot.name} ${t} ${d.id}: version 4.x`).toBe(true)
          // v3 keys must be gone.
          expect("elements" in (d.model as any), `${snapshot.name} ${t} ${d.id}: no v3 'elements'`).toBe(false)
          expect("relationships" in (d.model as any), `${snapshot.name} ${t} ${d.id}: no v3 'relationships'`).toBe(false)
        }
      }

      // 3. GUI / Quantum diagrams unchanged (deep-equal).
      expect(migrated.diagrams.GUINoCodeDiagram).toEqual(before.diagrams.GUINoCodeDiagram)
      expect(migrated.diagrams.QuantumCircuitDiagram).toEqual(before.diagrams.QuantumCircuitDiagram)

      // 4. No diagram-content loss: per-diagram element/relationship counts.
      for (const t of umlTypes) {
        const beforeDiagrams = before.diagrams[t] ?? []
        const afterDiagrams = migrated.diagrams[t] ?? []
        expect(afterDiagrams.length).toBe(beforeDiagrams.length)
        for (let i = 0; i < beforeDiagrams.length; i++) {
          const v3Model = beforeDiagrams[i].model as any
          const v4Model = afterDiagrams[i].model as any
          if (!v3Model || !v4Model) continue
          const expectedNodes = v3SurvivingElementCount(v3Model)
          const actualNodes = v4Model.nodes?.length ?? 0
          const expectedEdges = v3RelationshipCount(v3Model)
          const actualEdges = v4Model.edges?.length ?? 0
          expect(actualNodes,
            `${snapshot.name} ${t} ${beforeDiagrams[i].id}: node count`).toBe(expectedNodes)
          expect(actualEdges,
            `${snapshot.name} ${t} ${beforeDiagrams[i].id}: edge count`).toBe(expectedEdges)

          totals.v3Nodes += expectedNodes
          totals.v4Nodes += actualNodes
          totals.v3Edges += expectedEdges
          totals.v4Edges += actualEdges
        }
      }
    })
  }

  it("corpus aggregate: total node + edge counts preserved across all snapshots", () => {
    // This test runs after the per-snapshot tests have populated `totals`.
    // eslint-disable-next-line no-console
    console.log("[A9 corpus totals]", JSON.stringify(totals))
    expect(totals.v4Nodes).toBe(totals.v3Nodes)
    expect(totals.v4Edges).toBe(totals.v3Edges)
    // Sanity: corpus is non-trivial.
    expect(totals.v3Nodes).toBeGreaterThan(15)
    expect(totals.v3Edges).toBeGreaterThan(5)
  })
})
