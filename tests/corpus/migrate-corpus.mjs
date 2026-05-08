#!/usr/bin/env node
/**
 * SA-DEEP-MIGRATOR-CORPUS sweep.
 *
 * Run every available v3-shape JSON fixture through the per-diagram-type
 * migrators exposed by `@besser/wme` and look specifically for data-loss
 * patterns (fields that were populated in v3 but ended up empty/missing
 * on the v4 node/edge `data.X`).
 *
 * Sources scanned:
 *   - /tmp/wme-develop/packages/...                                 (last v3 develop branch)
 *   - frontend submodule packages/library/tests/fixtures/v3/        (canonical round-trip set)
 *   - frontend submodule packages/webapp/src/main/templates/        (only stray v3 entries)
 *   - frontend submodule packages/editor/src/tests/...              (legacy v3 test JSON)
 *
 * Per-fixture status:
 *   PASS  — migration completed; node/edge counts preserved; no data-loss
 *           pattern triggered.
 *   WARN  — migration completed but at least one v3 attribute/method/relationship
 *           field is missing or empty on the corresponding v4 `data.*` field.
 *   FAIL  — migrator threw or returned a non-object / lost nodes/edges.
 *
 * Output:
 *   - prints a Markdown report to stdout
 *   - if `--out <path>` is provided, also writes the report there
 *   - if `--json <path>` is provided, dumps the structured results
 *
 * The script is read-only; it never mutates source fixtures.
 */

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = path.resolve(__dirname, '..', '..')
const FRONTEND = path.join(REPO_ROOT, 'besser/utilities/web_modeling_editor/frontend')
const LIB_DIST = path.join(FRONTEND, 'packages/library/dist/index.js')
const JSDOM_ENTRY = path.join(FRONTEND, 'node_modules/jsdom/lib/api.js')

// ── Browser-globals shim so the bundled library can load ──────────────────
const { JSDOM } = await import(pathToFileURL(JSDOM_ENTRY).href)
const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>')
globalThis.window = dom.window
globalThis.document = dom.window.document
globalThis.HTMLElement = dom.window.HTMLElement
globalThis.SVGElement = dom.window.SVGElement
// Node 22 has a built-in `navigator` getter that we cannot redefine with `=`,
// so only override it when not already present.
if (!('navigator' in globalThis) || globalThis.navigator == null) {
  try { globalThis.navigator = dom.window.navigator } catch { /* read-only — ignore */ }
}

// Silence the JSDOM canvas-not-implemented log noise.
const origStderr = process.stderr.write.bind(process.stderr)
process.stderr.write = (chunk, ...rest) => {
  if (typeof chunk === 'string' && chunk.includes('HTMLCanvasElement.prototype.getContext')) return true
  return origStderr(chunk, ...rest)
}

const lib = await import(LIB_DIST)
const MIGRATORS = {
  ClassDiagram: lib.migrateClassDiagramV3ToV4,
  ObjectDiagram: lib.migrateObjectDiagramV3ToV4,
  StateMachineDiagram: lib.migrateStateMachineDiagramV3ToV4,
  AgentDiagram: lib.migrateAgentDiagramV3ToV4,
  UserDiagram: lib.migrateUserDiagramV3ToV4,
  NNDiagram: lib.migrateNNDiagramV3ToV4,
}

// ── Fixture discovery ───────────────────────────────────────────────────────

const SOURCE_ROOTS = [
  '/tmp/wme-develop/packages/library/tests/fixtures/v3',
  '/tmp/wme-develop/packages/webapp/src/main/templates',
  '/tmp/wme-develop/packages/editor/src/tests/unit/test-resources',
  '/tmp/wme-develop/packages/editor/src/main/packages/user-modeling',
  path.join(FRONTEND, 'packages/library/tests/fixtures/v3'),
  path.join(FRONTEND, 'packages/webapp/src/main/templates'),
  path.join(FRONTEND, 'packages/editor/src/tests/unit/test-resources'),
  path.join(FRONTEND, 'packages/editor/src/main/packages/user-modeling'),
]

function listJson (dir) {
  if (!fs.existsSync(dir)) return []
  const out = []
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) out.push(...listJson(full))
    else if (entry.isFile() && entry.name.endsWith('.json')) out.push(full)
  }
  return out
}

function readJson (p) {
  try { return JSON.parse(fs.readFileSync(p, 'utf8')) } catch (err) { return { __readError: err.message } }
}

/**
 * Yield every v3 model embedded in `raw`. Handles three wrapping shapes:
 *  - direct:        { version: "3.x", type, elements, relationships }
 *  - {model: ...}:   wraps a direct model
 *  - project:       { project: { diagrams: { TYPE: [{model:...}, ...] } } }
 *
 * Each yielded entry is { label, model } where label includes the original
 * file path plus a child suffix for project/diagram entries.
 */
function * extractV3Models (file, raw) {
  if (!raw || typeof raw !== 'object' || raw.__readError) return
  if (typeof raw.version === 'string' && raw.version.startsWith('3') && typeof raw.type === 'string') {
    yield { label: file, model: raw }
    return
  }
  if (raw.model && typeof raw.model === 'object' && typeof raw.model.version === 'string' && raw.model.version.startsWith('3')) {
    yield { label: file, model: raw.model }
    return
  }
  const proj = raw.project ?? raw
  if (proj && proj.diagrams && typeof proj.diagrams === 'object') {
    for (const [diagramType, list] of Object.entries(proj.diagrams)) {
      if (!Array.isArray(list)) continue
      for (let idx = 0; idx < list.length; idx++) {
        const d = list[idx]
        const m = d && d.model
        if (m && typeof m.version === 'string' && m.version.startsWith('3') && m.type) {
          yield { label: `${file}::diagrams.${diagramType}[${idx}]`, model: m }
        }
      }
    }
  }
}

// ── Data-loss probes ───────────────────────────────────────────────────────

/**
 * Map a v3 element's "interesting" data fields to the v4 `data.*` fields
 * we expect them to land on. The migrator lifts attribute/method rows into
 * `data.attributes` / `data.methods`, classifier name → `data.name`, etc.
 *
 * Each probe returns a list of issue strings ("attribute.defaultValue empty
 * after migration"). Empty list means no data loss observed.
 */
function probeV3v4Diff (v3Model, v4Model) {
  const issues = []

  const v3Elements = v3Model.elements || {}
  const v3Rels = v3Model.relationships || {}
  const v4Nodes = v4Model.nodes || []
  const v4Edges = v4Model.edges || []

  // Build look-ups for quick field comparisons.
  const nodesById = new Map(v4Nodes.map(n => [n.id, n]))
  const edgesById = new Map(v4Edges.map(e => [e.id, e]))

  // 1) Classifier-level checks: every v3 top-level element with attributes/methods
  //    must surface as a node. Classifier name must be preserved on data.name.
  for (const [id, el] of Object.entries(v3Elements)) {
    const isClassifier = (el.attributes && Array.isArray(el.attributes)) ||
      (el.methods && Array.isArray(el.methods))
    if (!isClassifier) continue
    const node = nodesById.get(id)
    if (!node) {
      issues.push(`node missing in v4 for v3 classifier ${el.type ?? '?'} (${id})`)
      continue
    }
    if (el.name && !(node.data && (node.data.name === el.name || node.data.name?.startsWith?.(el.name)))) {
      // Some types format the name; allow startsWith match.
      issues.push(`classifier.name "${el.name}" lost: v4 data.name=${JSON.stringify(node.data?.name)}`)
    }

    // 2) Attribute checks – v3 attribute rows live as separate elements; v4
    //    embeds them in node.data.attributes (array of row objects).
    const attrRows = (node.data && Array.isArray(node.data.attributes)) ? node.data.attributes : []
    const v3AttrIds = (el.attributes || [])
    for (const attrId of v3AttrIds) {
      const attrEl = v3Elements[attrId]
      if (!attrEl) continue
      const v4Row = attrRows.find(r => r && (r.id === attrId || r.elementId === attrId || r.name === attrEl.name))
      if (!v4Row) {
        issues.push(`attribute "${attrEl.name ?? '?'}" missing from v4 data.attributes`)
        continue
      }
      // Field-by-field defaultValue/visibility/attributeType/isId/isOptional/isDerived/isExternalId checks
      for (const field of ['defaultValue', 'visibility', 'attributeType', 'isId', 'isOptional', 'isDerived', 'isExternalId']) {
        if (!(field in attrEl)) continue
        const v3Val = attrEl[field]
        if (v3Val === undefined || v3Val === null || v3Val === '' || v3Val === false) continue
        // Map v3 attributeType -> v4 type (some migrators normalize this).
        const v4Field = field === 'attributeType' ? 'type' : field
        const v4Val = v4Row[v4Field] ?? v4Row[field]
        if (v4Val === undefined || v4Val === null || v4Val === '') {
          issues.push(`attribute.${field} empty after migration (v3=${JSON.stringify(v3Val)})`)
        }
      }
    }

    // 3) Method checks – analogous structure to attributes.
    const methodRows = (node.data && Array.isArray(node.data.methods)) ? node.data.methods : []
    const v3MethodIds = (el.methods || [])
    for (const methodId of v3MethodIds) {
      const methodEl = v3Elements[methodId]
      if (!methodEl) continue
      const v4Row = methodRows.find(r => r && (r.id === methodId || r.elementId === methodId || r.name === methodEl.name))
      if (!v4Row) {
        issues.push(`method "${methodEl.name ?? '?'}" missing from v4 data.methods`)
        continue
      }
      for (const field of ['code', 'visibility', 'implementationType', 'attributeType']) {
        if (!(field in methodEl)) continue
        const v3Val = methodEl[field]
        if (v3Val === undefined || v3Val === null || v3Val === '' || v3Val === false) continue
        const v4Field = field === 'attributeType' ? 'type' : field
        const v4Val = v4Row[v4Field] ?? v4Row[field]
        if (v4Val === undefined || v4Val === null || v4Val === '') {
          issues.push(`method.${field} empty after migration (v3=${JSON.stringify(v3Val).slice(0, 80)})`)
        }
      }
    }
  }

  // 4) Relationship checks – every v3 relationship must produce a v4 edge.
  for (const [id, rel] of Object.entries(v3Rels)) {
    const edge = edgesById.get(id)
    if (!edge) {
      issues.push(`edge missing in v4 for v3 relationship ${rel.type ?? '?'} (${id})`)
      continue
    }
    // Source / target endpoint preservation.
    const srcId = rel.source?.element ?? rel.source?.id
    const tgtId = rel.target?.element ?? rel.target?.id
    if (srcId && edge.source !== srcId) {
      issues.push(`edge.source mismatch: v3=${srcId} v4=${edge.source}`)
    }
    if (tgtId && edge.target !== tgtId) {
      issues.push(`edge.target mismatch: v3=${tgtId} v4=${edge.target}`)
    }
    // Multiplicity / role labels live on data.* in v4.
    const data = edge.data || {}
    const labelChecks = [
      ['source.multiplicity', rel.source?.multiplicity, data.sourceMultiplicity ?? data.source?.multiplicity],
      ['target.multiplicity', rel.target?.multiplicity, data.targetMultiplicity ?? data.target?.multiplicity],
      ['source.role', rel.source?.role, data.sourceRole ?? data.source?.role],
      ['target.role', rel.target?.role, data.targetRole ?? data.target?.role],
    ]
    for (const [label, v3Val, v4Val] of labelChecks) {
      if (v3Val === undefined || v3Val === null || v3Val === '') continue
      if (v4Val === undefined || v4Val === null || v4Val === '') {
        issues.push(`relationship.${label} empty after migration (v3=${JSON.stringify(v3Val)})`)
      }
    }
    // Name on a relationship.
    if (rel.name && !(data.name === rel.name)) {
      // Only flag when v4 has no name representation at all.
      if (!data.name && !data.label) {
        issues.push(`relationship.name "${rel.name}" lost`)
      }
    }
  }

  return issues
}

// ── Main sweep ─────────────────────────────────────────────────────────────

const candidates = []
for (const root of SOURCE_ROOTS) {
  for (const file of listJson(root)) {
    const raw = readJson(file)
    if (raw.__readError) {
      candidates.push({ file, error: `read error: ${raw.__readError}` })
      continue
    }
    for (const entry of extractV3Models(file, raw)) {
      candidates.push(entry)
    }
  }
}

// De-duplicate identical models that appear in both /tmp/wme-develop and the
// submodule (the develop snapshot may match HEAD). Keep the first occurrence.
const seen = new Set()
const dedup = []
for (const c of candidates) {
  if (c.error) { dedup.push(c); continue }
  const key = JSON.stringify(c.model).slice(0, 4096)
  if (seen.has(key)) continue
  seen.add(key)
  dedup.push(c)
}

const results = []
let totalNodesPreserved = 0
let totalNodesExpected = 0
let totalEdgesPreserved = 0
let totalEdgesExpected = 0
const lossPatternCounts = new Map()

for (const c of dedup) {
  if (c.error) {
    results.push({ label: c.file, type: '?', status: 'FAIL', reason: c.error, issues: [] })
    continue
  }
  const { label, model } = c
  const type = model.type
  const migrator = MIGRATORS[type]
  if (!migrator) {
    results.push({ label, type, status: 'SKIP', reason: `no migrator for type ${type}`, issues: [] })
    continue
  }

  // Count expected classifier nodes (top-level elements that aren't sub-rows).
  const v3Elements = model.elements || {}
  const v3Rels = model.relationships || {}
  const expectedNodes = Object.values(v3Elements).filter(e => e && e.owner == null).length
  const expectedEdges = Object.keys(v3Rels).length

  let v4
  try {
    v4 = migrator(model)
  } catch (err) {
    results.push({
      label, type, status: 'FAIL',
      reason: `threw: ${err.message?.split('\n')[0]}`,
      issues: [],
      expectedNodes, expectedEdges, actualNodes: 0, actualEdges: 0,
    })
    continue
  }
  if (!v4 || typeof v4 !== 'object') {
    results.push({
      label, type, status: 'FAIL',
      reason: `migrator returned non-object`,
      issues: [],
      expectedNodes, expectedEdges, actualNodes: 0, actualEdges: 0,
    })
    continue
  }

  const actualNodes = Array.isArray(v4.nodes) ? v4.nodes.length : 0
  const actualEdges = Array.isArray(v4.edges) ? v4.edges.length : 0

  totalNodesExpected += expectedNodes
  totalNodesPreserved += Math.min(actualNodes, expectedNodes)
  totalEdgesExpected += expectedEdges
  totalEdgesPreserved += Math.min(actualEdges, expectedEdges)

  const issues = probeV3v4Diff(model, v4)
  for (const i of issues) {
    // Normalise the prefix so similar messages cluster.
    const key = i.replace(/"[^"]*"/g, '"…"').replace(/v3=[^ )]+/, 'v3=…')
    lossPatternCounts.set(key, (lossPatternCounts.get(key) ?? 0) + 1)
  }

  let status = 'PASS'
  let reason = 'ok'
  if (actualNodes < expectedNodes || actualEdges < expectedEdges) {
    status = 'FAIL'
    reason = `count drop: nodes ${actualNodes}/${expectedNodes}, edges ${actualEdges}/${expectedEdges}`
  } else if (issues.length > 0) {
    status = 'WARN'
    reason = `${issues.length} data-loss issue(s)`
  }

  results.push({
    label, type, status, reason,
    expectedNodes, expectedEdges, actualNodes, actualEdges,
    issues,
  })
}

// ── Render Markdown report ─────────────────────────────────────────────────

const totals = {
  total: results.length,
  pass: results.filter(r => r.status === 'PASS').length,
  warn: results.filter(r => r.status === 'WARN').length,
  fail: results.filter(r => r.status === 'FAIL').length,
  skip: results.filter(r => r.status === 'SKIP').length,
}

const topPatterns = [...lossPatternCounts.entries()]
  .sort((a, b) => b[1] - a[1])
  .slice(0, 10)

const lines = []
lines.push('# SA-DEEP-MIGRATOR-CORPUS sweep')
lines.push('')
lines.push('Real production v3 fixtures piped through the per-diagram-type v3→v4 migrators.')
lines.push('Counts only data fields that **were populated in v3** but vanished in v4.')
lines.push('')
lines.push('## Totals')
lines.push('')
lines.push(`- Fixtures discovered: **${totals.total}**`)
lines.push(`- PASS: **${totals.pass}**`)
lines.push(`- WARN (data loss): **${totals.warn}**`)
lines.push(`- FAIL (crash / count drop): **${totals.fail}**`)
lines.push(`- SKIP (no migrator for type): **${totals.skip}**`)
lines.push(`- Nodes preserved: **${totalNodesPreserved}** / **${totalNodesExpected}**`)
lines.push(`- Edges preserved: **${totalEdgesPreserved}** / **${totalEdgesExpected}**`)
lines.push('')

lines.push('## By diagram type')
lines.push('')
const byType = new Map()
for (const r of results) {
  const t = r.type ?? '?'
  const b = byType.get(t) ?? { pass: 0, warn: 0, fail: 0, skip: 0 }
  b[r.status.toLowerCase()] = (b[r.status.toLowerCase()] ?? 0) + 1
  byType.set(t, b)
}
lines.push('| Type | PASS | WARN | FAIL | SKIP |')
lines.push('|------|-----:|-----:|-----:|-----:|')
for (const [t, b] of [...byType.entries()].sort()) {
  lines.push(`| ${t} | ${b.pass ?? 0} | ${b.warn ?? 0} | ${b.fail ?? 0} | ${b.skip ?? 0} |`)
}
lines.push('')

lines.push('## Top data-loss patterns')
lines.push('')
if (topPatterns.length === 0) {
  lines.push('_No data-loss patterns detected._')
} else {
  lines.push('| # | Count | Pattern |')
  lines.push('|--:|------:|---------|')
  topPatterns.forEach(([pat, n], idx) => {
    lines.push(`| ${idx + 1} | ${n} | \`${pat}\` |`)
  })
}
lines.push('')

lines.push('## Per-fixture results')
lines.push('')
lines.push('| Status | Type | Nodes | Edges | Issues | Fixture |')
lines.push('|--------|------|------:|------:|------:|---------|')
for (const r of results) {
  const nodes = r.expectedNodes != null ? `${r.actualNodes}/${r.expectedNodes}` : '—'
  const edges = r.expectedEdges != null ? `${r.actualEdges}/${r.expectedEdges}` : '—'
  const issues = r.issues?.length ?? 0
  // Make the path repo-relative for readability.
  const rel = r.label.replace(REPO_ROOT, '<repo>').replace('/tmp/wme-develop', '<develop>')
  lines.push(`| ${r.status} | ${r.type ?? '?'} | ${nodes} | ${edges} | ${issues} | \`${rel}\` |`)
}
lines.push('')

lines.push('## Recommended migrator fixes')
lines.push('')
if (topPatterns.length === 0) {
  lines.push('No fixes required from this corpus run; every populated v3 field is mirrored in v4.')
} else {
  lines.push('Order matches the top-pattern table above:')
  lines.push('')
  topPatterns.forEach(([pat, n], idx) => {
    let recommendation
    if (pat.includes('attribute.defaultValue')) {
      recommendation = 'Lift `defaultValue` from the v3 ClassAttribute element onto the v4 `data.attributes[].defaultValue` row.'
    } else if (pat.includes('attribute.attributeType')) {
      recommendation = 'Map v3 `attributeType` onto v4 `data.attributes[].type` (or keep `attributeType` for parity).'
    } else if (pat.includes('attribute.visibility')) {
      recommendation = 'Preserve attribute visibility on v4 `data.attributes[].visibility` (default `public` if absent).'
    } else if (pat.includes('method.code')) {
      recommendation = 'Carry the method body string onto v4 `data.methods[].code`; today it is dropped during the row collapse.'
    } else if (pat.includes('method.implementationType')) {
      recommendation = 'Forward `implementationType` (`code` / `bal`) to v4 `data.methods[].implementationType`.'
    } else if (pat.includes('relationship.source.multiplicity') || pat.includes('relationship.target.multiplicity')) {
      recommendation = 'Lift v3 `relationships[].{source,target}.multiplicity` to v4 edge `data.{source,target}Multiplicity`.'
    } else if (pat.includes('relationship.source.role') || pat.includes('relationship.target.role')) {
      recommendation = 'Carry v3 `relationships[].{source,target}.role` onto v4 edge `data.{source,target}Role`.'
    } else if (pat.includes('relationship.name')) {
      recommendation = 'Forward `relationship.name` to v4 edge `data.name` (currently lost for some edge types).'
    } else if (pat.includes('classifier.name')) {
      recommendation = 'Preserve the classifier name verbatim on v4 `data.name`; today the migrator may rename when the v3 name embeds a parameter list.'
    } else if (pat.startsWith('node missing') || pat.startsWith('edge missing')) {
      recommendation = 'Investigate why this element is filtered out — count drops indicate the migrator silently discards unrecognised v3 types.'
    } else {
      recommendation = 'Audit the migrator branch responsible for this field family and add the missing copy.'
    }
    lines.push(`${idx + 1}. **${pat}** _(seen ${n}×)_ — ${recommendation}`)
  })
}
lines.push('')
lines.push('## Source roots scanned')
lines.push('')
for (const root of SOURCE_ROOTS) {
  lines.push(`- \`${root.replace(REPO_ROOT, '<repo>').replace('/tmp/wme-develop', '<develop>')}\` ` +
    (fs.existsSync(root) ? '(present)' : '(missing — skipped)'))
}
lines.push('')

const md = lines.join('\n')
process.stdout.write(md)

const outIdx = process.argv.indexOf('--out')
if (outIdx >= 0 && process.argv[outIdx + 1]) {
  fs.writeFileSync(process.argv[outIdx + 1], md)
}
const jsonIdx = process.argv.indexOf('--json')
if (jsonIdx >= 0 && process.argv[jsonIdx + 1]) {
  fs.writeFileSync(process.argv[jsonIdx + 1], JSON.stringify({
    totals,
    totalNodesExpected, totalNodesPreserved,
    totalEdgesExpected, totalEdgesPreserved,
    topPatterns,
    results,
  }, null, 2))
}

// Non-zero exit only on FAIL (crashes / count drops). WARN is informational.
process.exit(totals.fail > 0 ? 1 : 0)
