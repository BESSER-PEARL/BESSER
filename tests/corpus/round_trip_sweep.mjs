#!/usr/bin/env node
/**
 * A1 round-trip corpus sweep.
 *
 * Finds every v3-shape and v4-shape JSON fixture in:
 *   - packages/library/tests/fixtures/v3/
 *   - packages/editor/src/tests/unit/test-resources/  (filter version "3")
 *   - packages/editor/src/main/packages/user-modeling/  (filter version "3")
 *   - packages/webapp/src/main/templates/              (filter for stray v3)
 *   - tests/fixtures/v3/                                (currently empty)
 *   - tests/fixtures/v4/                                (reverse direction)
 *
 * For v3 inputs:
 *   v3 -> migrate*V3ToV4(model) -> convertV4ToV3*(v4) -> migrate*V3ToV4(v3') -> v4'
 *   Assert canonical(v4) === canonical(v4')   (idempotent round-trip)
 *
 * For v4 inputs:
 *   v4 -> convertV4ToV3*(v4) -> migrate*V3ToV4(v3) -> v4'
 *   Assert canonical(v4) === canonical(v4')   (lossless reverse round-trip)
 *
 * Output: per-fixture PASS/FAIL with diagnostic, plus per-diagram summary.
 *
 * Run: node tests/corpus/round_trip_sweep.mjs [--json out.json]
 */

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = path.resolve(__dirname, '..', '..')
const FRONTEND = path.join(REPO_ROOT, 'besser/utilities/web_modeling_editor/frontend')
const LIB_DIST = path.join(FRONTEND, 'packages/library/dist/index.js')
const JSDOM_ENTRY = path.join(FRONTEND, 'node_modules/jsdom/lib/api.js')

// jsdom is hoisted into the frontend submodule; import via absolute file URL
// so this script can run from the repo root regardless of cwd.
const { JSDOM } = await import(pathToFileURL(JSDOM_ENTRY).href)

// JSDOM setup so the editor library can load (it touches document on import).
const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>')
globalThis.window = dom.window
globalThis.document = dom.window.document
globalThis.HTMLElement = dom.window.HTMLElement

// Silence the JSDOM canvas-not-implemented log noise; we don't render.
const origStderr = process.stderr.write.bind(process.stderr)
process.stderr.write = (chunk, ...rest) => {
  if (typeof chunk === 'string' && chunk.includes('HTMLCanvasElement.prototype.getContext')) return true
  return origStderr(chunk, ...rest)
}

const lib = await import(LIB_DIST)
const {
  migrateClassDiagramV3ToV4,
  migrateObjectDiagramV3ToV4,
  migrateStateMachineDiagramV3ToV4,
  migrateAgentDiagramV3ToV4,
  migrateUserDiagramV3ToV4,
  migrateNNDiagramV3ToV4,
  convertV4ToV3Class,
  convertV4ToV3StateMachine,
  convertV4ToV3Agent,
  convertV4ToV3User,
  convertV4ToV3NN,
} = lib

// ObjectDiagram has no dedicated reverse converter; the existing round-trip
// tests piggy-back on convertV4ToV3Class. We mirror that.
const MIGRATORS = {
  ClassDiagram: migrateClassDiagramV3ToV4,
  ObjectDiagram: migrateObjectDiagramV3ToV4,
  StateMachineDiagram: migrateStateMachineDiagramV3ToV4,
  AgentDiagram: migrateAgentDiagramV3ToV4,
  UserDiagram: migrateUserDiagramV3ToV4,
  NNDiagram: migrateNNDiagramV3ToV4,
}
const REVERSE = {
  ClassDiagram: convertV4ToV3Class,
  ObjectDiagram: convertV4ToV3Class,           // shares Class reverse converter
  StateMachineDiagram: convertV4ToV3StateMachine,
  AgentDiagram: convertV4ToV3Agent,
  UserDiagram: convertV4ToV3User,
  NNDiagram: convertV4ToV3NN,
}

// Volatile keys the converter regenerates on each call (Date.now()-based ids).
// Stripping these gives us "real" round-trip semantics — id is not user data.
const VOLATILE_TOP_LEVEL_KEYS = new Set(['id'])

// Stable canonical JSON: sort keys deterministically AND drop top-level volatile
// id fields so converter-generated `converted-diagram-<timestamp>` ids don't
// register as failures.
function canon (v, depth = 0) {
  if (v === null || typeof v !== 'object') return v
  if (Array.isArray(v)) return v.map(x => canon(x, depth + 1))
  const out = {}
  for (const k of Object.keys(v).sort()) {
    if (depth === 0 && VOLATILE_TOP_LEVEL_KEYS.has(k)) {
      const val = v[k]
      // Only drop if it's the auto-generated converted-diagram-<ts> form.
      if (typeof val === 'string' && /^converted-diagram-\d+$/.test(val)) continue
    }
    out[k] = canon(v[k], depth + 1)
  }
  return out
}
const canonStr = (v) => JSON.stringify(canon(v))

// Discover candidate fixtures.
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

// Identify the v3 model (or v4 model) inside a possibly wrapped fixture.
// Returns { kind: 'v3'|'v4'|'unknown', model, type } or null when irrelevant.
function classify (raw) {
  if (!raw || typeof raw !== 'object' || raw.__readError) return null
  // Direct shape: { version, type, elements? | nodes? }
  if (typeof raw.version === 'string' && typeof raw.type === 'string') {
    const major = raw.version.split('.')[0]
    if (major === '3') return { kind: 'v3', model: raw, type: raw.type }
    if (major === '4') return { kind: 'v4', model: raw, type: raw.type }
    return null
  }
  // v4 fixture wrapper: { title, model: {...} }
  if (raw.model && typeof raw.model === 'object' && raw.model.version) {
    const major = String(raw.model.version).split('.')[0]
    if (major === '3') return { kind: 'v3', model: raw.model, type: raw.model.type, wrapper: 'model' }
    if (major === '4') return { kind: 'v4', model: raw.model, type: raw.model.type, wrapper: 'model' }
  }
  // ProjectInput wrapper: { project: { diagrams: { type: [{model:{...}}] } } } — not a single
  // diagram fixture; we'll surface its children as separate entries.
  return null
}

function expandProjectInputs (rawPath, raw) {
  if (!raw || typeof raw !== 'object' || raw.__readError) return []
  const proj = raw.project ?? raw
  if (!proj || !proj.diagrams || typeof proj.diagrams !== 'object') return []
  const out = []
  for (const [diagramType, list] of Object.entries(proj.diagrams)) {
    if (!Array.isArray(list)) continue
    list.forEach((d, idx) => {
      if (d && d.model && d.model.version) {
        out.push({
          path: `${rawPath}::diagrams.${diagramType}[${idx}]`,
          raw: d.model,
        })
      }
    })
  }
  return out
}

function attemptRoundTrip ({ kind, type, model }) {
  const migrator = MIGRATORS[type]
  const reverse = REVERSE[type]
  if (!migrator || !reverse) {
    return { ok: false, reason: `no migrator/reverse for type ${type}` }
  }
  try {
    if (kind === 'v3') {
      const v4a = migrator(model)
      const v3b = reverse(v4a)
      const v4b = migrator(v3b)
      const a = canonStr(v4a)
      const b = canonStr(v4b)
      if (a !== b) {
        const sample = firstDiff(JSON.parse(a), JSON.parse(b))
        return { ok: false, reason: `v3->v4->v3->v4 not idempotent: ${sample}` }
      }
      return { ok: true, reason: 'v3 idempotent' }
    } else {
      // v4 reverse: v4 -> v3 -> v4'
      const v3a = reverse(model)
      const v4b = migrator(v3a)
      // canonicalize input v4 model the same way for comparison
      const a = canonStr(model)
      const b = canonStr(v4b)
      if (a !== b) {
        const sample = firstDiff(JSON.parse(a), JSON.parse(b))
        return { ok: false, reason: `v4->v3->v4 not lossless: ${sample}` }
      }
      return { ok: true, reason: 'v4 lossless' }
    }
  } catch (err) {
    return { ok: false, reason: `threw: ${err.message?.split('\n')[0]}` }
  }
}

// Walk both objects and report the first key path where they diverge.
function firstDiff (a, b, p = '$') {
  if (typeof a !== typeof b) return `${p}: type ${typeof a} vs ${typeof b}`
  if (a === null || b === null) {
    if (a !== b) return `${p}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`
    return null
  }
  if (typeof a !== 'object') {
    if (a !== b) return `${p}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`
    return null
  }
  if (Array.isArray(a) !== Array.isArray(b)) return `${p}: array vs object`
  if (Array.isArray(a)) {
    if (a.length !== b.length) return `${p}: array length ${a.length} vs ${b.length}`
    for (let i = 0; i < a.length; i++) {
      const d = firstDiff(a[i], b[i], `${p}[${i}]`)
      if (d) return d
    }
    return null
  }
  const ak = Object.keys(a).sort(); const bk = Object.keys(b).sort()
  if (ak.join(',') !== bk.join(',')) {
    const onlyA = ak.filter(k => !bk.includes(k))
    const onlyB = bk.filter(k => !ak.includes(k))
    return `${p}: keys differ (only-left=[${onlyA.join(',')}] only-right=[${onlyB.join(',')}])`
  }
  for (const k of ak) {
    const d = firstDiff(a[k], b[k], `${p}.${k}`)
    if (d) return d
  }
  return null
}

// ── Build the corpus ────────────────────────────────────────────────────────

const dirs = [
  path.join(FRONTEND, 'packages/library/tests/fixtures/v3'),
  path.join(FRONTEND, 'packages/library/tests/fixtures/v4'), // may not exist
  path.join(FRONTEND, 'packages/editor/src/tests/unit/test-resources'),
  path.join(FRONTEND, 'packages/editor/src/main/packages/user-modeling'),
  path.join(FRONTEND, 'packages/webapp/src/main/templates'),
  path.join(REPO_ROOT, 'tests/fixtures/v3'),
  path.join(REPO_ROOT, 'tests/fixtures/v4'),
]

const candidates = []
for (const d of dirs) {
  for (const f of listJson(d)) {
    const raw = readJson(f)
    if (raw.__readError) {
      candidates.push({ path: f, raw, error: raw.__readError })
      continue
    }
    const top = classify(raw)
    if (top) {
      candidates.push({ path: f, raw, classified: top })
    } else {
      // Attempt to expand a project wrapper (webapp templates / smoke project).
      const children = expandProjectInputs(f, raw)
      for (const child of children) {
        const c = classify(child.raw)
        if (c) candidates.push({ path: child.path, raw: child.raw, classified: c })
      }
    }
  }
}

// ── Run the sweep ───────────────────────────────────────────────────────────

const results = []
for (const c of candidates) {
  if (c.error) {
    results.push({ path: c.path, type: '?', kind: '?', ok: false, reason: `read error: ${c.error}` })
    continue
  }
  const cls = c.classified
  const r = attemptRoundTrip(cls)
  results.push({ path: c.path, type: cls.type, kind: cls.kind, ok: r.ok, reason: r.reason })
}

// ── Report ──────────────────────────────────────────────────────────────────

const byCategory = {}
for (const r of results) {
  const cat = `${r.type} (${r.kind})`
  byCategory[cat] = byCategory[cat] || { pass: 0, fail: 0 }
  byCategory[cat][r.ok ? 'pass' : 'fail']++
}

const total = results.length
const pass = results.filter(r => r.ok).length
const fail = total - pass

console.log('# A1 Round-trip Corpus Sweep')
console.log('')
console.log(`Total fixtures: ${total}   PASS: ${pass}   FAIL: ${fail}`)
console.log('')
console.log('## By diagram type')
for (const [cat, c] of Object.entries(byCategory).sort()) {
  console.log(`  ${cat.padEnd(36)} pass=${c.pass}  fail=${c.fail}`)
}
console.log('')
console.log('## Per-fixture results')
for (const r of results) {
  const status = r.ok ? 'PASS' : 'FAIL'
  console.log(`[${status}] (${r.kind} ${r.type}) ${r.path}`)
  console.log(`        ${r.reason}`)
}

const outIdx = process.argv.indexOf('--json')
if (outIdx >= 0 && process.argv[outIdx + 1]) {
  fs.writeFileSync(process.argv[outIdx + 1], JSON.stringify({ total, pass, fail, byCategory, results }, null, 2))
  console.log(`\nJSON report -> ${process.argv[outIdx + 1]}`)
}

process.exit(fail === 0 ? 0 : 1)
