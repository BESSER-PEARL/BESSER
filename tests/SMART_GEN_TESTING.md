# Testing the Spec-Driven Agent & the keyless Free tier

Everything you need to test **"describe an app → generate code"**, including the
**keyless free tier** (server-hosted `qwen3-coder`, no API key). The tests live in
three repos — `BESSER` (backend), the frontend submodule, and `modeling-agent` —
so this is the single place that indexes them and shows how to run each one.

> 🧪 Prefer a visual version? Open **[`SMART_GEN_TESTING.html`](./SMART_GEN_TESTING.html)**
> in a browser — the same content as a styled, light/dark page (a test-pyramid
> diagram, layer cards, console-styled commands). This Markdown stays the source
> of truth; keep the two in sync when you edit.

**Contents:** [Mental model](#mental-model) · [Quick start](#quick-start) ·
[Prerequisites](#prerequisites) · [The five layers](#the-five-layers) ·
[Environment flags](#environment-flags) · [What is NOT tested](#what-is-not-tested) ·
[Troubleshooting](#troubleshooting) · [CI](#ci-integration) · [Caveats](#caveats)

---

## Mental model

The suite is a **pyramid**: many fast deterministic tests at the base, a couple of
slow real-world smokes at the top. Pick the lowest layer that can prove what you
changed.

```
        ▲  slow, real, non-deterministic   ── run before a demo / nightly
        │   5. live browser vibe e2e        (real classifier + real GPU)
        │   4. live backend free e2e        (real free generation)
        │  ───────────────────────────────
        │   3. mocked browser e2e           ── every PR (CI-safe)
        │   2. frontend unit (Vitest)
        ▼   1. backend contract (pytest)    ── fastest, most reliable
   many, fast, deterministic
```

Rule of thumb:

- Changed **backend generation / the free provider / pricing**? → layer 1 (+ run 4 before shipping).
- Changed the **run dialog / trigger / free UI**? → layers 2 & 3.
- Preparing a **demo** or want end-to-end confidence? → layers 4 & 5.

---

## Quick start

> **Windows / PowerShell users — read this.** `FLAG=1 cmd` is **bash only**; in
> PowerShell it errors with `CommandNotFoundException`. Use `$env:FLAG=1; cmd`.
> The env var then persists for the whole terminal — clear it with
> `Remove-Item Env:FLAG`. Every gated command below is shown for both shells.

| I want to… | Run |
|---|---|
| Check the free-tier **backend contract** | `python -m pytest tests/utilities/web_modeling_editor/backend/smart_generation/test_free_tier.py` |
| Check the **frontend logic** | `npm run test --workspace=webapp` |
| Check the **free-run UI** in a real browser (mocked, fast) | `npx playwright test smart-gen-free-tier` |
| Prove a **real free generation** works | `python tests/live/test_vibe_free_e2e.py` |
| Watch the **whole vibe pipeline** run for real | *(gated — see layer 5)* |

---

## Prerequisites

**Backend (layers 1, 4)** — Python env with the project installed (`pip install -e .`).
The live test (layer 4) also needs `requests` and network access to the deployed
stack.

**Frontend (layers 2, 3, 5)** — from the frontend root
(`besser/utilities/web_modeling_editor/frontend`):

```bash
npm install                       # once
npx playwright install chromium   # once, for the e2e layers
```

**Free tier must be configured on the server** (layers 4 & 5) — the deployed backend
needs these env vars set (values are a **server secret**, never client-side):

```
BESSER_FREE_LLM_BASE_URL   BESSER_FREE_LLM_TOKEN   BESSER_FREE_LLM_MODEL
```

Verify it's live: `GET /besser_api/smart-gen/config` should return
`free_tier: { available: true, model: "qwen3-coder:30b" }`.

---

## The five layers

### 1 · Backend contract — `test_free_tier.py`
`tests/utilities/web_modeling_editor/backend/smart_generation/test_free_tier.py`

**Asserts:** `provider="free"` needs no key while other providers still require one;
the request **and** `StartEvent` Literals both accept `"free"` (the bug that once
killed every free run at the start frame); local/`:`-tagged models price at `$0` so
a free run can't trip the cost cap; the factory injects the server endpoint + bearer
header and ignores any client-supplied key/model/URL.

```bash
cd BESSER
python -m pytest tests/utilities/web_modeling_editor/backend/smart_generation/test_free_tier.py -q
```
Deterministic, ~3s. **Runs in normal CI.**

### 2 · Frontend logic — Vitest
`packages/webapp/src/main/**/__tests__/` (notably `SmartGenByokDialog.test.tsx`)

**Asserts:** the BYOK dialog's **keyless-approve contract** — clicking "Use the free
model" produces an *approved, keyless* pending trigger (free flag set, no BYOK key
written, no cancel event) — plus the smart-gen trigger/SSE/Redux logic. All mocked.

```bash
cd BESSER/besser/utilities/web_modeling_editor/frontend
npm run test --workspace=webapp
# or just the free-tier dialog file:
npx vitest run src/main/features/smart-generation/components/__tests__/SmartGenByokDialog.test.tsx
```
Deterministic, ~15s. **Runs in normal CI.**

### 3 · Mocked browser e2e — `smart-gen-free-tier.spec.ts`
`packages/webapp/tests/e2e/smart-gen-free-tier.spec.ts`

**Asserts (real browser, mocked network):** injects the trigger over a mocked
WebSocket, mocks `/smart-gen/config` + the `/smart-generate` SSE, then clicks "Use
the free model" and checks the POST body is `provider:'free'` with **no**
`api_key`/`base_url`, the run reaches completion, and the "no API key — did not run"
message **never** shows.

```bash
cd BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp
npx playwright test smart-gen-free-tier --project=chromium
#   --headed  to watch it   ·   --ui  for step-through mode
```
Deterministic, ~15s (auto-starts Vite; needs no backend). **CI-safe.**

### 4 · Live backend free generation — `test_vibe_free_e2e.py`
`tests/live/test_vibe_free_e2e.py`

**Asserts (real free tier):** a class model → a FastAPI app (`main_api.py`,
`pydantic_classes.py`, …); a class model → Rust classes (`.rs` with `struct`s). It
checks the run **produces the right output**, not that it boots.

```bash
# bash
cd BESSER
RUN_LIVE_FREE_E2E=1 python -m pytest tests/live/test_vibe_free_e2e.py -s
```
```powershell
# PowerShell
cd BESSER
$env:RUN_LIVE_FREE_E2E=1; python -m pytest tests/live/test_vibe_free_e2e.py -s
```
Or the **standalone demo runner** — streams each phase live, prints PASS/FAIL, and
self-enables (no env var needed):
```bash
python tests/live/test_vibe_free_e2e.py
```
Gated, ~1–3 min each. **Not in CI.**

### 5 · Live full vibe pipeline — `smart-gen-vibe-live.spec.ts`
`packages/webapp/tests/e2e/smart-gen-vibe-live.spec.ts`

**Asserts (the whole thing, no mocks):** fresh browser → create a project → describe
an app in plain words so the agent **models** a class diagram → **spec-driven
generate** a full app on the **free tier** → the run finishes. This is the
"no-API-key demo path", automated.

```bash
# bash
cd BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp
RUN_LIVE_E2E=1 npx playwright test smart-gen-vibe-live --project=chromium
```
```powershell
# PowerShell
cd BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp
$env:RUN_LIVE_E2E=1; npx playwright test smart-gen-vibe-live --headed   # --headed to watch
Remove-Item Env:RUN_LIVE_E2E                                            # when done
```
Gated, ~5 min. **Nightly smoke, never a CI gate** (see [Caveats](#caveats)).

---

## Environment flags

| Flag | Enables | Default |
|---|---|---|
| `RUN_LIVE_FREE_E2E=1` | Backend live free tests (layer 4) | off — tests skip |
| `RUN_LIVE_E2E=1` | Live browser vibe e2e (layer 5) | off — tests skip |
| `BACKEND_URL` | Host for backend live tests | `https://experimental.besser-pearl.org/besser_api` |
| `LIVE_E2E_BASE_URL` | Host for the browser live test | `https://experimental.besser-pearl.org` |
| `BESSER_FREE_LLM_BASE_URL` / `_TOKEN` / `_MODEL` | Free tier config, **server-side only** | unset → free tier off |

---

## What is NOT tested

Every layer asserts generation **produced the right output** — **not** that the
generated app **boots or runs**. That fidelity check (import-smoke + start the
generated backend, hit `/openapi.json`) is the deferred **Phase-3 boot check**.
It's out of scope on purpose: free-model output often doesn't boot yet, so a boot
assertion would be a *known-flaky* gate. When the boot check lands, it becomes both
a Phase-3 fix-loop input and the fidelity assertion these tests are missing.

---

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `RUN_LIVE_E2E=1 : CommandNotFoundException` | Bash syntax in PowerShell. Use `$env:RUN_LIVE_E2E=1; <cmd>`. |
| Live test reports **`1 skipped`** | The gate flag isn't set. Set `RUN_LIVE_E2E` / `RUN_LIVE_FREE_E2E` (correct shell syntax). |
| `browserType.launch: Executable doesn't exist` | Playwright browser missing. `npx playwright install chromium`. |
| Python live test → **HTTP 403** | Cloudflare bot filter blocks the default UA. The tests already send a browser UA; if you hand-roll a request, add one. |
| Layer 5 fails at the **modeling** step | The agent's classifier occasionally times out ("taking too long, try again"). Re-run; the test already retries once. |
| Layer 5 fails at **"Use the free model"** | Widget+drawer render duplicate buttons and the classifier can lag. The test targets `:visible`/`.last()` and falls back to typing "yes" — re-run if the GPU was cold. |
| First free run is very slow (~60s before phases) | Model reload into VRAM after ~5 min idle. Expected; timeouts are generous. |

---

## CI integration

- **Runs on every PR (deterministic):** layers 1–3. Backend `pytest` + `ruff`
  already run in `.github/workflows/ci.yml`; the Playwright + Vitest layers are the
  natural additions.
- **Nightly / manual (gated):** layers 4–5. Set the `RUN_LIVE_*` flag as a scheduled
  job; treat failures as a **fidelity signal**, not a blocking gate.

---

## Caveats

- **Layer 5 is a nightly smoke, not a CI gate.** It drives the *real* classifier LLM
  and the *real* shared GPU, so it will occasionally go red for infra reasons (GPU
  load, a classifier hiccup) unrelated to any code change. Layers 1–3 are the
  dependable everyday coverage.
- **The "Use-the-free-model does nothing" race is production-build-only.** It doesn't
  reproduce in Vite dev (the resume effect wins the timing) or jsdom (Radix doesn't
  fire `onOpenChange` on a controlled close) — only a production build. It's held by
  the `startingRunRef` guard in `SmartGenByokDialog`; **keep that guard**, and don't
  expect an automated test to catch a regression of it.
- **The free tier is a single shared GPU** (one run at a time) — don't run the live
  layers in parallel.
