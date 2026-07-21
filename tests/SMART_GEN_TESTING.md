# Testing the Spec-Driven Agent & the keyless Free tier

How to test "describe an app → generate code", including the **keyless free tier**
(server-hosted `qwen3-coder`, no API key). Tests span **three** repos — this one
(`BESSER`, backend), the frontend submodule, and `modeling-agent` — so this doc
is the single index for all of it.

> **Windows note (read this first):** the `VAR=value cmd` form is **bash only**.
> In PowerShell it errors with `CommandNotFoundException`. Every gated command
> below is given for **both** shells. In PowerShell, `$env:FLAG=1` stays set for
> the rest of the terminal session — clear it with `Remove-Item Env:FLAG`.

---

## The layers (what runs where, and how much to trust it)

| # | Test | Proves | Speed / reliability |
|---|---|---|---|
| 1 | `tests/utilities/web_modeling_editor/backend/smart_generation/test_free_tier.py` | The free-tier **contract**: `provider="free"` needs no key; `StartEvent`/request Literals accept `"free"`; local models price `$0`; the factory injects the server endpoint + bearer header. | Fast, deterministic — **normal CI** |
| 2 | `packages/webapp/src/main/**/__tests__/` (Vitest) | Frontend logic in isolation (mocked): the BYOK dialog's **keyless-approve** contract, the smart-gen trigger/SSE/Redux. | Fast, deterministic — **normal CI** |
| 3 | `packages/webapp/tests/e2e/smart-gen-free-tier.spec.ts` (Playwright, **mocked**) | The free-run **UI wiring** in a real browser: click "Use the free model" → POST is `provider:'free'` with no `api_key`/`base_url`, run completes, no "no API key" message. WS/SSE/config are mocked. | ~15s, deterministic — **CI-safe** |
| 4 | `tests/live/test_vibe_free_e2e.py` (Python, **live**) | The whole backend generation pipeline on the **real free tier**: a model → a FastAPI app; a model → Rust classes. Asserts output is **produced**. | ~1–3 min each, gated |
| 5 | `packages/webapp/tests/e2e/smart-gen-vibe-live.spec.ts` (Playwright, **live**) | The **full vibe pipeline, no mocks**: fresh browser → vibe-model a diagram from plain words → spec-driven generate on the free tier → asserts it finishes. | ~5 min, gated, **flaky — nightly only** |

Related (not free-tier-specific): `modeling-agent/tests/live/test_nl_generation_scenarios.py`
(NL routing matrix) and `tests/live/test_generation_workflows_live.py` (deterministic
generator output). See `packages/webapp/tests/e2e/README.md` for the full e2e catalogue.

**Scope on purpose:** every test here asserts generation *produced the right kind
of output* — **not** that the produced app *boots/runs*. That boot/run fidelity
check (import-smoke + start the generated backend) is the deferred **Phase-3 boot
check**; free-model output often doesn't boot yet, so we don't want a known-flaky
fidelity gate blocking these plumbing checks.

---

## How to run

### 1 — Backend contract (deterministic)
```bash
cd BESSER
python -m pytest tests/utilities/web_modeling_editor/backend/smart_generation/test_free_tier.py -q
```

### 2 — Frontend Vitest (deterministic)
```bash
cd BESSER/besser/utilities/web_modeling_editor/frontend
npm run test --workspace=webapp
# or just the free-tier dialog tests:
npx vitest run src/main/features/smart-generation/components/__tests__/SmartGenByokDialog.test.tsx
```

### 3 — Frontend e2e, mocked (deterministic, CI-safe)
```bash
cd BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp
npx playwright test smart-gen-free-tier --project=chromium
# watch it: add --headed, or --ui for step-through mode
```
Auto-starts Vite on :8080; needs no backend (everything is mocked).

### 4 — Backend live free generation (gated, ~1–3 min each)
Needs the deployed stack + the free tier configured (`BESSER_FREE_LLM_*`), and the
`requests` package.

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
Or the **standalone demo runner** (streams each phase live, prints PASS/FAIL, no
env var needed — it self-enables):
```bash
python tests/live/test_vibe_free_e2e.py
```

### 5 — Full live vibe e2e (gated, ~5 min, nightly)
The complete "describe → model → generate on the free tier" flow in a real browser
against the deployed stack.

```bash
# bash
cd BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp
RUN_LIVE_E2E=1 npx playwright test smart-gen-vibe-live --project=chromium
```
```powershell
# PowerShell
cd BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp
$env:RUN_LIVE_E2E=1; npx playwright test smart-gen-vibe-live --headed   # --headed to watch
# when done:  Remove-Item Env:RUN_LIVE_E2E
```

---

## Environment flags

| Flag | Enables | Default |
|---|---|---|
| `RUN_LIVE_FREE_E2E=1` | Backend live free-tier tests (layer 4) | off (skipped) |
| `RUN_LIVE_E2E=1` | Full live browser vibe e2e (layer 5) | off (skipped) |
| `BACKEND_URL` / `LIVE_E2E_BASE_URL` | Point live tests at another host | `https://experimental.besser-pearl.org…` |
| `BESSER_FREE_LLM_BASE_URL` / `_TOKEN` / `_MODEL` | Configure the free tier **on the server** (never client-side) | unset = free tier off |

---

## Caveats & gotchas (learned the hard way)

- **The full live browser test (layer 5) is a nightly smoke, not a CI gate.** It
  drives the *real* classifier LLM and *real* shared GPU, so it goes red for
  reasons unrelated to a code change (GPU load, a classifier timeout). Use layers
  1–3 as everyday coverage; run 4–5 before a demo / on a schedule.
- **The "does nothing on Use-the-free-model" race is production-build-only.** It
  does not reproduce in Vite dev (the resume effect wins) or jsdom (Radix doesn't
  fire `onOpenChange` on a controlled close) — only a production build. It is held
  by the `startingRunRef` guard in `SmartGenByokDialog`; **do not remove it**, and
  don't expect an automated test to catch a regression of it.
- **Live browser flow quirks:** a fresh context inconsistently lands in a
  "Describe Your App" wizard vs. the empty editor (the test handles both); the
  widget + drawer render **duplicate** composers/buttons (target `:visible` /
  `.last()`); the Run→dialog click is flaky (the test falls back to typing "yes").
  Cloudflare does **not** block headless Playwright.
- **First free call after ~5 min idle adds ~60s** while the model reloads into
  VRAM. Timeouts here are generous on purpose.
- The free tier is a **single shared GPU** (one run at a time) — don't run the
  live tests in parallel.
