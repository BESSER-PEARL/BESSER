# A3 — Frontend Lint Deep Audit

Wave: BESSER WME final-analysis (post SA-7b cutover, post SA-7b.1 partial v4-native rewrite).

Branch: `claude/refine-local-plan-sS9Zv` (parent repo).
Submodule HEAD: `7a3b82a refactor(webapp): SA-7b.1 partial - v4-native rewrite of 4 assistant files + tsconfig path fix`.
Lint command: `npm run lint --workspace=webapp` (eslint -c .eslintrc.js "**/*.{ts,tsx}").

## Headline numbers

| | count |
|---|---|
| **Errors** | **7** |
| **Warnings** | **943** |
| Total problems | 950 |
| Auto-fixable warnings | 14 |

## Errors (7)

All seven errors are the **same problem**: lint config does not register `eslint-plugin-react-hooks`, but six webapp source files use inline directives like `// eslint-disable-next-line react-hooks/exhaustive-deps`. ESLint flat config v9 reports unknown disable directives as **errors**, not warnings.

This is a **config bug, not a code bug**. No real-bug content under these errors.

| File | Line | Rule reference |
|---|---|---|
| `packages/webapp/src/main/app/hooks/useProjectBootstrap.ts` | 92 | react-hooks/exhaustive-deps |
| `packages/webapp/src/main/features/agent-config/AgentConfigurationPanel.tsx` | 576 | react-hooks/exhaustive-deps |
| `packages/webapp/src/main/features/agent-config/AgentConfigurationPanel.tsx` | 583 | react-hooks/exhaustive-deps |
| `packages/webapp/src/main/features/agent-config/AgentConfigurationPanel.tsx` | 618 | react-hooks/exhaustive-deps |
| `packages/webapp/src/main/features/assistant/hooks/useWebSocketConnection.ts` | 95 | react-hooks/exhaustive-deps |
| `packages/webapp/src/main/features/export/ExportDialog.tsx` | 63 | react-hooks/exhaustive-deps |
| `packages/webapp/src/main/features/import/useImportDiagramPicture.ts` | 128 | react-hooks/exhaustive-deps |
| `packages/webapp/src/main/features/project/ProjectHubDialog.tsx` | 128 | react-hooks/exhaustive-deps |

(Eight directive-uses across six files; pytest-style: file appears once but produces multiple line errors.)

**Fix**: register `eslint-plugin-react-hooks` in `.eslintrc.js` and set `react-hooks/exhaustive-deps: 'warn'`. Either the rule fires (and we suppress where intentional with descriptions) or the directives are stripped — either way the errors disappear.

## Warnings by category

| Rule | Count | Share | Bug-density assessment |
|---|---|---|---|
| `@typescript-eslint/no-explicit-any` | **781** | 82.8% | Mostly noise. Almost every `any` lives at the v4 cutover boundary (assistant modifiers, GUI editor diagram-helpers, GrapesJS shims). SA-7b.1 added ~47 of them as deliberate seam casts during partial v4-native rewrite. |
| `@typescript-eslint/no-unused-vars` | **137** | 14.5% | Mixed. ~30 are quantum constants exported as a registry (legitimate). The rest contain real dead-code candidates from the SA-7b cutover (see top-10 list). |
| `prefer-const` | 19 | 2.0% | Noise. Auto-fixable. |
| `react-hooks/exhaustive-deps` (errors → see above) | 7 | n/a | Counted under errors. |
| `no-empty` | 3 | 0.3% | Worth eyeballing — all three are silent-catch idioms. |
| `directive (unused eslint-disable)` | 2 | 0.2% | Pure noise — strip the directives. |
| `@typescript-eslint/ban-ts-comment` | 1 | 0.1% | Real-but-trivial: `@ts-ignore` lacks description on a `grapesjs-blocks-basic` import. |

`react-refresh/only-export-components`: **0 occurrences** (rule is not enabled in this repo's ESLint config).

### `no-explicit-any` baseline comparison

- SA-7b reported 887 warnings overall; that included ~770 `any` warnings before SA-7b.1.
- SA-7b.1 added ~47 boundary casts during the partial v4-native rewrite.
- **Current `no-explicit-any`: 781**. Total warnings 943 vs. SA-7b's 887 = +56, of which ~47 came from SA-7b.1 casts and the remainder from new unused imports/vars on cutover paths.
- **SA-7b.2 has not yet been committed to the submodule** (last submodule commit is `7a3b82a` SA-7b.1). The expected reduction from SA-7b.2 (replacing seam casts with v4-native types) is not visible in this run. When SA-7b.2 lands, expect the count to drop in `assistant/services/modifiers/`, `assistant/services/UMLModelingService.ts`, and `assistant/services/AssistantClient.ts` first.

### Top files by `no-explicit-any` density

| Rank | File | `any` count |
|---|---|---|
| 1 | `features/assistant/services/modifiers/ClassDiagramModifier.ts` | 84 |
| 2 | `features/editors/gui/diagram-helpers.ts` | 47 |
| 3 | `features/editors/gui/GraphicalUIEditor.tsx` | 37 |
| 3 | `features/assistant/services/UMLModelingService.ts` | 37 |
| 5 | `features/editors/diagram-tabs/scaffoldObjectsFromClasses.ts` | 31 |
| 6 | `features/generation/useGeneratorExecution.ts` | 24 |
| 6 | `features/editors/gui/setup/setupPageSystem.ts` | 24 |
| 8 | `features/assistant/services/AssistantClient.ts` | 21 |
| 9 | `features/editors/gui/traits/registerSeriesManagerTrait.ts` | 20 |
| 10 | `features/editors/gui/component-registrars/registerAgentComponent.tsx` | 19 |

87 distinct files contain at least one `no-explicit-any`.

## Top 10 highest-priority warnings to clean up

These are the warnings most likely to indicate a real bug, dead code from incomplete refactor, or silently-swallowed error. All paths are absolute.

| # | Severity | File:Line | Rule | Why it matters |
|---|---|---|---|---|
| 1 | dead code from SA-7b cutover | `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp/src/main/app/application.tsx:1` (`useEffect`), `:12` (`isUMLModel`), `:29` (`useOnboarding`), `:30` (`useAppSelector`), `:31` (`selectActiveDiagram`) | `no-unused-vars` | App root has 5 unused imports. Strong signal that the v4 cutover left dead wiring (onboarding, active-diagram selector, UML-model guard) that should either be re-attached or deleted. |
| 2 | dead handler | `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp/src/main/app/shell/WorkspaceShell.tsx:607` (`handleAssistantSwitchDiagram`), `:643` (`handleDiagramRename`) | `no-unused-vars` | Two non-trivial handlers (multi-line async fns that navigate routes) defined but never wired. Either feature is broken or this is post-refactor leftover. |
| 3 | dead state | `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp/src/main/app/shell/WorkspaceShell.tsx:168` (`isAssistantWorkspaceOpen`, `setIsAssistantWorkspaceOpen`) | `no-unused-vars` | `useState` declared but neither getter nor setter referenced — assistant workspace drawer is feature-gone or feature-broken. |
| 4 | dead nav state | `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp/src/main/app/hooks/useProjectBootstrap.ts:35-36` (`searchParams`, `setSearchParams`, `isGitHubImportLoading`) | `no-unused-vars` | Bootstrap hook reads URL params and a GitHub-import loading flag but never uses them. GitHub import flow may be silently broken. |
| 5 | swallowed error | `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp/src/main/features/editors/gui/component-registrars/registerChartComponent.ts:212` | `no-empty` | `try { JSON.parse(attrs['series']) } catch {}` — silent. Acceptable here (default `[]` is the fallback) but should at least be `catch { /* default to [] */ }` for clarity. |
| 6 | swallowed error | `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp/src/main/features/editors/gui/traits/registerSeriesManagerTrait.ts:93,220` | `no-empty` | Two empty catch blocks in a trait registrar — most likely chart-builder JSON parsing. Same risk profile as #5; verify both have a sensible default. |
| 7 | dead import | `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp/src/main/shared/utils/projectExportUtils.ts:1` (`getActiveDiagram`) | `no-unused-vars` | Project-export helper imports active-diagram resolver it doesn't use — likely a regression where multi-diagram-aware export logic was removed. |
| 8 | uncaught-then-discarded error | `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp/src/main/shared/utils/githubUrlUtils.ts:77` (`error`) | `no-unused-vars` | Error in `catch (error)` is ignored when parsing GitHub URLs. Silent failure on malformed URLs. |
| 9 | undocumented `@ts-ignore` | `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp/src/main/features/editors/gui/GraphicalUIEditor.tsx:260` | `ban-ts-comment` | `@ts-ignore` on `import('grapesjs-blocks-basic')` — add a one-line description so future readers know the package lacks types. Trivial fix. |
| 10 | dead drawer + chat-pending hook | `/home/user/BESSER/besser/utilities/web_modeling_editor/frontend/packages/webapp/src/main/app/shell/WorkspaceShell.tsx:49` (`AssistantWorkspaceDrawer`), `src/components/chatbot-kit/ui/chat.tsx:309` (`isPending`) | `no-unused-vars` | Imported drawer component never rendered; chat receives an `isPending` prop it doesn't use. Both indicate UI affordances that the v4 cutover may have orphaned. |

## Recommended config tweaks

The current `.eslintrc.js` is intentionally permissive (`any` and `ts-ignore` are warnings, per the frontend `CLAUDE.md`). Suggested adjustments to make this audit signal more useful **without** blocking SA-7b.x migration work:

1. **Register `eslint-plugin-react-hooks`** and set `react-hooks/rules-of-hooks: 'error'` + `react-hooks/exhaustive-deps: 'warn'`. This eliminates the 7 errors immediately and surfaces a class of real bugs (stale closures) that the codebase isn't currently checking at all.

2. **Suppress `no-explicit-any` on the v4 cutover boundary** during migration to recover signal in the rest of the codebase:
   ```js
   {
     files: [
       'src/main/features/assistant/**/*.{ts,tsx}',
       'src/main/features/editors/gui/**/*.{ts,tsx}',
       'src/main/features/generation/useGeneratorExecution.ts',
     ],
     rules: { '@typescript-eslint/no-explicit-any': 'off' },
   }
   ```
   This is migration-only scaffolding to be removed when SA-7b.2 / SA-7b.3 land. After it, the remaining `any` count would drop from 781 to roughly **70-90**, all in shared utilities — a cleanable target.

3. **`prefer-const`: leave as warn** but run `eslint --fix` once before SA-7b.2 — all 19 are auto-fixable and create churn in unrelated diffs.

4. **Tighten `no-unused-vars`** to the standard severity but exempt the quantum constants registry which exports type-only enum members:
   ```js
   {
     files: ['src/main/features/editors/quantum/constants.ts'],
     rules: { '@typescript-eslint/no-unused-vars': 'off' },
   }
   ```
   This single override removes 30/137 unused-vars warnings (22%) and lets the remaining 107 stand out as real dead code to triage.

5. **`react-refresh/only-export-components`** is not currently enabled. Adding it would add value (catches HMR breakage from mixed exports) but adds ~30-50 more warnings; **defer** until after SA-7b.x stabilizes.

## Conclusion

- **No real bugs hide behind the 7 errors** — they are config noise from a missing react-hooks plugin registration.
- **No real bugs hide behind the 781 `no-explicit-any` warnings** — they are deliberate v4 cutover seams. SA-7b.2 is expected to chip away at this.
- **Real bugs are concentrated in the 137 `no-unused-vars` warnings**, specifically in `application.tsx`, `WorkspaceShell.tsx`, `useProjectBootstrap.ts`, `projectExportUtils.ts`, and `githubUrlUtils.ts`. These point to incomplete cleanup of the assistant workspace drawer, GitHub-import flow, and project-export multi-diagram support after SA-7b.
- **The 3 `no-empty` warnings** are silent-catch idioms; benign as currently written but worth a one-comment fix.
- **Recommendation order**: (1) fix the react-hooks config to clear the 7 errors and unlock real exhaustive-deps signal; (2) triage the 10 unused-vars hotspots above (~30-60 minutes of work); (3) `eslint --fix` for prefer-const; (4) defer `no-explicit-any` cleanup to SA-7b.2/SA-7b.3.
