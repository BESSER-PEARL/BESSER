## TypeScript Guidelines
- TypeScript code only appears inside the editor submodule at `besser/utilities/web_modeling_editor/`.
- Match the conventions from the upstream `BESSER_WME_standalone` project: strict compiler options, ESLint, and Prettier.
- Prefer `interface` declarations, `const` bindings, and explicit return types.
- Keep component logic lean—delegate backend interactions to the Python services exposed by this repository.
- Run the submodule's npm scripts (lint, test, build) whenever you modify its sources.
