/**
 * Vitest config for the A9 migration corpus test.
 *
 * `node_modules` is symlinked from the frontend submodule so Node can
 * resolve `vitest/config`, `@besser/wme`, etc.
 */
import { defineConfig } from "vitest/config"
import { resolve } from "path"

const FRONTEND = resolve(
  __dirname,
  "../../besser/utilities/web_modeling_editor/frontend",
)

export default defineConfig({
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: [resolve(__dirname, "setup.ts")],
    include: [resolve(__dirname, "*.test.ts")],
  },
  resolve: {
    alias: {
      "@besser/wme": resolve(FRONTEND, "packages/library/dist/index.js"),
    },
  },
})
