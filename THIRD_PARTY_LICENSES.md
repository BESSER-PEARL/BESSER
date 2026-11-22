# Third-Party Licenses

BESSER is distributed under the MIT License (see `LICENSE.md`). This file
lists third-party software that BESSER redistributes or bundles as part of
its build/runtime artifacts, along with its license.

## Alloy Analyzer

- **Component**: [Alloy Analyzer](https://alloytools.org/) 6.2
  (`org.alloytools.alloy.dist.jar`)
- **Used by**: the object diagram generator, the semantic check of the web modeling editor backend
  (`besser/utilities/web_modeling_editor/backend/services/validators/alloy/`), which analyzes the
  specifications written by the Alloy generator (`besser/generators/alloy/`)
- **License**: MIT License
- **Copyright**: Copyright (c) the Alloy project and contributors
- **License text**: <https://github.com/AlloyTools/org.alloytools.alloy/blob/v6.2.0/LICENSE>
- **Source / release**: <https://github.com/AlloyTools/org.alloytools.alloy/releases/tag/v6.2.0>

Alloy Analyzer is downloaded and bundled into the `besser` Docker image at
build time (see `Dockerfile`), where its jar is placed at `/alloy/` alongside
a copy of its MIT license text (`/alloy/LICENSE`), fulfilling the license's
notice requirement for that redistributed copy.

When running BESSER outside Docker, the jar is not vendored in this
repository; it is installed and configured separately by the user (via the
`JAVA_HOME` and `BESSER_ALLOY_JAR` environment variables, see
`docs/source/utilities/object_diagram_generator.rst`) and invoked by BESSER as an
external process at runtime.
